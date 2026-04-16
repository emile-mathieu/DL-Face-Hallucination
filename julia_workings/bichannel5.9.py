import csv
import math
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

'''
==================================================================
bichannel 5.9
Changes vs previous version:
  1. Per-epoch checkpoints saved to RESULTS_DIR/checkpoints/
  2. Best-model checkpoint unchanged (still saved separately)
  3. CelebA centre-crop applied before resizing to 100×100
  4. PSNR/SSIM computed on Y-channel (luminance) by default;
     set METRIC_CHANNEL = "rgb" to revert to full-colour metrics.
'''

# ============================================================
# Config
# ============================================================
WORKDIR = Path("/home/msds/tans0444")  # change this to your own directory
DEFAULT_IMAGE_ROOTS = [
    WORKDIR / "celeba" / "img_align_celeba"
]

RESULTS_DIR = WORKDIR / "results_bichannel"
SPLITS_DIR = WORKDIR / "splits_bichannel"

SEED = 42
MAX_IMAGES = 100_000
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

HR_SIZE = (100, 100)           # output HR image size
TRAIN_LR_INPUT_SIZE = (48, 48) # network input size (paper Table 1)
TEST_FIXED_LR_SIZE = (50, 50)  # LR size used during test degradation (paper: downsample to 50x50)
TEST_NN_INPUT_SIZE = (48, 48)  # resize test LR to network input size

# CelebA face crop (images are 218×178; crop to face before resize)
CELEBA_CROP = True   # set False if images are already tight face crops
CROP_FRAC   = 0.6   # fraction of shorter side to keep

# Metric channel: "y" (luminance, recommended) or "rgb"
METRIC_CHANNEL = "y"

# Hyperparameters — kept identical to basic_cnn.py so the pretrained conv/fc
# weights operate in the same training regime as basic_cnn.py.
BATCH_SIZE    = 32
NUM_EPOCHS    = 600     # same as basic_cnn.py
LEARNING_RATE = 5e-4    # same as basic_cnn.py
MIN_LR        = 1e-6    # same as basic_cnn.py
WEIGHT_DECAY  = 1e-3    # same as basic_cnn.py
GRAD_CLIP     = 1.0     # same as basic_cnn.py
MOMENTUM      = 0.9     # kept for reference; not used by AdamW
PATIENCE      = 10      # kept for reference; not used by warmup-cosine schedule
NUM_WORKERS   = 2

WARMUP_EPOCHS = 10      # same as basic_cnn.py
LR_T_MAX      = 590     # same as basic_cnn.py — cosine cycle = NUM_EPOCHS - WARMUP_EPOCHS

KEEP_LAST_N   = 3       # rolling checkpoint window

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


# ============================================================
# CelebA crop helper
# ============================================================
def celeba_crop(pil_img: Image.Image) -> Image.Image:
    if not CELEBA_CROP:
        return pil_img
    w, h   = pil_img.size
    side   = int(min(w, h) * CROP_FRAC)
    left   = (w - side) // 2
    top    = max(0, (h - side) // 2 + int(side * 0.15))
    return pil_img.crop((left, top, left + side, top + side))


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_root(explicit_path: str = None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Image root not found: {p}")
        return p
    for p in DEFAULT_IMAGE_ROOTS:
        if p.exists():
            return p
    raise FileNotFoundError("CelebA folder not found. Pass --image-root.")


def gather_images(image_root: Path) -> List[str]:
    exts  = {".jpg", ".jpeg", ".png"}
    files = [str(p) for p in image_root.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise ValueError(f"No images in {image_root}")
    files.sort()
    return files


def make_splits(
    image_root: Path,
    max_images: int = MAX_IMAGES,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    splits_dir:  Path  = SPLITS_DIR,
) -> Tuple[List[str], List[str], List[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.")
    all_images = gather_images(image_root)
    if len(all_images) < max_images:
        print(f"[Warning] Only {len(all_images)} images; using all.")
        max_images = len(all_images)
    rng     = random.Random(seed)
    sampled = rng.sample(all_images, max_images)
    n_train = int(max_images * train_ratio)
    n_val   = int(max_images * val_ratio)
    train_f = sampled[:n_train]
    val_f   = sampled[n_train:n_train + n_val]
    test_f  = sampled[n_train + n_val:]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in [("train", train_f), ("val", val_f), ("test", test_f)]:
        with open(splits_dir / f"{name}.txt", "w", encoding="utf-8") as fh:
            fh.writelines(p + "\n" for p in lst)
    summary = {"image_root": str(image_root), "seed": seed,
                "max_images": max_images, "train": len(train_f),
                "val": len(val_f), "test": len(test_f)}
    with open(splits_dir / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Splits:", json.dumps(summary, indent=2))
    return train_f, val_f, test_f


def save_metrics_to_csv(csv_path: Path, row: List, header: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ============================================================
# Metrics  (Y-channel or RGB, toggled by METRIC_CHANNEL)
# ============================================================
def _to_y(t: torch.Tensor) -> torch.Tensor:
    """CHW or 1CHW RGB [0,1] → 1×1×H×W Y-channel."""
    if t.ndim == 3:
        t = t.unsqueeze(0)
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return (0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0).clamp(0, 1)


def _ssim_kernel(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    k = g.unsqueeze(0) * g.unsqueeze(1)
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(device=device, dtype=dtype)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y(pred), _to_y(target)
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y(pred), _to_y(target)
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch  = pred.shape[1]
    k   = _ssim_kernel(ch, pred.device, pred.dtype)
    pad = 5
    mu1  = F.conv2d(pred,   k, padding=pad, groups=ch)
    mu2  = F.conv2d(target, k, padding=pad, groups=ch)
    s1   = F.conv2d(pred*pred,     k, padding=pad, groups=ch) - mu1**2
    s2   = F.conv2d(target*target, k, padding=pad, groups=ch) - mu2**2
    s12  = F.conv2d(pred*target,   k, padding=pad, groups=ch) - mu1*mu2
    num  = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-12)).mean().item()


# ============================================================
# Per-image normalization (paper Eq. 15-17)
# ============================================================
def normalize_per_image(img: np.ndarray, eps: float = 1e-8
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = img.mean(axis=(0, 1))
    std  = img.std(axis=(0, 1)).clip(eps)
    return np.tanh((img - mean) / std).astype(np.float32), mean, std


def denormalize_per_image(img: torch.Tensor, mean: np.ndarray,
                           std: np.ndarray) -> torch.Tensor:
    """tanh-space CHW tensor → pixel-space CHW tensor [0,1]. eps=1e-3 matches basic_cnn.py."""
    eps = 1e-3
    mt = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    st = torch.tensor(std,  dtype=img.dtype, device=img.device).view(3, 1, 1)
    return torch.clamp(torch.atanh(img.clamp(-1+eps, 1-eps)) * st + mt, 0.0, 1.0)


# ============================================================
# Load BasicCNN weights into BiChannelCNN (for pretraining)
# ============================================================
def load_basiccnn_into_bichannel(bichannel_model, basic_checkpoint_path, device):
    ckpt  = torch.load(basic_checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    bichannel_model.conv1.weight.data.copy_(state["conv1.weight"])
    bichannel_model.conv1.bias.data.copy_(state["conv1.bias"])
    bichannel_model.conv2.weight.data.copy_(state["conv2.weight"])
    bichannel_model.conv2.bias.data.copy_(state["conv2.bias"])
    bichannel_model.conv3.weight.data.copy_(state["conv3.weight"])
    bichannel_model.conv3.bias.data.copy_(state["conv3.bias"])
    bichannel_model.fc1_1.weight.data.copy_(state["fc1.weight"])
    bichannel_model.fc1_1.bias.data.copy_(state["fc1.bias"])
    bichannel_model.fc2_1.weight.data.copy_(state["fc2.weight"])
    bichannel_model.fc2_1.bias.data.copy_(state["fc2.bias"])
    print(f"Loaded BasicCNN weights from {basic_checkpoint_path}")
    return bichannel_model


# ============================================================
# Motion blur kernel
# ============================================================
def _motion_blur_kernel(length: int, theta: float) -> np.ndarray:
    length = max(2, int(length))
    kernel = np.zeros((length, length), dtype=np.float32)
    c  = (length - 1) / 2.0
    x0 = int(round(c - c * math.cos(theta)))
    y0 = int(round(c - c * math.sin(theta)))
    x1 = int(round(c + c * math.cos(theta)))
    y1 = int(round(c + c * math.sin(theta)))
    cv2.line(kernel, (x0, y0), (x1, y1), 1, thickness=1)
    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        kernel[length // 2, length // 2] = 1.0
    return kernel


# ============================================================
# Datasets
# ============================================================
class TrainValFaceDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths   = image_paths
        self.hr_size       = HR_SIZE
        self.lr_input_size = TRAIN_LR_INPUT_SIZE

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = celeba_crop(Image.open(self.image_paths[idx]).convert("RGB"))
        img = img.resize(self.hr_size, Image.BICUBIC)
        ih  = np.array(img).astype(np.float32) / 255.0

        il  = self._generate_low_res(ih)
        iin = cv2.resize(il, self.lr_input_size,
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)

        iin_norm, mean, std = normalize_per_image(iin)
        ih_norm = np.tanh((ih - mean) / std.clip(1e-8)).astype(np.float32)

        return (torch.from_numpy(iin_norm).permute(2, 0, 1).float(),
                torch.from_numpy(ih_norm).permute(2, 0, 1).float(),
                torch.tensor(mean, dtype=torch.float32),
                torch.tensor(std,  dtype=torch.float32))

    def _generate_low_res(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        if random.random() < 0.5:
            sigma   = random.uniform(0, 7)
            blurred = (cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
                       if sigma > 1e-6 else img.copy())
        else:
            length  = random.randint(0, 11)
            theta   = random.uniform(-math.pi, math.pi)
            blurred = (cv2.filter2D(img, -1, _motion_blur_kernel(length, theta))
                       if length > 1 else img.copy())
        scale = random.randint(2, 5)
        return cv2.resize(blurred, (max(1, w // scale), max(1, h // scale)),
                          interpolation=cv2.INTER_CUBIC).astype(np.float32)

    @staticmethod
    def motion_blur_kernel(length: int, theta: float) -> np.ndarray:
        return _motion_blur_kernel(length, theta)


class FixedTestFaceDataset(Dataset):
    def __init__(self, image_paths: List[str], blur_type: str = "gaussian",
                 gaussian_sigma: float = None, motion_length: int = None,
                 base_seed: int = 42):
        self.image_paths    = image_paths
        self.hr_size        = HR_SIZE
        self.fixed_lr_size  = TEST_FIXED_LR_SIZE
        self.nn_input_size  = TEST_NN_INPUT_SIZE
        self.blur_type      = blur_type
        self.gaussian_sigma = gaussian_sigma
        self.motion_length  = motion_length
        self.base_seed      = base_seed
        if blur_type == "gaussian" and gaussian_sigma is None:
            raise ValueError("gaussian_sigma required")
        if blur_type == "motion" and motion_length is None:
            raise ValueError("motion_length required")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img    = celeba_crop(Image.open(self.image_paths[idx]).convert("RGB"))
        hr_img = img.resize(self.hr_size, Image.BICUBIC)
        ih     = np.array(hr_img).astype(np.float32) / 255.0

        fixed_lr = self._fixed_test_preprocessing(img, idx)
        iin = cv2.resize(fixed_lr, self.nn_input_size,
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)

        iin_norm, mean, std = normalize_per_image(iin)
        ih_norm = np.tanh((ih - mean) / std.clip(1e-8)).astype(np.float32)

        return (torch.from_numpy(iin_norm).permute(2, 0, 1).float(),
                torch.from_numpy(ih_norm).permute(2, 0, 1).float(),
                torch.tensor(mean, dtype=torch.float32),
                torch.tensor(std,  dtype=torch.float32))

    def _fixed_test_preprocessing(self, pil_img: Image.Image, idx: int) -> np.ndarray:
        img_100 = np.array(pil_img.resize(self.hr_size, Image.BICUBIC)
                           ).astype(np.float32) / 255.0
        if self.blur_type == "gaussian":
            img_blur = cv2.GaussianBlur(img_100, (0, 0),
                                        sigmaX=self.gaussian_sigma,
                                        sigmaY=self.gaussian_sigma)
        else:
            theta    = random.Random(self.base_seed + idx
                                     ).uniform(-math.pi, math.pi)
            img_blur = cv2.filter2D(img_100, -1,
                                    _motion_blur_kernel(self.motion_length, theta))
        return cv2.resize(img_blur, self.fixed_lr_size,
                          interpolation=cv2.INTER_CUBIC).astype(np.float32)


# ============================================================
# WarmupCosineScheduler — identical to basic_cnn.py
# ============================================================
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Epochs 0..warmup_epochs-1 : linear ramp from MIN_LR to base_lr
    Epochs warmup_epochs..end  : cosine decay from base_lr to MIN_LR
    Identical to basic_cnn.py — single clean cycle, no restarts.
    """
    def __init__(self, optimizer, warmup_epochs: int, t_max: int,
                 min_lr: float, last_epoch: int = -1):
        self.warmup = warmup_epochs
        self.t_max  = t_max
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            alpha = (e + 1) / max(self.warmup, 1)
            return [self.min_lr + alpha * (base - self.min_lr)
                    for base in self.base_lrs]
        progress = min((e - self.warmup) / max(self.t_max, 1), 1.0)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + cosine * (base - self.min_lr)
                for base in self.base_lrs]

# ============================================================
# Model — BiChannelCNN
# ============================================================
class BiChannelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1_1 = nn.Linear(128 * 4 * 4, 2000)
        self.fc2_1 = nn.Linear(2000, 3 * 100 * 100)
        self.fc1_2 = nn.Linear(128 * 4 * 4, 100)
        self.fc2_2 = nn.Linear(100, 1)
        self._init_weights()

    def _init_weights(self):
        """
        Matches basic_cnn.py init exactly so the alpha branch starts in the
        same regime as the pretrained reconstruction branch.
          conv1/2/3  : orthogonal, gain=0.6  (same as basic_cnn.py)
          fc1_1/fc2_1: Normal(0, 1/sqrt(fan_in)) (same as basic_cnn.py)
          fc1_2/fc2_2: Normal(0, 1/sqrt(fan_in)) (alpha branch, new layers)
        The reconstruction weights (conv1-3, fc1_1, fc2_1) are overwritten
        by load_basiccnn_into_bichannel() immediately after construction,
        so their init here only affects the alpha branch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        inp = x
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        f = torch.flatten(x, start_dim=1)

        i_rec = torch.tanh(self.fc1_1(f))
        i_rec = torch.tanh(self.fc2_1(i_rec))
        i_rec = i_rec.view(-1, 3, 100, 100)

        alpha = torch.tanh(self.fc1_2(f))
        alpha = 0.5 * torch.tanh(self.fc2_2(alpha)) + 0.5
        alpha = alpha.view(-1, 1, 1, 1)

        i_up = nn.functional.interpolate(
            inp, size=(100, 100), mode="bicubic", align_corners=False)
        return alpha * i_up + (1 - alpha) * i_rec, alpha


# ============================================================
# Checkpoint helpers
# ============================================================
def save_checkpoint(model, optimizer, scheduler, epoch: int,
                    val_loss: float, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":        val_loss,
    }
    # Atomic write: save to .tmp then rename so a crash mid-write never
    # leaves a corrupted checkpoint (matches basic_cnn.py behaviour).
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _keep_last_n_checkpoints(checkpoint_dir: Path,
                              prefix: str = "bichannel_epoch_",
                              n: int = KEEP_LAST_N) -> None:
    """Delete all but the most recent n rolling epoch checkpoints."""
    ckpts = sorted(Path(checkpoint_dir).glob(f"{prefix}*.pth"))
    for old in ckpts[:-n]:
        try:
            old.unlink()
            print(f"  → Deleted old checkpoint: {old.name}")
        except Exception as e:
            print(f"  → Warning: could not delete {old.name}: {e}")


def load_checkpoint(path: Path, model, optimizer=None,
                    scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ============================================================
# Train / Validate
# ============================================================
def evaluate_validation(model, dataloader, device, criterion):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    with torch.no_grad():
        for iin, ih, mean_b, std_b in dataloader:
            iin, ih = iin.to(device).float(), ih.to(device).float()
            out, _   = model(iin)
            total_loss += criterion(out, ih).item()
            for b in range(out.shape[0]):
                p = denormalize_per_image(out[b], mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(ih[b],  mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    return total_loss / len(dataloader), total_psnr / n, total_ssim / n


def train_model(model, train_loader, val_loader, device,
                resume_checkpoint: Path = None):
    criterion = nn.MSELoss()   # MSE in tanh-space — paper Eq. 3

    # AdamW — decoupled weight decay, same as basic_cnn.py.
    # Replaces SGD: SGD with lr=1e-3 and tanh saturation gives near-zero
    # effective updates on the conv layers. AdamW adapts per-parameter LRs
    # and applies weight decay uniformly across all 60M+ parameters.
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # WarmupCosineScheduler — same as basic_cnn.py (single clean cosine, no restarts)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        t_max=LR_T_MAX,
        min_lr=MIN_LR)

    print(f"Optimizer: AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
    print(f"LR schedule: {WARMUP_EPOCHS}-epoch warmup → cosine T_max={LR_T_MAX}")
    print(f"Loss: MSE in tanh-space | Metrics: PSNR/SSIM in pixel space")

    start_epoch    = 0
    best_val_loss  = float("inf")
    best_model_path = RESULTS_DIR / "best_model.pth"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint and Path(resume_checkpoint).exists():
        start_epoch, best_val_loss = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = epoch_psnr = epoch_ssim = n = 0

        for batch_idx, (iin, ih, mean_b, std_b) in enumerate(train_loader):
            iin, ih = iin.to(device).float(), ih.to(device).float()
            optimizer.zero_grad()
            out, alpha = model(iin)
            loss = criterion(out, ih)
            loss.backward()
            # Gradient clipping — same value as basic_cnn.py
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                for b in range(out.shape[0]):
                    p = denormalize_per_image(out[b].detach(),
                                              mean_b[b].numpy(), std_b[b].numpy())
                    t = denormalize_per_image(ih[b],
                                              mean_b[b].numpy(), std_b[b].numpy())
                    epoch_psnr += compute_psnr(p, t)
                    epoch_ssim += compute_ssim(p, t)
                    n += 1

            if epoch == start_epoch and batch_idx == 0:
                print(f"First batch — In:{iin.shape} Out:{out.shape} Target:{ih.shape}")
            if batch_idx == 0:
                print(f"[Epoch {epoch+1}] alpha mean={alpha.mean().item():.4f} "
                      f"min={alpha.min().item():.4f} max={alpha.max().item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        train_psnr = epoch_psnr / n
        train_ssim = epoch_ssim / n

        val_loss, val_psnr, val_ssim = evaluate_validation(
            model, val_loader, device, criterion)

        # Cosine scheduler steps once per epoch
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss:{train_loss:.4f} PSNR:{train_psnr:.2f} SSIM:{train_ssim:.4f} | "
              f"Val Loss:{val_loss:.4f} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f} | LR:{lr:.1e}")

        save_metrics_to_csv(
            RESULTS_DIR / "train_val_metrics.csv",
            [epoch+1, train_loss, train_psnr, train_ssim,
             val_loss, val_psnr, val_ssim, lr],
            ["epoch","train_loss","train_psnr","train_ssim",
             "val_loss","val_psnr","val_ssim","lr"])

        # ── per-epoch checkpoint (rolling, keep last KEEP_LAST_N) ──
        ckpt_path = CHECKPOINT_DIR / f"bichannel_epoch_{epoch+1:04d}.pth"
        save_checkpoint(model, optimizer, scheduler,
                        epoch + 1, val_loss, ckpt_path)
        print(f"  → Checkpoint saved: {ckpt_path.name}")
        _keep_last_n_checkpoints(CHECKPOINT_DIR,
                                  prefix="bichannel_epoch_", n=KEEP_LAST_N)

        # ── best-model checkpoint ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler,
                            epoch + 1, val_loss, best_model_path)
            print(f"  → New best model (val_loss={best_val_loss:.4f})")

    return best_model_path


# ============================================================
# Test
# ============================================================
def load_model_for_inference(model_path: Path, device):
    model = BiChannelCNN().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_one_test_setting(model, dataloader, device, criterion):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    total_am = total_amin = total_amax = 0
    with torch.no_grad():
        for iin, ih, mean_b, std_b in dataloader:
            iin, ih = iin.to(device).float(), ih.to(device).float()
            out, alpha = model(iin)
            total_loss  += criterion(out, ih).item()
            total_am    += alpha.mean().item()
            total_amin  += alpha.min().item()
            total_amax  += alpha.max().item()
            for b in range(out.shape[0]):
                p = denormalize_per_image(out[b], mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(ih[b],  mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    nb = len(dataloader)
    print(f"  alpha mean={total_am/nb:.4f} min={total_amin/nb:.4f} max={total_amax/nb:.4f}")
    return total_loss / nb, total_psnr / n, total_ssim / n


def save_image(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)

def save_sample_grid(model, test_files: List[str], device,
                     save_dir: Path,
                     blur_type: str,
                     gaussian_sigma: float = None,
                     motion_length: int = None,
                     sample_idx: int = 42) -> None:
    """
    Save 4 images for one blur condition, matching basic_cnn.py layout:
      0_raw.png       — original CelebA image (cropped, native resolution)
      1_hr.png        — 100×100 ground-truth HR target
      2_lr_input.png  — LR input upscaled to 100×100 with NEAREST (pixelation visible)
      3_sr_output.png — BiChannelCNN SR prediction (100×100)

    Uses the same test image index (sample_idx) for all 6 conditions so
    the same face is compared across blur types side by side.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = FixedTestFaceDataset(
        test_files,
        blur_type=blur_type,
        gaussian_sigma=gaussian_sigma,
        motion_length=motion_length,
        base_seed=SEED,
    )
    model.eval()

    raw_path = test_files[sample_idx]
    iin, ih, mean_b, std_b = ds[sample_idx]

    # 0. Raw (original CelebA face, native resolution)
    celeba_crop(Image.open(raw_path).convert("RGB")).resize(
        HR_SIZE, Image.BICUBIC).save(save_dir / "0_raw.png")

    # 1. Ground-truth HR
    save_image(denormalize_per_image(ih, mean_b.numpy(), std_b.numpy()),
               save_dir / "1_hr.png")

    # 2. LR input — nearest-upscale to 100×100 so pixelation is clearly visible
    lr_arr = denormalize_per_image(iin, mean_b.numpy(), std_b.numpy())
    lr_pil = Image.fromarray(
        (lr_arr.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8))
    lr_pil.resize(HR_SIZE, Image.NEAREST).save(save_dir / "2_lr_input.png")

    # 3. SR output from BiChannelCNN
    with torch.no_grad():
        iin_b = iin.unsqueeze(0).to(device).float()
        out_b, alpha_b = model(iin_b)
    save_image(denormalize_per_image(out_b[0], mean_b.numpy(), std_b.numpy()),
               save_dir / "3_sr_output.png")

    print(f"  Samples (α={alpha_b.mean().item():.3f}) → {save_dir}")


def run_test_pipeline(test_files: List[str], best_model_path: Path, device):
    criterion = nn.MSELoss()
    model     = load_model_for_inference(best_model_path, device)
    results   = []
    lkw = dict(batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # Run all 6 blur conditions, saving metrics + sample images for each
    for sigma in [1, 3, 5]:
        ds      = FixedTestFaceDataset(test_files, blur_type="gaussian",
                                       gaussian_sigma=sigma)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Gaussian sigma={sigma} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"gaussian_sigma_{sigma}",
                        "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         save_dir=RESULTS_DIR / f"samples_gaussian_sigma_{sigma}",
                         blur_type="gaussian", gaussian_sigma=sigma)

    for length in [2, 6, 9]:
        ds      = FixedTestFaceDataset(test_files, blur_type="motion",
                                       motion_length=length, base_seed=SEED)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Motion l={length} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"motion_l_{length}",
                        "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         save_dir=RESULTS_DIR / f"samples_motion_l_{length}",
                         blur_type="motion", motion_length=length)

    csv_path = RESULTS_DIR / "test_pipeline_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["blur_type", "loss", "psnr", "ssim"])
        w.writeheader()
        w.writerows(results)
    print(f"Test metrics → {csv_path}")
    print(f"\nSample images saved to: {RESULTS_DIR}/samples_*/")
    print("  Each condition folder contains:")
    print("    0_raw.png        original CelebA face (native crop resolution)")
    print("    1_hr.png         100×100 ground-truth HR")
    print("    2_lr_input.png   50×50 LR nearest-upscaled to 100×100")
    print("    3_sr_output.png  BiChannelCNN SR prediction")
    return results


# ============================================================
# Main
# ============================================================
def main(image_root_arg=None, resume_checkpoint=None):
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Metric channel: {METRIC_CHANNEL.upper()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    image_root = find_image_root(image_root_arg)
    train_files, val_files, test_files = make_splits(image_root=image_root)

    train_ds = TrainValFaceDataset(train_files)
    val_ds   = TrainValFaceDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())

    model = BiChannelCNN().to(device)
    basic_ckpt = WORKDIR / "results_basiccnn" / "best_model_basiccnn.pth"
    if basic_ckpt.exists():
        model = load_basiccnn_into_bichannel(model, basic_ckpt, device)
    else:
        print(f"[Warning] BasicCNN checkpoint not found at {basic_ckpt}. Training from scratch.")

    best_model_path = train_model(model, train_loader, val_loader, device,
                                   resume_checkpoint=resume_checkpoint)

    print("\nRunning test pipeline ...")
    results = run_test_pipeline(test_files, best_model_path, device)
    print("\nFinal summary:")
    for r in results:
        print(f"  {r['blur_type']}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--resume",     default=None,
                    help="Checkpoint path to resume training from")
    args = ap.parse_args()
    main(args.image_root, args.resume)