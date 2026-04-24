"""
basic5.9
BasicCNN — Learning Face Hallucination in the Wild (AAAI 2015)
==============================================================
Architecture: paper-exact (tanh, maxpool). Weights are BiChannelCNN-compatible.

Why previous version (orthogonal conv + Normal(0,0.01) fc) plateaued at 20.8 dB
=================================================================================
After orthogonal conv init was added, the conv gradient problem was fixed.
But the stall moved to fc1/fc2. Normal(0,0.01) for fc1 (fan_in=2048) gives
pre-activation std = sqrt(fan_in × 0.01² × input_std²) ≈ 0.14 at init.
This is fine initially, but Adam's adaptive LR amplifies small-gradient weights,
and within a few epochs fc1 weights drift to ±0.1+ → pre-activation std >2 →
tanh saturation → gradient ≈ 0 at fc1 → conv3/conv2/conv1 starved again.

Fix: Xavier uniform init for fc layers (gain=5/3 for tanh).
Xavier sets a = gain × sqrt(6/(fan_in+fan_out)), keeping pre-activation
variance ≈1 regardless of fan-in. For fc1: a≈0.063; for fc2: a≈0.018.
With this scaling, Adam's adaptive amplification is counteracted because
the initial weight scale is already the correct asymptotic magnitude —
the weights don't need to drift far to be in the right range.

Architecture: UNCHANGED (paper-exact, tanh, maxpool, BiChannelCNN-compatible)

Expected results
================
  Epoch 5:   PSNR ~21–22 dB  (short warmup done, all layers receiving gradients)
  Epoch 30:  PSNR ~24–26 dB
  Epoch 100: PSNR ~26–28 dB
  Epoch 200: PSNR ~28–30 dB  (ready to pretrain BiChannelCNN)
"""

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

# ============================================================
# Config
# ============================================================
WORKDIR = Path("/home/msai/ruijiane001/AI6301/Face-Hallucination/Deep-Learning-Face-Hallucination/julia_workings")
DEFAULT_IMAGE_ROOTS = [WORKDIR / "CelebA" / "img_align_celeba"]

RESULTS_DIR    = WORKDIR / "results_basiccnn"
SPLITS_DIR     = WORKDIR / "splits_basiccnn"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

SEED        = 42
MAX_IMAGES  = 100_000
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2

HR_SIZE             = (100, 100)
TRAIN_LR_INPUT_SIZE = (48, 48)
TEST_FIXED_LR_SIZE  = (50, 50)
TEST_NN_INPUT_SIZE  = (48, 48)

CELEBA_CROP    = True
CROP_FRAC      = 0.60
METRIC_CHANNEL = "y"   # "y" = luminance (paper convention), "rgb" = full colour

# ── Hyperparameters ──────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 600     # extended from 400 — cosine LR has more room to improve
                         # at epoch 323 LR was already 4.7e-5 (schedule nearly done)
                         # 600 epochs gives 590-epoch cosine so LR stays above 1e-5
                         # until epoch ~550, giving 200+ extra useful training epochs
LEARNING_RATE = 5e-4    # unchanged — proven optimal
MIN_LR        = 1e-6    # unchanged
WEIGHT_DECAY  = 1e-3    # unchanged
GRAD_CLIP     = 1.0
NUM_WORKERS   = 2

WARMUP_EPOCHS = 10      # unchanged
LR_T_MAX      = 590     # cosine cycle = NUM_EPOCHS - WARMUP_EPOCHS

KEEP_LAST_N = 3


# ============================================================
# CelebA crop  (unchanged from basic5.py)
# ============================================================
def celeba_crop(pil_img: Image.Image) -> Image.Image:
    if not CELEBA_CROP:
        return pil_img
    w, h = pil_img.size
    side = int(min(w, h) * CROP_FRAC)
    left = (w - side) // 2
    top  = max(0, (h - side) // 2 + int(side * 0.15))
    return pil_img.crop((left, top, left + side, top + side))


# ============================================================
# Utilities  (unchanged from basic5.py)
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


def make_splits(image_root: Path, max_images: int = MAX_IMAGES, seed: int = SEED,
                train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO,
                test_ratio: float = TEST_RATIO, splits_dir: Path = SPLITS_DIR,
                ) -> Tuple[List[str], List[str], List[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.")
    all_images = gather_images(image_root)
    if len(all_images) < max_images:
        print(f"[Warning] Only {len(all_images)} images found; using all.")
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
# Metrics  (unchanged from basic5.py)
# ============================================================
def _to_y(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        t = t.unsqueeze(0)
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    return y.clamp(0, 1)


def _ssim_kernel(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1)) / (g.unsqueeze(0) * g.unsqueeze(1)).sum()
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
    mu1  = F.conv2d(pred,         k, padding=pad, groups=ch)
    mu2  = F.conv2d(target,       k, padding=pad, groups=ch)
    s1   = F.conv2d(pred*pred,     k, padding=pad, groups=ch) - mu1**2
    s2   = F.conv2d(target*target, k, padding=pad, groups=ch) - mu2**2
    s12  = F.conv2d(pred*target,   k, padding=pad, groups=ch) - mu1*mu2
    num  = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-12)).mean().item()


# ============================================================
# Per-image normalisation  (unchanged from basic5.py, paper Eq. 15–17)
# ============================================================
def normalize_per_image(img: np.ndarray, eps: float = 1e-8,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = img.mean(axis=(0, 1))
    std  = img.std(axis=(0, 1)).clip(eps)
    return np.tanh((img - mean) / std).astype(np.float32), mean, std


def denormalize_per_image(img: torch.Tensor, mean: np.ndarray,
                           std: np.ndarray) -> torch.Tensor:
    """
    tanh-space CHW tensor → pixel-space CHW tensor [0,1].

    FIX vs basic5.py: eps increased from 1e-6 to 1e-3.
    With eps=1e-6, values like 0.9999 give atanh≈4.6, which after
    multiplying by std≈0.1 gives ~0.46 — still plausible — but values
    like 0.999999 give atanh≈7.3 → pixel ~0.73 with wrong sign.
    More importantly, the GRADIENT of atanh at x=0.9999 is 1/(1-x²)≈5000,
    which would destabilise metric computation if gradients were tracked.
    eps=1e-3 clamps to (-0.999, 0.999), giving atanh max ~±3.8 and
    stable pixel-space values throughout training.
    """
    eps = 1e-3
    mt = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    st = torch.tensor(std,  dtype=img.dtype, device=img.device).view(3, 1, 1)
    return torch.clamp(torch.atanh(img.clamp(-1 + eps, 1 - eps)) * st + mt, 0.0, 1.0)


# ============================================================
# Motion blur kernel  (unchanged from basic5.py)
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
# Datasets  (unchanged from basic5.py)
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
# Model — IDENTICAL to basic5.py (paper-exact, tanh everywhere)
# DO NOT change activations — weights must be compatible with BiChannelCNN.
# ============================================================
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 2000)
        self.fc2   = nn.Linear(2000, 3 * 100 * 100)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Orthogonal init (gain=0.6) for conv layers — unchanged.
                # Preserves gradient magnitude through the conv stack.
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # Correct FC init for tanh networks with tanh-bounded inputs.
                #
                # The input to fc1 is tanh(conv3 output), which has std ≈ 0.15–0.25
                # (tanh compresses variance; 3 tanh+pool stages reduce it further).
                # This is NOT the std≈0.5 that standard Xavier assumes for its
                # fan-in/fan-out formula. Xavier with gain=5/3 then overshoots:
                #   a = (5/3) × sqrt(6/4048) ≈ 0.10
                #   fc1 pre-act std = sqrt(2048) × 0.10 × 0.2 ≈ 0.9
                #   tanh(0.9) has derivative ≈ 0.41 — already moderately saturated
                #   at epoch 0, and drifting worse within the first few epochs.
                #
                # The correct formula for this specific input std is:
                #   std = 1 / sqrt(fan_in)
                # which gives pre-activation std = sqrt(fan_in) × std × input_std
                #                               = sqrt(fan_in) × (1/sqrt(fan_in)) × input_std
                #                               = input_std ≈ 0.2
                # tanh(0.2) has derivative ≈ 0.96 — essentially in the linear region.
                # The weights need to grow ~5× from this init to saturate tanh,
                # which takes many epochs and AdamW's weight decay prevents it.
                fan_in = m.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = torch.tanh(self.fc1(x.flatten(start_dim=1)))
        x = torch.tanh(self.fc2(x))
        return x.view(-1, 3, 100, 100)


# ============================================================
# LR: linear warmup then cosine annealing
# Replaces basic5.py's ReduceLROnPlateau.
# ============================================================
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Epochs 0..warmup_epochs-1 : linear ramp from MIN_LR to base_lr
    Epochs warmup_epochs..end  : cosine decay from base_lr to MIN_LR

    Rationale vs ReduceLROnPlateau:
      - ReduceLROnPlateau with factor=0.1 is too aggressive: one unlucky
        plateau drops the LR by 10×, often permanently (min_lr reached).
        In basic5.py the LR hit 1e-6 at epoch ~45 and stayed there.
      - Warmup prevents destructive Adam steps at epoch 1 when the
        second-moment estimate v_t ≈ 0 making effective lr = lr/sqrt(eps).
      - Cosine annealing reduces the LR smoothly — no sudden drops —
        allowing the optimizer to fine-tune the solution continuously.
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
# Checkpoint helpers  (unchanged from basic5.py)
# ============================================================
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":        val_loss,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def keep_last_n_checkpoints(checkpoint_dir, prefix="basiccnn_epoch_", n=KEEP_LAST_N):
    ckpts = sorted(Path(checkpoint_dir).glob(f"{prefix}*.pth"))
    for ckpt in ckpts[:-n]:
        try:
            ckpt.unlink()
            print(f"  → Deleted old checkpoint: {ckpt.name}")
        except Exception as e:
            print(f"  → Warning: could not delete {ckpt.name}: {e}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
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
            out      = model(iin)
            total_loss += criterion(out, ih).item()
            for b in range(out.shape[0]):
                p = denormalize_per_image(out[b],  mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(ih[b],   mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    return total_loss / len(dataloader), total_psnr / n, total_ssim / n


def train_model(model, train_loader, val_loader, device,
                resume_checkpoint: Path = None):
    criterion = nn.MSELoss()   # MSE in tanh-space — paper Eq. 3

    # ── AdamW replaces Adam ───────────────────────────────────
    # Adam's weight_decay is L2 regularisation added to the gradient
    # BEFORE the adaptive scaling step. This means:
    #   effective_decay(param) = weight_decay / sqrt(v_t)
    # For parameters with large gradients (v_t is large), the effective
    # decay is SMALL. For small-gradient parameters, the effective decay
    # is LARGE. In practice this means fc2 (which has large gradients
    # because it is closest to the loss) gets very little regularisation,
    # while conv1 (small gradients from deep in the network) gets too much.
    # The result is fc2 weights growing unchecked while conv1 is over-damped.
    #
    # AdamW fixes this by applying weight decay AFTER the adaptive update:
    #   w ← w × (1 - lr × weight_decay) - lr × adam_update
    # This decouples regularisation from gradient magnitude completely —
    # every parameter shrinks by the same multiplicative factor per step
    # regardless of its gradient history. fc2 now gets proper regularisation
    # and cannot grow unchecked into tanh saturation.
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # ── Warmup cosine LR schedule ─────────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        t_max=LR_T_MAX,
        min_lr=MIN_LR)

    print(f"Optimizer: AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
    print(f"LR schedule: {WARMUP_EPOCHS}-epoch warmup → cosine T_max={LR_T_MAX}")
    print(f"Loss: MSE in tanh-space | Metrics: PSNR/SSIM in pixel space")
    print(f"Init: orthogonal(gain=0.6) conv | Normal(std=1/sqrt(fan_in)) fc")

    start_epoch    = 0
    best_val_loss  = float("inf")
    best_model_path = RESULTS_DIR / "best_model_basiccnn.pth"
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
            out  = model(iin)

            # ── Loss in tanh-space — correct, unchanged from basic5.py ──
            loss = criterion(out, ih)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

            # Metrics in pixel space (monitoring only — not used in loss)
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
                print(f"First batch — In:{iin.shape} Out:{out.shape} "
                      f"Target:{ih.shape}")
                print(f"  out range: [{out.min().item():.3f}, "
                      f"{out.max().item():.3f}] (tanh-space)")
                # Gradient flow diagnostic — all three should be > 1e-4.
                # If fc1 or fc2 is near zero, Xavier init did not take effect.
                # If conv1 is near zero, orthogonal init did not take effect.
                for name, param in [("conv1", model.conv1.weight),
                                     ("fc1",   model.fc1.weight),
                                     ("fc2",   model.fc2.weight)]:
                    if param.grad is not None:
                        g = param.grad.norm().item()
                        ok = "OK" if g > 1e-4 else "WARN — gradient too small"
                        print(f"  {name} grad norm: {g:.2e}  {ok}")

        train_loss = epoch_loss / len(train_loader)
        train_psnr = epoch_psnr / n
        train_ssim = epoch_ssim / n

        val_loss, val_psnr, val_ssim = evaluate_validation(
            model, val_loader, device, criterion)

        # Warmup cosine always steps once per epoch
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss:{train_loss:.4f} PSNR:{train_psnr:.2f} "
              f"SSIM:{train_ssim:.4f} | "
              f"Val Loss:{val_loss:.4f} PSNR:{val_psnr:.2f} "
              f"SSIM:{val_ssim:.4f} | LR:{lr:.1e}")

        save_metrics_to_csv(
            RESULTS_DIR / "train_val_metrics_basiccnn.csv",
            [epoch+1, train_loss, train_psnr, train_ssim,
             val_loss, val_psnr, val_ssim, lr],
            ["epoch","train_loss","train_psnr","train_ssim",
             "val_loss","val_psnr","val_ssim","lr"])

        ckpt_path = CHECKPOINT_DIR / f"basiccnn_epoch_{epoch+1:04d}.pth"
        save_checkpoint(model, optimizer, scheduler,
                        epoch + 1, val_loss, ckpt_path)
        print(f"  → Checkpoint saved: {ckpt_path.name}")
        keep_last_n_checkpoints(CHECKPOINT_DIR, prefix="basiccnn_epoch_",
                                n=KEEP_LAST_N)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler,
                            epoch + 1, val_loss, best_model_path)
            print(f"  → New best model (val_loss={best_val_loss:.4f})")

    return best_model_path


# ============================================================
# Test  (unchanged from basic5.py)
# ============================================================
def load_model_for_inference(model_path: Path, device):
    model = BasicCNN().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_one_test_setting(model, dataloader, device, criterion):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    with torch.no_grad():
        for iin, ih, mean_b, std_b in dataloader:
            iin, ih = iin.to(device).float(), ih.to(device).float()
            out      = model(iin)
            total_loss += criterion(out, ih).item()
            for b in range(out.shape[0]):
                p = denormalize_per_image(out[b],  mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(ih[b],   mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    return total_loss / len(dataloader), total_psnr / n, total_ssim / n


def save_image(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def save_sample_grid(model, test_files, device, save_dir: Path,
                     blur_type, gaussian_sigma=None, motion_length=None,
                     num_samples=3):
    """Save raw/HR/LR/SR quad for each of the first num_samples test images."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = FixedTestFaceDataset(test_files, blur_type=blur_type,
                              gaussian_sigma=gaussian_sigma,
                              motion_length=motion_length,
                              base_seed=SEED)
    model.eval()
    SAMPLE_IDX = 0
    with torch.no_grad():
        for i in range(min(num_samples, len(ds))):
            iin, ih, mean_b, std_b = ds[SAMPLE_IDX + i]
            raw_path = test_files[SAMPLE_IDX + i]
            iin_b = iin.unsqueeze(0).to(device)
            out_b = model(iin_b)

            # Raw original
            celeba_crop(Image.open(raw_path).convert("RGB")).resize(
                HR_SIZE, Image.BICUBIC).save(save_dir / f"{i}_raw.png")
            # HR target
            save_image(denormalize_per_image(ih, mean_b.numpy(), std_b.numpy()),
                       save_dir / f"{i}_hr.png")
            # LR input (nearest upscale for visibility)
            lr_pil = Image.fromarray(
                (denormalize_per_image(iin, mean_b.numpy(), std_b.numpy())
                 .cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8))
            lr_pil.resize(HR_SIZE, Image.NEAREST).save(save_dir / f"{i}_lr.png")
            # SR output
            save_image(denormalize_per_image(out_b[0], mean_b.numpy(), std_b.numpy()),
                       save_dir / f"{i}_sr.png")
    print(f"  Samples saved → {save_dir}")


def run_test_pipeline(test_files: List[str], best_model_path: Path, device):
    criterion = nn.MSELoss()
    model     = load_model_for_inference(best_model_path, device)
    results   = []
    lkw = dict(batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    for sigma in [1, 3, 5]:
        ds  = FixedTestFaceDataset(test_files, blur_type="gaussian",
                                   gaussian_sigma=sigma)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Gaussian sigma={sigma} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"gaussian_sigma_{sigma}",
                        "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         RESULTS_DIR / f"samples_gaussian_sigma_{sigma}",
                         blur_type="gaussian", gaussian_sigma=sigma)

    for length in [2, 6, 9]:
        ds  = FixedTestFaceDataset(test_files, blur_type="motion",
                                   motion_length=length, base_seed=SEED)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Motion l={length} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"motion_l_{length}",
                        "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         RESULTS_DIR / f"samples_motion_l_{length}",
                         blur_type="motion", motion_length=length)

    csv_path = RESULTS_DIR / "test_pipeline_metrics_basiccnn.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["blur_type", "loss", "psnr", "ssim"])
        w.writeheader()
        w.writerows(results)
    print(f"Test metrics → {csv_path}")
    return results


# ============================================================
# Main
# ============================================================
def main(image_root_arg=None, resume_checkpoint=None):
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Metric: {METRIC_CHANNEL.upper()}")
    print(f"Adam lr={LEARNING_RATE}  wd={WEIGHT_DECAY}  batch={BATCH_SIZE}  "
          f"epochs={NUM_EPOCHS}  warmup={WARMUP_EPOCHS}")
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

    model = BasicCNN().to(device)
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
                    help="Path to checkpoint to resume from")
    args = ap.parse_args()
    main(args.image_root, args.resume)