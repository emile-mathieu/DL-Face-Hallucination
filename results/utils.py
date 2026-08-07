"""
utils.py
========
Shared utilities used by all three methods:
  - set_seed
  - CelebA crop
  - image splitting / loading
  - per-image tanh normalisation (paper Eq. 15-17)
  - motion blur kernel
  - PSNR / SSIM (torch + numpy variants)
  - checkpoint save/load
  - CSV metric writer
"""

import csv
import json
import math
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

from config import (
    SEED, MAX_IMAGES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SPLITS_DIR, CELEBA_CROP, CROP_FRAC, CROP_TOP_OFFSET,
    METRIC_CHANNEL, HR_SIZE, KEEP_LAST_N,
)


# ── Reproducibility ───────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── CelebA crop ───────────────────────────────────────────────
def celeba_crop(pil_img: Image.Image) -> Image.Image:
    """
    Centre-crop
    CROP_FRAC=0.60, top = (h-side)//2 + int(side * 0.15).
    Applied to ALL methods (deep and classical) for a fair comparison.
    """
    if not CELEBA_CROP:
        return pil_img
    w, h  = pil_img.size
    side  = int(min(w, h) * CROP_FRAC)
    left  = (w - side) // 2
    top   = max(0, (h - side) // 2 + int(side * CROP_TOP_OFFSET))
    return pil_img.crop((left, top, left + side, top + side))


# ── Dataset splits ────────────────────────────────────────────
def gather_images(image_root: Path) -> List[str]:
    print(f"Scanning directory: {image_root} ...", flush=True) 
    exts  = {".jpg", ".jpeg", ".png"}
    files = [
        str(image_root / f) for f in os.listdir(image_root) 
        if Path(f).suffix.lower() in exts
    ]
    if not files:
        raise ValueError(f"No images found in {image_root}")
    files.sort()
    print(f"Found {len(files)} images.", flush=True)
    return files


def make_splits(
    image_root: Path,
    max_images: int  = MAX_IMAGES,
    seed: int        = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    splits_dir:  Path  = SPLITS_DIR,
) -> Tuple[List[str], List[str], List[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.")
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


def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> List[str]:
    p = splits_dir / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_image_root(explicit_path: str = None,
                    default_root: Path = None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Image root not found: {p}")
        return p
    if default_root is not None and default_root.exists():
        return default_root
    raise FileNotFoundError(
        "CelebA image directory not found. Pass --image-root.")


# ── Per-image tanh normalisation (paper Eq. 15-17) ────────────
def normalize_per_image(
    img: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HWC float32 [0,1] → tanh-normalised HWC + per-channel mean, std."""
    mean = img.mean(axis=(0, 1))
    std  = img.std(axis=(0, 1)).clip(eps)
    return np.tanh((img - mean) / std).astype(np.float32), mean, std


def denormalize_per_image(
    img: torch.Tensor, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    """
    Tanh-space CHW tensor → pixel-space CHW tensor [0,1].
    eps=1e-3 prevents atanh blow-up near ±1.
    """
    eps = 1e-3
    mt  = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    st  = torch.tensor(std,  dtype=img.dtype, device=img.device).view(3, 1, 1)
    return torch.clamp(
        torch.atanh(img.clamp(-1 + eps, 1 - eps)) * st + mt, 0.0, 1.0)


# ── Blur kernels ──────────────────────────────────────────────
def motion_blur_kernel(length: int, theta: float) -> np.ndarray:
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


# ── Metrics (PyTorch — used by deep models) ───────────────────
def _to_y_torch(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        t = t.unsqueeze(0)
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return (0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0).clamp(0, 1)


def _ssim_kernel_torch(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1))
    k /= k.sum()
    return k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(device=device, dtype=dtype)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y_torch(pred), _to_y_torch(target)
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y_torch(pred), _to_y_torch(target)
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch  = pred.shape[1]
    k   = _ssim_kernel_torch(ch, pred.device, pred.dtype)
    pad = 5
    mu1  = F.conv2d(pred,         k, padding=pad, groups=ch)
    mu2  = F.conv2d(target,       k, padding=pad, groups=ch)
    s1   = F.conv2d(pred * pred,   k, padding=pad, groups=ch) - mu1 ** 2
    s2   = F.conv2d(target*target, k, padding=pad, groups=ch) - mu2 ** 2
    s12  = F.conv2d(pred * target, k, padding=pad, groups=ch) - mu1 * mu2
    num  = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den  = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-12)).mean().item()


# ── Metrics (NumPy — used by classical models) ────────────────
def rgb_to_y_np(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim_np(pred: np.ndarray, target: np.ndarray, win: int = 11) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k1d = np.exp(-(np.arange(win) - win // 2) ** 2 / (2 * 1.5 ** 2))
    k1d /= k1d.sum()
    k2d = np.outer(k1d, k1d).astype(np.float64)
    scores = []
    if pred.ndim == 2:
        pred   = pred[:, :, None]
    if target.ndim == 2:
        target = target[:, :, None]
    for c in range(pred.shape[2]):
        p = pred[:, :, c].astype(np.float64)
        t = target[:, :, c].astype(np.float64)
        kw = dict(ddepth=-1, kernel=k2d, borderType=cv2.BORDER_REFLECT_101)
        mu1, mu2 = cv2.filter2D(p, **kw), cv2.filter2D(t, **kw)
        s1   = cv2.filter2D(p * p, **kw) - mu1 ** 2
        s2   = cv2.filter2D(t * t, **kw) - mu2 ** 2
        s12  = cv2.filter2D(p * t, **kw) - mu1 * mu2
        num  = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den  = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
        scores.append(float(np.mean(num / (den + 1e-12))))
    return float(np.mean(scores))


# ── Checkpoint helpers ────────────────────────────────────────
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None,
                    scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def keep_last_n_checkpoints(checkpoint_dir: Path, prefix: str,
                             n: int = KEEP_LAST_N) -> None:
    ckpts = sorted(Path(checkpoint_dir).glob(f"{prefix}*.pth"))
    for old in ckpts[:-n]:
        try:
            old.unlink()
            print(f"  → Deleted old checkpoint: {old.name}")
        except Exception as e:
            print(f"  → Warning: could not delete {old.name}: {e}")


# ── CSV metric writer ─────────────────────────────────────────
def save_metrics_to_csv(csv_path: Path, row: list,
                        header: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ── Image save helper ─────────────────────────────────────────
def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)
