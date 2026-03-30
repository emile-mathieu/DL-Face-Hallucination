'''
Can run this on CPU cos it doesn't use GPU
Classical Baseline Methods for Face Hallucination
===================================================
Implements the four comparison methods from Table 2 of:
  Zhou et al., "Learning Face Hallucination in the Wild", AAAI 2015.

Key fixes vs previous version:
  1. Face-region centre-crop before resizing to 100x100 (matches paper's dataset)
  2. SC1/SC2 HR patch placement corrected
  3. Stronger default dictionary sizes
  4. SFH gradient integration improved
  5. PSNR/SSIM computed on Y-channel or RGB using the same toggle style
     as bichannel.py, so metrics are directly comparable

For very fast debugging you can change these parameters below
SC1_N_TRAIN   = 500
SC2_N_TRAIN   = 300
SC2_N_BASIS   = 30
SC1_DICT_SIZE = 64

also can change max_images below under main to 300 for debugging 

'''

import csv
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ============================================================
# Config
# ============================================================
WORKDIR     = Path("C:/Users/julia/Desktop/celeba")
SPLITS_DIR  = Path("C:/Users/julia/Desktop/splits_basiccnn")
RESULTS_DIR = Path("C:/Users/julia/Desktop/results_classical")

SEED         = 42
HR_SIZE      = (100, 100)
TEST_LR_SIZE = (50, 50)

# CelebA face crop: images are 218x178, face is roughly centred.
# Crop a square region around the face centre before resizing.
CELEBA_CROP = True      # set False if your images are already face-only
CROP_FRAC   = 0.85      # fraction of shorter side to keep

# the last parameter you can change is max_images under main below, now set at 2500 (if just for debugging can use 300 images)

# Metric channel: "y" (luminance, recommended) or "rgb"
METRIC_CHANNEL = "y"

# SC1 parameters
SC1_PATCH_SIZE = 5         # low-res patch size (Yang 2008 uses 5×5 on upsampled LR), patches sampled to buuild dictionary
SC1_UPSCALE    = 2          # upscale factor LR→HR  (50→100)
SC1_DICT_SIZE  = 512        # 512+ recommended; 256 minimum
SC1_LAMBDA     = 0.1        # sparsity regularisation weight
SC1_N_TRAIN    = 20_000

# SC2 parameters
SC2_PATCH_SIZE = 5
SC2_SIGMA_K    = 0.05       # Gaussian kernel width (Table 1 of Kim & Kwon 2010, factor 2)
SC2_LAMBDA_REG = 5e-8       # ridge regularisation
SC2_N_BASIS    = 300        # sparse basis points (Table 1), 30
SC2_N_TRAIN    = 10_000     # training patches

# SFH parameters
SFH_MAX_EXEMPLARS = 300


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(split_name: str) -> List[str]:
    path = SPLITS_DIR / f"{split_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def face_crop_and_resize(
    pil_img: Image.Image,
    hr_size: Tuple[int, int] = HR_SIZE
) -> np.ndarray:
    """
    Centre-crop to face region then resize to hr_size.
    Returns HWC float32 in [0,1].
    """
    w, h = pil_img.size
    if CELEBA_CROP:
        side   = int(min(w, h) * CROP_FRAC)
        left   = (w - side) // 2
        top    = max(0, (h - side) // 2 - int(side * 0.05))
        right  = left + side
        bottom = top + side
        pil_img = pil_img.crop((left, top, right, bottom))
    return np.array(pil_img.resize(hr_size, Image.BICUBIC)).astype(np.float32) / 255.0


# ============================================================
# Blur / degradation  (paper Eq. 1)
# ============================================================
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
    kernel = kernel / s if s > 0 else kernel
    if kernel.sum() == 0:
        kernel[length // 2, length // 2] = 1.0
    return kernel


def degrade(
    hr: np.ndarray,
    blur_type: str,
    gaussian_sigma: float = None,
    motion_length: int = None,
    theta: float = None
) -> np.ndarray:
    """HWC float32 [0,1] -> LR HWC float32 [0,1] at TEST_LR_SIZE."""
    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(
            hr, (0, 0),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma
        )
    else:
        blurred = cv2.filter2D(
            hr, -1,
            motion_blur_kernel(motion_length, theta)
        )
    return cv2.resize(
        blurred, TEST_LR_SIZE,
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)


# ============================================================
# Metrics  (same Y/RGB logic as bichannel.py)
# ============================================================
def _to_tensor_hwc(img: np.ndarray) -> torch.Tensor:
    """HWC numpy [0,1] -> CHW torch float32."""
    return torch.from_numpy(img).permute(2, 0, 1).float()


def _to_y(t: torch.Tensor) -> torch.Tensor:
    """CHW or NCHW RGB [0,1] -> Y-channel tensor."""
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


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = _to_tensor_hwc(pred)
    target_t = _to_tensor_hwc(target)

    if METRIC_CHANNEL == "y":
        pred_t, target_t = _to_y(pred_t), _to_y(target_t)

    mse = torch.mean((pred_t - target_t) ** 2).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = _to_tensor_hwc(pred)
    target_t = _to_tensor_hwc(target)

    if METRIC_CHANNEL == "y":
        pred_t, target_t = _to_y(pred_t), _to_y(target_t)
    else:
        pred_t = pred_t.unsqueeze(0)
        target_t = target_t.unsqueeze(0)

    if pred_t.ndim == 3:
        pred_t = pred_t.unsqueeze(0)
        target_t = target_t.unsqueeze(0)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = pred_t.shape[1]
    k = _ssim_kernel(ch, pred_t.device, pred_t.dtype)
    pad = 5

    mu1 = F.conv2d(pred_t, k, padding=pad, groups=ch)
    mu2 = F.conv2d(target_t, k, padding=pad, groups=ch)
    s1 = F.conv2d(pred_t * pred_t, k, padding=pad, groups=ch) - mu1 ** 2
    s2 = F.conv2d(target_t * target_t, k, padding=pad, groups=ch) - mu2 ** 2
    s12 = F.conv2d(pred_t * target_t, k, padding=pad, groups=ch) - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-12)).mean().item()


# ============================================================
# Shared feature extractor (Yang 2008 Eq. 12)
# ============================================================
_FEAT_FILTERS = [
    np.array([[-1, 0, 1]], dtype=np.float32),
    np.array([[-1], [0], [1]], dtype=np.float32),
    np.array([[1, 0, -2, 0, 1]], dtype=np.float32),
    np.array([[1], [0], [-2], [0], [1]], dtype=np.float32),
]

def extract_features(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([
        cv2.filter2D(patch, -1, f, borderType=cv2.BORDER_REFLECT).ravel()
        for f in _FEAT_FILTERS
    ]).astype(np.float32)


# ============================================================
# Method 1: Bicubic
# ============================================================
def bicubic_sr(lr: np.ndarray) -> np.ndarray:
    return cv2.resize(
        lr, HR_SIZE,
        interpolation=cv2.INTER_CUBIC
    ).clip(0, 1).astype(np.float32)


# ============================================================
# Method 2: SC1 — Sparse Coding SR (Yang et al. 2008/2010)
# ============================================================
class SC1Solver:

    def __init__(
        self,
        patch_size=SC1_PATCH_SIZE,
        upscale=SC1_UPSCALE,
        dict_size=SC1_DICT_SIZE,
        lam=SC1_LAMBDA,
        n_train=SC1_N_TRAIN
    ):
        self.ps = patch_size
        self.us = upscale
        self.dict_size = dict_size
        self.lam = lam
        self.n_train = n_train
        self.Dl = None
        self.Dh = None

    def _collect_patches(self, paths, rng):
        ps, us, hps = self.ps, self.us, self.ps * self.us
        lr_f, hr_p = [], []
        per = max(1, self.n_train // max(1, len(paths)))

        for path in rng.sample(paths, min(len(paths), self.n_train // per + 1)):
            try:
                hr = face_crop_and_resize(Image.open(path).convert("L"))
                lr_up = cv2.resize(
                    cv2.resize(hr, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC),
                    HR_SIZE, interpolation=cv2.INTER_CUBIC
                )
            except Exception:
                continue

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 2)
                           for c in range(0, W - ps + 1, 2)]

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_features(lr_up[r:r+ps, c:c+ps])
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch = r * us, c * us
                if rh + hps > hr.shape[0] or ch + hps > hr.shape[1]:
                    continue

                hp = hr[rh:rh+hps, ch:ch+hps].ravel()
                lr_f.append(feat / fn)
                hr_p.append(hp - hp.mean())

                if len(lr_f) >= self.n_train:
                    break
            if len(lr_f) >= self.n_train:
                break

        return (
            np.stack(lr_f).astype(np.float32),
            np.stack(hr_p).astype(np.float32)
        )

    def build(self, paths: List[str], seed: int = SEED) -> None:
        print(f"[SC1] Building dictionary (n_train={self.n_train}, dict_size={self.dict_size}) ...")
        rng = random.Random(seed)
        Xtr, Ytr = self._collect_patches(paths, rng)
        n = Xtr.shape[0]
        print(f"[SC1] Collected {n} patches")

        if n > self.dict_size:
            try:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(
                    n_clusters=self.dict_size,
                    random_state=seed,
                    batch_size=min(4096, n),
                    max_iter=200,
                    n_init=5
                )
                lbl = km.fit_predict(Xtr)
                Dl, Dh = [], []
                for k in range(self.dict_size):
                    idx = np.where(lbl == k)[0]
                    if len(idx) == 0:
                        idx = [rng.randint(0, n - 1)]
                    best = int(idx[np.argmin(
                        np.linalg.norm(Xtr[idx] - km.cluster_centers_[k], axis=1)
                    )])
                    Dl.append(Xtr[best])
                    Dh.append(Ytr[best])
                self.Dl = np.stack(Dl, axis=1).astype(np.float32)
                self.Dh = np.stack(Dh, axis=1).astype(np.float32)
            except ImportError:
                idx = np.random.default_rng(seed).choice(n, self.dict_size, replace=False)
                self.Dl = Xtr[idx].T.astype(np.float32)
                self.Dh = Ytr[idx].T.astype(np.float32)
        else:
            self.Dl = Xtr.T.astype(np.float32)
            self.Dh = Ytr.T.astype(np.float32)

        norms = np.linalg.norm(self.Dl, axis=0, keepdims=True).clip(1e-8)
        self.Dl /= norms
        self.Dh /= norms
        self._DtD = self.Dl.T @ self.Dl
        self._L = float(np.linalg.norm(self._DtD, ord=2)) + 1e-6
        print(f"[SC1] Done — Dl={self.Dl.shape}")

    def _ista(self, y: np.ndarray, iters: int = 100) -> np.ndarray:
        Dty = self.Dl.T @ y
        thresh = self.lam / self._L
        a = np.zeros(self.Dl.shape[1], dtype=np.float32)
        for _ in range(iters):
            a -= (self._DtD @ a - Dty) / self._L
            a = np.sign(a) * np.maximum(np.abs(a) - thresh, 0)
        return a

    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        ps, us, hps = self.ps, self.us, self.ps * self.us
        step = max(1, ps - 2)
        lr_up = cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W = lr_up.shape
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        for r in range(0, H - ps + 1, step):
            for c in range(0, W - ps + 1, step):
                feat = extract_features(lr_up[r:r+ps, c:c+ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-8:
                    continue

                hp = (self.Dh @ self._ista(feat / fn)).reshape(hps, hps)

                rh, ch = r * us, c * us
                rh2, ch2 = min(rh + hps, H), min(ch + hps, W)
                if rh >= H or ch >= W:
                    continue

                acc[rh:rh2, ch:ch2] += hp[:rh2-rh, :ch2-ch]
                cnt[rh:rh2, ch:ch2] += 1.0

        hr_rec = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
        hr_rec += cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        for _ in range(15):
            lr_est = cv2.resize(hr_rec, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.6 * cv2.resize(lr - lr_est, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0, 1)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2
        ).clip(0, 1).astype(np.float32)


# ============================================================
# Method 3: SC2 — Kernel Ridge Regression SR (Kim & Kwon 2010)
# ============================================================
class SC2Solver:

    def __init__(
        self,
        patch_size=SC2_PATCH_SIZE,
        sigma_k=SC2_SIGMA_K,
        lam=SC2_LAMBDA_REG,
        n_basis=SC2_N_BASIS,
        n_train=SC2_N_TRAIN
    ):
        self.ps = patch_size
        self.sigma_k = sigma_k
        self.lam = lam
        self.n_basis = n_basis
        self.n_train = n_train
        self.B = None
        self.A = None

    def _kern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        d = (
            np.sum(X**2, 1, keepdims=True)
            + np.sum(Y**2, 1, keepdims=True).T
            - 2 * X @ Y.T
        ).clip(0)
        return np.exp(-d / self.sigma_k).astype(np.float32)

    def build(self, paths: List[str], seed: int = SEED) -> None:
        print(f"[SC2] Building KRR model (n_train={self.n_train}, n_basis={self.n_basis}) ...")
        rng = random.Random(seed)
        ps, us, hps = self.ps, 2, self.ps * 2
        per = max(1, self.n_train // max(1, len(paths)))
        Xl, Yl = [], []

        for path in rng.sample(paths, min(len(paths), self.n_train // per + 1)):
            try:
                hr = face_crop_and_resize(Image.open(path).convert("L"))
                lr_up = cv2.resize(
                    cv2.resize(hr, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC),
                    HR_SIZE, interpolation=cv2.INTER_CUBIC
                )
            except Exception:
                continue

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 3)
                           for c in range(0, W - ps + 1, 3)]

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_features(lr_up[r:r+ps, c:c+ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch = r * us, c * us
                if rh + hps > hr.shape[0] or ch + hps > hr.shape[1]:
                    continue

                hp = hr[rh:rh+hps, ch:ch+hps].ravel()
                Xl.append(feat / fn)
                Yl.append(hp - hp.mean())

                if len(Xl) >= self.n_train:
                    break
            if len(Xl) >= self.n_train:
                break

        Xtr = np.stack(Xl).astype(np.float32)
        Ytr = np.stack(Yl).astype(np.float32)
        n = Xtr.shape[0]
        print(f"[SC2] Collected {n} patches")

        bidx = np.random.default_rng(seed).choice(n, min(self.n_basis, n), replace=False)
        self.B = Xtr[bidx]
        Kbx = self._kern(self.B, Xtr)
        Kbb = self._kern(self.B, self.B)
        M = Kbx @ Kbx.T + self.lam * Kbb + 1e-8 * np.eye(len(bidx))
        self.A = np.linalg.solve(
            M.astype(np.float64),
            (Kbx @ Ytr).astype(np.float64)
        ).astype(np.float32)
        print(f"[SC2] Done — B={self.B.shape}, A={self.A.shape}")

    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        ps, us, hps = self.ps, 2, self.ps * 2
        step = max(1, ps - 2)
        lr_up = cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W = lr_up.shape
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        for r in range(0, H - ps + 1, step):
            for c in range(0, W - ps + 1, step):
                feat = extract_features(lr_up[r:r+ps, c:c+ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-8:
                    continue

                k = self._kern(self.B, (feat / fn).reshape(1, -1)).ravel()
                hp = (k @ self.A).reshape(hps, hps)

                rh, ch = r * us, c * us
                rh2, ch2 = min(rh + hps, H), min(ch + hps, W)
                if rh >= H or ch >= W:
                    continue

                acc[rh:rh2, ch:ch2] += hp[:rh2-rh, :ch2-ch]
                cnt[rh:rh2, ch:ch2] += 1.0

        hr_rec = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
        hr_rec += cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        for _ in range(15):
            lr_est = cv2.resize(hr_rec, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.6 * cv2.resize(lr - lr_est, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0, 1)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2
        ).clip(0, 1).astype(np.float32)


# ============================================================
# Method 4: SFH — Structured Face Hallucination (Yang et al. 2013)
# ============================================================
class SFHSolver:

    def __init__(self, patch_size=7, max_exemplars=SFH_MAX_EXEMPLARS):
        self.ps = patch_size
        self.max_exemplars = max_exemplars
        self.ex_lr: List[np.ndarray] = []
        self.ex_hr: List[np.ndarray] = []

    def build(self, paths: List[str], seed: int = SEED) -> None:
        print(f"[SFH] Building exemplar DB ({self.max_exemplars} images) ...")
        rng = random.Random(seed)
        for path in rng.sample(paths, min(self.max_exemplars, len(paths))):
            try:
                hr = face_crop_and_resize(Image.open(path).convert("L"))
                lr = cv2.resize(hr, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC)
                self.ex_lr.append(lr)
                self.ex_hr.append(hr)
            except Exception:
                continue
        print(f"[SFH] Exemplar DB: {len(self.ex_lr)} images")

    def _best_exemplar(self, lr_q: np.ndarray) -> np.ndarray:
        q = lr_q.ravel().astype(np.float64)
        q -= q.mean()
        qn = np.linalg.norm(q) + 1e-8

        best_score = -np.inf
        best_hr = self.ex_hr[0]

        for lr_e, hr_e in zip(self.ex_lr, self.ex_hr):
            e = lr_e.ravel().astype(np.float64)
            e -= e.mean()
            score = np.dot(q, e) / (qn * (np.linalg.norm(e) + 1e-8))
            if score > best_score:
                best_score = score
                best_hr = hr_e

        return best_hr

    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        best_hr = self._best_exemplar(lr)
        lr_up = cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        gx_lr = cv2.Sobel(lr_up, cv2.CV_64F, 1, 0, ksize=3)
        gy_lr = cv2.Sobel(lr_up, cv2.CV_64F, 0, 1, ksize=3)
        gx_hr = cv2.Sobel(best_hr, cv2.CV_64F, 1, 0, ksize=3)
        gy_hr = cv2.Sobel(best_hr, cv2.CV_64F, 0, 1, ksize=3)

        mag_lr = np.sqrt(gx_lr**2 + gy_lr**2)
        mag_hr = np.sqrt(gx_hr**2 + gy_hr**2)
        w = np.clip(mag_lr / (mag_lr.max() + 1e-8), 0, 1)

        scale = np.where(
            mag_hr > 1e-6,
            mag_lr / (mag_hr + 1e-6),
            1.0
        ).clip(0, 4.0)

        gx_out = w * scale * gx_hr + (1 - w) * gx_lr
        gy_out = w * scale * gy_hr + (1 - w) * gy_lr

        div = (
            cv2.Sobel(gx_out, cv2.CV_64F, 1, 0, ksize=3) +
            cv2.Sobel(gy_out, cv2.CV_64F, 0, 1, ksize=3)
        )
        hr_rec = (lr_up + 0.08 * div).astype(np.float32)

        for _ in range(20):
            lr_est = cv2.resize(hr_rec, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.5 * cv2.resize(lr - lr_est, HR_SIZE, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0, 1)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2
        ).clip(0, 1).astype(np.float32)


# ============================================================
# Evaluation
# ============================================================
def evaluate_method(
    name: str,
    sr_fn,
    test_paths: List[str],
    blur_type: str,
    gaussian_sigma=None,
    motion_length=None,
    base_seed=SEED,
    save_sample=True
) -> Dict:
    total_p = 0.0
    total_s = 0.0
    n = 0
    sample_saved = False

    for idx, path in enumerate(test_paths):
        try:
            hr = face_crop_and_resize(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] {path}: {e}")
            continue

        theta = (
            random.Random(base_seed + idx).uniform(-math.pi, math.pi)
            if blur_type == "motion" else None
        )

        lr = degrade(
            hr,
            blur_type,
            gaussian_sigma=gaussian_sigma,
            motion_length=motion_length,
            theta=theta
        )
        sr = sr_fn(lr)

        total_p += compute_psnr(sr, hr)
        total_s += compute_ssim(sr, hr)
        n += 1

        if save_sample and not sample_saved:
            tag = (
                f"gaussian_s{gaussian_sigma}"
                if blur_type == "gaussian"
                else f"motion_l{motion_length}"
            )
            sd = RESULTS_DIR / "samples" / name / tag
            sd.mkdir(parents=True, exist_ok=True)

            def sv(arr, p):
                Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(p)

            sv(
                cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC).clip(0, 1),
                sd / "lr_bicubic.png"
            )
            sv(sr, sd / "sr.png")
            sv(hr, sd / "hr.png")
            sample_saved = True

    avg_p = total_p / max(n, 1)
    avg_s = total_s / max(n, 1)
    blur_label = (
        f"gaussian_sigma_{gaussian_sigma}"
        if blur_type == "gaussian"
        else f"motion_l_{motion_length}"
    )

    print(
        f"  {name:8s} | {blur_label:25s} | "
        f"PSNR={avg_p:.2f} dB | SSIM={avg_s:.4f}  "
        f"(n={n}, ch={METRIC_CHANNEL.upper()})"
    )

    return {
        "method": name,
        "blur_type": blur_label,
        "psnr": round(avg_p, 4),
        "ssim": round(avg_s, 4)
    }


def create_splits_from_folder(image_dir, train_ratio=0.8, max_images=None, seed=42):
    paths = sorted(Path(image_dir).glob("*.jpg"))
    if not paths:
        raise ValueError(f"No .jpg images in {image_dir}")
    if max_images:
        paths = paths[:max_images]
    random.Random(seed).shuffle(paths)
    cut = int(len(paths) * train_ratio)
    return [str(p) for p in paths[:cut]], [str(p) for p in paths[cut:]]


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Metric channel: {METRIC_CHANNEL.upper()}")

    IMAGE_DIR = WORKDIR / "img_align_celeba"

    train_paths, test_paths = create_splits_from_folder(
        IMAGE_DIR,
        train_ratio=0.80,
        max_images=2500
    )
    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

    sc1 = SC1Solver()
    sc1.build(train_paths)

    sc2 = SC2Solver()
    sc2.build(train_paths)

    sfh = SFHSolver()
    sfh.build(train_paths)

    methods = {
        "Bicubic": bicubic_sr,
        "SC1": sc1.sr_rgb,
        "SC2": sc2.sr_rgb,
        "SFH": sfh.sr_rgb,
    }

    settings = [
        dict(blur_type="gaussian", gaussian_sigma=1),
        dict(blur_type="gaussian", gaussian_sigma=3),
        dict(blur_type="gaussian", gaussian_sigma=5),
        dict(blur_type="motion", motion_length=2),
        dict(blur_type="motion", motion_length=6),
        dict(blur_type="motion", motion_length=9),
    ]

    rows = []
    t0 = time.time()

    for mname, sr_fn in methods.items():
        print(f"\n{'='*60}\nMethod: {mname}\n{'='*60}")
        for cfg in settings:
            rows.append(
                evaluate_method(mname, sr_fn, test_paths, **cfg, save_sample=True)
            )

    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")

    out_csv = RESULTS_DIR / "classical_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "blur_type", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {out_csv}")

    blur_tags = [r["blur_type"] for r in rows if r["method"] == "Bicubic"]
    header = f"{'Blur':<26}" + "".join(f"{m:>10}" for m in methods)

    for metric in ("psnr", "ssim"):
        print(f"\n{metric.upper()}")
        print(header)
        for bt in blur_tags:
            row_str = f"{bt:<26}"
            for m in methods:
                v = next(
                    r[metric] for r in rows
                    if r["method"] == m and r["blur_type"] == bt
                )
                row_str += f"{v:>10.2f}" if metric == "psnr" else f"{v:>10.4f}"
            print(row_str)

    print(f"\nMetric channel used: {METRIC_CHANNEL.upper()}")


if __name__ == "__main__":
    main()