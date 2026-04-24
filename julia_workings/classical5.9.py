"""
Classical9: Optimized SC1, SC2, and SFH
Improvements for Mild Blur (motion_l_2):
- SC1: Replaced fixed 0.5 shrinkage with a Feature-Confidence Gate.
- SFH: Added an Input-Quality Gate to the Gain to prevent over-hallucination on mild blurs.
- SC2: Maintained Confidence Gating.
"""

import json
import csv
import math
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from common_config import CLASSICAL_CONFIG

# ============================================================
# Config
# ============================================================
WORKDIR = CLASSICAL_CONFIG["WORKDIR"]
SPLITS_DIR = CLASSICAL_CONFIG["SPLITS_DIR"]
RESULTS_DIR = CLASSICAL_CONFIG["RESULTS_DIR"]
IMAGE_DIR = CLASSICAL_CONFIG["IMAGE_DIR"]

SEED = CLASSICAL_CONFIG["SEED"]
HR_SIZE = CLASSICAL_CONFIG["HR_SIZE"]
TEST_LR_SIZE = CLASSICAL_CONFIG["TEST_LR_SIZE"]

CELEBA_CROP = CLASSICAL_CONFIG["CELEBA_CROP"]
CROP_FRAC = CLASSICAL_CONFIG["CROP_FRAC"]

max_images = CLASSICAL_CONFIG["max_images"]
N_TRAIN = CLASSICAL_CONFIG["N_TRAIN"]
N_TEST = CLASSICAL_CONFIG["N_TEST"]
USE_SPLITS = CLASSICAL_CONFIG["USE_SPLITS"]

SC1_PATCH_SIZE = CLASSICAL_CONFIG["SC1_PATCH_SIZE"]
SC1_DICT_SIZE = CLASSICAL_CONFIG["SC1_DICT_SIZE"]
SC1_LAMBDA = CLASSICAL_CONFIG["SC1_LAMBDA"]
SC1_N_TRAIN = CLASSICAL_CONFIG["SC1_N_TRAIN"]
SC1_ISTA_ITERS = CLASSICAL_CONFIG["SC1_ISTA_ITERS"]

SC2_PATCH_SIZE = CLASSICAL_CONFIG["SC2_PATCH_SIZE"]
SC2_SIGMA_K = CLASSICAL_CONFIG["SC2_SIGMA_K"]
SC2_LAMBDA_REG = CLASSICAL_CONFIG["SC2_LAMBDA_REG"]
SC2_N_BASIS = CLASSICAL_CONFIG["SC2_N_BASIS"]
SC2_N_TRAIN = CLASSICAL_CONFIG["SC2_N_TRAIN"]

SFH_MAX_EXEMPLARS = CLASSICAL_CONFIG["SFH_MAX_EXEMPLARS"]

# ============================================================
# Dataset Utilities
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def celeba_crop(pil_img: Image.Image) -> Image.Image:
    if not CELEBA_CROP:
        return pil_img
    w, h   = pil_img.size
    side   = int(min(w, h) * CROP_FRAC)
    left   = (w - side) // 2
    top    = max(0, (h - side) // 2 + int(side * 0.15))
    return pil_img.crop((left, top, left + side, top + side))

def load_hr(path: str) -> np.ndarray:
    img = celeba_crop(Image.open(path).convert("RGB"))
    return np.array(img.resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0

def make_splits(image_root: Path, max_images: int = max_images, seed: int = SEED,
                train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, 
                splits_dir: Path = SPLITS_DIR) -> Tuple[List[str], List[str], List[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.")
    all_images = sorted(str(p) for p in Path(image_root).glob("*.jpg"))
    if len(all_images) < max_images:
        max_images = len(all_images)
    rng     = random.Random(seed)
    sampled = rng.sample(all_images, max_images)
    n_train = int(max_images * train_ratio)
    n_val   = int(max_images * val_ratio)
    train_f = sampled[:n_train]
    val_f   = sampled[n_train:n_train + n_val]
    test_f  = sampled[n_train + n_val:]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in[("train", train_f), ("val", val_f), ("test", test_f)]:
        with open(splits_dir / f"{name}.txt", "w", encoding="utf-8") as fh:
            fh.writelines(p + "\n" for p in lst)
    summary = {"image_root": str(image_root), "seed": seed, "max_images": max_images, "train": len(train_f), "val": len(val_f), "test": len(test_f)}
    with open(splits_dir / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return train_f, val_f, test_f

def load_split(name: str) -> List[str]:
    p = SPLITS_DIR / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return[l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

# ============================================================
# Preprocessing
# ============================================================
def motion_kernel(length: int, theta: float) -> np.ndarray:
    length = max(2, int(length))
    kernel = np.zeros((length, length), dtype=np.float32)
    c  = (length - 1) / 2.0
    x0 = int(round(c - c * math.cos(theta)))
    y0 = int(round(c - c * math.sin(theta)))
    x1 = int(round(c + c * math.cos(theta)))
    y1 = int(round(c + c * math.sin(theta)))
    cv2.line(kernel, (x0, y0), (x1, y1), 1, thickness=1)
    s = kernel.sum()
    if s > 0: kernel /= s
    else: kernel[length // 2, length // 2] = 1.0
    return kernel

def generate_train_lr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if random.random() < 0.5:
        sigma = random.uniform(0, 7)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 1e-6 else img.copy()
    else:
        length = random.randint(0, 11)
        theta = random.uniform(-math.pi, math.pi)
        blurred = cv2.filter2D(img, -1, motion_kernel(length, theta)) if length > 1 else img.copy()
    scale = random.randint(2, 5)
    lr_temp = cv2.resize(blurred, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(lr_temp, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC).astype(np.float32)

def degrade(hr: np.ndarray, blur_type: str, gaussian_sigma=None, motion_length=None, theta=None) -> np.ndarray:
    if hr.ndim == 2: hr = np.stack([hr]*3, axis=2)
    if blur_type == "gaussian":
        bl = cv2.GaussianBlur(hr, (0,0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    else:
        bl = cv2.filter2D(hr, -1, motion_kernel(motion_length, theta))
    return cv2.resize(bl, TEST_LR_SIZE, interpolation=cv2.INTER_CUBIC).astype(np.float32)

# ============================================================
# Metrics
# ============================================================
def rgb_to_y(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else float(10*np.log10(1.0/mse))

def ssim_np(pred: np.ndarray, target: np.ndarray, win=11) -> float:
    C1, C2 = 0.01**2, 0.03**2
    k1d = np.exp(-(np.arange(win)-win//2)**2/(2*1.5**2)); k1d/=k1d.sum()
    k2d = np.outer(k1d, k1d).astype(np.float64)
    scores =[]
    if pred.ndim == 2: pred = pred[:, :, None]
    if target.ndim == 2: target = target[:, :, None]
    for c in range(pred.shape[2]):
        p = pred[:, :, c].astype(np.float64); t = target[:, :, c].astype(np.float64)
        kw = dict(ddepth=-1, kernel=k2d, borderType=cv2.BORDER_REFLECT_101)
        mu1,mu2 = cv2.filter2D(p,**kw), cv2.filter2D(t,**kw)
        s1  = cv2.filter2D(p*p,**kw)-mu1**2
        s2  = cv2.filter2D(t*t,**kw)-mu2**2
        s12 = cv2.filter2D(p*t,**kw)-mu1*mu2
        num = (2*mu1*mu2+C1)*(2*s12+C2)
        den = (mu1**2+mu2**2+C1)*(s1+s2+C2)
        scores.append(float(np.mean(num/(den+1e-12))))
    return float(np.mean(scores))

_FEAT_FILTERS =[
    np.array([[-1,0,1]],  dtype=np.float32),              
    np.array([[-1],[0],[1]], dtype=np.float32),           
    np.array([[1,0,-2,0,1]],  dtype=np.float32),          
    np.array([[1],[0],[-2],[0],[1]], dtype=np.float32),   
]

def extract_features(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([
        cv2.filter2D(patch,-1,f,borderType=cv2.BORDER_REFLECT).ravel()
        for f in _FEAT_FILTERS
    ]).astype(np.float32)

def bicubic_sr(lr: np.ndarray) -> np.ndarray:
    return cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC).clip(0,1).astype(np.float32)

# ============================================================
# Method 2: SC1 — Sparse Coding SR (Adaptive Confidence Improvement)
# ============================================================
class SC1Solver:
    def __init__(self, patch_size=SC1_PATCH_SIZE, dict_size=SC1_DICT_SIZE, 
                 lam=SC1_LAMBDA, n_train=SC1_N_TRAIN, ista_iters=SC1_ISTA_ITERS):
        self.ps = patch_size
        self.ds = dict_size
        self.lam = lam
        self.nt = n_train
        self.ista_iters = ista_iters
        self.Dl = self.Dh = self._DtD = self._L = None

    def _collect(self, paths, rng):
        ps = self.ps
        Xl, Yl = [],[]
        per = max(1, (self.ds * 5) // max(1, len(paths)))
        for path in rng.sample(paths, min(len(paths), self.nt)):
            try:
                img_raw = celeba_crop(Image.open(path).convert("RGB"))
                hr_rgb = np.array(img_raw.resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_small = generate_train_lr(hr_g)
                lr_up = cv2.resize(lr_small, HR_SIZE, interpolation=cv2.INTER_CUBIC)
            except Exception: continue
            H, W = lr_up.shape
            step = 2 
            pos =[(r, c) for r in range(0, H - ps + 1, step) for c in range(0, W - ps + 1, step)]
            for r, c in rng.sample(pos, min(per, len(pos))):
                lr_patch = lr_up[r:r+ps, c:c+ps]
                hr_patch = hr_g[r:r+ps, c:c+ps]
                feat = extract_features(lr_patch)
                fn = np.linalg.norm(feat)
                if fn < 1e-4: continue
                residual = hr_patch.ravel() - lr_patch.ravel()
                Xl.append(feat / fn)
                Yl.append(residual / fn)
        return (np.stack(Xl).astype(np.float32), np.stack(Yl).astype(np.float32))

    def build(self, paths: List[str], seed=SEED):
        rng = random.Random(seed)
        Xtr, Ytr = self._collect(paths, rng)
        n = Xtr.shape[0]
        if n > self.ds:
            try:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(n_clusters=self.ds, random_state=seed, batch_size=min(4096, n), n_init=3)
                lbl = km.fit_predict(Xtr)
                Dl, Dh = [],[]
                for k in range(self.ds):
                    idx = np.where(lbl == k)[0]
                    if len(idx) == 0: idx =[rng.randint(0, n - 1)]
                    center = km.cluster_centers_[k]
                    best = int(idx[np.argmin(np.linalg.norm(Xtr[idx] - center, axis=1))])
                    Dl.append(Xtr[best])
                    Dh.append(Ytr[best])
                self.Dl = np.stack(Dl, axis=1).astype(np.float32)
                self.Dh = np.stack(Dh, axis=1).astype(np.float32)
            except ImportError:
                idx = np.random.default_rng(seed).choice(n, self.ds, replace=False)
                self.Dl = Xtr[idx].T.astype(np.float32)
                self.Dh = Ytr[idx].T.astype(np.float32)
        else:
            self.Dl = Xtr.T.astype(np.float32)
            self.Dh = Ytr.T.astype(np.float32)
        norms = np.linalg.norm(self.Dl, axis=0, keepdims=True).clip(1e-8)
        self.Dl /= norms
        self.Dh /= norms 
        self._DtD = self.Dl.T @ self.Dl
        self._L   = float(np.linalg.norm(self._DtD, ord=2)) + 1e-6

    def _ista(self, y):
        Dty, thresh = self.Dl.T @ y, self.lam / self._L
        a = np.zeros(self.Dl.shape[1], dtype=np.float32)
        for _ in range(self.ista_iters):
            a -= (self._DtD@a - Dty) / self._L
            a  = np.sign(a) * np.maximum(np.abs(a)-thresh, 0)
        return a

    def _recon_channel(self, lr_ch: np.ndarray) -> np.ndarray:
        ps = self.ps
        step = 2 
        lr_up = cv2.resize(lr_ch, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W  = lr_up.shape
        acc, cnt = np.zeros((H,W)), np.zeros((H,W))
        for r in range(0, H-ps+1, step):
            for c in range(0, W-ps+1, step):
                lr_patch = lr_up[r:r+ps, c:c+ps]
                feat = extract_features(lr_patch).astype(np.float32)
                fn   = np.linalg.norm(feat)
                if fn < 1e-4: continue
                alpha = self._ista(feat / fn)
                
                # IMPROVEMENT: Feature Confidence Gating
                # Instead of fixed 0.5, scale residual by how well the sparse code represents the input
                recon_feat = self.Dl @ alpha
                conf = 1.0 - np.linalg.norm((feat/fn) - recon_feat) / (np.linalg.norm(feat/fn) + 1e-6)
                conf = np.clip(conf, 0, 1)
                
                residual_pred = (self.Dh @ alpha) * fn * conf * 0.4
                acc[r:r+ps, c:c+ps] += residual_pred.reshape(ps, ps)
                cnt[r:r+ps, c:c+ps] += 1.0
        return (lr_up + (acc / np.maximum(cnt, 1.0))).clip(0, 1).astype(np.float32)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        lr_ycrcb = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        hr_y = self._recon_channel(lr_ycrcb[:,:,0])
        hr_cr = cv2.resize(lr_ycrcb[:,:,1], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(lr_ycrcb[:,:,2], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_ycrcb = np.stack([hr_y, hr_cr, hr_cb], axis=2)
        return cv2.cvtColor(hr_ycrcb, cv2.COLOR_YCrCb2RGB).clip(0,1).astype(np.float32)

# ============================================================
# Method 3: SC2 — Full Kernel Ridge Regression SR
# ============================================================
class SC2Solver:
    def __init__(self, patch_size=SC2_PATCH_SIZE, sigma_k=SC2_SIGMA_K, 
                 lam=SC2_LAMBDA_REG, n_basis=SC2_N_BASIS, n_train=SC2_N_TRAIN):
        self.ps = patch_size
        self.sk = sigma_k
        self.lam = lam
        self.nb = n_basis
        self.nt = n_train
        self.B = self.A = None

    def _kern(self, X, Y):
        d2 = (np.sum(X**2, 1, keepdims=True) + np.sum(Y**2, 1, keepdims=True).T - 2 * X @ Y.T).clip(0)
        return np.exp(-d2 / (2 * self.sk**2)).astype(np.float32)

    def build(self, paths: List[str], seed=SEED):
        rng = random.Random(seed)
        per = max(1, (self.nb * 5) // max(1, len(paths)))
        Xl, Yl =[],[]
        for path in rng.sample(paths, min(len(paths), self.nt)):
            try:
                img_raw = celeba_crop(Image.open(path).convert("RGB"))
                hr_rgb = np.array(img_raw.resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_small = generate_train_lr(hr_g)
                lr_up = cv2.resize(lr_small, HR_SIZE, interpolation=cv2.INTER_CUBIC)
            except Exception: continue
            H, W = lr_up.shape
            step = 2 
            pos =[(r, c) for r in range(0, H - self.ps + 1, step) for c in range(0, W - self.ps + 1, step)]
            for r, c in rng.sample(pos, min(per, len(pos))):
                lr_patch = lr_up[r:r+self.ps, c:c+self.ps]
                hr_patch = hr_g[r:r+self.ps, c:c+self.ps]
                l1_norm = np.sum(np.abs(lr_patch))
                if l1_norm < 1e-4: continue
                feat = extract_features(lr_patch / l1_norm)
                residual = hr_patch.ravel() - lr_patch.ravel()
                Xl.append(feat)
                Yl.append(residual)
        Xtr = np.stack(Xl).astype(np.float32)
        Ytr = np.stack(Yl).astype(np.float32)
        n = Xtr.shape[0]
        if n > self.nb:
            try:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(n_clusters=self.nb, random_state=seed, batch_size=min(4096, n), n_init=3)
                lbl = km.fit_predict(Xtr)
                B =[]
                for k in range(self.nb):
                    idx = np.where(lbl == k)[0]
                    if len(idx) == 0: idx =[rng.randint(0, n - 1)]
                    center = km.cluster_centers_[k]
                    best = int(idx[np.argmin(np.linalg.norm(Xtr[idx] - center, axis=1))])
                    B.append(Xtr[best])
                self.B = np.stack(B).astype(np.float32)
            except ImportError:
                bidx = np.random.default_rng(seed).choice(n, self.nb, replace=False)
                self.B = Xtr[bidx]
        else:
            self.B = Xtr
            self.nb = n
        Kxb = self._kern(self.B, Xtr).astype(np.float64) 
        Kbb = self._kern(self.B, self.B).astype(np.float64) 
        matrix_to_inv = (Kxb @ Kxb.T) + (self.lam * Kbb) + (1e-6 * np.eye(self.nb))
        rhs = Kxb @ Ytr.astype(np.float64)
        self.A = np.linalg.solve(matrix_to_inv, rhs).astype(np.float32)

    def _recon_channel(self, lr_ch: np.ndarray) -> np.ndarray:
        lr_up = cv2.resize(lr_ch, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W = lr_up.shape
        acc, cnt = np.zeros((H, W)), np.zeros((H, W))
        step = 2  
        for r in range(0, H - self.ps + 1, step):
            for c in range(0, W - self.ps + 1, step):
                lr_patch = lr_up[r:r+self.ps, c:c+self.ps]
                l1_norm = np.sum(np.abs(lr_patch))
                if l1_norm < 1e-4: continue
                feat = extract_features(lr_patch / l1_norm).astype(np.float32)
                k_vec = self._kern(feat.reshape(1, -1), self.B) 
                conf = np.max(k_vec)
                residual_pred = (k_vec @ self.A).ravel() * conf
                acc[r:r+self.ps, c:c+self.ps] += residual_pred.reshape(self.ps, self.ps)
                cnt[r:r+self.ps, c:c+self.ps] += 1.0
        return (lr_up + (acc / np.maximum(cnt, 1.0))).clip(0, 1).astype(np.float32)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        lr_ycrcb = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        hr_y = self._recon_channel(lr_ycrcb[:,:,0])
        hr_cr = cv2.resize(lr_ycrcb[:,:,1], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(lr_ycrcb[:,:,2], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_ycrcb = np.stack([hr_y, hr_cr, hr_cb], axis=2)
        return cv2.cvtColor(hr_ycrcb, cv2.COLOR_YCrCb2RGB).clip(0,1).astype(np.float32)

# ============================================================
# Method 4: SFH — Structured Face Hallucination (Yang et al. Optimized)
# ============================================================
class SFHSolver:
    def __init__(self, max_exemplars=SFH_MAX_EXEMPLARS):
        self.max_exemplars = max_exemplars
        self.ex_hr = []
        self.ex_lr = []
        self.zones = {"upper": (20, 45), "middle": (45, 65), "lower": (65, 90)}

    def build(self, paths: List[str], seed=SEED):
        rng = random.Random(seed)
        for path in rng.sample(paths, min(self.max_exemplars, len(paths))):
            try:
                img_raw = celeba_crop(Image.open(path).convert("RGB"))
                hr_rgb = np.array(img_raw.resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_small = generate_train_lr(hr_g)
                self.ex_hr.append(hr_rgb)
                self.ex_lr.append(lr_small)
            except Exception: continue

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        hr_bicubic = bicubic_sr(lr)
        lr_g = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
        hr_bicubic_g = cv2.cvtColor(hr_bicubic, cv2.COLOR_RGB2GRAY)
        hallucinated_y = hr_bicubic_g.copy()
        
        # Global Image Quality Gate: Check if image is already "sharp"
        # This prevents over-hallucinating on mild blurs
        global_lap = cv2.Laplacian(lr_g, cv2.CV_32F).var()
        quality_gate = 1.0 if global_lap < 50 else 0.4 

        for zone_name, (y0, y1) in self.zones.items():
            ly0, ly1 = y0 // 2, y1 // 2
            lr_zone = lr_g[ly0:ly1, :]
            q = lr_zone.ravel().astype(np.float64); q -= q.mean(); qn = np.linalg.norm(q) + 1e-8
            
            best_idx, max_sim = 0, -np.inf
            for i, ex_lr in enumerate(self.ex_lr):
                ex_z = ex_lr[ly0:ly1, :]
                e = ex_z.ravel().astype(np.float64); e -= e.mean()
                sim = np.dot(q, e) / (qn * (np.linalg.norm(e) + 1e-8))
                if sim > max_sim: max_sim, best_idx = sim, i
            
            ex_hr_g = cv2.cvtColor(self.ex_hr[best_idx], cv2.COLOR_RGB2GRAY)
            ex_hr_zone = ex_hr_g[y0:y1, :]
            lap = cv2.Laplacian(ex_hr_zone, cv2.CV_32F, ksize=3)
            target_lap = cv2.Laplacian(cv2.resize(lr_zone, (100, y1-y0), interpolation=cv2.INTER_CUBIC), cv2.CV_32F)
            contrast_mask = np.abs(target_lap) / (np.max(np.abs(target_lap)) + 1e-8)
            
            # IMPROVEMENT: Combined confidence and quality gate
            gain = 0.12 * max(0, max_sim) * quality_gate
            hallucinated_y[y0:y1, :] += gain * lap * contrast_mask

        # Back-Projection Correction
        lr_est = cv2.resize(hallucinated_y, TEST_LR_SIZE, interpolation=cv2.INTER_AREA)
        correction = cv2.resize(lr_g - lr_est, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        final_y = np.clip(hallucinated_y + correction, 0, 1)
        
        lr_ycrcb = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        hr_ycrcb = np.stack([final_y, cv2.resize(lr_ycrcb[:,:,1], HR_SIZE, interpolation=cv2.INTER_CUBIC), 
                             cv2.resize(lr_ycrcb[:,:,2], HR_SIZE, interpolation=cv2.INTER_CUBIC)], axis=2)
        return cv2.cvtColor(hr_ycrcb, cv2.COLOR_YCrCb2RGB).clip(0,1).astype(np.float32)

# ============================================================
# Evaluation & Main
# ============================================================
def evaluate_method(name, sr_fn, test_paths, blur_type, gaussian_sigma=None, motion_length=None, base_seed=SEED, save_sample=True) -> Dict:
    total_p = total_s = n = 0
    for idx, path in enumerate(test_paths):
        try: hr = load_hr(path)
        except Exception: continue
        theta = random.Random(base_seed+idx).uniform(-math.pi, math.pi) if blur_type == "motion" else None
        lr = degrade(hr, blur_type, gaussian_sigma=gaussian_sigma, motion_length=motion_length, theta=theta)
        try: sr = sr_fn(lr)
        except Exception: continue
        sr_y, hr_y = rgb_to_y(sr), rgb_to_y(hr)
        if sr_y.ndim != 2 or hr_y.ndim != 2: continue
        total_p += psnr_np(sr_y, hr_y)
        total_s += ssim_np(sr_y, hr_y)
        n += 1
        if save_sample and idx == 0:
            tag = f"gaussian_s{gaussian_sigma}" if blur_type=="gaussian" else f"motion_l{motion_length}"
            sd = RESULTS_DIR/"samples"/name/tag
            sd.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.clip(sr,0,1)*255).astype(np.uint8)).save(sd/"sr.png")

    avg_p, avg_s = total_p / max(n, 1), total_s / max(n, 1)
    label = f"gaussian_sigma_{gaussian_sigma}" if blur_type=="gaussian" else f"motion_l_{motion_length}"
    print(f"  {name:8s} | {label:25s} | PSNR_Y={avg_p:.2f} | SSIM_Y={avg_s:.4f}")
    return {"method":name, "blur_type":label, "psnr":round(avg_p,4), "ssim":round(avg_s,4)}

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if USE_SPLITS:
        try: train_paths, test_paths = load_split("train"), load_split("test")
        except FileNotFoundError: train_paths, _, test_paths = make_splits(IMAGE_DIR)
    else: train_paths, _, test_paths = make_splits(IMAGE_DIR)
    train_paths, test_paths = train_paths[:N_TRAIN], test_paths[:N_TEST]

    sc1 = SC1Solver(); sc1.build(train_paths)
    sc2 = SC2Solver(); sc2.build(train_paths)
    sfh = SFHSolver(); sfh.build(train_paths)

    methods = {"Bicubic": bicubic_sr, "SC1": sc1.sr_rgb, "SC2": sc2.sr_rgb, "SFH": sfh.sr_rgb}
    settings =[dict(blur_type="gaussian", gaussian_sigma=1), dict(blur_type="gaussian", gaussian_sigma=3),
               dict(blur_type="gaussian", gaussian_sigma=5), dict(blur_type="motion", motion_length=2),
               dict(blur_type="motion", motion_length=6), dict(blur_type="motion", motion_length=9)]

    rows = []; t0 = time.time()
    for mname, sr_fn in methods.items():
        print(f"\nMethod: {mname}")
        for cfg in settings: rows.append(evaluate_method(mname, sr_fn, test_paths, **cfg))

    out_csv = RESULTS_DIR / "classical_metrics.csv"
    with open(out_csv,"w",newline="",encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method","blur_type","psnr","ssim"])
        w.writeheader(); w.writerows(rows)

    blur_tags = [r["blur_type"] for r in rows if r["method"]=="Bicubic"]
    for metric in ("psnr","ssim"):
        print(f"\n{metric.upper()}\n{'Blur':<26}" + "".join(f"{m:>10}" for m in methods))
        for bt in blur_tags:
            row_str = f"{bt:<26}"
            for m in methods:
                v = next(r[metric] for r in rows if r["method"]==m and r["blur_type"]==bt)
                row_str += f"{v:>10.2f}" if metric=="psnr" else f"{v:>10.4f}"
            print(row_str)

if __name__ == "__main__":
    main()