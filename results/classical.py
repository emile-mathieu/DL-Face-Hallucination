"""
classical.py
============
Classical SR methods:
  - bicubic_sr()
  - SC1Solver  (sparse coding, Yang et al. 2008/2010)
  - SC2Solver  (kernel ridge regression, Kim & Kwon 2010)
  - SFHSolver  (structured face hallucination, Yang et al. 2013)
  - evaluate_method()
  - run_classical_pipeline()
"""

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(0) 
from sklearn.cluster import MiniBatchKMeans

from config import (
    HR_SIZE, TEST_FIXED_LR_SIZE, CL_SEED,
    SC1_PATCH_SIZE, SC1_DICT_SIZE, SC1_LAMBDA, SC1_N_TRAIN, SC1_ISTA_ITERS,
    SC2_PATCH_SIZE, SC2_SIGMA_K, SC2_LAMBDA_REG, SC2_N_BASIS, SC2_N_TRAIN,
    SFH_MAX_EXEMPLARS, RESULTS_CL_DIR,
)
from utils import celeba_crop, psnr_np, ssim_np, rgb_to_y_np


# ── Degradation helpers ───────────────────────────────────────
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
    if s > 0:
        kernel /= s
    else:
        kernel[length // 2, length // 2] = 1.0
    return kernel


def generate_train_lr(img: np.ndarray) -> np.ndarray:
    """Random blur + downsample for classical model training patches."""
    h, w = img.shape[:2]
    if random.random() < 0.5:
        sigma   = random.uniform(0, 7)
        blurred = (cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
                   if sigma > 1e-6 else img.copy())
    else:
        length  = random.randint(0, 11)
        theta   = random.uniform(-math.pi, math.pi)
        blurred = (cv2.filter2D(img, -1, motion_kernel(length, theta))
                   if length > 1 else img.copy())
    scale   = random.randint(2, 5)
    lr_temp = cv2.resize(blurred, (max(1, w // scale), max(1, h // scale)),
                         interpolation=cv2.INTER_CUBIC)
    return cv2.resize(lr_temp, TEST_FIXED_LR_SIZE,
                      interpolation=cv2.INTER_CUBIC).astype(np.float32)


def degrade(hr: np.ndarray, blur_type: str,
            gaussian_sigma=None, motion_length=None, theta=None) -> np.ndarray:
    """Apply fixed test degradation to an HR image."""
    if hr.ndim == 2:
        hr = np.stack([hr] * 3, axis=2)
    if blur_type == "gaussian":
        bl = cv2.GaussianBlur(hr, (0, 0),
                              sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    else:
        bl = cv2.filter2D(hr, -1, motion_kernel(motion_length, theta))
    return cv2.resize(bl, TEST_FIXED_LR_SIZE,
                      interpolation=cv2.INTER_CUBIC).astype(np.float32)


def load_hr(path: str) -> np.ndarray:
    img = celeba_crop(Image.open(path).convert("RGB"))
    return np.array(img.resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0


# ── Feature extraction (shared by SC1 and SC2) ───────────────
_FEAT_FILTERS = [
    np.array([[-1, 0, 1]],               dtype=np.float32),
    np.array([[-1], [0], [1]],           dtype=np.float32),
    np.array([[1, 0, -2, 0, 1]],         dtype=np.float32),
    np.array([[1], [0], [-2], [0], [1]], dtype=np.float32),
]


def extract_features(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([
        cv2.filter2D(patch, -1, f, borderType=cv2.BORDER_REFLECT).ravel()
        for f in _FEAT_FILTERS
    ]).astype(np.float32)


def bicubic_sr(lr: np.ndarray) -> np.ndarray:
    return cv2.resize(lr, HR_SIZE, interpolation=cv2.INTER_CUBIC).clip(0, 1).astype(np.float32)


# ── SC1 ───────────────────────────────────────────────────────
class SC1Solver:
    """
    Sparse coding SR (Yang et al. 2008 / 2010).
    ISTA sparse coding with confidence-gated residual prediction.
    Operates on Y channel only; CrCb are bicubic-upsampled.
    """
    def __init__(self, patch_size=SC1_PATCH_SIZE, dict_size=SC1_DICT_SIZE,
                 lam=SC1_LAMBDA, n_train=SC1_N_TRAIN, ista_iters=SC1_ISTA_ITERS):
        self.ps         = patch_size
        self.ds         = dict_size
        self.lam        = lam
        self.nt         = n_train
        self.ista_iters = ista_iters
        self.Dl = self.Dh = self._DtD = self._L = None

    def _collect(self, paths, rng):
        ps  = self.ps
        Xl, Yl = [], []
        sampled_paths = rng.sample(paths, min(len(paths), self.nt))
        per = max(1, (self.ds * 5) // max(1, len(paths)))
        
        print(f"  [SC1] Collecting patches from {len(sampled_paths)} images...", flush=True)
        
        for i, path in enumerate(sampled_paths):
            if i % 20 == 0: # Print progress every 20 images
                print(f"    Processing image {i}/{len(sampled_paths)}...", flush=True)
            try:
                hr_rgb  = np.array(
                    celeba_crop(Image.open(path).convert("RGB"))
                    .resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g    = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_s    = generate_train_lr(hr_g)
                lr_up   = cv2.resize(lr_s, HR_SIZE, interpolation=cv2.INTER_CUBIC)
            except Exception:
                print(f"  [SC1] Error processing image {path}", flush=True)
                continue
            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 2)
                           for c in range(0, W - ps + 1, 2)]
            for r, c in rng.sample(pos, min(per, len(pos))):
                lp   = lr_up[r:r+ps, c:c+ps]
                hp   = hr_g [r:r+ps, c:c+ps]
                feat = extract_features(lp)
                fn   = np.linalg.norm(feat)
                if fn < 1e-4:
                    continue
                Xl.append(feat / fn)
                Yl.append((hp.ravel() - lp.ravel()) / fn)
        return (np.stack(Xl).astype(np.float32),
                np.stack(Yl).astype(np.float32))

    def build(self, paths: List[str], seed=CL_SEED):
        print("[SC1] Building dictionary ...", flush=True)
        rng      = random.Random(seed)
        Xtr, Ytr = self._collect(paths, rng)

        n        = Xtr.shape[0]
        print(f"[SC1] Collected {n} patches. Starting KMeans...", flush=True)

        if n > self.ds:
            km  = MiniBatchKMeans(n_clusters=self.ds, random_state=seed,
                                  batch_size=min(4096, n), n_init=3)
            lbl = km.fit_predict(Xtr)
            print("[SC1] KMeans finished. Mapping centroids to patches...", flush=True)
            Dl, Dh = [], []
            for k in range(self.ds):
                idx  = np.where(lbl == k)[0]
                if len(idx) == 0:
                    idx = [rng.randint(0, n - 1)]
                best = int(idx[np.argmin(
                    np.linalg.norm(Xtr[idx] - km.cluster_centers_[k], axis=1))])
                Dl.append(Xtr[best]); Dh.append(Ytr[best])
            self.Dl = np.stack(Dl, axis=1).astype(np.float32)
            self.Dh = np.stack(Dh, axis=1).astype(np.float32)
        else:
            self.Dl = Xtr.T.astype(np.float32)
            self.Dh = Ytr.T.astype(np.float32)
        norms      = np.linalg.norm(self.Dl, axis=0, keepdims=True).clip(1e-8)
        self.Dl   /= norms;  self.Dh /= norms
        self._DtD  = self.Dl.T @ self.Dl
        self._L    = float(np.linalg.norm(self._DtD, ord=2)) + 1e-6
        print(f"[SC1] Done — dict shape: {self.Dl.shape}")

    def _ista(self, y: np.ndarray) -> np.ndarray:
        Dty    = self.Dl.T @ y
        thresh = self.lam / self._L
        a      = np.zeros(self.Dl.shape[1], dtype=np.float32)
        for _ in range(self.ista_iters):
            a -= (self._DtD @ a - Dty) / self._L
            a  = np.sign(a) * np.maximum(np.abs(a) - thresh, 0)
        return a

    def _recon_channel(self, lr_ch: np.ndarray) -> np.ndarray:
        ps    = self.ps
        lr_up = cv2.resize(lr_ch, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W  = lr_up.shape
        acc   = np.zeros((H, W)); cnt = np.zeros((H, W))
        for r in range(0, H - ps + 1, 2):
            for c in range(0, W - ps + 1, 2):
                lp        = lr_up[r:r+ps, c:c+ps]
                feat      = extract_features(lp).astype(np.float32)
                fn        = np.linalg.norm(feat)
                if fn < 1e-4:
                    continue
                alpha     = self._ista(feat / fn)
                recon_f   = self.Dl @ alpha
                conf      = np.clip(
                    1.0 - np.linalg.norm((feat/fn) - recon_f) /
                    (np.linalg.norm(feat/fn) + 1e-6), 0, 1)
                res_pred  = (self.Dh @ alpha) * fn * conf * 0.4
                acc[r:r+ps, c:c+ps] += res_pred.reshape(ps, ps)
                cnt[r:r+ps, c:c+ps] += 1.0
        return (lr_up + acc / np.maximum(cnt, 1.0)).clip(0, 1).astype(np.float32)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        ycc   = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        hr_y  = self._recon_channel(ycc[:, :, 0])
        hr_cr = cv2.resize(ycc[:, :, 1], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(ycc[:, :, 2], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(
            np.stack([hr_y, hr_cr, hr_cb], axis=2),
            cv2.COLOR_YCrCb2RGB).clip(0, 1).astype(np.float32)


# ── SC2 ───────────────────────────────────────────────────────
class SC2Solver:
    """
    Kernel ridge regression SR (Kim & Kwon 2010).
    Confidence-gated residual prediction with Gaussian kernel.
    """
    def __init__(self, patch_size=SC2_PATCH_SIZE, sigma_k=SC2_SIGMA_K,
                 lam=SC2_LAMBDA_REG, n_basis=SC2_N_BASIS, n_train=SC2_N_TRAIN):
        self.ps  = patch_size
        self.sk  = sigma_k
        self.lam = lam
        self.nb  = n_basis
        self.nt  = n_train
        self.B   = self.A = None

    def _kern(self, X, Y):
        d2 = (np.sum(X**2, 1, keepdims=True)
              + np.sum(Y**2, 1, keepdims=True).T
              - 2 * X @ Y.T).clip(0)
        return np.exp(-d2 / (2 * self.sk**2)).astype(np.float32)

    def build(self, paths: List[str], seed=CL_SEED):
        print("[SC2] Building KRR model ...", flush=True)
        rng = random.Random(seed)
        per = max(1, (self.nb * 5) // max(1, len(paths)))
        Xl, Yl = [], []
        for path in rng.sample(paths, min(len(paths), self.nt)):
            try:
                hr_rgb  = np.array(
                    celeba_crop(Image.open(path).convert("RGB"))
                    .resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g    = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_s    = generate_train_lr(hr_g)
                lr_up   = cv2.resize(lr_s, HR_SIZE, interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            H, W = lr_up.shape
            pos  = [(r, c) for r in range(0, H - self.ps + 1, 2)
                            for c in range(0, W - self.ps + 1, 2)]
            for r, c in rng.sample(pos, min(per, len(pos))):
                lp  = lr_up[r:r+self.ps, c:c+self.ps]
                hp  = hr_g [r:r+self.ps, c:c+self.ps]
                l1n = np.sum(np.abs(lp))
                if l1n < 1e-4:
                    continue
                Xl.append(extract_features(lp / l1n))
                Yl.append(hp.ravel() - lp.ravel())
        Xtr = np.stack(Xl).astype(np.float32)
        Ytr = np.stack(Yl).astype(np.float32)
        n   = Xtr.shape[0]
        if n > self.nb:
            from sklearn.cluster import MiniBatchKMeans
            km  = MiniBatchKMeans(n_clusters=self.nb, random_state=seed,
                                  batch_size=min(4096, n), n_init=3)
            lbl = km.fit_predict(Xtr)
            B   = []
            for k in range(self.nb):
                idx  = np.where(lbl == k)[0]
                if len(idx) == 0:
                    idx = [rng.randint(0, n - 1)]
                best = int(idx[np.argmin(
                    np.linalg.norm(Xtr[idx] - km.cluster_centers_[k], axis=1))])
                B.append(Xtr[best])
            self.B = np.stack(B).astype(np.float32)
        else:
            self.B  = Xtr; self.nb = n
        Kxb  = self._kern(self.B, Xtr).astype(np.float64)
        Kbb  = self._kern(self.B, self.B).astype(np.float64)
        M    = Kxb @ Kxb.T + self.lam * Kbb + 1e-6 * np.eye(self.nb)
        self.A = np.linalg.solve(M, Kxb @ Ytr.astype(np.float64)).astype(np.float32)
        print(f"[SC2] Done — basis shape: {self.B.shape}")

    def _recon_channel(self, lr_ch: np.ndarray) -> np.ndarray:
        lr_up = cv2.resize(lr_ch, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        H, W  = lr_up.shape
        acc   = np.zeros((H, W)); cnt = np.zeros((H, W))
        for r in range(0, H - self.ps + 1, 2):
            for c in range(0, W - self.ps + 1, 2):
                lp  = lr_up[r:r+self.ps, c:c+self.ps]
                l1n = np.sum(np.abs(lp))
                if l1n < 1e-4:
                    continue
                feat     = extract_features(lp / l1n).astype(np.float32)
                k_vec    = self._kern(feat.reshape(1, -1), self.B)
                conf     = float(np.max(k_vec))
                res_pred = (k_vec @ self.A).ravel() * conf
                acc[r:r+self.ps, c:c+self.ps] += res_pred.reshape(self.ps, self.ps)
                cnt[r:r+self.ps, c:c+self.ps] += 1.0
        return (lr_up + acc / np.maximum(cnt, 1.0)).clip(0, 1).astype(np.float32)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        ycc   = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        hr_y  = self._recon_channel(ycc[:, :, 0])
        hr_cr = cv2.resize(ycc[:, :, 1], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        hr_cb = cv2.resize(ycc[:, :, 2], HR_SIZE, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(
            np.stack([hr_y, hr_cr, hr_cb], axis=2),
            cv2.COLOR_YCrCb2RGB).clip(0, 1).astype(np.float32)


# ── SFH ───────────────────────────────────────────────────────
class SFHSolver:
    """
    Structured Face Hallucination (Yang, Liu & Yang 2013).
    Zone-based exemplar matching with Laplacian gradient transfer
    and input-quality gating to prevent over-hallucination.
    """
    def __init__(self, max_exemplars=SFH_MAX_EXEMPLARS):
        self.max_exemplars = max_exemplars
        self.ex_hr = []; self.ex_lr = []
        self.zones = {"upper": (20, 45), "middle": (45, 65), "lower": (65, 90)}

    def build(self, paths: List[str], seed=CL_SEED):
        print(f"[SFH] Loading {self.max_exemplars} exemplars ...", flush=True)
        rng = random.Random(seed)
        for path in rng.sample(paths, min(self.max_exemplars, len(paths))):
            try:
                hr_rgb = np.array(
                    celeba_crop(Image.open(path).convert("RGB"))
                    .resize(HR_SIZE, Image.BICUBIC)).astype(np.float32) / 255.0
                hr_g   = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
                lr_s   = generate_train_lr(hr_g)
                self.ex_hr.append(hr_rgb); self.ex_lr.append(lr_s)
            except Exception:
                continue
        print(f"[SFH] {len(self.ex_hr)} exemplars loaded.", flush=True)

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        hr_bic  = bicubic_sr(lr)
        lr_g    = cv2.cvtColor(lr,     cv2.COLOR_RGB2GRAY)
        hall_y  = cv2.cvtColor(hr_bic, cv2.COLOR_RGB2GRAY).copy()
        gl      = cv2.Laplacian(lr_g, cv2.CV_32F).var()
        quality = 1.0 if gl < 50 else 0.4

        for _, (y0, y1) in self.zones.items():
            ly0, ly1 = y0 // 2, y1 // 2
            lr_zone  = lr_g[ly0:ly1, :]
            q  = lr_zone.ravel().astype(np.float64); q -= q.mean()
            qn = np.linalg.norm(q) + 1e-8
            best_idx, max_sim = 0, -np.inf
            for i, ex_l in enumerate(self.ex_lr):
                e  = ex_l[ly0:ly1, :].ravel().astype(np.float64); e -= e.mean()
                s  = np.dot(q, e) / (qn * (np.linalg.norm(e) + 1e-8))
                if s > max_sim:
                    max_sim, best_idx = s, i
            ex_g    = cv2.cvtColor(self.ex_hr[best_idx], cv2.COLOR_RGB2GRAY)
            lap     = cv2.Laplacian(ex_g[y0:y1, :], cv2.CV_32F, ksize=3)
            t_lap   = cv2.Laplacian(
                cv2.resize(lr_zone, (100, y1 - y0),
                           interpolation=cv2.INTER_CUBIC), cv2.CV_32F)
            cmask   = np.abs(t_lap) / (np.max(np.abs(t_lap)) + 1e-8)
            gain    = 0.12 * max(0, max_sim) * quality
            hall_y[y0:y1, :] += gain * lap * cmask

        lr_est  = cv2.resize(hall_y, TEST_FIXED_LR_SIZE, interpolation=cv2.INTER_AREA)
        corr    = cv2.resize(lr_g - lr_est, HR_SIZE, interpolation=cv2.INTER_CUBIC)
        final_y = np.clip(hall_y + corr, 0, 1)
        ycc     = cv2.cvtColor(lr, cv2.COLOR_RGB2YCrCb)
        return cv2.cvtColor(
            np.stack([final_y,
                      cv2.resize(ycc[:, :, 1], HR_SIZE, interpolation=cv2.INTER_CUBIC),
                      cv2.resize(ycc[:, :, 2], HR_SIZE, interpolation=cv2.INTER_CUBIC)],
                     axis=2),
            cv2.COLOR_YCrCb2RGB).clip(0, 1).astype(np.float32)


# ── Evaluation ────────────────────────────────────────────────
def evaluate_method(name: str, sr_fn, test_paths: List[str],
                    blur_type: str, gaussian_sigma=None,
                    motion_length=None, base_seed=CL_SEED,
                    save_sample=True) -> Dict:
    total_p = total_s = n = 0
    for idx, path in enumerate(test_paths):
        try:
            hr = load_hr(path)
        except Exception:
            continue
        theta = (random.Random(base_seed + idx).uniform(-math.pi, math.pi)
                 if blur_type == "motion" else None)
        lr = degrade(hr, blur_type, gaussian_sigma=gaussian_sigma,
                     motion_length=motion_length, theta=theta)
        try:
            sr = sr_fn(lr)
        except Exception:
            continue
        sr_y, hr_y = rgb_to_y_np(sr), rgb_to_y_np(hr)
        if sr_y.ndim != 2 or hr_y.ndim != 2:
            continue
        total_p += psnr_np(sr_y, hr_y)
        total_s += ssim_np(sr_y, hr_y)
        n += 1
        if save_sample and idx == 0:
            tag = (f"gaussian_s{gaussian_sigma}" if blur_type == "gaussian"
                   else f"motion_l{motion_length}")
            sd = RESULTS_CL_DIR / "samples" / name / tag
            sd.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                (np.clip(sr, 0, 1) * 255).astype(np.uint8)
            ).save(sd / "sr.png")
            Image.fromarray(
                (np.clip(hr, 0, 1) * 255).astype(np.uint8)
            ).save(sd / "hr.png")
            Image.fromarray(
                (np.clip(cv2.resize(lr, HR_SIZE,
                                    interpolation=cv2.INTER_NEAREST),
                         0, 1) * 255).astype(np.uint8)
            ).save(sd / "lr_upscaled.png")

    avg_p = total_p / max(n, 1)
    avg_s = total_s / max(n, 1)
    label = (f"gaussian_sigma_{gaussian_sigma}" if blur_type == "gaussian"
             else f"motion_l_{motion_length}")
    print(f"  {name:8s} | {label:25s} | PSNR_Y={avg_p:.2f} | SSIM_Y={avg_s:.4f}", flush=True)
    return {"method": name, "blur_type": label,
            "psnr": round(avg_p, 4), "ssim": round(avg_s, 4)}

def run_classical_pipeline(train_paths: List[str],
                           test_paths:  List[str]) -> List[Dict]:
    """Build all classical solvers then evaluate on all 6 conditions."""
    RESULTS_CL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    settings = [
        dict(blur_type="gaussian", gaussian_sigma=1),
        dict(blur_type="gaussian", gaussian_sigma=3),
        dict(blur_type="gaussian", gaussian_sigma=5),
        dict(blur_type="motion",   motion_length=2),
        dict(blur_type="motion",   motion_length=6),
        dict(blur_type="motion",   motion_length=9),
    ]

    # 1. Evaluate Bicubic FIRST (requires no training/building)
    print("\nMethod: Bicubic", flush=True)
    for cfg in settings:
        rows.append(evaluate_method("Bicubic", bicubic_sr, test_paths, **cfg))

    # 2. Build and Evaluate SC1
    sc1 = SC1Solver()
    sc1.build(train_paths)
    print("\nMethod: SC1", flush=True)
    for cfg in settings:
        rows.append(evaluate_method("SC1", sc1.sr_rgb, test_paths, **cfg))
    del sc1 # Free up memory

    # 3. Build and Evaluate SC2
    sc2 = SC2Solver()
    sc2.build(train_paths)
    print("\nMethod: SC2", flush=True)
    for cfg in settings:
        rows.append(evaluate_method("SC2", sc2.sr_rgb, test_paths, **cfg))
    del sc2

    # 4. Build and Evaluate SFH
    sfh = SFHSolver()
    sfh.build(train_paths)
    print("\nMethod: SFH", flush=True)
    for cfg in settings:
        rows.append(evaluate_method("SFH", sfh.sr_rgb, test_paths, **cfg))

    out_csv = RESULTS_CL_DIR / "classical_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "blur_type", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nClassical metrics → {out_csv}", flush=True)

    # Define the list of methods manually since we don't have the dict anymore
    method_names = ["Bicubic", "SC1", "SC2", "SFH"]
    
    # Get unique blur tags from the rows we actually managed to finish
    blur_tags = []
    for r in rows:
        if r["blur_type"] not in blur_tags:
            blur_tags.append(r["blur_type"])

    for metric in ("psnr", "ssim"):
        print(f"\n{metric.upper()}\n{'Blur':<26}" +
              "".join(f"{m:>10}" for m in method_names), flush=True)
        
        for bt in blur_tags:
            row_str = f"{bt:<26}"
            for m in method_names:
                try:
                    v = next(r[metric] for r in rows
                             if r["method"] == m and r["blur_type"] == bt)
                    row_str += (f"{v:>10.2f}" if metric == "psnr"
                                else f"{v:>10.4f}")
                except StopIteration:
                    row_str += f"{'N/A':>10}" 
            print(row_str, flush=True)
    return rows