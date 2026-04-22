"""
reconstruction.py
=================
Classical super-resolution methods for face hallucination.

All classes share the same public interface:
  .fit(lr_imgs_rgb, hr_imgs_rgb)  – train on list of HWC float32 [0,1] arrays
  .predict(lr_rgb)                – return SR HWC float32 [0,1] array

"""

import math
import random
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


# ============================================================
# Colour-space helpers
# ============================================================

def rgb_to_ycbcr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HWC float32 [0,1] RGB -> (Y, Cb, Cr) each float32 [0,1]."""
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    return ycrcb[:, :, 0], ycrcb[:, :, 2], ycrcb[:, :, 1]   # Y, Cb, Cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """(Y, Cb, Cr) float32 [0,1] -> HWC float32 [0,1] RGB."""
    ycrcb = np.stack([y, cr, cb], axis=-1)
    ycrcb_u8 = np.clip(ycrcb * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# Patch helpers  (used by SR1_SC1_Yang_FeatureResidual)
# ============================================================

def extract_patches_2d(
    img: np.ndarray,
    patch: int,
    stride: int,
):
    """Extract overlapping square patches from a 2-D (H, W) image."""
    h, w = img.shape
    patches, coords = [], []
    for i in range(0, h - patch + 1, stride):
        for j in range(0, w - patch + 1, stride):
            patches.append(img[i:i + patch, j:j + patch].reshape(-1))
            coords.append((i, j))
    if not patches:
        return np.empty((0, patch * patch), dtype=np.float32), []
    return np.asarray(patches, dtype=np.float32), coords


def extract_feature_patches(
    feat_img: np.ndarray,
    patch: int,
    stride: int,
):
    """Extract overlapping patches from a (H, W, C) feature map."""
    h, w, c = feat_img.shape
    patches, coords = [], []
    for i in range(0, h - patch + 1, stride):
        for j in range(0, w - patch + 1, stride):
            patches.append(feat_img[i:i + patch, j:j + patch].reshape(-1))
            coords.append((i, j))
    if not patches:
        return np.empty((0, patch * patch * c), dtype=np.float32), []
    return np.asarray(patches, dtype=np.float32), coords


def aggregate_patches_2d(
    patches: np.ndarray,
    coords: List[Tuple[int, int]],
    out_shape: Tuple[int, int],
    patch: int,
):
    """Average-aggregate vectorised patches back into a 2-D image."""
    out = np.zeros(out_shape, dtype=np.float32)
    weight = np.zeros(out_shape, dtype=np.float32)
    for vec, (i, j) in zip(patches, coords):
        p = vec.reshape(patch, patch)
        out[i:i + patch, j:j + patch] += p
        weight[i:i + patch, j:j + patch] += 1.0
    weight[weight == 0] = 1.0
    return out / weight


# ============================================================
# Feature extractors
# ============================================================

def gradient_features(y: np.ndarray) -> np.ndarray:
    """
    4-channel gradient feature map used by SR1_SC1_Yang_FeatureResidual.
    Returns (H, W, 4) float32.
    """
    g1 = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    g2 = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    g3 = cv2.Sobel(y, cv2.CV_32F, 2, 0, ksize=3)
    g4 = cv2.Sobel(y, cv2.CV_32F, 0, 2, ksize=3)
    return np.stack([g1, g2, g3, g4], axis=2).astype(np.float32)


# Derivative filters used by the classical SC1/SC2 methods (Yang 2008 Eq. 12)
_FEAT_FILTERS = [
    np.array([[-1,  0,  1]], dtype=np.float32),
    np.array([[-1], [0], [1]], dtype=np.float32),
    np.array([[ 1,  0, -2,  0,  1]], dtype=np.float32),
    np.array([[ 1], [0], [-2], [0], [1]], dtype=np.float32),
]


def extract_classical_features(patch: np.ndarray) -> np.ndarray:
    """
    Filter-bank feature vector used by SR1_SC1_Classical and SR2_SC2_Kim_KRR.
    Input patch should be a 2-D float array.  Returns 1-D float32.
    """
    return np.concatenate([
        cv2.filter2D(patch, -1, f, borderType=cv2.BORDER_REFLECT).ravel()
        for f in _FEAT_FILTERS
    ]).astype(np.float32)


# ============================================================
# Misc utilities
# ============================================================

def _safe_concat(xs: List[np.ndarray]) -> np.ndarray:
    if not xs:
        raise ValueError("No patches collected.")
    return np.concatenate(xs, axis=0).astype(np.float32)


def row_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-8)


# ============================================================
# Method 0 : Bicubic baseline - interpolation
# ============================================================

class SR0_Bicubic:
    """Simple bicubic upsampling to HR_SIZE.  No training required."""

    def __init__(self, hr_size: Tuple[int, int] = (100, 100)):
        self.hr_size = hr_size

    def fit(self, *args, **kwargs) -> "SR0_Bicubic":
        return self

    def predict(self, lr_rgb: np.ndarray) -> np.ndarray:
        out = cv2.resize(lr_rgb, self.hr_size, interpolation=cv2.INTER_CUBIC)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # path-list API alias (no-op)
    def build(self, paths=None) -> "SR0_Bicubic":
        return self

    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        return self.predict(lr)


# ============================================================
# Method 1 : SR1_SC1_Yang_FeatureResidual
# ============================================================

class SR1_SC1_Yang_FeatureResidual:
    """
    Sparse-coding SR using gradient-feature patches and sklearn DictionaryLearning.

    Training API : .fit(lr_imgs_rgb, hr_imgs_rgb)
      lr_imgs_rgb : list of HWC float32 [0,1] — LR images upsampled to HR size (e.g. 100×100)
      hr_imgs_rgb : list of HWC float32 [0,1] — corresponding HR ground-truth images

    Inference API : .predict(lr100_rgb) -> HWC float32 [0,1]
    """

    def __init__(
        self,
        patch: int = 5,
        stride: int = 2,
        n_atoms: int = 256,
        lasso_alpha: float = 0.001,
        lasso_max_iter: int = 5000,
        lasso_tol: float = 1e-4,
        dl_alpha: float = 1.0,
        dl_iter: int = 120,
        max_train_patches: int = 120_000,
        per_image_patch_cap: int = 2_000,
        random_state: int = 42,
    ):
        self.patch = patch
        self.stride = stride
        self.n_atoms = n_atoms
        self.lasso_alpha = lasso_alpha
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.dl_alpha = dl_alpha
        self.dl_iter = dl_iter
        self.max_train_patches = max_train_patches
        self.per_image_patch_cap = per_image_patch_cap
        self.random_state = random_state

        self.Dl: Optional[np.ndarray] = None
        self.Dh: Optional[np.ndarray] = None

        self._lasso = Lasso(
            alpha=self.lasso_alpha,
            fit_intercept=False,
            max_iter=self.lasso_max_iter,
            tol=self.lasso_tol,
            warm_start=False,
        )

    def fit(
        self,
        lr100_imgs_rgb: List[np.ndarray],
        hr100_imgs_rgb: List[np.ndarray],
    ) -> "SR1_SC1_Yang_FeatureResidual":
        if len(lr100_imgs_rgb) != len(hr100_imgs_rgb):
            raise ValueError("lr100_imgs_rgb and hr100_imgs_rgb must have the same length.")

        Xl_list, Xh_list = [], []
        rng = np.random.RandomState(self.random_state)
        total = 0

        for lr_rgb, hr_rgb in zip(lr100_imgs_rgb, hr100_imgs_rgb):
            yl, _, _ = rgb_to_ycbcr(lr_rgb)
            yh, _, _ = rgb_to_ycbcr(hr_rgb)

            feat_l = gradient_features(yl)
            Xl, _ = extract_feature_patches(feat_l, self.patch, self.stride)
            Xh, _ = extract_patches_2d(yh, self.patch, self.stride)
            Yb, _ = extract_patches_2d(yl, self.patch, self.stride)

            if Xl.shape[0] == 0 or Xh.shape[0] == 0 or Yb.shape[0] == 0:
                continue

            n = min(Xl.shape[0], Xh.shape[0], Yb.shape[0])
            Xl, Xh, Yb = Xl[:n], Xh[:n], Yb[:n]

            if n > self.per_image_patch_cap:
                idx = rng.choice(n, size=self.per_image_patch_cap, replace=False)
                Xl, Xh, Yb = Xl[idx], Xh[idx], Yb[idx]

            patch_mean = Yb.mean(axis=1, keepdims=True).astype(np.float32)
            Xl_list.append(Xl.astype(np.float32))
            Xh_list.append((Xh - patch_mean).astype(np.float32))

            total += Xl.shape[0]
            if total >= self.max_train_patches:
                break

        Xl_all = row_normalize(_safe_concat(Xl_list)).astype(np.float32)
        Xh_all = row_normalize(_safe_concat(Xh_list)).astype(np.float32)

        dl = DictionaryLearning(
            n_components=self.n_atoms,
            alpha=self.dl_alpha,
            max_iter=self.dl_iter,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            random_state=self.random_state,
        )
        A = dl.fit_transform(Xl_all).astype(np.float32)
        self.Dl = dl.components_.astype(np.float32)

        AtA = A.T @ A + 1e-4 * np.eye(self.n_atoms, dtype=np.float32)
        self.Dh = np.linalg.solve(AtA, A.T @ Xh_all).astype(np.float32)
        return self

    def _code(self, x: np.ndarray) -> np.ndarray:
        if self.Dl is None:
            raise RuntimeError("Model has not been fit yet.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self._lasso.fit(self.Dl.T, x)
        return self._lasso.coef_.astype(np.float32)

    def predict(self, lr100_rgb: np.ndarray) -> np.ndarray:
        if self.Dl is None or self.Dh is None:
            raise RuntimeError("Model has not been fit yet.")

        y, cb, cr = rgb_to_ycbcr(lr100_rgb)
        feat_l = gradient_features(y)
        Xl, coords = extract_feature_patches(feat_l, self.patch, self.stride)
        Yb, _ = extract_patches_2d(y, self.patch, self.stride)

        if Xl.shape[0] == 0:
            return lr100_rgb.copy()

        Xl = row_normalize(Xl).astype(np.float32)
        A = np.zeros((Xl.shape[0], self.n_atoms), dtype=np.float32)
        for i in range(Xl.shape[0]):
            A[i] = self._code(Xl[i])

        patch_mean = Yb.mean(axis=1, keepdims=True).astype(np.float32)
        Xh_hat = A @ self.Dh + patch_mean

        y_hat = np.clip(aggregate_patches_2d(Xh_hat, coords, y.shape, self.patch), 0.0, 1.0)
        return ycbcr_to_rgb(y_hat, cb, cr)


# ============================================================
# Method 1b : SR1_SC1_Classical
#             This is referencing original CelebA2.py
# ============================================================

class SR1_SC1_Classical:
    """
    Sparse-coding SR using a K-Means dictionary and an ISTA sparse solver.
    Operates per-channel (grayscale feature patches -> HR residual patches).

    Training API : .fit(lr_imgs_rgb, hr_imgs_rgb)   — image-list API
                   .build(paths)                      — path-list API (classical5 style)
    Inference API: .predict(lr_rgb) -> HWC float32   — unified interface
                   .sr_rgb(lr_rgb)                    — alias
    """

    def __init__(
        self,
        patch_size: int = 5,
        upscale: int = 2,
        dict_size: int = 512,
        lam: float = 0.1,
        n_train: int = 20_000,
        hr_size: Tuple[int, int] = (100, 100),
        lr_size: Tuple[int, int] = (50, 50),
        seed: int = 42,
    ):
        self.ps = patch_size
        self.us = upscale
        self.dict_size = dict_size
        self.lam = lam
        self.n_train = n_train
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.seed = seed

        self.Dl: Optional[np.ndarray] = None   # (feat_dim, dict_size)
        self.Dh: Optional[np.ndarray] = None   # (hr_patch**2, dict_size)
        self._DtD: Optional[np.ndarray] = None
        self._L: float = 1.0

    # ----------------------------------------------------------
    # Internal patch collection
    # ----------------------------------------------------------
    def _collect_patches_from_images(
        self,
        lr_imgs: List[np.ndarray],
        hr_imgs: List[np.ndarray],
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps, us, hps = self.ps, self.us, self.ps * self.us
        per = max(1, self.n_train // max(1, len(lr_imgs)))
        lr_f, hr_p = [], []

        for lr_rgb, hr_rgb in zip(lr_imgs, hr_imgs):
            # Work on luminance only
            yl, _, _ = rgb_to_ycbcr(lr_rgb)   # already at HR size (100x100)
            yh, _, _ = rgb_to_ycbcr(hr_rgb)
            # Simulate LR-upsampled image from the provided LR
            lr_up = yl  # lr_rgb is already upsampled to hr_size by caller

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 2)
                           for c in range(0, W - ps + 1, 2)]
            if not pos:
                continue

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps])
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch_idx = r * us, c * us
                if rh + hps > yh.shape[0] or ch_idx + hps > yh.shape[1]:
                    continue

                hp = yh[rh:rh + hps, ch_idx:ch_idx + hps].ravel()
                lr_f.append(feat / fn)
                hr_p.append(hp - hp.mean())

                if len(lr_f) >= self.n_train:
                    break
            if len(lr_f) >= self.n_train:
                break

        if not lr_f:
            raise ValueError("[SC1_Classical] No patches collected during training.")
        return (
            np.stack(lr_f).astype(np.float32),
            np.stack(hr_p).astype(np.float32),
        )

    def _collect_patches_from_paths(
        self,
        paths: List[str],
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Path-list training (classical5 style)."""
        from PIL import Image as _Image

        ps, us, hps = self.ps, self.us, self.ps * self.us
        per = max(1, self.n_train // max(1, len(paths)))
        lr_f, hr_p = [], []

        for path in rng.sample(paths, min(len(paths), self.n_train // per + 1)):
            try:
                pil = _Image.open(path).convert("L")
                hr = np.array(pil.resize(self.hr_size, _Image.BICUBIC)).astype(np.float32) / 255.0
                lr_up = cv2.resize(
                    cv2.resize(hr, self.lr_size, interpolation=cv2.INTER_CUBIC),
                    self.hr_size, interpolation=cv2.INTER_CUBIC,
                )
            except Exception:
                continue

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 2)
                           for c in range(0, W - ps + 1, 2)]

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps])
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch_idx = r * us, c * us
                if rh + hps > hr.shape[0] or ch_idx + hps > hr.shape[1]:
                    continue

                hp = hr[rh:rh + hps, ch_idx:ch_idx + hps].ravel()
                lr_f.append(feat / fn)
                hr_p.append(hp - hp.mean())

                if len(lr_f) >= self.n_train:
                    break
            if len(lr_f) >= self.n_train:
                break

        if not lr_f:
            raise ValueError("[SC1_Classical] No patches collected from paths.")
        return (
            np.stack(lr_f).astype(np.float32),
            np.stack(hr_p).astype(np.float32),
        )

    # ----------------------------------------------------------
    # Dictionary construction (shared)
    # ----------------------------------------------------------
    def _build_dict(self, Xtr: np.ndarray, Ytr: np.ndarray) -> None:
        n = Xtr.shape[0]
        print(f"[SC1_Classical] Collected {n} patches — building dictionary (size={self.dict_size}) ...")

        if n > self.dict_size:
            try:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(
                    n_clusters=self.dict_size,
                    random_state=self.seed,
                    batch_size=min(4096, n),
                    max_iter=200,
                    n_init=5,
                )
                lbl = km.fit_predict(Xtr)
                Dl, Dh = [], []
                rng_np = np.random.default_rng(self.seed)
                for k in range(self.dict_size):
                    idx = np.where(lbl == k)[0]
                    if len(idx) == 0:
                        idx = rng_np.integers(0, n, size=1)
                    best = int(idx[np.argmin(
                        np.linalg.norm(Xtr[idx] - km.cluster_centers_[k], axis=1)
                    )])
                    Dl.append(Xtr[best])
                    Dh.append(Ytr[best])
                self.Dl = np.stack(Dl, axis=1).astype(np.float32)
                self.Dh = np.stack(Dh, axis=1).astype(np.float32)
            except ImportError:
                idx = np.random.default_rng(self.seed).choice(n, self.dict_size, replace=False)
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
        print(f"[SC1_Classical] Done — Dl={self.Dl.shape}")

    # ----------------------------------------------------------
    # Public training entry-points
    # ----------------------------------------------------------
    def fit(
        self,
        lr100_imgs_rgb: List[np.ndarray],
        hr100_imgs_rgb: List[np.ndarray],
    ) -> "SR1_SC1_Classical":
        """Train from pre-loaded image arrays (unified interface)."""
        rng = random.Random(self.seed)
        Xtr, Ytr = self._collect_patches_from_images(lr100_imgs_rgb, hr100_imgs_rgb, rng)
        self._build_dict(Xtr, Ytr)
        return self

    def build(self, paths: List[str]) -> "SR1_SC1_Classical":
        """Train from a list of image file paths (classical5 style)."""
        rng = random.Random(self.seed)
        Xtr, Ytr = self._collect_patches_from_paths(paths, rng)
        self._build_dict(Xtr, Ytr)
        return self

    # ----------------------------------------------------------
    # ISTA sparse coder
    # ----------------------------------------------------------
    def _ista(self, y: np.ndarray, iters: int = 100) -> np.ndarray:
        Dty = self.Dl.T @ y
        thresh = self.lam / self._L
        a = np.zeros(self.Dl.shape[1], dtype=np.float32)
        for _ in range(iters):
            a -= (self._DtD @ a - Dty) / self._L
            a = np.sign(a) * np.maximum(np.abs(a) - thresh, 0.0)
        return a

    # ----------------------------------------------------------
    # Per-channel reconstruction
    # ----------------------------------------------------------
    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        """lr: 2-D float32 at LR resolution."""
        ps, us, hps = self.ps, self.us, self.ps * self.us
        step = max(1, ps - 2)
        lr_up = cv2.resize(lr, self.hr_size, interpolation=cv2.INTER_CUBIC)
        H, W = lr_up.shape
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        for r in range(0, H - ps + 1, step):
            for c in range(0, W - ps + 1, step):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-8:
                    continue

                feat_n = feat / fn
                alpha = self._ista(feat_n)
                # Confidence-gated residual limits over-sharpening on mild blur.
                recon_feat = self.Dl @ alpha
                conf = 1.0 - np.linalg.norm(feat_n - recon_feat) / (np.linalg.norm(feat_n) + 1e-6)
                conf = float(np.clip(conf, 0.0, 1.0))
                hp = ((self.Dh @ alpha) * conf * 0.4).reshape(hps, hps)

                rh, ch_idx = r * us, c * us
                rh2 = min(rh + hps, H)
                ch2 = min(ch_idx + hps, W)
                if rh >= H or ch_idx >= W:
                    continue

                acc[rh:rh2, ch_idx:ch2] += hp[:rh2 - rh, :ch2 - ch_idx]
                cnt[rh:rh2, ch_idx:ch2] += 1.0

        hr_rec = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
        hr_rec += cv2.resize(lr, self.hr_size, interpolation=cv2.INTER_CUBIC)

        # Back-projection refinement
        for _ in range(15):
            lr_est = cv2.resize(hr_rec, self.lr_size, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.6 * cv2.resize(lr - lr_est, self.hr_size, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0.0, 1.0)

    # ----------------------------------------------------------
    # Public inference
    # ----------------------------------------------------------
    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        """lr: HWC float32 [0,1] at original LR resolution."""
        if self.Dl is None:
            raise RuntimeError("Model has not been fit / built yet.")
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2,
        ).clip(0.0, 1.0).astype(np.float32)

    def predict(self, lr_rgb: np.ndarray) -> np.ndarray:
        """Unified predict interface — alias for sr_rgb."""
        return self.sr_rgb(lr_rgb)


# ============================================================
# Method 2 : SR2_SC2_Kim_KRR
#             Kernel Ridge Regression SR (Kim & Kwon 2010)
# ============================================================

class SR2_SC2_Kim_KRR:
    """
    Super-resolution via Kernel Ridge Regression on gradient-feature patches.

    Training API : .fit(lr_imgs_rgb, hr_imgs_rgb)
                   .build(paths)
    Inference API: .predict(lr_rgb)
                   .sr_rgb(lr_rgb)
    """

    def __init__(
        self,
        patch_size: int = 5,
        sigma_k: float = 0.05,
        lam: float = 5e-8,
        n_basis: int = 300,
        n_train: int = 10_000,
        hr_size: Tuple[int, int] = (100, 100),
        lr_size: Tuple[int, int] = (50, 50),
        seed: int = 42,
    ):
        self.ps = patch_size
        self.sigma_k = sigma_k
        self.lam = lam
        self.n_basis = n_basis
        self.n_train = n_train
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.seed = seed

        self.B: Optional[np.ndarray] = None   # (n_basis, feat_dim)
        self.A: Optional[np.ndarray] = None   # (n_basis, hr_patch**2)

    def _kern(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        d = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2.0 * X @ Y.T
        ).clip(0.0)
        return np.exp(-d / self.sigma_k).astype(np.float32)

    # ----------------------------------------------------------
    # Internal patch collection
    # ----------------------------------------------------------
    def _collect_patches_from_images(
        self,
        lr_imgs: List[np.ndarray],
        hr_imgs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps, us, hps = self.ps, 2, self.ps * 2
        per = max(1, self.n_train // max(1, len(lr_imgs)))
        rng = random.Random(self.seed)
        Xl, Yl = [], []

        for lr_rgb, hr_rgb in zip(lr_imgs, hr_imgs):
            yl, _, _ = rgb_to_ycbcr(lr_rgb)
            yh, _, _ = rgb_to_ycbcr(hr_rgb)
            lr_up = yl  # already at hr_size

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 3)
                           for c in range(0, W - ps + 1, 3)]
            if not pos:
                continue

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch_idx = r * us, c * us
                if rh + hps > yh.shape[0] or ch_idx + hps > yh.shape[1]:
                    continue

                hp = yh[rh:rh + hps, ch_idx:ch_idx + hps].ravel()
                Xl.append(feat / fn)
                Yl.append(hp - hp.mean())

                if len(Xl) >= self.n_train:
                    break
            if len(Xl) >= self.n_train:
                break

        if not Xl:
            raise ValueError("[SC2_Kim] No patches collected during training.")
        return np.stack(Xl).astype(np.float32), np.stack(Yl).astype(np.float32)

    def _collect_patches_from_paths(
        self,
        paths: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        from PIL import Image as _Image

        ps, us, hps = self.ps, 2, self.ps * 2
        per = max(1, self.n_train // max(1, len(paths)))
        rng = random.Random(self.seed)
        Xl, Yl = [], []

        for path in rng.sample(paths, min(len(paths), self.n_train // per + 1)):
            try:
                pil = _Image.open(path).convert("L")
                hr = np.array(pil.resize(self.hr_size, _Image.BICUBIC)).astype(np.float32) / 255.0
                lr_up = cv2.resize(
                    cv2.resize(hr, self.lr_size, interpolation=cv2.INTER_CUBIC),
                    self.hr_size, interpolation=cv2.INTER_CUBIC,
                )
            except Exception:
                continue

            H, W = lr_up.shape
            pos = [(r, c) for r in range(0, H - ps + 1, 3)
                           for c in range(0, W - ps + 1, 3)]

            for r, c in rng.sample(pos, min(per, len(pos))):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-6:
                    continue

                rh, ch_idx = r * us, c * us
                if rh + hps > hr.shape[0] or ch_idx + hps > hr.shape[1]:
                    continue

                hp = hr[rh:rh + hps, ch_idx:ch_idx + hps].ravel()
                Xl.append(feat / fn)
                Yl.append(hp - hp.mean())

                if len(Xl) >= self.n_train:
                    break
            if len(Xl) >= self.n_train:
                break

        if not Xl:
            raise ValueError("[SC2_Kim] No patches collected from paths.")
        return np.stack(Xl).astype(np.float32), np.stack(Yl).astype(np.float32)

    # ----------------------------------------------------------
    # Solver
    # ----------------------------------------------------------
    def _solve(self, Xtr: np.ndarray, Ytr: np.ndarray) -> None:
        n = Xtr.shape[0]
        print(f"[SC2_Kim] Collected {n} patches — solving KRR (n_basis={self.n_basis}) ...")
        bidx = np.random.default_rng(self.seed).choice(n, min(self.n_basis, n), replace=False)
        self.B = Xtr[bidx]
        Kbx = self._kern(self.B, Xtr)
        Kbb = self._kern(self.B, self.B)
        M = Kbx @ Kbx.T + self.lam * Kbb + 1e-8 * np.eye(len(bidx))
        self.A = np.linalg.solve(
            M.astype(np.float64),
            (Kbx @ Ytr).astype(np.float64),
        ).astype(np.float32)
        print(f"[SC2_Kim] Done — B={self.B.shape}, A={self.A.shape}")

    # ----------------------------------------------------------
    # Public training
    # ----------------------------------------------------------
    def fit(
        self,
        lr100_imgs_rgb: List[np.ndarray],
        hr100_imgs_rgb: List[np.ndarray],
    ) -> "SR2_SC2_Kim_KRR":
        Xtr, Ytr = self._collect_patches_from_images(lr100_imgs_rgb, hr100_imgs_rgb)
        self._solve(Xtr, Ytr)
        return self

    def build(self, paths: List[str]) -> "SR2_SC2_Kim_KRR":
        Xtr, Ytr = self._collect_patches_from_paths(paths)
        self._solve(Xtr, Ytr)
        return self

    # ----------------------------------------------------------
    # Per-channel reconstruction
    # ----------------------------------------------------------
    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        ps, us, hps = self.ps, 2, self.ps * 2
        step = max(1, ps - 2)
        lr_up = cv2.resize(lr, self.hr_size, interpolation=cv2.INTER_CUBIC)
        H, W = lr_up.shape
        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        for r in range(0, H - ps + 1, step):
            for c in range(0, W - ps + 1, step):
                feat = extract_classical_features(lr_up[r:r + ps, c:c + ps]).astype(np.float32)
                fn = np.linalg.norm(feat)
                if fn < 1e-8:
                    continue

                k = self._kern(self.B, (feat / fn).reshape(1, -1)).ravel()
                conf = float(np.clip(np.max(k), 0.0, 1.0))
                hp = ((k @ self.A) * conf).reshape(hps, hps)

                rh, ch_idx = r * us, c * us
                rh2 = min(rh + hps, H)
                ch2 = min(ch_idx + hps, W)
                if rh >= H or ch_idx >= W:
                    continue

                acc[rh:rh2, ch_idx:ch2] += hp[:rh2 - rh, :ch2 - ch_idx]
                cnt[rh:rh2, ch_idx:ch2] += 1.0

        hr_rec = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
        hr_rec += cv2.resize(lr, self.hr_size, interpolation=cv2.INTER_CUBIC)

        # Back-projection refinement
        for _ in range(15):
            lr_est = cv2.resize(hr_rec, self.lr_size, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.6 * cv2.resize(lr - lr_est, self.hr_size, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0.0, 1.0)

    # ----------------------------------------------------------
    # Public inference
    # ----------------------------------------------------------
    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        if self.B is None:
            raise RuntimeError("Model has not been fit / built yet.")
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2,
        ).clip(0.0, 1.0).astype(np.float32)

    def predict(self, lr_rgb: np.ndarray) -> np.ndarray:
        return self.sr_rgb(lr_rgb)


# ============================================================
# Method 3 : SR3_SFH
#             Structured Face Hallucination (Yang et al. 2013)
# ============================================================

class SR3_SFH:
    """
    Exemplar-based SR using gradient-guided back-projection.

    Training API : .fit(lr_imgs_rgb, hr_imgs_rgb)   — stores (LR, HR) exemplars
                   .build(paths)                      — path-list API
    Inference API: .predict(lr_rgb)
                   .sr_rgb(lr_rgb)
    """

    def __init__(
        self,
        max_exemplars: int = 300,
        hr_size: Tuple[int, int] = (100, 100),
        lr_size: Tuple[int, int] = (50, 50),
        seed: int = 42,
    ):
        self.max_exemplars = max_exemplars
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.seed = seed

        self.ex_lr: List[np.ndarray] = []   # 2-D float32 grayscale at lr_size
        self.ex_hr: List[np.ndarray] = []   # 2-D float32 grayscale at hr_size

    # ----------------------------------------------------------
    # Public training
    # ----------------------------------------------------------
    def fit(
        self,
        lr_imgs_rgb: List[np.ndarray],
        hr_imgs_rgb: List[np.ndarray],
    ) -> "SR3_SFH":
        """Store up to max_exemplars (LR-Y, HR-Y) pairs from image arrays."""
        rng = random.Random(self.seed)
        n = min(self.max_exemplars, len(lr_imgs_rgb))
        indices = rng.sample(range(len(lr_imgs_rgb)), n)

        self.ex_lr, self.ex_hr = [], []
        for i in indices:
            yl, _, _ = rgb_to_ycbcr(lr_imgs_rgb[i])
            yh, _, _ = rgb_to_ycbcr(hr_imgs_rgb[i])
            # Store LR at its native LR size
            lr_small = cv2.resize(yl, self.lr_size, interpolation=cv2.INTER_CUBIC)
            self.ex_lr.append(lr_small.astype(np.float32))
            self.ex_hr.append(yh.astype(np.float32))

        print(f"[SFH] Exemplar DB: {len(self.ex_lr)} images")
        return self

    def build(self, paths: List[str]) -> "SR3_SFH":
        """Store exemplars from a list of file paths."""
        from PIL import Image as _Image

        rng = random.Random(self.seed)
        self.ex_lr, self.ex_hr = [], []
        sample = rng.sample(paths, min(self.max_exemplars, len(paths)))

        for path in sample:
            try:
                pil = _Image.open(path).convert("L")
                hr = np.array(pil.resize(self.hr_size, _Image.BICUBIC)).astype(np.float32) / 255.0
                lr = cv2.resize(hr, self.lr_size, interpolation=cv2.INTER_CUBIC)
                self.ex_lr.append(lr)
                self.ex_hr.append(hr)
            except Exception:
                continue

        print(f"[SFH] Exemplar DB: {len(self.ex_lr)} images")
        return self

    # ----------------------------------------------------------
    # Exemplar search
    # ----------------------------------------------------------
    def _best_exemplar(self, lr_q: np.ndarray) -> Tuple[np.ndarray, float]:
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

        return best_hr, float(best_score)

    # ----------------------------------------------------------
    # Per-channel reconstruction
    # ----------------------------------------------------------
    def _reconstruct_channel(self, lr: np.ndarray) -> np.ndarray:
        """lr: 2-D float32 at LR resolution."""
        best_hr, exemplar_conf = self._best_exemplar(lr)
        lr_up = cv2.resize(lr, self.hr_size, interpolation=cv2.INTER_CUBIC)

        # Global quality gate dampens hallucination when LR input is already sharp.
        lap_var = cv2.Laplacian(lr.astype(np.float32), cv2.CV_32F).var()
        quality_gate = 1.0 if lap_var < 0.01 else 0.4

        gx_lr = cv2.Sobel(lr_up, cv2.CV_64F, 1, 0, ksize=3)
        gy_lr = cv2.Sobel(lr_up, cv2.CV_64F, 0, 1, ksize=3)
        gx_hr = cv2.Sobel(best_hr, cv2.CV_64F, 1, 0, ksize=3)
        gy_hr = cv2.Sobel(best_hr, cv2.CV_64F, 0, 1, ksize=3)

        mag_lr = np.sqrt(gx_lr ** 2 + gy_lr ** 2)
        mag_hr = np.sqrt(gx_hr ** 2 + gy_hr ** 2)
        w = np.clip(mag_lr / (mag_lr.max() + 1e-8), 0.0, 1.0)
        scale = np.where(mag_hr > 1e-6, mag_lr / (mag_hr + 1e-6), 1.0).clip(0.0, 4.0)

        gx_out = w * scale * gx_hr + (1 - w) * gx_lr
        gy_out = w * scale * gy_hr + (1 - w) * gy_lr

        div = (
            cv2.Sobel(gx_out, cv2.CV_64F, 1, 0, ksize=3)
            + cv2.Sobel(gy_out, cv2.CV_64F, 0, 1, ksize=3)
        )
        gain = 0.08 * max(0.0, exemplar_conf) * quality_gate
        hr_rec = (lr_up + gain * div).astype(np.float32)

        # Back-projection refinement
        for _ in range(20):
            lr_est = cv2.resize(hr_rec, self.lr_size, interpolation=cv2.INTER_CUBIC)
            hr_rec += 0.5 * cv2.resize(lr - lr_est, self.hr_size, interpolation=cv2.INTER_CUBIC)

        return hr_rec.clip(0.0, 1.0)

    # ----------------------------------------------------------
    # Public inference
    # ----------------------------------------------------------
    def sr_rgb(self, lr: np.ndarray) -> np.ndarray:
        if not self.ex_lr:
            raise RuntimeError("Model has not been fit / built yet.")
        return np.stack(
            [self._reconstruct_channel(lr[:, :, c]) for c in range(3)],
            axis=2,
        ).clip(0.0, 1.0).astype(np.float32)

    def predict(self, lr_rgb: np.ndarray) -> np.ndarray:
        return self.sr_rgb(lr_rgb)
