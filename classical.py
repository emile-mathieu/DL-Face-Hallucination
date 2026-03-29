import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.decomposition import DictionaryLearning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.datasets import CelebA


# =========================================================
# Seed
# =========================================================
def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# Data structures
# =========================================================
@dataclass(frozen=True)
class Degradation:
    kind: str
    sigma: Optional[float] = None
    length: Optional[float] = None
    theta: Optional[float] = None


@dataclass(frozen=True)
class MethodResult:
    psnr: float
    ssim: float


# =========================================================
# Color space helpers
# =========================================================
def rgb_to_ycbcr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycbcr = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycbcr[:, :, 0] / 255.0
    cr = ycbcr[:, :, 1] / 255.0
    cb = ycbcr[:, :, 2] / 255.0
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    ycrcb = np.stack([y, cr, cb], axis=-1)
    rgb = cv2.cvtColor((ycrcb * 255.0).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    return np.clip(rgb / 255.0, 0.0, 1.0)


# =========================================================
# Patch helpers
# =========================================================
def extract_patches_2d(
    img2d: np.ndarray,
    patch: int,
    stride: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    H, W = img2d.shape
    patches = []
    coords = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            patches.append(img2d[y : y + patch, x : x + patch].reshape(-1))
            coords.append((y, x))
    return np.stack(patches, axis=0).astype(np.float32), coords


def aggregate_patches_2d(
    patches: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    out_hw: Tuple[int, int],
    patch: int,
) -> np.ndarray:
    H, W = out_hw
    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)

    for p, (y, x) in zip(patches, coords):
        block = p.reshape(patch, patch)
        out[y : y + patch, x : x + patch] += block
        wgt[y : y + patch, x : x + patch] += 1.0

    out /= np.maximum(wgt, 1e-6)
    return out


# =========================================================
# Metrics
# =========================================================
def compute_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    hr_eval = np.clip(hr.astype(np.float32), 0.0, 1.0)
    sr_eval = np.clip(sr.astype(np.float32), 0.0, 1.0)
    return peak_signal_noise_ratio(hr_eval, sr_eval, data_range=1.0)


def compute_ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    hr_eval = np.clip(hr.astype(np.float32), 0.0, 1.0)
    sr_eval = np.clip(sr.astype(np.float32), 0.0, 1.0)
    try:
        return structural_similarity(hr_eval, sr_eval, channel_axis=-1, data_range=1.0)
    except TypeError:
        return structural_similarity(hr_eval, sr_eval, multichannel=True, data_range=1.0)


# =========================================================
# Blur helpers
# =========================================================
def gaussian_blur(
    img: np.ndarray,
    sigma_x: float,
    sigma_y: Optional[float] = None,
) -> np.ndarray:
    if sigma_y is None:
        sigma_y = sigma_x
    sigma_x = float(max(0.0, sigma_x))
    sigma_y = float(max(0.0, sigma_y))
    if sigma_x <= 0.0 and sigma_y <= 0.0:
        return img
    return cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REFLECT101,
    )


def motion_blur_kernel(length: int, angle_rad: float) -> np.ndarray:
    length = max(1, int(round(length)))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    angle_deg = angle_rad * 180.0 / math.pi
    M = cv2.getRotationMatrix2D(
        (length / 2 - 0.5, length / 2 - 0.5),
        angle_deg,
        1.0,
    )
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def apply_motion_blur(img: np.ndarray, length: float, angle_rad: float) -> np.ndarray:
    if length <= 0:
        return img
    k = motion_blur_kernel(length, angle_rad)
    channels = []
    for c in range(img.shape[2]):
        channels.append(
            cv2.filter2D(
                img[:, :, c],
                -1,
                k,
                borderType=cv2.BORDER_REFLECT101,
            )
        )
    return np.stack(channels, axis=2)


# =========================================================
# Fixed degradation for classical evaluation
# =========================================================
def degrade_hr_to_lr50(hr100_rgb: np.ndarray, deg: Degradation) -> np.ndarray:
    """
    HR (100x100) -> fixed blur -> fixed downsample to 50x50.
    """
    if deg.kind == "gaussian":
        assert deg.sigma is not None
        blurred = gaussian_blur(hr100_rgb, float(deg.sigma))
    elif deg.kind == "motion":
        assert deg.length is not None and deg.theta is not None
        blurred = apply_motion_blur(hr100_rgb, float(deg.length), float(deg.theta))
    else:
        raise ValueError(f"Unknown degradation kind: {deg.kind}")

    lr50 = cv2.resize(blurred, (50, 50), interpolation=cv2.INTER_CUBIC)
    return np.clip(lr50.astype(np.float32), 0.0, 1.0)


def lr50_to_input100(lr50_rgb: np.ndarray) -> np.ndarray:
    """
    Bicubic upsample LR 50x50 to 100x100.
    """
    out = cv2.resize(lr50_rgb, (100, 100), interpolation=cv2.INTER_CUBIC)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


# =========================================================
# Dataset split + loading
# =========================================================
def create_splits(
    n_total: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    rng = torch.Generator().manual_seed(seed)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    all_indices = torch.randperm(n_total, generator=rng).tolist()
    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train : n_train + n_val]
    test_idx = all_indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def load_celeba_splits(
    root: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    base = CelebA(
        root=root,
        split="all",
        target_type="attr",
        download=False,
        transform=None,
    )
    n_total = len(base)
    return create_splits(n_total, train_ratio, val_ratio, seed)


def get_hr_arrays_from_celeba(
    root: str,
    indices: Sequence[int],
    limit: int = 5000,
) -> List[np.ndarray]:
    base = CelebA(
        root=root,
        split="all",
        target_type="attr",
        download=False,
        transform=None,
    )

    hr_transform = transforms.Compose(
        [
            transforms.CenterCrop(160),
            transforms.Resize((100, 100), interpolation=Image.BICUBIC),
        ]
    )

    out = []
    for idx in indices[:limit]:
        img, _ = base[idx]
        img = hr_transform(img)
        hr_np = np.array(img).astype(np.float32) / 255.0
        out.append(np.clip(hr_np, 0.0, 1.0))
    return out


# =========================================================
# Training-pair sampling for SC1 / SC2
# =========================================================
def sample_training_pairs(
    hr100_imgs: Sequence[np.ndarray],
    n_pairs: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create LR100/HR100 pairs for training classical methods.
    LR100 is produced by:
        HR100 -> random blur -> LR50 -> bicubic LR100
    """
    lr_list, hr_list = [], []
    rng = np.random.RandomState(seed)

    for _ in range(n_pairs):
        hr = hr100_imgs[int(rng.randint(0, len(hr100_imgs)))]

        if rng.rand() < 0.5:
            deg = Degradation(
                kind="gaussian",
                sigma=float(rng.uniform(0.0, 7.0)),
            )
        else:
            deg = Degradation(
                kind="motion",
                length=float(rng.uniform(0.0, 11.0)),
                theta=float(rng.uniform(-math.pi, math.pi)),
            )

        lr50 = degrade_hr_to_lr50(hr, deg)
        lr100 = lr50_to_input100(lr50)

        lr_list.append(lr100)
        hr_list.append(hr)

    return lr_list, hr_list


# =========================================================
# Bicubic baseline
# =========================================================
def bicubic_predict(lr100_rgb: np.ndarray) -> np.ndarray:
    return np.clip(lr100_rgb.astype(np.float32), 0.0, 1.0)

# =========================================================
# SC1 (FULL)
# =========================================================


class SC1:
    """
    SC1: Sparse-coding super-resolution (Yang et al. 2008; 2010).
    Coupled dictionary learning on Y-channel patches; Cb/Cr from bicubic input.
    """

    def __init__(
        self,
        patch: int = 5,
        stride: int = 2,
        n_atoms: int = 256,
        lasso_alpha: float = 0.002,
        lasso_max_iter: int = 3000,
        lasso_tol: float = 2e-4,
        dl_iter: int = 120,
        max_train_patches: int = 120_000,
    ):
        self.patch = patch
        self.stride = stride
        self.n_atoms = n_atoms
        self.lasso_alpha = lasso_alpha
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.dl_iter = dl_iter
        self.max_train_patches = max_train_patches
        self.Dl: Optional[np.ndarray] = None
        self.Dh: Optional[np.ndarray] = None
        self._lasso = Lasso(
            alpha=lasso_alpha,
            fit_intercept=False,
            max_iter=lasso_max_iter,
            tol=lasso_tol,
            warm_start=True,
        )

    @staticmethod
    def _row_normalize(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-8)

    def fit(
        self,
        lr100_imgs_rgb: Sequence[np.ndarray],
        hr100_imgs_rgb: Sequence[np.ndarray],
    ) -> "SC1":
        assert len(lr100_imgs_rgb) == len(hr100_imgs_rgb)
        Xl_list, Xh_list = [], []
        rng = np.random.RandomState(42)
        for lr_rgb, hr_rgb in zip(lr100_imgs_rgb, hr100_imgs_rgb):
            yl, _, _ = rgb_to_ycbcr(lr_rgb)
            yh, _, _ = rgb_to_ycbcr(hr_rgb)
            Xl, _ = extract_patches_2d(yl, self.patch, self.stride)
            Xh, _ = extract_patches_2d(yh, self.patch, self.stride)
            if Xl.shape[0] > 2000:
                idx = rng.choice(Xl.shape[0], size=2000, replace=False)
                Xl, Xh = Xl[idx], Xh[idx]
            Xl_list.append(Xl)
            Xh_list.append(Xh)
            if sum(x.shape[0] for x in Xl_list) >= self.max_train_patches:
                break
        Xl_all = self._row_normalize(np.concatenate(Xl_list, axis=0))
        Xh_all = self._row_normalize(np.concatenate(Xh_list, axis=0))
        dl = DictionaryLearning(
            n_components=self.n_atoms,
            alpha=1.0,
            max_iter=self.dl_iter,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            random_state=42,
        )
        A = dl.fit_transform(Xl_all)
        self.Dl = dl.components_.astype(np.float32)
        AtA = A.T @ A + 1e-4 * np.eye(self.n_atoms, dtype=np.float32)
        self.Dh = np.linalg.solve(AtA, (A.T @ Xh_all).astype(np.float32)).astype(np.float32)
        return self

    def _code(self, x: np.ndarray) -> np.ndarray:
        assert self.Dl is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self._lasso.fit(self.Dl.T, x)
        return self._lasso.coef_.astype(np.float32)

    def predict(self, lr100_rgb: np.ndarray) -> np.ndarray:
        assert self.Dl is not None and self.Dh is not None
        y, cb, cr = rgb_to_ycbcr(lr100_rgb)
        Xl, coords = extract_patches_2d(y, self.patch, self.stride)
        Xl = self._row_normalize(Xl)
        A = np.zeros((Xl.shape[0], self.n_atoms), dtype=np.float32)
        for i in range(Xl.shape[0]):
            A[i] = self._code(Xl[i])
        Xh_hat = A @ self.Dh
        y_hat = aggregate_patches_2d(Xh_hat, coords, (100, 100), self.patch)
        y_hat = np.clip(y_hat, 0.0, 1.0)
        return ycbcr_to_rgb(y_hat, cb, cr)

# =========================================================
# SC2
# =========================================================

class SC2:
    """
    SC2: Sparse regression SR (Kim & Kwon 2010). Anchor-based K-NN + ridge regression on Y-channel.
    """

    def __init__(
        self,
        patch: int = 5,
        stride: int = 2,
        n_anchors: int = 8000,
        k_nn: int = 64,
        ridge: float = 1e-3,
        max_train_patches: int = 250_000,
    ):
        self.patch = patch
        self.stride = stride
        self.n_anchors = n_anchors
        self.k_nn = k_nn
        self.ridge = ridge
        self.max_train_patches = max_train_patches
        self.lr_anchors: Optional[np.ndarray] = None
        self.hr_anchors: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

    def fit(
        self,
        lr100_imgs_rgb: Sequence[np.ndarray],
        hr100_imgs_rgb: Sequence[np.ndarray],
    ) -> "SC2":
        assert len(lr100_imgs_rgb) == len(hr100_imgs_rgb)
        Xl_list, Xh_list = [], []
        rng = np.random.RandomState(42)
        for lr_rgb, hr_rgb in zip(lr100_imgs_rgb, hr100_imgs_rgb):
            yl, _, _ = rgb_to_ycbcr(lr_rgb)
            yh, _, _ = rgb_to_ycbcr(hr_rgb)
            Xl, _ = extract_patches_2d(yl, self.patch, self.stride)
            Xh, _ = extract_patches_2d(yh, self.patch, self.stride)
            if Xl.shape[0] > 4000:
                idx = rng.choice(Xl.shape[0], size=4000, replace=False)
                Xl, Xh = Xl[idx], Xh[idx]
            Xl_list.append(Xl)
            Xh_list.append(Xh)
            if sum(x.shape[0] for x in Xl_list) >= self.max_train_patches:
                break
        Xl_all = np.concatenate(Xl_list, axis=0).astype(np.float32)
        Xh_all = np.concatenate(Xh_list, axis=0).astype(np.float32)
        n = Xl_all.shape[0]
        take = min(self.n_anchors, n)
        idx = rng.choice(n, size=take, replace=False)
        self.lr_anchors = Xl_all[idx]
        self.hr_anchors = Xh_all[idx]
        k = min(self.k_nn, take)
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        self.nn.fit(self.lr_anchors)
        return self

    def predict(self, lr100_rgb: np.ndarray) -> np.ndarray:
        assert self.lr_anchors is not None and self.hr_anchors is not None and self.nn is not None
        y, cb, cr = rgb_to_ycbcr(lr100_rgb)
        Xl, coords = extract_patches_2d(y, self.patch, self.stride)
        Xh_hat = np.zeros_like(Xl, dtype=np.float32)
        _, neigh = self.nn.kneighbors(Xl, return_distance=True)
        for i in range(Xl.shape[0]):
            Zl = self.lr_anchors[neigh[i]]
            Zh = self.hr_anchors[neigh[i]]
            A = (Zl @ Zl.T).astype(np.float32) + self.ridge * np.eye(Zl.shape[0], dtype=np.float32)
            b = (Zl @ Xl[i]).astype(np.float32)
            w = np.linalg.solve(A, b).astype(np.float32)
            Xh_hat[i] = w @ Zh
        y_hat = aggregate_patches_2d(Xh_hat, coords, (100, 100), self.patch)
        y_hat = np.clip(y_hat, 0.0, 1.0)
        return ycbcr_to_rgb(y_hat, cb, cr)


# =========================================================
# SFH
# =========================================================

def _try_import_face_alignment() -> Optional[object]:
    try:
        import face_alignment  # type: ignore
        return face_alignment
    except Exception:
        return None


class SFH:
    """
    SFH: Structured Face Hallucination (Yang, Liu, Yang 2013). Landmark-guided component mask + gradient refinement.
    Uses SR1 as base; optional face_alignment for 68 landmarks.
    """

    def __init__(self, base_sr: SC1, use_gpu: bool = True):
        self.base = base_sr
        fa_mod = _try_import_face_alignment()
        self.fa = None
        if fa_mod is not None:
            device_str = "cuda" if (use_gpu and hasattr(fa_mod, "FaceAlignment")) else "cpu"
            try:
                self.fa = fa_mod.FaceAlignment(
                    fa_mod.LandmarksType.TWO_D,
                    flip_input=False,
                    device=device_str,
                )
            except Exception:
                self.fa = None

    @staticmethod
    def _poly_mask(h: int, w: int, pts_xy: np.ndarray) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        if pts_xy.shape[0] < 3:
            return mask
        poly = np.round(pts_xy).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 1.0)
        return mask

    def _component_mask(self, img100_rgb: np.ndarray) -> np.ndarray:
        h, w = img100_rgb.shape[:2]
        if self.fa is None:
            return np.zeros((h, w), dtype=np.float32)
        try:
            lms = self.fa.get_landmarks((img100_rgb * 255.0).astype(np.uint8))
        except Exception:
            lms = None
        if lms is None or len(lms) == 0:
            return np.zeros((h, w), dtype=np.float32)
        lm = np.asarray(lms[0], dtype=np.float32)
        eye = lm[np.r_[36:42, 42:48]]
        nose = lm[27:36]
        mouth = lm[48:68]
        m_eye = self._poly_mask(h, w, eye)
        m_nose = self._poly_mask(h, w, nose)
        m_mouth = self._poly_mask(h, w, mouth)
        comp = np.clip(m_eye + m_nose + m_mouth, 0.0, 1.0)
        comp = cv2.GaussianBlur(comp, (9, 9), 0).astype(np.float32)
        return np.clip(comp, 0.0, 1.0)

    @staticmethod
    def _screened_poisson(
        target: np.ndarray, guidance: np.ndarray, mask: np.ndarray, lam: float = 0.15
    ) -> np.ndarray:
        H, W = target.shape
        N = H * W
        idx = np.arange(N, dtype=np.int32).reshape(H, W)
        rows, cols, data = [], [], []
        b = np.zeros((N,), dtype=np.float32)

        def add(r: int, c: int, v: float) -> None:
            rows.append(r)
            cols.append(c)
            data.append(v)

        for y in range(H):
            for x in range(W):
                p = int(idx[y, x])
                if mask[y, x] < 0.5:
                    add(p, p, 1.0)
                    b[p] = target[y, x]
                    continue
                add(p, p, 1.0 + 4.0 * lam)
                for (yy, xx) in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < H and 0 <= xx < W:
                        add(p, int(idx[yy, xx]), -lam)
                gyx = guidance[y, x]
                g_up = guidance[y - 1, x] if y - 1 >= 0 else gyx
                g_dn = guidance[y + 1, x] if y + 1 < H else gyx
                g_lf = guidance[y, x - 1] if x - 1 >= 0 else gyx
                g_rt = guidance[y, x + 1] if x + 1 < W else gyx
                lap_g = 4.0 * gyx - g_up - g_dn - g_lf - g_rt
                b[p] = target[y, x] + lam * lap_g
        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        sol = spsolve(A, b).astype(np.float32)
        return np.clip(sol.reshape(H, W), 0.0, 1.0)

    def predict(self, lr100_rgb: np.ndarray) -> np.ndarray:
        base = self.base.predict(lr100_rgb)
        bic = np.clip(lr100_rgb, 0.0, 1.0)
        comp = self._component_mask(lr100_rgb)
        blended = base * comp[:, :, None] + bic * (1.0 - comp[:, :, None])
        y_t, cb_t, cr_t = rgb_to_ycbcr(blended)
        y_g, _, _ = rgb_to_ycbcr(bic)
        y_ref = self._screened_poisson(y_t, y_g, mask=comp, lam=0.15)
        return ycbcr_to_rgb(y_ref, cb_t, cr_t)

# =========================================================
# Evaluation
# =========================================================
def eval_method_on_conditions(
    hr100_test: Sequence[np.ndarray],
    method_name: str,
    predict_fn,
    conditions: Sequence[Degradation],
    seed: int = 42,
) -> Dict[str, MethodResult]:
    """
    For each condition:
        HR100 -> fixed blur -> LR50 -> bicubic LR100 -> method -> PSNR/SSIM
    """
    rng = np.random.RandomState(seed)
    out: Dict[str, MethodResult] = {}

    for cond in conditions:
        psnrs, ssims = [], []
        total = len(hr100_test)

        for i, hr in enumerate(hr100_test, start=1):
            if cond.kind == "motion":
                deg = Degradation(
                    kind="motion",
                    length=cond.length,
                    theta=float(rng.uniform(-math.pi, math.pi)),
                )
            else:
                deg = cond

            lr50 = degrade_hr_to_lr50(hr, deg)
            lr100 = lr50_to_input100(lr50)

            pred = predict_fn(lr100)
            pred = np.clip(pred, 0.0, 1.0)

            psnrs.append(compute_psnr(hr, pred))
            ssims.append(compute_ssim(hr, pred))

            if i % 25 == 0 or i == total:
                ckey = f"sigma={cond.sigma:g}" if cond.kind == "gaussian" else f"l={cond.length:g}"
                print(f"[{method_name}] {ckey}: processed {i}/{total} images...", flush=True)

        key = f"sigma={cond.sigma:g}" if cond.kind == "gaussian" else f"l={cond.length:g}"
        out[key] = MethodResult(
            psnr=float(np.mean(psnrs)),
            ssim=float(np.mean(ssims)),
        )
        print(f"[{method_name}] {key}: PSNR={out[key].psnr:.2f}, SSIM={out[key].ssim:.3f}", flush=True)

    return out


# =========================================================
# Main pipeline
# =========================================================
def run_classical_pipeline(
    data_root: str,
    seed: int = 42,
    train_hr_limit: int = 50,
    test_hr_limit: int = 20,
    train_pairs: int = 80,
    sr1_atoms: int = 128,
    sr2_anchors: int = 2000,
    sr2_knn: int = 32,
):
    set_global_seed(seed)

    train_idx, _, test_idx = load_celeba_splits(data_root, seed=seed)

    print("Loading HR training images...")
    hr_train = get_hr_arrays_from_celeba(data_root, train_idx, limit=train_hr_limit)

    print("Loading HR test images...")
    hr_test = get_hr_arrays_from_celeba(data_root, test_idx, limit=test_hr_limit)

    gaussian_conds = [
        Degradation(kind="gaussian", sigma=1.0),
        Degradation(kind="gaussian", sigma=3.0),
        Degradation(kind="gaussian", sigma=5.0),
    ]

    motion_conds = [
        Degradation(kind="motion", length=2.0, theta=0.0),
        Degradation(kind="motion", length=6.0, theta=0.0),
        Degradation(kind="motion", length=9.0, theta=0.0),
    ]

    print("Sampling classical training pairs...")
    lr_train, hr_train_pairs = sample_training_pairs(
        hr_train,
        n_pairs=train_pairs,
        seed=seed,
    )

    print("Training SC1...")
    sr1 = SC1(n_atoms=sr1_atoms)
    sr1.fit(lr_train, hr_train_pairs)

    print("Training SC2...")
    sr2 = SC2(n_anchors=sr2_anchors, k_nn=sr2_knn)
    sr2.fit(lr_train, hr_train_pairs)

    print("Initializing SFH...")
    sfh = SFH(base_sr=sr1, use_gpu=torch.cuda.is_available())

    print("\n=== TEST RESULTS: GAUSSIAN ===")
    g_bic = eval_method_on_conditions(hr_test, "Bicubic", bicubic_predict, gaussian_conds, seed=seed)
    g_sc1 = eval_method_on_conditions(hr_test, "SC1", sr1.predict, gaussian_conds, seed=seed)
    g_sc2 = eval_method_on_conditions(hr_test, "SC2", sr2.predict, gaussian_conds, seed=seed)
    g_sfh = eval_method_on_conditions(hr_test, "SFH", sfh.predict, gaussian_conds, seed=seed)

    print("\n=== TEST RESULTS: MOTION ===")
    m_bic = eval_method_on_conditions(hr_test, "Bicubic", bicubic_predict, motion_conds, seed=seed)
    m_sc1 = eval_method_on_conditions(hr_test, "SC1", sr1.predict, motion_conds, seed=seed)
    m_sc2 = eval_method_on_conditions(hr_test, "SC2", sr2.predict, motion_conds, seed=seed)
    m_sfh = eval_method_on_conditions(hr_test, "SFH", sfh.predict, motion_conds, seed=seed)

    return {
        "gaussian": {
            "Bicubic": g_bic,
            "SC1": g_sc1,
            "SC2": g_sc2,
            "SFH": g_sfh,
        },
        "motion": {
            "Bicubic": m_bic,
            "SC1": m_sc1,
            "SC2": m_sc2,
            "SFH": m_sfh,
        },
    }


if __name__ == "__main__":
    ROOT = "/home/msds/tans0444"

    results = run_classical_pipeline(
        data_root=ROOT,
        seed=42,
        train_hr_limit=50,
        test_hr_limit=20,
        train_pairs=80,
        sr1_atoms=128,
        sr2_anchors=2000,
        sr2_knn=32,
    )

    print("\nDone.")