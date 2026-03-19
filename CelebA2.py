import os
import math
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

METRIC_COLOR_SPACE = "rgb"

try:
    from sklearn.decomposition import DictionaryLearning
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:
    raise RuntimeError("scikit-learn required for SC1/SC2 baselines. pip install scikit-learn") from e
try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except ImportError as e:
    raise RuntimeError("scipy required for SFH baseline. pip install scipy") from e


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


############################################
# Models
############################################


class FeatureExtractor(nn.Module):
    """
    Three conv + pool layers as in Table 1 of the paper.
    Input: (B, 3, 48, 48)
    Output: flattened features of size 2048.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor: #conv> tanh > pooling=conv>pool>tanh (this is computationally easier)
        x = torch.tanh(self.pool(self.conv1(x)))  # (B, 32, 22, 22)
        x = torch.tanh(self.pool(self.conv2(x)))  # (B, 64, 10, 10)
        x = torch.tanh(self.pool(self.conv3(x)))  # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 2048)
        return x


class BasicCNN(nn.Module):
    """
    Strict straightforward CNN from the paper (no fusion).
    Produces a 100x100 RGB hallucinated face from normalized input.
    """

    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.fc1 = nn.Linear(2048, 2000)
        self.fc2 = nn.Linear(2000, 3 * 100 * 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: normalized (B, 3, 48, 48)
        f = self.features(x)  # (B, 2048)
        f = torch.tanh(self.fc1(f))  # (B, 2000)
        out = torch.tanh(self.fc2(f))  # (B, 30000)
        out = out.view(-1, 3, 100, 100)  # normalized HR
        return out


class BiChannelCNN(nn.Module):
    """
    Bi-channel CNN with fusion:
      alpha * upsampled_input + (1 - alpha) * I_rec
    All images in normalized space.

    Training configuration (following the paper):
      1) Parameter initialization: all conv filters and fully-connected
         weights ~ N(0, 0.001), all biases = 0.
      2) Gradients are computed by back-propagation using SGD.
      3) Momentum-based SGD update: v_k = 0.9 * v_{k-1} + lr * dL/dw,
         w_k = w_{k-1} - v_k.
      4) Learning-rate schedule: initial lr = 0.00001. When validation
         error stops decreasing, lr is reduced by a factor of 10.
         Mini-batch size is 200, and training is run for ~5000 cycles.
    """

    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()

        # Branch 1: intermediate image I_rec
        self.fc1_1 = nn.Linear(2048, 2000)
        self.fc2_1 = nn.Linear(2000, 3 * 100 * 100)

        # Branch 2: fusion coefficient alpha
        self.fc1_2 = nn.Linear(2048, 100)
        self.fc2_2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor, upsampled_norm: torch.Tensor) -> torch.Tensor:
        """
        x: normalized LR input (B, 3, 48, 48)
        upsampled_norm: normalized upsampled LR (B, 3, 100, 100)
        Returns normalized hallucinated HR (B, 3, 100, 100).
        """
        f = self.features(x)  # (B, 2048)

        # I_rec branch
        i4 = torch.tanh(self.fc1_1(f))  # (B, 2000)
        irec = torch.tanh(self.fc2_1(i4))  # (B, 30000)
        irec = irec.view(-1, 3, 100, 100)

        # alpha branch
        i5 = torch.tanh(self.fc1_2(f))  # (B, 100)
        alpha_raw = self.fc2_2(i5)  # (B, 1)
        alpha = 0.5 * torch.tanh(alpha_raw) + 0.5  # normalize to [0,1]
        alpha = alpha.view(-1, 1, 1, 1)  # broadcast

        out = alpha * upsampled_norm + (1.0 - alpha) * irec
        return out


############################################
# Data and preprocessing
############################################


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
        (length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0
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


def degrade_image(
    hr_img: np.ndarray,
    blur_type: str,
    sigma: Optional[float] = None,
    length: Optional[float] = None,
    train_random: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    hr_img: float32, [0,1], shape (100,100,3)
    Returns LR (possibly not 48x48) and degradation params.
    """
    H, W, _ = hr_img.shape
    assert H == 100 and W == 100

    params = {}

    # For training with blur_type == "mixed", randomly choose
    # between Gaussian and motion blur with 50/50 probability.
    if train_random and blur_type == "mixed":
        if random.random() < 0.5:
            blur_type_effective = "gaussian"
        else:
            blur_type_effective = "motion"
    else:
        blur_type_effective = blur_type

    if blur_type_effective == "gaussian":
        if train_random:
            # Training/validation: anisotropic Gaussian blur (independent sigmas).
            sigma_x = random.uniform(0.0, 7.0)
            sigma_y = random.uniform(0.0, 7.0)
        else:
            # Testing/evaluation: isotropic Gaussian blur.
            sigma_iso = random.uniform(0.0, 7.0) if sigma is None else float(sigma)
            sigma_x = sigma_iso
            sigma_y = sigma_iso
        blurred = gaussian_blur(hr_img, sigma_x, sigma_y)
        params["sigma_x"] = sigma_x
        params["sigma_y"] = sigma_y
        if abs(sigma_x - sigma_y) < 1e-12:
            params["sigma"] = sigma_x
    elif blur_type_effective == "motion":
        if train_random or length is None:
            length_val = random.uniform(0.0, 11.0)
        else:
            length_val = float(length)
        theta = random.uniform(-math.pi, math.pi)
        blurred = apply_motion_blur(hr_img, length_val, theta)
        params["length"] = length_val
        params["theta"] = theta
    else:
        blurred = hr_img

    if train_random:
        factor = random.choice([2, 3, 4, 5])
    else:
        factor = 2  # 100 -> 50 for evaluation, as in the paper

    new_size = (W // factor, H // factor)
    lr = cv2.resize(blurred, new_size, interpolation=cv2.INTER_CUBIC)
    params["down_factor"] = factor

    return lr, params


############################################
# Classical SR baselines (SC1, SC2, SFH) — Y/Cb/Cr, patches, degradation
############################################


def rgb_to_ycbcr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """img: RGB [0,1]. Returns Y, Cr, Cb in [0,1]."""
    ycbcr = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycbcr[:, :, 0] / 255.0
    cr = ycbcr[:, :, 1] / 255.0
    cb = ycbcr[:, :, 2] / 255.0
    return y, cr, cb


def ycbcr_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ycrcb = np.stack([y, cr, cb], axis=-1)
    rgb = cv2.cvtColor((ycrcb * 255.0).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    return np.clip(rgb / 255.0, 0.0, 1.0)


def extract_patches_2d(
    img2d: np.ndarray, patch: int, stride: int
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


@dataclass(frozen=True)
class Degradation:
    """Fixed degradation for baseline eval (Gaussian or motion blur -> 50x50)."""
    kind: str  # "gaussian" or "motion"
    sigma: Optional[float] = None
    length: Optional[float] = None
    theta: Optional[float] = None


def degrade_hr_to_lr50(hr100_rgb: np.ndarray, deg: Degradation) -> np.ndarray:
    """HR (100x100) -> blur -> downsample to 50x50. Uses existing gaussian/motion blur."""
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
    """Bicubic upsample LR 50x50 to 100x100 for classical SR input."""
    out = cv2.resize(lr50_rgb, (100, 100), interpolation=cv2.INTER_CUBIC)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def preprocess_for_network(lr_img: np.ndarray, hr_img: np.ndarray):
    """
    lr_img: float32 [0,1], shape (h,w,3)
    hr_img: float32 [0,1], shape (100,100,3)

    Returns:
        in_norm: (3,48,48) normalized input
        up_norm: (3,100,100) normalized upsampled LR
        target_norm: (3,100,100) normalized HR
        mean: (3,) mean of input (per channel)
        std: (3,) std of input (per channel)
    """
    lr_t = torch.from_numpy(lr_img.transpose(2, 0, 1)).float()
    hr_t = torch.from_numpy(hr_img.transpose(2, 0, 1)).float()

    h, w = lr_t.shape[1:]
    if (h, w) != (48, 48):
        in_t = F.interpolate(
            lr_t.unsqueeze(0),
            size=(48, 48),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    else:
        in_t = lr_t

    up_t = F.interpolate(
        lr_t.unsqueeze(0),
        size=(100, 100),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    mean = in_t.view(3, -1).mean(dim=1)
    std = in_t.view(3, -1).std(dim=1) + 1e-6 #1e-6 is called epsilon stabilisation -to avoid division by zero

    def norm_and_tanh(img_t: torch.Tensor) -> torch.Tensor:
        m = mean.view(3, 1, 1)
        s = std.view(3, 1, 1)
        x = (img_t - m) / s
        return torch.tanh(x)

    in_norm = norm_and_tanh(in_t)
    up_norm = norm_and_tanh(up_t)
    target_norm = norm_and_tanh(hr_t)

    return in_norm, up_norm, target_norm, mean, std


class CelebAFaceHallucination(Dataset):
    """
    Wraps CelebA and generates:
      in_norm, up_norm, target_norm, mean, std, hr
    """

    def __init__(
        self,
        root: str,
        split_indices: List[int],
        blur_type: str,
        train_random: bool,
        sigma: Optional[float] = None,
        length: Optional[float] = None,
    ):
        super().__init__()
        self.root = root
        self.blur_type = blur_type
        self.train_random = train_random
        self.sigma = sigma
        self.length = length

        try:
            self.base = CelebA(
                root=root,
                split="all",
                target_type="attr",
                download=False,
                transform=None,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"CelebA not found under root='{root}' with download disabled. "
                "Please place the dataset manually using torchvision's expected layout, "
                "for example: <root>/celeba/img_align_celeba and annotation files."
            ) from e

        self.indices = split_indices

        self.hr_transform = transforms.Compose(
            [
                transforms.CenterCrop(160),
                transforms.Resize((100, 100), interpolation=Image.BICUBIC),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, _ = self.base[real_idx]
        img = self.hr_transform(img)
        hr_np = np.array(img).astype(np.float32) / 255.0

        lr_np, _ = degrade_image(
            hr_np,
            blur_type=self.blur_type,
            sigma=self.sigma,
            length=self.length,
            train_random=self.train_random,
        )

        in_norm, up_norm, target_norm, mean, std = preprocess_for_network(
            lr_np,
            hr_np,
        )

        hr_tensor = torch.from_numpy(hr_np.transpose(2, 0, 1)).float()

        return {
            "in_norm": in_norm,
            "up_norm": up_norm,
            "target_norm": target_norm,
            "mean": mean,
            "std": std,
            "hr": hr_tensor,
        }


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
    try:
        base = CelebA(
            root=root,
            split="all",
            target_type="attr",
            download=False,
            transform=None,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"CelebA not found under root='{root}' with download disabled. "
            "Please place the dataset manually using torchvision's expected layout, "
            "for example: <root>/celeba/img_align_celeba and annotation files."
        ) from e
    n_total = len(base)
    return create_splits(n_total, train_ratio, val_ratio, seed)


############################################
# Classical SR: SC1 (Yang et al.), SC2 (Kim & Kwon), SFH (Yang, Liu, Yang)
############################################


class SR1_SC1_Yang:
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
    ) -> "SR1_SC1_Yang":
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


class SR2_SC2_KimKwon:
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
    ) -> "SR2_SC2_KimKwon":
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


def _try_import_face_alignment() -> Optional[object]:
    try:
        import face_alignment  # type: ignore
        return face_alignment
    except Exception:
        return None


class SFH_YangProxy:
    """
    SFH: Structured Face Hallucination (Yang, Liu, Yang 2013). Landmark-guided component mask + gradient refinement.
    Uses SR1 as base; optional face_alignment for 68 landmarks.
    """

    def __init__(self, base_sr: SR1_SC1_Yang, use_gpu: bool = True):
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


def bicubic_predict(lr100_rgb: np.ndarray) -> np.ndarray:
    """
    Bicubic baseline: input is already bicubic-upsampled LR (50->100).
    Returns it as-is (no learned super-resolution). Used in Table 2 as the simple baseline.
    """
    return np.clip(lr100_rgb.astype(np.float32), 0.0, 1.0)


############################################
# Metrics
############################################


def compute_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    hr_eval, sr_eval = _prepare_metric_images(hr, sr)
    return peak_signal_noise_ratio(hr_eval, sr_eval, data_range=1.0)


def compute_ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    hr_eval, sr_eval = _prepare_metric_images(hr, sr)
    # skimage>=0.19 expects channel_axis; older versions expect multichannel.
    if hr_eval.ndim == 2:
        return structural_similarity(hr_eval, sr_eval, data_range=1.0)
    try:
        return structural_similarity(hr_eval, sr_eval, channel_axis=-1, data_range=1.0)
    except TypeError:
        return structural_similarity(hr_eval, sr_eval, multichannel=True, data_range=1.0)


def _prepare_metric_images(hr: np.ndarray, sr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hr_eval = np.clip(hr.astype(np.float32), 0.0, 1.0)
    sr_eval = np.clip(sr.astype(np.float32), 0.0, 1.0)
    if METRIC_COLOR_SPACE == "y":
        hr_eval, _, _ = rgb_to_ycbcr(hr_eval)
        sr_eval, _, _ = rgb_to_ycbcr(sr_eval)
    return hr_eval, sr_eval


############################################
# Paper Table 2 (AAAI 2015) — all 6 methods
# Source: Zhou et al., "Learning Face Hallucination in the Wild", AAAI 2015.
# Methods: Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN. SSIM: Wang et al. 2004.
############################################

PAPER_TABLE_2_SC1_SC2_SFH: Dict[str, Dict[str, Dict[str, float]]] = {
    "gaussian": {
        "PSNR": {
            "sigma=1": {"SC1": 32.89, "SC2": 32.98, "SFH": 32.56},
            "sigma=3": {"SC1": 30.30, "SC2": 30.31, "SFH": 30.06},
            "sigma=5": {"SC1": 29.48, "SC2": 29.49, "SFH": 29.29},
        },
        "SSIM": {
            "sigma=1": {"SC1": 0.87, "SC2": 0.87, "SFH": 0.88},
            "sigma=3": {"SC1": 0.68, "SC2": 0.68, "SFH": 0.67},
            "sigma=5": {"SC1": 0.58, "SC2": 0.58, "SFH": 0.58},
        },
    },
    "motion": {
        "PSNR": {
            "l=2": {"SC1": 34.05, "SC2": 34.23, "SFH": 33.59},
            "l=6": {"SC1": 29.68, "SC2": 29.69, "SFH": 29.53},
            "l=9": {"SC1": 28.76, "SC2": 28.76, "SFH": 28.67},
        },
        "SSIM": {
            "l=2": {"SC1": 0.91, "SC2": 0.91, "SFH": 0.91},
            "l=6": {"SC1": 0.65, "SC2": 0.65, "SFH": 0.64},
            "l=9": {"SC1": 0.48, "SC2": 0.48, "SFH": 0.48},
        },
    },
}

# Full Table 2: Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN (paper-reported values)
PAPER_TABLE_2_ALL_METHODS: List[str] = [
    "Bicubic", "SC1", "SC2", "SFH", "Basic CNN", "Bi-channel CNN"
]
PAPER_TABLE_2_FULL: Dict[str, Dict[str, Dict[str, float]]] = {
    "gaussian": {
        "PSNR": {
            "sigma=1": {"Bicubic": 32.15, "SC1": 32.89, "SC2": 32.98, "SFH": 32.56, "Basic CNN": 29.78, "Bi-channel CNN": 33.11},
            "sigma=3": {"Bicubic": 30.33, "SC1": 30.30, "SC2": 30.31, "SFH": 30.06, "Basic CNN": 29.78, "Bi-channel CNN": 30.35},
            "sigma=5": {"Bicubic": 29.52, "SC1": 29.48, "SC2": 29.49, "SFH": 29.29, "Basic CNN": 29.61, "Bi-channel CNN": 29.71},
        },
        "SSIM": {
            "sigma=1": {"Bicubic": 0.84, "SC1": 0.87, "SC2": 0.87, "SFH": 0.88, "Basic CNN": 0.70, "Bi-channel CNN": 0.89},
            "sigma=3": {"Bicubic": 0.68, "SC1": 0.68, "SC2": 0.68, "SFH": 0.67, "Basic CNN": 0.68, "Bi-channel CNN": 0.71},
            "sigma=5": {"Bicubic": 0.58, "SC1": 0.58, "SC2": 0.58, "SFH": 0.58, "Basic CNN": 0.65, "Bi-channel CNN": 0.65},
        },
    },
    "motion": {
        "PSNR": {
            "l=2": {"Bicubic": 32.39, "SC1": 34.05, "SC2": 34.23, "SFH": 33.59, "Basic CNN": 29.75, "Bi-channel CNN": 34.63},
            "l=6": {"Bicubic": 30.11, "SC1": 29.68, "SC2": 29.69, "SFH": 29.53, "Basic CNN": 29.35, "Bi-channel CNN": 30.23},
            "l=9": {"Bicubic": 28.89, "SC1": 28.76, "SC2": 28.76, "SFH": 28.67, "Basic CNN": 29.07, "Bi-channel CNN": 29.33},
        },
        "SSIM": {
            "l=2": {"Bicubic": 0.85, "SC1": 0.91, "SC2": 0.91, "SFH": 0.91, "Basic CNN": 0.69, "Bi-channel CNN": 0.92},
            "l=6": {"Bicubic": 0.71, "SC1": 0.65, "SC2": 0.65, "SFH": 0.64, "Basic CNN": 0.65, "Bi-channel CNN": 0.77},
            "l=9": {"Bicubic": 0.52, "SC1": 0.48, "SC2": 0.48, "SFH": 0.48, "Basic CNN": 0.61, "Bi-channel CNN": 0.68},
        },
    },
}


def print_paper_table_2_sc1_sc2_sfh() -> None:
    """
    Print Table 2 (excerpt: SC1, SC2, SFH) from 'Learning Face Hallucination in the Wild' (AAAI 2015).
    """
    methods = ["SC1", "SC2", "SFH"]
    print("\n" + "=" * 70)
    print("Table 2 (excerpt): Learning Face Hallucination in the Wild (AAAI 2015)")
    print("Methods: SC1 (Yang et al. 2008; 2010), SC2 (Kim & Kwon 2010), SFH (Yang, Liu, Yang 2013)")
    print("Metrics: PSNR (dB), SSIM (Wang et al. 2004)")
    print("=" * 70)

    print("\n(a) Quantitative comparison under Gaussian blur")
    print("PSNR\t\t" + "\t".join(methods))
    for sigma_key in ["sigma=1", "sigma=3", "sigma=5"]:
        row = [sigma_key] + [f"{PAPER_TABLE_2_SC1_SC2_SFH['gaussian']['PSNR'][sigma_key][m]:.2f}" for m in methods]
        print("\t\t".join(row))
    print("SSIM\t\t" + "\t".join(methods))
    for sigma_key in ["sigma=1", "sigma=3", "sigma=5"]:
        row = [sigma_key] + [f"{PAPER_TABLE_2_SC1_SC2_SFH['gaussian']['SSIM'][sigma_key][m]:.2f}" for m in methods]
        print("\t\t".join(row))

    print("\n(b) Quantitative comparison under motion blur")
    print("PSNR\t\t" + "\t".join(methods))
    for l_key in ["l=2", "l=6", "l=9"]:
        row = [l_key] + [f"{PAPER_TABLE_2_SC1_SC2_SFH['motion']['PSNR'][l_key][m]:.2f}" for m in methods]
        print("\t\t".join(row))
    print("SSIM\t\t" + "\t".join(methods))
    for l_key in ["l=2", "l=6", "l=9"]:
        row = [l_key] + [f"{PAPER_TABLE_2_SC1_SC2_SFH['motion']['SSIM'][l_key][m]:.2f}" for m in methods]
        print("\t\t".join(row))
    print("=" * 70)


def print_paper_table_2_all_six() -> None:
    """
    Print full Table 2 from the paper: all 6 methods (Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN).
    """
    methods = PAPER_TABLE_2_ALL_METHODS
    print("\n" + "=" * 90)
    print("Table 2 (full): Learning Face Hallucination in the Wild (AAAI 2015)")
    print("Methods: Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN | Metrics: PSNR (dB), SSIM (Wang et al. 2004)")
    print("=" * 90)

    print("\n(a) Quantitative comparison under Gaussian blur")
    print("PSNR\t\t" + "\t".join(methods))
    for sigma_key in ["sigma=1", "sigma=3", "sigma=5"]:
        row = [sigma_key] + [f"{PAPER_TABLE_2_FULL['gaussian']['PSNR'][sigma_key][m]:.2f}" for m in methods]
        print("\t".join(row))
    print("SSIM\t\t" + "\t".join(methods))
    for sigma_key in ["sigma=1", "sigma=3", "sigma=5"]:
        row = [sigma_key] + [f"{PAPER_TABLE_2_FULL['gaussian']['SSIM'][sigma_key][m]:.2f}" for m in methods]
        print("\t".join(row))

    print("\n(b) Quantitative comparison under motion blur")
    print("PSNR\t\t" + "\t".join(methods))
    for l_key in ["l=2", "l=6", "l=9"]:
        row = [l_key] + [f"{PAPER_TABLE_2_FULL['motion']['PSNR'][l_key][m]:.2f}" for m in methods]
        print("\t".join(row))
    print("SSIM\t\t" + "\t".join(methods))
    for l_key in ["l=2", "l=6", "l=9"]:
        row = [l_key] + [f"{PAPER_TABLE_2_FULL['motion']['SSIM'][l_key][m]:.2f}" for m in methods]
        print("\t".join(row))
    print("=" * 90)


def get_hr_arrays_from_celeba(root: str, indices: List[int], limit: int = 5000) -> List[np.ndarray]:
    """Load CelebA images as 100x100 RGB float32 [0,1] for baseline training/eval."""
    base = CelebA(root=root, split="all", target_type="attr", download=False, transform=None)
    hr_transform = transforms.Compose(
        [transforms.CenterCrop(160), transforms.Resize((100, 100), interpolation=Image.BICUBIC)]
    )
    out = []
    for i in indices[:limit]:
        img, _ = base[i]
        hr_np = np.array(hr_transform(img)).astype(np.float32) / 255.0
        out.append(np.clip(hr_np, 0.0, 1.0))
    return out


def sample_training_pairs(
    hr100_imgs: Sequence[np.ndarray],
    n_pairs: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build (lr100_input, hr100_target) pairs for SC1/SC2: random blur -> 50x50 -> bicubic 100x100."""
    rng = np.random.RandomState(seed)
    lr_list, hr_list = [], []
    for _ in range(n_pairs):
        hr = hr100_imgs[int(rng.randint(0, len(hr100_imgs)))]
        if rng.rand() < 0.5:
            deg = Degradation(kind="gaussian", sigma=float(rng.uniform(0.0, 7.0)))
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


@dataclass(frozen=True)
class MethodResult:
    psnr: float
    ssim: float


def eval_method_on_conditions(
    hr100_test: Sequence[np.ndarray],
    method_name: str,
    predict_fn,
    conditions: Sequence[Degradation],
    seed: int = 42,
) -> Dict[str, MethodResult]:
    """For each condition, degrade HR->LR50->LR100, run predict_fn(LR100), compute mean PSNR/SSIM."""
    rng = np.random.RandomState(seed)
    out: Dict[str, MethodResult] = {}
    for cond in conditions:
        psnrs, ssims = [], []
        total = len(hr100_test)
        for i, hr in enumerate(hr100_test, start=1):
            if cond.kind == "motion":
                deg = Degradation(kind="motion", length=cond.length, theta=float(rng.uniform(-math.pi, math.pi)))
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
        out[key] = MethodResult(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))
        print(f"[{method_name}] {key}: PSNR={out[key].psnr:.2f}, SSIM={out[key].ssim:.3f}", flush=True)
    return out


def predict_hr_with_cnn(
    hr: np.ndarray,
    deg: Degradation,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Run CNN (Bi-channel or Basic) on one HR image under degradation; return predicted HR (100,100,3) in [0,1]."""
    lr50 = degrade_hr_to_lr50(hr, deg)
    in_norm, up_norm, _, mean, std = preprocess_for_network(lr50, hr)
    in_norm = in_norm.unsqueeze(0).to(device)
    up_norm = up_norm.unsqueeze(0).to(device)
    mean = mean.unsqueeze(0).to(device)
    std = std.unsqueeze(0).to(device)
    with torch.no_grad():
        if isinstance(model, BiChannelCNN):
            out_norm = model(in_norm, up_norm)
        else:
            out_norm = model(in_norm)
    out = inverse_normalize(out_norm, mean, std)
    out = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def eval_cnn_on_conditions(
    hr100_test: Sequence[np.ndarray],
    method_name: str,
    model: nn.Module,
    device: torch.device,
    gaussian_conds: Sequence[Degradation],
    motion_conds: Sequence[Degradation],
    seed: int = 42,
) -> Tuple[Dict[str, MethodResult], Dict[str, MethodResult]]:
    """Evaluate a trained CNN (Bi-channel or Basic) under Gaussian and motion conditions; returns (gaussian_results, motion_results)."""
    rng = np.random.RandomState(seed)
    model.eval()
    g_out: Dict[str, MethodResult] = {}
    for cond in gaussian_conds:
        psnrs, ssims = [], []
        for hr in hr100_test:
            pred = predict_hr_with_cnn(hr, cond, model, device)
            psnrs.append(compute_psnr(hr, pred))
            ssims.append(compute_ssim(hr, pred))
        key = f"sigma={cond.sigma:g}"
        g_out[key] = MethodResult(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))
        print(f"[{method_name}] {key}: PSNR={g_out[key].psnr:.2f}, SSIM={g_out[key].ssim:.3f}")
    m_out: Dict[str, MethodResult] = {}
    for cond in motion_conds:
        psnrs, ssims = [], []
        for hr in hr100_test:
            deg = Degradation(kind="motion", length=cond.length, theta=float(rng.uniform(-math.pi, math.pi)))
            pred = predict_hr_with_cnn(hr, deg, model, device)
            psnrs.append(compute_psnr(hr, pred))
            ssims.append(compute_ssim(hr, pred))
        key = f"l={cond.length:g}"
        m_out[key] = MethodResult(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))
        print(f"[{method_name}] {key}: PSNR={m_out[key].psnr:.2f}, SSIM={m_out[key].ssim:.3f}")
    return g_out, m_out


def print_table_like_paper(
    gaussian_results: Dict[str, Dict[str, MethodResult]],
    motion_results: Dict[str, Dict[str, MethodResult]],
    methods: Sequence[str],
) -> None:
    """Print Table-2-style PSNR/SSIM for baseline methods (e.g. SR1(SC1), SR2(SC2), SFH)."""
    def fmt(mr: MethodResult) -> str:
        return f"{mr.psnr:.2f} / {mr.ssim:.2f}"

    print("\n(a) Quantitative comparison under Gaussian blur (CelebA)")
    print("PSNR/SSIM\t" + "\t".join(methods))
    for sigma_key in ["sigma=1", "sigma=3", "sigma=5"]:
        row = [sigma_key] + [fmt(gaussian_results[m][sigma_key]) for m in methods]
        print("\t".join(row))
    print("\n(b) Quantitative comparison under motion blur (CelebA)")
    print("PSNR/SSIM\t" + "\t".join(methods))
    for l_key in ["l=2", "l=6", "l=9"]:
        row = [l_key] + [fmt(motion_results[m][l_key]) for m in methods]
        print("\t".join(row))


############################################
# Training & evaluation utilities
############################################


def init_weights_gaussian(m: nn.Module):
    """
    Initialize conv and linear layers with N(0, 0.001) for weights
    and 0 for biases, as specified in the paper.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        in_norm = batch["in_norm"].to(device)
        up_norm = batch["up_norm"].to(device)
        target_norm = batch["target_norm"].to(device)

        optimizer.zero_grad()
        if isinstance(model, BiChannelCNN):
            out = model(in_norm, up_norm)
        else:
            out = model(in_norm)
        loss = mse(out, target_norm)
        loss.backward()
        optimizer.step()

        bs = in_norm.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            in_norm = batch["in_norm"].to(device)
            up_norm = batch["up_norm"].to(device)
            target_norm = batch["target_norm"].to(device)

            if isinstance(model, BiChannelCNN):
                out = model(in_norm, up_norm)
            else:
                out = model(in_norm)

            loss = mse(out, target_norm)
            bs = in_norm.size(0)
            total_loss += loss.item() * bs
            n += bs
    return total_loss / max(1, n)


def inverse_normalize(
    out_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Implements I_out = arctanh(tilde_out) * S_in + M_in
    with arctanh computed via log formula.
    """
    t = torch.clamp(out_norm, -0.999, 0.999)
    atanh = 0.5 * torch.log((1 + t) / (1 - t))

    mean = mean.view(-1, 3, 1, 1)
    std = std.view(-1, 3, 1, 1)
    x = atanh * std + mean
    return x


def load_model_state(path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint safely on CPU and return model state_dict."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def evaluate_blur(
    model: nn.Module,
    blur_type: str,
    test_idx: list,
    root: str,
    device: torch.device,
    sigma_values: Optional[list] = None,
    length_values: Optional[list] = None,
    batch_size: int = 64,
):
    results = []

    if blur_type == "gaussian":
        assert sigma_values is not None
        for sigma in sigma_values:
            ds = CelebAFaceHallucination(
                root=root,
                split_indices=test_idx,
                blur_type="gaussian",
                train_random=False,
                sigma=sigma,
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            psnr_sum, ssim_sum, n = 0.0, 0.0, 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"σ={sigma}"):
                    in_norm = batch["in_norm"].to(device)
                    up_norm = batch["up_norm"].to(device)
                    hr = batch["hr"].to(device)
                    mean = batch["mean"].to(device)
                    std = batch["std"].to(device)

                    if isinstance(model, BiChannelCNN):
                        out_norm = model(in_norm, up_norm)
                    else:
                        out_norm = model(in_norm)

                    out = inverse_normalize(out_norm, mean, std)
                    out = torch.clamp(out, 0.0, 1.0)
                    hr = torch.clamp(hr, 0.0, 1.0)

                    out_np = out.cpu().numpy().transpose(0, 2, 3, 1)
                    hr_np = hr.cpu().numpy().transpose(0, 2, 3, 1)

                    for i in range(out_np.shape[0]):
                        psnr_sum += compute_psnr(hr_np[i], out_np[i])
                        ssim_sum += compute_ssim(hr_np[i], out_np[i])
                        n += 1

            results.append(
                {
                    "type": "gaussian",
                    "sigma": sigma,
                    "psnr": psnr_sum / max(1, n),
                    "ssim": ssim_sum / max(1, n),
                }
            )

    elif blur_type == "motion":
        assert length_values is not None
        for length in length_values:
            ds = CelebAFaceHallucination(
                root=root,
                split_indices=test_idx,
                blur_type="motion",
                train_random=False,
                length=length,
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            psnr_sum, ssim_sum, n = 0.0, 0.0, 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"l={length}"):
                    in_norm = batch["in_norm"].to(device)
                    up_norm = batch["up_norm"].to(device)
                    hr = batch["hr"].to(device)
                    mean = batch["mean"].to(device)
                    std = batch["std"].to(device)

                    if isinstance(model, BiChannelCNN):
                        out_norm = model(in_norm, up_norm)
                    else:
                        out_norm = model(in_norm)

                    out = inverse_normalize(out_norm, mean, std)
                    out = torch.clamp(out, 0.0, 1.0)
                    hr = torch.clamp(hr, 0.0, 1.0)

                    out_np = out.cpu().numpy().transpose(0, 2, 3, 1)
                    hr_np = hr.cpu().numpy().transpose(0, 2, 3, 1)

                    for i in range(out_np.shape[0]):
                        psnr_sum += compute_psnr(hr_np[i], out_np[i])
                        ssim_sum += compute_ssim(hr_np[i], out_np[i])
                        n += 1

            results.append(
                {
                    "type": "motion",
                    "length": length,
                    "psnr": psnr_sum / max(1, n),
                    "ssim": ssim_sum / max(1, n),
                }
            )

    return results


############################################
# Main entrypoints (train / eval)
############################################


def main():
    parser = argparse.ArgumentParser(
        description="Bi-channel CNN face hallucination on CelebA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = subparsers.add_parser("train", help="Train model")
    train_p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder where CelebA will be downloaded",
    )
    train_p.add_argument(
        "--model",
        type=str,
        default="bichannel",
        choices=["bichannel", "basic"],
    )
    train_p.add_argument(
        "--blur_type",
        type=str,
        default="mixed",
        choices=["gaussian", "motion", "mixed", "none"],
    )
    train_p.add_argument("--batch_size", type=int, default=200)
    # The paper mentions training with mini-batches of 200 for ~5000 cycles.
    train_p.add_argument("--epochs", type=int, default=5000)
    # Initial learning rate 0.0001 for all layers.
    train_p.add_argument("--lr", type=float, default=1e-5)
    train_p.add_argument("--min_lr", type=float, default=1e-6)
    train_p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_p.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
    )
    train_p.add_argument("--seed", type=int, default=42)

    # Eval Gaussian
    eval_g = subparsers.add_parser(
        "eval_gaussian",
        help="Evaluate trained model under Gaussian blur (σ=1,3,5)",
    )
    eval_g.add_argument("--data_root", type=str, required=True)
    eval_g.add_argument(
        "--model",
        type=str,
        default="bichannel",
        choices=["bichannel", "basic"],
    )
    eval_g.add_argument("--checkpoint", type=str, required=True)
    eval_g.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_g.add_argument("--metric_color_space", type=str, default="rgb", choices=["rgb", "y"])
    eval_g.add_argument("--batch_size", type=int, default=64)
    eval_g.add_argument("--seed", type=int, default=42)

    # Eval Motion
    eval_m = subparsers.add_parser(
        "eval_motion",
        help="Evaluate trained model under motion blur (l=2,6,9)",
    )
    eval_m.add_argument("--data_root", type=str, required=True)
    eval_m.add_argument(
        "--model",
        type=str,
        default="bichannel",
        choices=["bichannel", "basic"],
    )
    eval_m.add_argument("--checkpoint", type=str, required=True)
    eval_m.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_m.add_argument("--metric_color_space", type=str, default="rgb", choices=["rgb", "y"])
    eval_m.add_argument("--batch_size", type=int, default=64)
    eval_m.add_argument("--seed", type=int, default=42)

    # Print Table 2 (all 6 methods) from AAAI 2015 paper
    subparsers.add_parser(
        "print_paper_table",
        help="Print Table 2 (Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN) from Learning Face Hallucination in the Wild (AAAI 2015) and exit.",
    )

    # Eval classical baselines SC1, SC2, SFH (train on CelebA, then eval under Gaussian/motion blur)
    eval_baselines_p = subparsers.add_parser(
        "eval_baselines",
        help="Train and evaluate SC1, SC2, SFH on CelebA; print Table-2-style PSNR/SSIM.",
    )
    eval_baselines_p.add_argument("--data_root", type=str, required=True, help="CelebA root.")
    eval_baselines_p.add_argument("--seed", type=int, default=42)
    eval_baselines_p.add_argument("--train_hr_limit", type=int, default=1200, help="Max HR images for training pool.")
    eval_baselines_p.add_argument("--test_hr_limit", type=int, default=300, help="Max test images.")
    eval_baselines_p.add_argument("--train_pairs", type=int, default=600, help="Degraded pairs for SC1/SC2 training.")
    eval_baselines_p.add_argument("--sr1_atoms", type=int, default=256)
    eval_baselines_p.add_argument("--sr2_anchors", type=int, default=8000)
    eval_baselines_p.add_argument("--sr2_knn", type=int, default=64)
    eval_baselines_p.add_argument(
        "--bichannel_checkpoint",
        type=str,
        default=None,
        help="Optional path to Bi-channel CNN checkpoint to include in evaluation (all 6 methods in one table).",
    )
    eval_baselines_p.add_argument(
        "--basic_checkpoint",
        type=str,
        default=None,
        help="Optional path to Basic CNN checkpoint to include in evaluation (all 6 methods in one table).",
    )
    eval_baselines_p.add_argument(
        "--val_hr_limit",
        type=int,
        default=200,
        help="Max validation images for baseline validation PSNR/SSIM.",
    )
    eval_baselines_p.add_argument("--metric_color_space", type=str, default="rgb", choices=["rgb", "y"])

    # Show a sample image before/after Gaussian and motion blur
    sample_p = subparsers.add_parser(
        "show_sample",
        help="Save a sample HR image and its Gaussian/motion blurred versions",
    )
    sample_p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder where CelebA will be downloaded",
    )
    sample_p.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save sample images",
    )
    sample_p.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Sigma to use for Gaussian blur sample",
    )
    sample_p.add_argument(
        "--length",
        type=float,
        default=6.0,
        help="Length to use for motion blur sample",
    )
    sample_p.add_argument(
        "--bichannel_checkpoint",
        type=str,
        default="checkpoints/bichannel_mixed.pt",
        help="Path to Bi-channel CNN checkpoint for sample reconstruction.",
    )
    sample_p.add_argument(
        "--basic_checkpoint",
        type=str,
        default="checkpoints/basic_mixed.pt",
        help="Path to Basic CNN checkpoint for sample reconstruction.",
    )
    sample_p.add_argument(
        "--sample_train_hr_limit",
        type=int,
        default=300,
        help="HR training pool size for fitting SC1/SC2 used in show_sample.",
    )
    sample_p.add_argument(
        "--sample_train_pairs",
        type=int,
        default=200,
        help="Number of degraded pairs for fitting SC1/SC2 used in show_sample.",
    )
    sample_p.add_argument("--sr1_atoms", type=int, default=128, help="SC1 atoms for show_sample fitting.")
    sample_p.add_argument("--sr2_anchors", type=int, default=3000, help="SC2 anchors for show_sample fitting.")
    sample_p.add_argument("--sr2_knn", type=int, default=32, help="SC2 kNN for show_sample fitting.")
    sample_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    global METRIC_COLOR_SPACE
    METRIC_COLOR_SPACE = getattr(args, "metric_color_space", "rgb")
    print(f"Metric color space: {METRIC_COLOR_SPACE}")

    if args.command == "train":
        set_global_seed(args.seed)
        os.makedirs(args.save_dir, exist_ok=True)
        train_idx, val_idx, _ = load_celeba_splits(args.data_root, seed=args.seed)

        train_ds = CelebAFaceHallucination(
            root=args.data_root,
            split_indices=train_idx,
            blur_type=args.blur_type,
            train_random=True,
        )
        val_ds = CelebAFaceHallucination(
            root=args.data_root,
            split_indices=val_idx,
            blur_type=args.blur_type,
            train_random=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        if args.model == "bichannel":
            model = BiChannelCNN()
        else:
            model = BasicCNN()

        # Parameter initialization: N(0, 0.001) for weights, 0 for biases.
        model.apply(init_weights_gaussian)

        device = torch.device(args.device)
        model.to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
        )

        ckpt_path = os.path.join(
            args.save_dir,
            f"{args.model}_{args.blur_type}.pt",
        )

        # Initialize training state
        best_val = math.inf
        epochs_no_improve = 0
        patience = 10  # number of epochs without improvement before lr decay
        start_epoch = 1

        # If a checkpoint already exists, resume from it
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = max(args.min_lr, param_group["lr"])
            best_val = ckpt.get("best_val", ckpt.get("val_loss", math.inf))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(
                f"Resuming from checkpoint {ckpt_path} at epoch {start_epoch - 1} "
                f"with best_val={best_val:.6f}, lr={optimizer.param_groups[0]['lr']:.6e}"
            )

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )
            val_loss = validate(model, val_loader, device)
            print(f"  Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "epochs_no_improve": epochs_no_improve,
                    },
                    ckpt_path,
                )
                print(f"  Saved checkpoint to {ckpt_path}")
            else:
                epochs_no_improve += 1

            # When validation error stops decreasing, reduce lr by factor 10.
            if epochs_no_improve >= patience:
                old_lr = optimizer.param_groups[0]["lr"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = max(args.min_lr, param_group["lr"] * 0.1)
                print(
                    f"  Validation has not improved for {patience} epochs. "
                    f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.6e}"
                )
                if optimizer.param_groups[0]["lr"] >= old_lr:
                    print(f"  Learning rate already at minimum ({args.min_lr:.6e}).")
                epochs_no_improve = 0

    elif args.command in ("eval_gaussian", "eval_motion"):
        set_global_seed(args.seed)
        _, _, test_idx = load_celeba_splits(args.data_root, seed=args.seed)

        if args.model == "bichannel":
            model = BiChannelCNN()
        else:
            model = BasicCNN()

        device = torch.device(args.device)
        model.load_state_dict(load_model_state(args.checkpoint))
        model.to(device)

        if args.command == "eval_gaussian":
            sigmas = [1.0, 3.0, 5.0]
            res = evaluate_blur(
                model,
                blur_type="gaussian",
                test_idx=test_idx,
                root=args.data_root,
                device=device,
                sigma_values=sigmas,
                batch_size=args.batch_size,
            )
            print("Gaussian blur results (CelebA test set):")
            print("σ\tPSNR\tSSIM")
            for r in res:
                print(f"{r['sigma']}\t{r['psnr']:.2f}\t{r['ssim']:.4f}")
        else:
            lengths = [2.0, 6.0, 9.0]
            res = evaluate_blur(
                model,
                blur_type="motion",
                test_idx=test_idx,
                root=args.data_root,
                device=device,
                length_values=lengths,
                batch_size=args.batch_size,
            )
            print("Motion blur results (CelebA test set):")
            print("l\tPSNR\tSSIM")
            for r in res:
                print(f"{r['length']}\t{r['psnr']:.2f}\t{r['ssim']:.4f}")
    elif args.command == "print_paper_table":
        print_paper_table_2_all_six()
    elif args.command == "eval_baselines":
        # Pipeline: train (SC1, SC2; Bicubic has no training; SFH uses trained SC1) -> validate (all 4 baselines on val set) -> evaluate (all 4 or all 6 if CNN checkpoints provided)
        set_global_seed(args.seed)
        train_idx, val_idx, test_idx = load_celeba_splits(args.data_root, seed=args.seed)
        hr_train = get_hr_arrays_from_celeba(args.data_root, train_idx, limit=args.train_hr_limit)
        hr_val = get_hr_arrays_from_celeba(args.data_root, val_idx, limit=args.val_hr_limit)
        hr_test = get_hr_arrays_from_celeba(args.data_root, test_idx, limit=args.test_hr_limit)
        if len(hr_train) < 50 or len(hr_test) < 20:
            raise RuntimeError("Not enough CelebA images; check data_root and limits.")
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

        # ---------- Training (SC1, SC2; Bicubic and SFH do not have separate training) ----------
        lr_train, hr_train_pairs = sample_training_pairs(hr_train, args.train_pairs, seed=args.seed)
        sr1 = SR1_SC1_Yang(n_atoms=args.sr1_atoms)
        sr2 = SR2_SC2_KimKwon(n_anchors=args.sr2_anchors, k_nn=args.sr2_knn)
        print("Training SC1 (Yang et al.)...")
        sr1.fit(lr_train, hr_train_pairs)
        print("Training SC2 (Kim & Kwon)...")
        sr2.fit(lr_train, hr_train_pairs)
        sfh = SFH_YangProxy(base_sr=sr1, use_gpu=torch.cuda.is_available())

        # ---------- Validation (all 4 baselines on validation set) ----------
        if len(hr_val) >= 10:
            print("\n--- Validation (Bicubic, SC1, SC2, SFH) on validation set ---")
            _ = eval_method_on_conditions(hr_val, "Bicubic", bicubic_predict, gaussian_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "Bicubic", bicubic_predict, motion_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SR1(SC1)", sr1.predict, gaussian_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SR1(SC1)", sr1.predict, motion_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SR2(SC2)", sr2.predict, gaussian_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SR2(SC2)", sr2.predict, motion_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SFH", sfh.predict, gaussian_conds, seed=args.seed)
            _ = eval_method_on_conditions(hr_val, "SFH", sfh.predict, motion_conds, seed=args.seed)

        # ---------- Test evaluation (4 baselines + optional 2 CNNs = all 6) ----------
        methods = ["Bicubic", "SR1(SC1)", "SR2(SC2)", "SFH"]
        gaussian_results = {}
        motion_results = {}
        print("\n--- Test evaluation ---")
        print("\nEvaluating Bicubic...")
        gaussian_results["Bicubic"] = eval_method_on_conditions(
            hr_test, "Bicubic", bicubic_predict, gaussian_conds, seed=args.seed
        )
        motion_results["Bicubic"] = eval_method_on_conditions(
            hr_test, "Bicubic", bicubic_predict, motion_conds, seed=args.seed
        )
        print("\nEvaluating SR1(SC1)...")
        gaussian_results["SR1(SC1)"] = eval_method_on_conditions(
            hr_test, "SR1(SC1)", sr1.predict, gaussian_conds, seed=args.seed
        )
        motion_results["SR1(SC1)"] = eval_method_on_conditions(
            hr_test, "SR1(SC1)", sr1.predict, motion_conds, seed=args.seed
        )
        print("\nEvaluating SR2(SC2)...")
        gaussian_results["SR2(SC2)"] = eval_method_on_conditions(
            hr_test, "SR2(SC2)", sr2.predict, gaussian_conds, seed=args.seed
        )
        motion_results["SR2(SC2)"] = eval_method_on_conditions(
            hr_test, "SR2(SC2)", sr2.predict, motion_conds, seed=args.seed
        )
        print("\nEvaluating SFH...")
        gaussian_results["SFH"] = eval_method_on_conditions(
            hr_test, "SFH", sfh.predict, gaussian_conds, seed=args.seed
        )
        motion_results["SFH"] = eval_method_on_conditions(
            hr_test, "SFH", sfh.predict, motion_conds, seed=args.seed
        )

        if args.bichannel_checkpoint and os.path.isfile(args.bichannel_checkpoint):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bichannel = BiChannelCNN()
            bichannel.load_state_dict(load_model_state(args.bichannel_checkpoint))
            bichannel.to(device)
            print("\nEvaluating Bi-channel CNN...")
            g_bc, m_bc = eval_cnn_on_conditions(
                hr_test, "Bi-channel CNN", bichannel, device, gaussian_conds, motion_conds, seed=args.seed
            )
            gaussian_results["Bi-channel CNN"] = g_bc
            motion_results["Bi-channel CNN"] = m_bc
            methods = methods + ["Bi-channel CNN"]
        if args.basic_checkpoint and os.path.isfile(args.basic_checkpoint):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            basic = BasicCNN()
            basic.load_state_dict(load_model_state(args.basic_checkpoint))
            basic.to(device)
            print("\nEvaluating Basic CNN...")
            g_b, m_b = eval_cnn_on_conditions(
                hr_test, "Basic CNN", basic, device, gaussian_conds, motion_conds, seed=args.seed
            )
            gaussian_results["Basic CNN"] = g_b
            motion_results["Basic CNN"] = m_b
            methods = methods + ["Basic CNN"]

        print_table_like_paper(gaussian_results, motion_results, methods)
        print("\n--- Paper Table 2 (reference, all 6 methods) ---")
        print_paper_table_2_all_six()
    elif args.command == "show_sample":
        set_global_seed(args.seed)
        # Load CelebA and take a single random image from the training split
        train_idx, _, _ = load_celeba_splits(args.data_root, seed=args.seed)
        base = CelebA(
            root=args.data_root,
            split="all",
            target_type="attr",
            download=True,
            transform=None,
        )
        idx = train_idx[0]
        img, _ = base[idx]
        # Apply same HR preprocessing as training (center crop + resize to 100x100)
        hr_transform = transforms.Compose(
            [
                transforms.CenterCrop(160),
                transforms.Resize((100, 100), interpolation=Image.BICUBIC),
            ]
        )
        img_hr = hr_transform(img)
        hr_np = np.array(img_hr).astype(np.float32) / 255.0

        # Generate Gaussian-blurred and motion-blurred versions (no randomness)
        lr_gauss, params_g = degrade_image(
            hr_np,
            blur_type="gaussian",
            sigma=args.sigma,
            length=None,
            train_random=False,
        )
        lr_motion, params_m = degrade_image(
            hr_np,
            blur_type="motion",
            sigma=None,
            length=args.length,
            train_random=False,
        )

        lr100_gauss = lr50_to_input100(lr_gauss)
        lr100_motion = lr50_to_input100(lr_motion)

        # Fit classical methods quickly for sample visualization.
        print("Preparing SC1/SC2/SFH sample models...")
        hr_train_pool = get_hr_arrays_from_celeba(args.data_root, train_idx, limit=args.sample_train_hr_limit)
        if len(hr_train_pool) < 20:
            raise RuntimeError("Not enough training images for show_sample classical reconstructions.")
        lr_train_pairs, hr_train_pairs = sample_training_pairs(hr_train_pool, args.sample_train_pairs, seed=args.seed)
        sr1 = SR1_SC1_Yang(n_atoms=args.sr1_atoms)
        sr2 = SR2_SC2_KimKwon(n_anchors=args.sr2_anchors, k_nn=args.sr2_knn)
        sr1.fit(lr_train_pairs, hr_train_pairs)
        sr2.fit(lr_train_pairs, hr_train_pairs)
        sfh = SFH_YangProxy(base_sr=sr1, use_gpu=torch.cuda.is_available())

        # Optional CNN sample predictions if checkpoints are available.
        bichannel = None
        basic = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.bichannel_checkpoint and os.path.isfile(args.bichannel_checkpoint):
            bichannel = BiChannelCNN()
            bichannel.load_state_dict(load_model_state(args.bichannel_checkpoint))
            bichannel.to(device)
            bichannel.eval()
        else:
            print(f"Bi-channel checkpoint not found, skipping: {args.bichannel_checkpoint}")
        if args.basic_checkpoint and os.path.isfile(args.basic_checkpoint):
            basic = BasicCNN()
            basic.load_state_dict(load_model_state(args.basic_checkpoint))
            basic.to(device)
            basic.eval()
        else:
            print(f"Basic checkpoint not found, skipping: {args.basic_checkpoint}")

        deg_g = Degradation(kind="gaussian", sigma=float(params_g.get("sigma", args.sigma)))
        deg_m = Degradation(
            kind="motion",
            length=float(params_m.get("length", args.length)),
            theta=float(params_m.get("theta", 0.0)),
        )

        preds = {
            "gaussian": {
                "bicubic": bicubic_predict(lr100_gauss),
                "sc1": sr1.predict(lr100_gauss),
                "sc2": sr2.predict(lr100_gauss),
                "sfh": sfh.predict(lr100_gauss),
            },
            "motion": {
                "bicubic": bicubic_predict(lr100_motion),
                "sc1": sr1.predict(lr100_motion),
                "sc2": sr2.predict(lr100_motion),
                "sfh": sfh.predict(lr100_motion),
            },
        }
        if bichannel is not None:
            preds["gaussian"]["bichannel_cnn"] = predict_hr_with_cnn(hr_np, deg_g, bichannel, device)
            preds["motion"]["bichannel_cnn"] = predict_hr_with_cnn(hr_np, deg_m, bichannel, device)
        if basic is not None:
            preds["gaussian"]["basic_cnn"] = predict_hr_with_cnn(hr_np, deg_g, basic, device)
            preds["motion"]["basic_cnn"] = predict_hr_with_cnn(hr_np, deg_m, basic, device)

        os.makedirs(args.output_dir, exist_ok=True)
        # Save HR (100x100) and LR (resized back to 100x100 for easier visual comparison)
        hr_save = (hr_np * 255.0).clip(0, 255).astype(np.uint8)
        g_save = cv2.resize(
            lr_gauss,
            (100, 100),
            interpolation=cv2.INTER_CUBIC,
        )
        g_save = (g_save * 255.0).clip(0, 255).astype(np.uint8)
        m_save = cv2.resize(
            lr_motion,
            (100, 100),
            interpolation=cv2.INTER_CUBIC,
        )
        m_save = (m_save * 255.0).clip(0, 255).astype(np.uint8)

        hr_path = os.path.join(args.output_dir, "sample_hr.png")
        g_path = os.path.join(
            args.output_dir,
            f"sample_gaussian_sigma{args.sigma}.png",
        )
        m_path = os.path.join(
            args.output_dir,
            f"sample_motion_l{args.length}.png",
        )

        Image.fromarray(hr_save).save(hr_path)
        Image.fromarray(g_save).save(g_path)
        Image.fromarray(m_save).save(m_path)

        for blur_name, blur_preds in preds.items():
            for method_name, pred in blur_preds.items():
                out = (np.clip(pred, 0.0, 1.0) * 255.0).astype(np.uint8)
                out_path = os.path.join(args.output_dir, f"sample_{blur_name}_{method_name}.png")
                Image.fromarray(out).save(out_path)
                print(f"Saved sample {blur_name} {method_name} output to {out_path}")

        print(f"Saved sample HR image to {hr_path}")
        print(f"Saved sample Gaussian-blurred image to {g_path}")
        print(f"Saved sample motion-blurred image to {m_path}")


if __name__ == "__main__":
    main()

