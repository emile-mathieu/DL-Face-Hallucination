"""
dataset.py
==========
PyTorch Dataset classes for training, validation, and fixed test evaluation.
Used by both BasicCNN and BiChannelCNN training pipelines.
"""

import math
import random
from typing import List

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import HR_SIZE, TRAIN_LR_INPUT_SIZE, TEST_FIXED_LR_SIZE, TEST_NN_INPUT_SIZE, SEED
from utils import celeba_crop, normalize_per_image, motion_blur_kernel


class TrainValFaceDataset(Dataset):
    """
    Random degradation dataset for training and validation.
    Each call to __getitem__ applies a fresh random blur + downsample,
    providing implicit data augmentation through degradation diversity.

    Returns: (iin_norm, ih_norm, mean, std)
      iin_norm : (3, 48, 48) tanh-normalised LR input
      ih_norm  : (3, 100, 100) tanh-normalised HR target
      mean/std : per-channel normalisation stats (for denormalisation)
    """
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

        return (
            torch.from_numpy(iin_norm).permute(2, 0, 1).float(),
            torch.from_numpy(ih_norm).permute(2, 0, 1).float(),
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std,  dtype=torch.float32),
        )

    def _generate_low_res(self, img: np.ndarray) -> np.ndarray:
        """Random Gaussian or motion blur + random 2-5× downscale."""
        h, w, _ = img.shape
        if random.random() < 0.5:
            sigma   = random.uniform(0, 7)
            blurred = (cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
                       if sigma > 1e-6 else img.copy())
        else:
            length  = random.randint(0, 11)
            theta   = random.uniform(-math.pi, math.pi)
            blurred = (cv2.filter2D(img, -1, motion_blur_kernel(length, theta))
                       if length > 1 else img.copy())
        scale = random.randint(2, 5)
        return cv2.resize(
            blurred, (max(1, w // scale), max(1, h // scale)),
            interpolation=cv2.INTER_CUBIC).astype(np.float32)


class FixedTestFaceDataset(Dataset):
    """
    Deterministic fixed-degradation dataset for test evaluation.
    Blur type and parameters are fixed; motion blur angle is seeded
    per-image for reproducibility.

    Returns: (iin_norm, ih_norm, mean, std)
      same layout as TrainValFaceDataset
    """
    def __init__(self, image_paths: List[str], blur_type: str,
                 gaussian_sigma: float = None, motion_length: int = None,
                 base_seed: int = SEED):
        if blur_type == "gaussian" and gaussian_sigma is None:
            raise ValueError("gaussian_sigma required for gaussian blur_type")
        if blur_type == "motion" and motion_length is None:
            raise ValueError("motion_length required for motion blur_type")
        self.image_paths    = image_paths
        self.hr_size        = HR_SIZE
        self.fixed_lr_size  = TEST_FIXED_LR_SIZE
        self.nn_input_size  = TEST_NN_INPUT_SIZE
        self.blur_type      = blur_type
        self.gaussian_sigma = gaussian_sigma
        self.motion_length  = motion_length
        self.base_seed      = base_seed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img    = celeba_crop(Image.open(self.image_paths[idx]).convert("RGB"))
        hr_img = img.resize(self.hr_size, Image.BICUBIC)
        ih     = np.array(hr_img).astype(np.float32) / 255.0

        fixed_lr = self._make_lr(img, idx)
        iin = cv2.resize(fixed_lr, self.nn_input_size,
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)

        iin_norm, mean, std = normalize_per_image(iin)
        ih_norm = np.tanh((ih - mean) / std.clip(1e-8)).astype(np.float32)

        return (
            torch.from_numpy(iin_norm).permute(2, 0, 1).float(),
            torch.from_numpy(ih_norm).permute(2, 0, 1).float(),
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std,  dtype=torch.float32),
        )

    def _make_lr(self, pil_img: Image.Image, idx: int) -> np.ndarray:
        img_100 = np.array(
            pil_img.resize(self.hr_size, Image.BICUBIC)
        ).astype(np.float32) / 255.0
        if self.blur_type == "gaussian":
            img_blur = cv2.GaussianBlur(img_100, (0, 0),
                                        sigmaX=self.gaussian_sigma,
                                        sigmaY=self.gaussian_sigma)
        else:
            theta    = random.Random(
                self.base_seed + idx).uniform(-math.pi, math.pi)
            img_blur = cv2.filter2D(
                img_100, -1, motion_blur_kernel(self.motion_length, theta))
        return cv2.resize(img_blur, self.fixed_lr_size,
                          interpolation=cv2.INTER_CUBIC).astype(np.float32)
