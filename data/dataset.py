import os, torch, random, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#calculate mean and std of an image
def compute_mean_std(image_dir, image_size=(100, 100)):
    """
    Compute per-channel mean and std from the training images only.
    Images are resized to image_size and scaled to [0,1].

    Returns:
        mean: tuple of 3 floats
        std: tuple of 3 floats
    """
    image_paths = [
        os.path.join(image_dir, img)
        for img in os.listdir(image_dir)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(image_size, Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0  # [0,1]

        h, w, _ = img.shape
        total_pixels += h * w

        channel_sum += img.sum(axis=(0, 1))
        channel_sum_sq += (img ** 2).sum(axis=(0, 1))

    mean = channel_sum / total_pixels
    std = np.sqrt(channel_sum_sq / total_pixels - mean ** 2)

    # protect against division by zero
    std = np.clip(std, 1e-8, None)

    return tuple(mean.astype(np.float32)), tuple(std.astype(np.float32))

# This is the dataset class that loads an Image.
# It generates a low-res version (LR) by downsampling and optionally blurring the original high-res image (HR).
class FaceDataset(Dataset):
    def __init__(self, image_dir, mean, std):
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # HR target size (paper uses 100x100)
        self.hr_size = (100, 100)

        # network input size
        self.lr_input_size = (48, 48)

        # store mean/std for broadcasting
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        # torch versions for denormalize()
        self.mean_torch = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std_torch = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        self.eps = 1e-6

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # load HR image
        img = Image.open(path).convert("RGB")
        img = img.resize(self.hr_size, Image.BICUBIC)

        # convert to [0,1]
        IH = np.array(img).astype(np.float32) / 255.0

        # -------- Step 1: create LR image --------
        IL = self.generate_low_res(IH)

        # -------- Step 2: preprocess --------
        Iin = self.preprocess(IL)

        # normalize HR too
        IH = self.normalize(IH)

        # convert to tensor (C,H,W)
        Iin = torch.from_numpy(Iin).permute(2, 0, 1).float()
        IH = torch.from_numpy(IH).permute(2, 0, 1).float()

        return Iin, IH

    # Create LR image (blur + downsample)
    def generate_low_res(self, img):
        """
        img: numpy array in [0,1], shape (H,W,3)
        """
        h, w, _ = img.shape

        # random downsample factor (2 to 5)
        scale = random.randint(2, 5)

        # downsample (FIX: prevent 0 size + better interpolation)
        small = cv2.resize(
            img,
            (max(1, w // scale), max(1, h // scale)),
            interpolation=cv2.INTER_CUBIC
        )

        # Randomly apply either Gaussian blur or motion blur to simulate real-world degradation
        if random.random() > 0.5:
            # Gaussian blur
            ksize = random.choice([3, 5, 7])
            small = cv2.GaussianBlur(small, (ksize, ksize), 0)
        else:
            # Motion blur
            kernel_size = random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[(kernel_size - 1) // 2, :] = 1.0
            kernel /= kernel_size
            small = cv2.filter2D(small, -1, kernel)

        return small.astype(np.float32)

    # Preprocess (resize to 48x48)
    def preprocess(self, IL):
        IL = cv2.resize(
            IL,
            self.lr_input_size,
            interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

        IL = self.normalize(IL)
        return IL

    #normalise by z-normalisation then tanh , img: numpy array in [0,1], shape (H,W,3)
    def normalize(self, img):
        z = (img - self.mean) / self.std
        y = np.tanh(z)
        return y.astype(np.float32)

    #denormalize by inverse tanh (i.e. z=atanh(y)) then inverse-z-normalisation (i.e.x= z*std + mean). Input: torch tensor of shape (C,H,W) or (B,C,H,W). Output: torch tensor in [0,1]
    def denormalize(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError("denormalize expects a torch.Tensor")

        img = torch.clamp(img, -1.0 + self.eps, 1.0 - self.eps)

        mean = self.mean_torch.to(device=img.device, dtype=img.dtype)
        std = self.std_torch.to(device=img.device, dtype=img.dtype)

        z = torch.atanh(img)

        if img.ndim == 3:
            x = z * std + mean
        elif img.ndim == 4:
            x = z * std.unsqueeze(0) + mean.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported shape: {img.shape}")

        x = torch.clamp(x, 0.0, 1.0)
        return x
