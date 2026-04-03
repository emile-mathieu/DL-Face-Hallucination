import os, torch, random, cv2, math, numpy as np
from torch.utils.data import Dataset
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


# generic gaussian blur
def gaussian_blur(img):
    sigma = random.uniform(0, 7)
    if sigma > 1e-6:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return img.astype(np.float32)


# generic motion blur
def motion_blur(img):
    length = random.randint(0, 11)
    theta = random.uniform(-np.pi, np.pi)

    if length <= 1:
        return img.astype(np.float32)

    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0

    x0 = center - (length - 1) / 2.0 * np.cos(theta)
    y0 = center - (length - 1) / 2.0 * np.sin(theta)
    x1 = center + (length - 1) / 2.0 * np.cos(theta)
    y1 = center + (length - 1) / 2.0 * np.sin(theta)

    cv2.line(
        kernel,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        1,
        thickness=1
    )

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
        img = cv2.filter2D(img, -1, kernel)

    return img.astype(np.float32)


# This is the dataset class that loads an Image.
# It generates a low-res version (LR) by downsampling and optionally blurring the original high-res image (HR).
class FaceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # HR target size (paper uses 100x100)
        self.hr_size = (100, 100)

        # network input size
        self.lr_input_size = (48, 48)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # load HR image
        img = Image.open(path).convert("RGB")
        img = img.resize(self.hr_size, Image.BICUBIC)

        # convert to [0,1]
        IH = np.array(img).astype(np.float32) / 255.0

        # -------- Step 1: create LR image by Gaussian OR motion blurring, then downsample by factor of 2-5 (see details of function below)--------
        IL = self._generate_low_res(IH)

        # -------- Step 2: preprocess to resize to 48x48 then normalise --------
        Iin = cv2.resize(IL, self.lr_input_size,
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # normalize Iin and IH
        Iin_norm, mean, std = normalize_per_image(Iin)
        IH_norm = np.tanh((IH - mean) / std).astype(np.float32) #normalise IH using the mean and std from Iin as per the paper

        # convert to tensor (C,H,W)
        Iin = torch.from_numpy(Iin_norm).permute(2, 0, 1).float()
        IH = torch.from_numpy(IH_norm).permute(2, 0, 1).float()

        return Iin, IH, torch.tensor(mean, dtype=torch.float32), torch.tensor(std,  dtype=torch.float32)

    # Create LR image (blur then downsample)
    def _generate_low_res(self, img):
        """
        img: numpy array in [0,1], shape (H,W,3)
        """
        h, w, _ = img.shape

        # Randomly apply either Gaussian blur or motion blur to simulate real-world degradation
        if random.random() < 0.5:
            # Gaussian blur (applied to full-res img before downsampling)
            small = gaussian_blur(img)
        else:
            # Motion blur (applied to full-res img before downsampling)
            small = motion_blur(img)

        # random downsample factor (2 to 5)
        scale = random.randint(2, 5)

        # downsample the blurred result
        small = cv2.resize(
            small,
            (max(1, w // scale), max(1, h // scale)),
            interpolation=cv2.INTER_CUBIC
        )
        
        return small.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img    = Image.open(self.image_paths[idx]).convert("RGB")
        hr_img = img.resize(self.hr_size, Image.BICUBIC) #resize to 100x100 by bicubic method
        IH     = np.array(hr_img).astype(np.float32) / 255.0 #get the 100x100 image with pixels range (0,1) by dividing by 255 

        fixed_lr = self._fixed_test_preprocessing(img, idx) #see the function _fixed_test_preprocesing below: for single type of blur with single value on all the test images, then resize to 50x50
        #then resize to 48x48 to enter the bichannel CNN:
        Iin = cv2.resize(fixed_lr, self.nn_input_size,
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)

        Iin_norm, mean, std = normalize_per_image(Iin) #Z-normalise and tanh the Iin to get Iin_norm
        IH_norm = np.tanh((IH - mean) / std).astype(np.float32) #get IH Z-normalised and tanh done

        #return the Iin_norm, IH_norm, mean, std
        return (torch.from_numpy(Iin_norm).permute(2, 0, 1).float(),
                torch.from_numpy(IH_norm).permute(2, 0, 1).float(),
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







        x = torch.clamp(x, 0.0, 1.0)
        return x

class Classical_FaceDataset:
    def __init__(self, image_dir, hr_size=(100, 100)):
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.hr_size = hr_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        ih = self.generate_high_res(path)
        il = self.generate_low_res(ih)

        return il, ih

    def generate_high_res(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize(self.hr_size, Image.BICUBIC)
        ih = np.array(img).astype(np.float32) / 255.0
        return ih

    def generate_low_res(self, img):
        h, w, _ = img.shape

        degraded = img.copy()

        if random.random() < 0.5:
            degraded = gaussian_blur(degraded)
        else:
            degraded = motion_blur(degraded)

        # keep same order as FaceDataset for consistency with your current setup
        lr_w, lr_h = 50, 50

        degraded = cv2.resize(
            degraded,
            (lr_w, lr_h),
            interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

        return degraded.astype(np.float32)

    def upscale_to_hr(self, lr_img):
        return cv2.resize(
            lr_img,
            self.hr_size,
            interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)
