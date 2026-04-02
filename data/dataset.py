import os, torch, random, cv2, math, numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#Definitions: 
#IH: high-resolution ground truth (100×100)
#IL: degraded version (blur + downsample)
#Iin: resized LR input (48×48) i.e. resized from IL 
#Then perform normalisation on Iin and IH so that after Iin passes through the network to come out with Iout: Iout and IH can be compared on the same scale 

#per-image normalisation: in each R/G/B channel of an image, do Z-normalisation then tanh
def normalize_per_image(img: np.ndarray, eps: float = 1e-8
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = img.mean(axis=(0, 1))
    std  = img.std(axis=(0, 1)).clip(eps)
    return np.tanh((img - mean) / std).astype(np.float32), mean, std

#per-image denormalisation:  undo the tanh, then undo the z-normalisation
def denormalize_per_image(img: torch.Tensor, mean: np.ndarray,
                           std: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    mt = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    st = torch.tensor(std,  dtype=img.dtype, device=img.device).view(3, 1, 1)
    return torch.clamp(torch.atanh(img.clamp(-1+eps, 1-eps)) * st + mt, 0.0, 1.0)

#motion blur kernel
def _motion_blur_kernel(length: int, theta: float) -> np.ndarray:
    length = max(2, int(length)) #it's a minimum 2x2 kernel for blur, because 1x1 kernel will produce the original pixel and won't smear the pixels
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

# This is the dataset class that loads an Image for TRAINING or VALIDATION (not for testing)
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
            # Gaussian blur
            sigma = random.uniform(0, 7)
        
            # If sigma is extremely close to 0, leave image unchanged
            blurred = (cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
                       if sigma > 1e-6 else img.copy())
        
        else:
            # Motion blur
            length = random.randint(0, 11)
            theta = random.uniform(-math.pi, math.pi)
            blurred = (cv2.filter2D(img, -1, _motion_blur_kernel(length, theta)) #apply the _motion_blur_kernel 
                       if length > 1 else img.copy())
                    
        # random downsample factor (2 to 5)
        scale = random.randint(2, 5)
        return cv2.resize(blurred, (max(1, w // scale), max(1, h // scale)),
                          interpolation=cv2.INTER_CUBIC).astype(np.float32)
    
    @staticmethod
    def motion_blur_kernel(length: int, theta: float) -> np.ndarray:
        return _motion_blur_kernel(length, theta)



