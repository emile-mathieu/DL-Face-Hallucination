import os, torch, random, cv2, math, numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple

#Definitions: 
#IH: high-resolution ground truth (100×100)
#IL: degraded version (blur + downsample)
#Iin: resized LR input (48×48) i.e. resized from IL 
#Then perform normalisation on Iin and IH so that after Iin passes through the network to come out with Iout: Iout and IH can be compared on the same scale 

#removed the function compute_mean_std --> changed to compute by image, the function is called normalize_per_image
#Essentially, in each R/G/B channel of an image, do Z-normalisation then tanh 
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
    else: # Create a "Do Nothing" kernel and return the original image
        kernel[length // 2, length // 2] = 1.0
    
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

#Preprocessing done for test dataset with fixed blurs and specific downfactor  
class FixedTestFaceDataset(Dataset):
    def __init__(self, image_paths: List[str], blur_type: str = "gaussian",
                 gaussian_sigma: float = None, motion_length: int = None,
                 base_seed: int = 42):
        self.image_paths    = image_paths
        self.hr_size        = HR_SIZE
        self.fixed_lr_size  = TEST_FIXED_LR_SIZE
        self.nn_input_size  = TEST_NN_INPUT_SIZE
        self.blur_type      = blur_type
        self.gaussian_sigma = gaussian_sigma
        self.motion_length  = motion_length
        self.base_seed      = base_seed
        if blur_type == "gaussian" and gaussian_sigma is None:
            raise ValueError("gaussian_sigma required")
        if blur_type == "motion" and motion_length is None:
            raise ValueError("motion_length required")
    
    def __len__(self):
        return len(self.image_paths)

        #essentially will be resize to 100x100 first, then fixed preprocessing (fixed blur, then resize to 50x 50), then resize to 48x48
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
        
    #I need the _motion_blur_kernel to fix the length and theta for testing  
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

    #in the run_test_pipeline function, we will fix the sigma and length torun the test set 
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
                                    self._motion_blur_kernel(self.motion_length, theta))
        return cv2.resize(img_blur, self.fixed_lr_size,
                          interpolation=cv2.INTER_CUBIC).astype(np.float32)

#train dataset for classical methods, just random blur and fixed downfactor by 2, no normalisation, the HR is 100x100, LR is 50x50, we need to obtain a HR-LR pair also to eventually build our paired dictionary
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

    #this function no use? can delete? 
    def upscale_to_hr(self, lr_img):
        return cv2.resize(
            lr_img,
            self.hr_size,
            interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)

