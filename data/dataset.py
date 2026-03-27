import os, torch, random, cv2, numpy as np
from torch.utils.data import Dataset
from PIL import Image

# This is the dataset class that loads an Image.
# It generates a low-res version (LR) by downsampling and optionally blurring the original high-res image (HR).
class FaceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith(('.jpg', '.png'))
        ]
        
        # HR target size (paper uses 100x100)
        self.hr_size = (100, 100)
        
        # network input size
        self.lr_input_size = (48, 48)

    # These are the standard methods for a PyTorch Dataset. __getitem__ will be called by the DataLoader to get each sample.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # load HR image
        img = Image.open(path).convert("RGB")
        img = img.resize(self.hr_size, Image.BICUBIC)  # FIX: explicit interpolation

        IH = np.array(img)

        # -------- Step 1: create LR image --------
        IL = self.generate_low_res(IH)

        # -------- Step 2: preprocess --------
        Iin = self.preprocess(IL)

        # normalize HR as well
        IH = self.normalize(IH)

        # convert to float32 before tensor
        Iin = Iin.astype(np.float32)
        IH = IH.astype(np.float32)

        # convert to tensor (C, H, W)
        Iin = torch.from_numpy(Iin).permute(2, 0, 1)
        IH = torch.from_numpy(IH).permute(2, 0, 1)

        return Iin, IH

    # Create LR image (blur + downsample)
    def generate_low_res(self, img):
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
            # Motion blur (simple version)
            kernel_size = random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            small = cv2.filter2D(small, -1, kernel)

        return small

    # Preprocess (resize to 48x48)
    def preprocess(self, IL):
        IL = cv2.resize(IL, self.lr_input_size, interpolation=cv2.INTER_CUBIC)  # FIX
        IL = self.normalize(IL)
        return IL

    # Normalize to [-1, 1] (tanh output range is -1 to 1)
    def normalize(self, img):
        img = img / 127.5 - 1.0
        return img