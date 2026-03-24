import os, torch, random, cv2, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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
        img = img.resize(self.hr_size)

        IH = np.array(img)

        # -------- Step 1: create LR image --------
        IL = self.generate_low_res(IH)

        # -------- Step 2: preprocess --------
        Iin = self.preprocess(IL)

        # normalize HR as well
        IH = self.normalize(IH)

        # convert to tensor (C, H, W)
        Iin = torch.tensor(Iin).permute(2, 0, 1).float()
        IH = torch.tensor(IH).permute(2, 0, 1).float()

        return Iin, IH

    # Create LR image (blur + downsample)
    def generate_low_res(self, img):
        h, w, _ = img.shape

        # random downsample factor (2 to 5)
        scale = random.randint(2, 5)

        # downsample
        small = cv2.resize(img, (w // scale, h // scale))

        # optional blur (Gaussian), if we need to put gaussian for each image, just remove the random condition
        if random.random() > 0.5:
            ksize = random.choice([3, 5, 7])
            small = cv2.GaussianBlur(small, (ksize, ksize), 0)

        return small


    # Preprocess (resize to 48x48)
    def preprocess(self, IL):
        IL = cv2.resize(IL, self.lr_input_size)
        IL = self.normalize(IL)
        return IL

   
    # Normalize to [-1, 1] (tanh output range is -1 to 1)
    def normalize(self, img):
        img = img / 127.5 - 1.0
        return img