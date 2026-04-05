import os
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import cv2
import pickle

# General functions

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_rgb_image(path, img_float01):
    ensure_dir(os.path.dirname(path))
    img = np.clip(img_float01, 0.0, 1.0)
    img_u8 = (img * 255.0).round().astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)



# Metrics

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()

def ssim(pred, target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return ssim_metric(pred,target).item()



# Checkpoints

def save_bichannel_checkpoint(model, mean, std, path):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
        },
        path,
    )
    print(f"Saved BiChannel checkpoint to: {path}")

def load_bichannel_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    print(f"Loaded BiChannel checkpoint from: {path}")
    return model, mean, std

def save_pickle_model(obj, path, label):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {label} model to: {path}")


def load_pickle_model(path, label):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded {label} model from: {path}")
    return obj

