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

def _to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [0,1] to Y channel with BT.601 coefficients."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    r = img[:, 0:1]
    g = img[:, 1:2]
    b = img[:, 2:3]
    return (0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0).clamp(0.0, 1.0)

def psnr(pred, target, channel: str = "y"):
    if channel.lower() == "y":
        pred = _to_y_channel(pred)
        target = _to_y_channel(target)
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()

def ssim(pred, target, channel: str = "y"):
    if channel.lower() == "y":
        pred = _to_y_channel(pred)
        target = _to_y_channel(target)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim_metric(pred, target).item()



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


def load_basic_weights_into_bichannel(bichannel_model, basic_checkpoint_path, device):
    ckpt = torch.load(basic_checkpoint_path, map_location=device)
    state = ckpt.get("model_state") if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    required = [
        "conv1.weight", "conv1.bias",
        "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias",
        "fc1.weight", "fc1.bias",
        "fc2.weight", "fc2.bias",
    ]
    if not all(k in state for k in required):
        raise KeyError("Basic checkpoint does not contain expected BasicCNN weights.")

    bichannel_model.conv1.weight.data.copy_(state["conv1.weight"])
    bichannel_model.conv1.bias.data.copy_(state["conv1.bias"])
    bichannel_model.conv2.weight.data.copy_(state["conv2.weight"])
    bichannel_model.conv2.bias.data.copy_(state["conv2.bias"])
    bichannel_model.conv3.weight.data.copy_(state["conv3.weight"])
    bichannel_model.conv3.bias.data.copy_(state["conv3.bias"])
    bichannel_model.fc1_1.weight.data.copy_(state["fc1.weight"])
    bichannel_model.fc1_1.bias.data.copy_(state["fc1.bias"])
    bichannel_model.fc2_1.weight.data.copy_(state["fc2.weight"])
    bichannel_model.fc2_1.bias.data.copy_(state["fc2.bias"])

    print(f"Loaded BasicCNN weights into BiChannel from: {basic_checkpoint_path}")
    return bichannel_model

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

