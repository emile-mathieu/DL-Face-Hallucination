import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model import BiChannelCNN
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import denormalize_per_image


def _unpack_model_output(model_out):
    if isinstance(model_out, tuple):
        return model_out[0], model_out[1]
    return model_out, None


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_image(tensor, path):
    ensure_dir(os.path.dirname(path))
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    plt.imsave(path, img)

#Get IMAGES.png for each test criteria e.g. gaussian sigma = 1,3,5, and motion blur l=2,6,9), to generate a image from Iin, outputs and IH. This will generate when the run_test_pipeline is run 
def save_sample_outputs(model, dataset, device, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        Iin, IH, mean_b, std_b = next(iter(loader))
        Iin, IH = Iin.to(device).float(), IH.to(device).float()
        outputs, _ = _unpack_model_output(model(Iin))
        save_image(denormalize_per_image(Iin[0], mean_b[0].numpy(), std_b[0].numpy()),
                   save_dir / "sample_lr.png")
        save_image(denormalize_per_image(outputs[0], mean_b[0].numpy(), std_b[0].numpy()),
                   save_dir / "sample_sr.png")
        save_image(denormalize_per_image(IH[0],  mean_b[0].numpy(), std_b[0].numpy()),
                   save_dir / "sample_hr.png")
    print(f"Sample images saved to {save_dir}")

#load model into testing for bichannel CNN
def load_model(model_path, device):
    model = BiChannelCNN().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

#not sure if this is useful: 
def test_single_image(
    dataloader,
    device,
    dataset=None,
    model_path="checkpoints/best_bichannel.pth",
    save_dir="results/images",
    image_index_in_batch=0,
):
    ensure_dir(save_dir)

    if dataset is None:
        dataset = getattr(dataloader, "dataset", None)

    model = load_model(model_path, device)

    with torch.no_grad():
        Iin, IH = next(iter(dataloader))

        Iin = Iin.to(device).float()
        IH = IH.to(device).float()

        outputs, _ = _unpack_model_output(model(Iin))

        idx = image_index_in_batch

        if idx >= Iin.shape[0]:
            raise IndexError(
                f"image_index_in_batch={idx} is out of range for batch size {Iin.shape[0]}"
            )

        if dataset is not None and hasattr(dataset, "denormalize"):
            lr = dataset.denormalize(Iin[idx])
            sr = dataset.denormalize(outputs[idx])
            hr = dataset.denormalize(IH[idx])
        else:
            lr = Iin[idx]
            sr = outputs[idx]
            hr = IH[idx]

        save_image(lr, os.path.join(save_dir, "lr.png"))
        save_image(sr, os.path.join(save_dir, "sr.png"))
        save_image(hr, os.path.join(save_dir, "hr.png"))

        print(f"Saved images to {save_dir}")
