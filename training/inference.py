import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model import BiChannelCNN


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_image(tensor, path):
    ensure_dir(os.path.dirname(path))
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    plt.imsave(path, img)


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

        outputs = model(Iin)

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