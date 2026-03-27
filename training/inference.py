import torch
import os
import matplotlib.pyplot as plt
from models.model import BiChannelCNN


def denormalize(img):
    return (img + 1) / 2  # [-1,1] → [0,1]


def save_image(tensor, path):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    plt.imsave(path, img)


def load_model(model_path, device):
    model = BiChannelCNN().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def test_single_image(dataloader, device, model_path="results/best_model.pth", save_dir="results/images"):
    os.makedirs(save_dir, exist_ok=True)

    # LOAD TRAINED MODEL
    model = load_model(model_path, device)

    with torch.no_grad():
        Iin, IH = next(iter(dataloader))

        Iin = Iin.to(device).float()
        IH = IH.to(device).float()

        outputs = model(Iin)

        # take first image
        lr = denormalize(Iin[0])
        sr = denormalize(outputs[0])
        hr = denormalize(IH[0])

        save_image(lr, os.path.join(save_dir, "lr.png"))
        save_image(sr, os.path.join(save_dir, "sr.png"))
        save_image(hr, os.path.join(save_dir, "hr.png"))

        print(f"Saved images to {save_dir}")