import torch
import os
from models.model import BiChannelCNN
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import denormalize_per_image

#use this function for the save_sample_outputs
def save_image(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)

#Get IMAGES.png for each test criteria e.g. gaussian sigma = 1,3,5, and motion blur l=2,6,9), to generate a image from Iin, outputs and IH. This will generate when the run_test_pipeline is run 
def save_sample_outputs(model, dataset, device, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        Iin, IH, mean_b, std_b = next(iter(loader))
        Iin, IH = Iin.to(device).float(), IH.to(device).float()
        outputs, alpha = model(Iin)
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

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

#not sure if this is useful: 
def test_single_image(
    dataloader,
    device,
    dataset,
    model_path="results/best_model.pth",
    save_dir="results/images"
):
    os.makedirs(save_dir, exist_ok=True)

    model = load_model(model_path, device)

    with torch.no_grad():
        Iin, IH = next(iter(dataloader))

        Iin = Iin.to(device).float()
        IH = IH.to(device).float()

        outputs, _ = model(Iin)

        # take 1st image, denormalize already defined in dataset.py 
        lr = dataset.denormalize(Iin[0])
        sr = dataset.denormalize(outputs[0])
        hr = dataset.denormalize(IH[0])

        save_image(lr, os.path.join(save_dir, "lr.png"))
        save_image(sr, os.path.join(save_dir, "sr.png"))
        save_image(hr, os.path.join(save_dir, "hr.png"))

        print(f"Saved images to {save_dir}")


# ============================================================
# Checkpoint helpers
# ============================================================
def save_checkpoint(model, optimizer, scheduler, epoch: int,
                    val_loss: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss":        val_loss,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None,
                    scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


