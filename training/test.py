import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import csv
from torch.utils.data import DataLoader

from data.dataset import denormalize_per_image
from training.inference import save_sample_outputs, load_model
from utils.utils import psnr, ssim, save_rgb_image

# -------------------------
# Test on best model 
# -------------------------
def evaluate_one_test_setting(model, dataloader, device, save_cfg=None):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    total_loss = total_psnr = total_ssim = n = 0
    total_am = total_amin = total_amax = 0

    try:
        save_first_n = save_cfg["save_first_n"]
        images_dir = save_cfg["images_dir"]
        params = save_cfg["params"]
        params_val = save_cfg["params_val"]
    except:
        save_first_n = 0


    with torch.no_grad():
        for Iin, IH, mean_b, std_b in dataloader:
            Iin, IH = Iin.to(device).float(), IH.to(device).float()
            outputs, alpha = model(Iin)
            total_loss += criterion(outputs, IH).item()
            total_am    += alpha.mean().item()
            total_amin  += alpha.min().item()
            total_amax  += alpha.max().item()
            for b in range(outputs.shape[0]):
                pred = denormalize_per_image(outputs[b],  mean_b[b].numpy(), std_b[b].numpy())
                target = denormalize_per_image(IH[b],   mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += psnr(pred, target)
                total_ssim += ssim(pred, target)
                n += 1

                if n <= save_first_n:
                    bi_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_rgb_image(os.path.join(images_dir, f"{params}{params_val}_img{n-1:03d}_bichannel.png"), bi_np)
                    # Save Reference HR and LR for BiChannel part
                    hr_ref = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    save_rgb_image(os.path.join(images_dir, f"{params}{params_val}_img{n-1:03d}_hr.png"), hr_ref)

    nb = len(dataloader)
    print(f"  alpha mean={total_am/nb:.4f} min={total_amin/nb:.4f} max={total_amax/nb:.4f}")
    return total_loss / nb, total_psnr / n, total_ssim / n, n

