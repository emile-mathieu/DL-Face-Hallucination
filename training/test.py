import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import csv
from torch.utils.data import DataLoader

from data.dataset import denormalize_per_image, FixedTestFaceDataset
from training.train import compute_psnr, compute_ssim
from training.inference import save_sample_outputs, load_model

# -------------------------
# Test on best model 
# -------------------------
def evaluate_one_test_setting(model, dataloader, device, criterion):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    total_loss = total_psnr = total_ssim = n = 0
    total_am = total_amin = total_amax = 0
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
                total_psnr += compute_psnr(pred, target)
                total_ssim += compute_ssim(pred, target)
                n += 1
    nb = len(dataloader)
    print(f"  alpha mean={total_am/nb:.4f} min={total_amin/nb:.4f} max={total_amax/nb:.4f}")
    return total_loss / nb, total_psnr / n, total_ssim / n

def run_test_pipeline(test_files: List[str], best_model_path: Path, device):
    criterion = nn.MSELoss()
    model     = load_model(best_model_path, device)
    results   = []
    lkw = dict(batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    for sigma in [1, 3, 5]:
        ds      = FixedTestFaceDataset(test_files, blur_type="gaussian",
                                       gaussian_sigma=sigma)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Gaussian sigma={sigma} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"gaussian_sigma_{sigma}",
                         "loss": l, "psnr": p, "ssim": s})

    for length in [2, 6, 9]:
        ds      = FixedTestFaceDataset(test_files, blur_type="motion",
                                       motion_length=length, base_seed=SEED)
        l, p, s = evaluate_one_test_setting(
            model, DataLoader(ds, **lkw), device, criterion)
        print(f"Motion l={length} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}")
        results.append({"blur_type": f"motion_l_{length}",
                         "loss": l, "psnr": p, "ssim": s})

    csv_path = RESULTS_DIR / "test_pipeline_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["blur_type","loss","psnr","ssim"])
        w.writeheader(); w.writerows(results)
    print(f"Test metrics saved to {csv_path}")

    save_sample_outputs(model,
                        FixedTestFaceDataset(test_files, blur_type="gaussian",
                                             gaussian_sigma=3),
                        device, RESULTS_DIR / "sample_outputs")
    return results
