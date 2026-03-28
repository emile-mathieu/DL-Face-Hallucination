import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.logger import save_metrics_to_csv


# -------------------------
# PSNR (safe)
# -------------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)

def evaluate(model, dataloader, device, dataset):
    model.to(device)
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_batches = 0

    ssim_metric.reset()

    with torch.no_grad():
        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            outputs = model(Iin)

            #compare the denormalized output with the denormalized IH
            pred = dataset.denormalize(outputs)
            target = dataset.denormalize(IH)

            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_metric(pred, target)

            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_batches += 1

            if batch_idx == 0:
                print(f"Input (LR): {Iin.shape}")
                print(f"Output (SR): {outputs.shape}")
                print(f"Target (HR): {IH.shape}")

    avg_psnr = total_psnr / total_batches
    avg_ssim = total_ssim / total_batches

    print(f"\nEvaluation Results → PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    save_metrics_to_csv(
        "results/eval_metrics.csv",
        [avg_psnr, avg_ssim],
        header=["PSNR", "SSIM"]
    )

    return avg_psnr, avg_ssim
