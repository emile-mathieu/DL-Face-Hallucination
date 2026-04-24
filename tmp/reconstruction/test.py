import os
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.logger import save_metrics_to_csv


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def denormalize_batch(batch, dataset):
    if dataset is not None and hasattr(dataset, "denormalize"):
        return dataset.denormalize(batch)
    return batch


def evaluate(
    model,
    dataloader,
    device,
    dataset=None,
    metrics_csv_path="results/eval_metrics.csv",
):
    model.to(device)
    model.eval()

    ensure_dir(os.path.dirname(metrics_csv_path))

    if dataset is None:
        dataset = getattr(dataloader, "dataset", None)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_loss_like = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            outputs = model(Iin)

            pred = denormalize_batch(outputs, dataset)
            target = denormalize_batch(IH, dataset)

            batch_mse = torch.mean((pred - target) ** 2).item()
            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_metric(pred, target)

            total_loss_like += batch_mse
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_batches += 1

            if batch_idx == 0:
                print(f"Input (LR): {Iin.shape}")
                print(f"Output (SR): {outputs.shape}")
                print(f"Target (HR): {IH.shape}")

    if total_batches == 0:
        raise ValueError("Evaluation dataloader is empty.")

    avg_mse = total_loss_like / total_batches
    avg_psnr = total_psnr / total_batches
    avg_ssim = total_ssim / total_batches

    print(f"\nEvaluation Results -> MSE: {avg_mse:.6f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    save_metrics_to_csv(
        metrics_csv_path,
        [avg_mse, avg_psnr, avg_ssim],
        header=["MSE", "PSNR", "SSIM"],
    )

    return avg_psnr, avg_ssim