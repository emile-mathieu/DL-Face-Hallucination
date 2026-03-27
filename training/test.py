import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

# PSNR function
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(1.0 / mse)


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        # As a reminder = Iin: low-res input, IH: high-res target
        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            # forward pass
            outputs = model(Iin)

            # convert [-1,1] → [0,1]
            pred = (outputs + 1) / 2
            target = (IH + 1) / 2

            # compute metrics
            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_metric(pred, target)

            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_batches += 1

            # optional debug (only first batch)
            if batch_idx == 0:
                print(f"Input (LR): {Iin.shape}")
                print(f"Output (SR): {outputs.shape}")
                print(f"Target (HR): {IH.shape}")

    avg_psnr = total_psnr / total_batches
    avg_ssim = total_ssim / total_batches

    print(f"\nEvaluation Results → PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim