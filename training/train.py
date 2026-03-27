import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

# PSNR function
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(1.0 / mse)


def train(optimizer, criterion, model, dataloader, device, num_epochs=10):
    model.to(device)

    # SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            optimizer.zero_grad()

            # forward pass
            outputs = model(Iin)

            # DEBUG (only once)
            if epoch == 0 and batch_idx == 0:
                print("Output shape:", outputs.shape)
                print("Target shape:", IH.shape)

            # loss
            loss = criterion(outputs, IH)
            epoch_loss += loss.item()

            # convert from [-1,1] → [0,1] for metrics
            pred = (outputs + 1) / 2
            target = (IH + 1) / 2

            # PSNR
            batch_psnr = psnr(pred, target)
            epoch_psnr += batch_psnr.item()

            # SSIM
            batch_ssim = ssim_metric(pred, target)
            epoch_ssim += batch_ssim.item()

            # backward pass
            loss.backward()

            # gradient clipping (optional but safe)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB | "
            f"SSIM: {avg_ssim:.4f}"
        )