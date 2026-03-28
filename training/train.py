import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.logger import save_metrics_to_csv

# PSNR function (safe)
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)  # prevent log(0)
    return 10 * torch.log10(1.0 / mse)


# Training function
def train(optimizer, criterion, model, dataloader, device, dataset, num_epochs=20):
    model.to(device)

    # keep track of best PSNR to save best model for inference
    best_psnr = 0.0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        ssim_metric.reset()

        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            optimizer.zero_grad()

            outputs = model(Iin)

            # debug (only first batch)
            if epoch == 0 and batch_idx == 0:
                print("Output shape:", outputs.shape)
                print("Target shape:", IH.shape)

            # loss (MSE)
            loss = criterion(outputs, IH)
            epoch_loss += loss.item()

            # denormalize then compare
            pred = dataset.denormalize(outputs)
            target = dataset.denormalize(IH)

            # PSNR
            batch_psnr = psnr(pred, target)
            epoch_psnr += batch_psnr.item()

            # SSIM
            batch_ssim = ssim_metric(pred, target)
            epoch_ssim += batch_ssim.item()

            # backward
            loss.backward()

            # gradient clipping (optional but safe)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        # print results
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB | "
            f"SSIM: {avg_ssim:.4f}"
        )

        # save to CSV
        save_metrics_to_csv(
            "results/train_metrics.csv",
            [epoch + 1, avg_loss, avg_psnr, avg_ssim],
            header=["Epoch", "Loss", "PSNR", "SSIM"]
        )

        # save best model based on lowest training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "loss": best_loss,
                "epoch": epoch + 1
            }, "results/best_model.pth")
            print(f"New best model saved with training loss: {best_loss:.4f}")
