import os
import torch
from utils.logger import save_metrics_to_csv
from torchmetrics.image import StructuralSimilarityIndexMeasure

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

#deleted denormalise_batch function and updated functions below to denormalise within 
def evaluate_one_epoch(model, dataloader, criterion, device, eps=1e-6):
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_loss = total_psnr = total_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        # Unpack all 4 returns from your FaceDataset
        for batch_idx, (Iin, IH, means, stds) in enumerate(dataloader):
            Iin, IH = Iin.to(device), IH.to(device)
            # Reshape means/stds for broadcasting: (B, 3) -> (B, 3, 1, 1)
            ms = means.view(-1, 3, 1, 1).to(device)
            ss = stds.view(-1, 3, 1, 1).to(device)

            outputs = model(Iin)
            loss = criterion(outputs, IH)

            # --- Individual Denormalization Logic ---
            # 1. Undo Tanh (clamping to avoid log/atanh errors at -1 and 1)
            # 2. Multiply by individual std, add individual mean
            pred = torch.atanh(torch.clamp(outputs, -1 + eps, 1 - eps)) * ss + ms
            target = torch.atanh(torch.clamp(IH, -1 + eps, 1 - eps)) * ss + ms
            
            # Clip to [0, 1] for valid PSNR/SSIM
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)

            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_metric(pred, target)
            
            total_loss += loss.item()
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_batches += 1

            if batch_idx == 0:
                print(
                    f"[VAL] Input: {Iin.shape} | Output: {outputs.shape} | Target: {IH.shape}"
                )

    if total_batches == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / total_batches
    avg_psnr = total_psnr / total_batches
    avg_ssim = total_ssim / total_batches

    return avg_loss, avg_psnr, avg_ssim

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=20,
    scheduler=None,
    eps=1e-6,
    train_metrics_csv_path="results/train_metrics.csv",
    val_metrics_csv_path="results/val_metrics.csv",
    best_model_path="checkpoints/best_bichannel.pth",
):

    ensure_dir(os.path.dirname(train_metrics_csv_path))
    ensure_dir(os.path.dirname(val_metrics_csv_path))
    ensure_dir(os.path.dirname(best_model_path))

    model.to(device)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        epoch_loss = epoch_psnr = epoch_ssim = 0.0
        total_batches = 0

        for batch_idx, (Iin, IH, means, stds) in enumerate(train_loader):
            Iin, IH = Iin.to(device), IH.to(device)
            ms = means.view(-1, 3, 1, 1).to(device)
            ss = stds.view(-1, 3, 1, 1).to(device)

            optimizer.zero_grad()
            outputs = model(Iin)

            if epoch == 0 and batch_idx == 0:
                print(
                    f"[TRAIN] Input: {Iin.shape} | Output: {outputs.shape} | Target: {IH.shape}"
                )
            
            loss = criterion(outputs, IH)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Inline Individual Denormalization ---
            with torch.no_grad(): # No need to track gradients for metrics
                pred = torch.atanh(torch.clamp(outputs, -1 + eps, 1 - eps)) * ss + ms
                target = torch.atanh(torch.clamp(IH, -1 + eps, 1 - eps)) * ss + ms
                pred, target = pred.clamp(0, 1), target.clamp(0, 1)

            epoch_loss += loss.item()
            epoch_psnr += psnr(pred, target).item()
            epoch_ssim += train_ssim_metric(pred, target).item()
            total_batches += 1

        if total_batches == 0:
            raise ValueError("Training loader is empty.")

        train_loss = epoch_loss / total_batches
        train_psnr = epoch_psnr / total_batches
        train_ssim = epoch_ssim / total_batches

        val_loss, val_psnr, val_ssim = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            eps=eps
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Train SSIM: {train_ssim:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        save_metrics_to_csv(
            train_metrics_csv_path,
            [epoch + 1, train_loss, train_psnr, train_ssim, current_lr],
            header=["Epoch", "Loss", "PSNR", "SSIM", "LR"],
        )

        save_metrics_to_csv(
            val_metrics_csv_path,
            [epoch + 1, val_loss, val_psnr, val_ssim, current_lr],
            header=["Epoch", "Loss", "PSNR", "SSIM", "LR"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_psnr": train_psnr,
                    "train_ssim": train_ssim,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "lr": current_lr,
                },
                best_model_path,
            )
            print(f"New best model saved to {best_model_path} with val loss: {best_val_loss:.4f}")
