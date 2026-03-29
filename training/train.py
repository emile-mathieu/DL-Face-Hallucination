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
    """
    Uses dataset.denormalize() if available.
    Otherwise returns the batch unchanged.
    """
    if dataset is not None and hasattr(dataset, "denormalize"):
        return dataset.denormalize(batch)
    return batch


def evaluate_one_epoch(model, dataloader, criterion, device, dataset=None):
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (Iin, IH) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            outputs = model(Iin)
            loss = criterion(outputs, IH)

            pred = denormalize_batch(outputs, dataset)
            target = denormalize_batch(IH, dataset)

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
    train_metrics_csv_path="results/train_metrics.csv",
    val_metrics_csv_path="results/val_metrics.csv",
    best_model_path="checkpoints/best_bichannel.pth",
):

    ensure_dir(os.path.dirname(train_metrics_csv_path))
    ensure_dir(os.path.dirname(val_metrics_csv_path))
    ensure_dir(os.path.dirname(best_model_path))

    model.to(device)

    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_loader, "dataset", None)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        train_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        total_batches = 0

        for batch_idx, (Iin, IH) in enumerate(train_loader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

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

            pred = denormalize_batch(outputs, train_dataset)
            target = denormalize_batch(IH, train_dataset)

            batch_psnr = psnr(pred, target)
            batch_ssim = train_ssim_metric(pred, target)

            epoch_loss += loss.item()
            epoch_psnr += batch_psnr.item()
            epoch_ssim += batch_ssim.item()
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
            dataset=val_dataset,
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