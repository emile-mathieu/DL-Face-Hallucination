import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import save_metrics_to_csv
from utils.utils import (
    ensure_dir,
    psnr, 
    ssim,
    save_bichannel_checkpoint
)


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, t_max: int, min_lr: float, last_epoch: int = -1):
        self.warmup = max(0, int(warmup_epochs))
        self.t_max = max(1, int(t_max))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            alpha = (e + 1) / max(self.warmup, 1)
            return [self.min_lr + alpha * (base - self.min_lr) for base in self.base_lrs]

        progress = min((e - self.warmup) / self.t_max, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + cosine * (base - self.min_lr) for base in self.base_lrs]


def _unpack_model_output(model_out):
    if isinstance(model_out, tuple):
        return model_out[0], model_out[1]
    return model_out, None


def _save_epoch_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path):
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    tmp_path = ckpt_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _keep_last_n_checkpoints(checkpoint_dir, prefix, keep_last_n):
    files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".pth")]
    )
    for old_name in files[:-keep_last_n]:
        old_path = os.path.join(checkpoint_dir, old_name)
        try:
            os.remove(old_path)
        except OSError:
            pass

#deleted denormalise_batch function and updated functions below to denormalise within 
def evaluate_one_epoch(model, dataloader, criterion, device, eps=1e-6):
    model.eval()

    total_loss = total_psnr = total_ssim = 0.0
    total_batches = 0

    with torch.no_grad():
        # Unpack all 4 returns from your FaceDataset
        for batch_idx, (Iin, IH, means, stds) in enumerate(dataloader):
            Iin, IH = Iin.to(device), IH.to(device)
            # Reshape means/stds for broadcasting: (B, 3) -> (B, 3, 1, 1)
            ms = means.view(-1, 3, 1, 1).to(device)
            ss = stds.view(-1, 3, 1, 1).to(device)

            outputs, _ = _unpack_model_output(model(Iin))
            loss = criterion(outputs, IH)

            # --- Individual Denormalization Logic ---
            # 1. Undo Tanh (clamping to avoid log/atanh errors at -1 and 1)
            # 2. Multiply by individual std, add individual mean
            pred = torch.atanh(torch.clamp(outputs, -1 + eps, 1 - eps)) * ss + ms
            target = torch.atanh(torch.clamp(IH, -1 + eps, 1 - eps)) * ss + ms
            
            # Clip to [0, 1] for valid PSNR/SSIM
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)

            total_loss += loss.item()
            total_psnr += psnr(pred, target)
            total_ssim += ssim(pred, target)
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
        
        epoch_loss = epoch_psnr = epoch_ssim = 0.0
        total_batches = 0

        for batch_idx, (Iin, IH, means, stds) in enumerate(train_loader):
            Iin, IH = Iin.to(device), IH.to(device)
            ms = means.view(-1, 3, 1, 1).to(device)
            ss = stds.view(-1, 3, 1, 1).to(device)

            optimizer.zero_grad()
            outputs, _ = _unpack_model_output(model(Iin))

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
            epoch_psnr += psnr(pred, target)
            epoch_ssim += ssim(pred, target)
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
            scheduler.step()

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



def run_train_pipeline(
    model, 
    model_cfg, 
    train_loader, 
    val_loader, 
    device,
    model_ckpt_path
):

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
    )

    warmup_epochs = int(model_cfg.get("warmup_epochs", 10))
    t_max = int(model_cfg.get("lr_t_max", max(1, model_cfg["num_epochs"] - warmup_epochs)))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        t_max=t_max,
        min_lr=model_cfg["min_lr"],
    )

    ckpt_dir = os.path.join(os.path.dirname(model_ckpt_path), f"{model.__class__.__name__.lower()}_epochs")
    ensure_dir(ckpt_dir)

    # Train with per-epoch rolling checkpoints and best checkpoint tracking.
    best_val = float("inf")
    for epoch in range(model_cfg["num_epochs"]):
        model.train()
        epoch_loss = epoch_psnr = epoch_ssim = 0.0
        total_batches = 0

        for Iin, IH, means, stds in train_loader:
            Iin, IH = Iin.to(device), IH.to(device)
            ms = means.view(-1, 3, 1, 1).to(device)
            ss = stds.view(-1, 3, 1, 1).to(device)

            optimizer.zero_grad()
            outputs, _ = _unpack_model_output(model(Iin))
            loss = criterion(outputs, IH)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = torch.atanh(torch.clamp(outputs, -1 + 1e-6, 1 - 1e-6)) * ss + ms
                target = torch.atanh(torch.clamp(IH, -1 + 1e-6, 1 - 1e-6)) * ss + ms
                pred, target = pred.clamp(0, 1), target.clamp(0, 1)

            epoch_loss += loss.item()
            epoch_psnr += psnr(pred, target)
            epoch_ssim += ssim(pred, target)
            total_batches += 1

        train_loss = epoch_loss / max(total_batches, 1)
        train_psnr = epoch_psnr / max(total_batches, 1)
        train_ssim = epoch_ssim / max(total_batches, 1)

        val_loss, val_psnr, val_ssim = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            eps=1e-6,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{model_cfg['num_epochs']}] | "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Train SSIM: {train_ssim:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch + 1:04d}.pth")
        _save_epoch_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, epoch_ckpt)
        _keep_last_n_checkpoints(ckpt_dir, "epoch_", int(model_cfg.get("keep_last_n", 3)))

        if val_loss < best_val:
            best_val = val_loss
            save_bichannel_checkpoint(model, None, None, model_ckpt_path)

    if model_cfg.get("save_after_train", True):
        save_bichannel_checkpoint(model, None, None, model_ckpt_path)

    return model