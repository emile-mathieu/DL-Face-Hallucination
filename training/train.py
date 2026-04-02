import torch
from utils.logger import save_metrics_to_csv

# ============================================================
# Metrics  (Y-channel or RGB, toggled by METRIC_CHANNEL)
# ============================================================
def _to_y(t: torch.Tensor) -> torch.Tensor:
    """CHW or 1CHW RGB [0,1] → 1×1×H×W Y-channel."""
    if t.ndim == 3:
        t = t.unsqueeze(0)
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return (0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0).clamp(0, 1)


def _ssim_kernel(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    k = g.unsqueeze(0) * g.unsqueeze(1)
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(device=device, dtype=dtype)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y(pred), _to_y(target)
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if METRIC_CHANNEL == "y":
        pred, target = _to_y(pred), _to_y(target)
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch  = pred.shape[1]
    k   = _ssim_kernel(ch, pred.device, pred.dtype)
    pad = 5
    mu1  = F.conv2d(pred,   k, padding=pad, groups=ch)
    mu2  = F.conv2d(target, k, padding=pad, groups=ch)
    s1   = F.conv2d(pred*pred,     k, padding=pad, groups=ch) - mu1**2
    s2   = F.conv2d(target*target, k, padding=pad, groups=ch) - mu2**2
    s12  = F.conv2d(pred*target,   k, padding=pad, groups=ch) - mu1*mu2
    num  = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-12)).mean().item()


# ============================================================
# Train / Validate
# ============================================================
def evaluate_validation(model, dataloader, device, criterion):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    with torch.no_grad():
        for Iin, IH, mean_b, std_b in dataloader:
            Iin, IH = Iin.to(device).float(), IH.to(device).float()
            outputs, _   = model(Iin)
            total_loss += criterion(outputs, IH).item()
            for b in range(outputs.shape[0]):
                pred = denormalize_per_image(outputs[b], mean_b[b].numpy(), std_b[b].numpy())
                target = denormalize_per_image(IH[b],  mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(pred, target)
                total_ssim += compute_ssim(pred, target)
                n += 1
    return total_loss / len(dataloader), total_psnr / n, total_ssim / n


# Training function
def train(optimizer, criterion, model, dataloader, device, dataset, num_epochs=20, scheduler=None):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        n = 0.0

        for batch_idx, (Iin, IH, mean_b, std_b) in enumerate(dataloader):
            Iin = Iin.to(device).float()
            IH = IH.to(device).float()

            optimizer.zero_grad()

            outputs, alpha = model(Iin)

            # loss (MSE)
            loss = criterion(outputs, IH)
            
            # backward
            loss.backward()
            # gradient clipping (optional but safe), if want to gradient clipping can use 'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)'
            
            optimizer.step()
            epoch_loss += loss.item()

            # denormalize then compare PSNR and SSIM (in the range 0,1)
            with torch.no_grad():
                for b in range(outputs.shape[0]):
                    pred = denormalize_per_image(outputs[b].detach(),
                                              mean_b[b].numpy(), std_b[b].numpy())
                    target = denormalize_per_image(IH[b],
                                              mean_b[b].numpy(), std_b[b].numpy())
                    epoch_psnr += compute_psnr(pred, target)
                    epoch_ssim += compute_ssim(pred, target)
                    n += 1

            #debug 
            if epoch == start_epoch and batch_idx == 0:
                print(f"First batch — In:{Iin.shape} Out:{outputs.shape} Target:{IH.shape}")
            if batch_idx == 0:
                print(f"[Epoch {epoch+1}] alpha mean={alpha.mean().item():.4f} "
                      f"min={alpha.min().item():.4f} max={alpha.max().item():.4f}")

        train_loss = epoch_loss / len(dataloader)
        train_psnr = epoch_psnr / n
        train_ssim = epoch_ssim / n

        val_loss, val_psnr, val_ssim = evaluate_validation(
            model, val_loader, device, criterion)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        #print results
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss:{train_loss:.4f} PSNR:{train_psnr:.2f} SSIM:{train_ssim:.4f} | "
              f"Val Loss:{val_loss:.4f} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f} | LR:{lr:.1e}")

        #save to csv
        save_metrics_to_csv(
            RESULTS_DIR / "train_val_metrics.csv",
            [epoch+1, train_loss, train_psnr, train_ssim,
             val_loss, val_psnr, val_ssim, lr],
            ["epoch","train_loss","train_psnr","train_ssim",
             "val_loss","val_psnr","val_ssim","lr"])
        
        # save best model based on lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler,
                            epoch + 1, val_loss, best_model_path)
            print(f"  → New best model (val_loss={best_val_loss:.4f})")

    return best_model_path
