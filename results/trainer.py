"""
trainer.py
==========
Training, validation, and test functions for BasicCNN and BiChannelCNN.
"""

import csv
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
    WARMUP_EPOCHS, LR_T_MAX, MIN_LR,
    BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, KEEP_LAST_N,
    RESULTS_BASIC_DIR, RESULTS_BI_DIR,
    BASIC_CKPT_DIR, BI_CKPT_DIR, HR_SIZE, SEED,
)
from models import WarmupCosineScheduler
from dataset import FixedTestFaceDataset
from utils import (
    denormalize_per_image, compute_psnr, compute_ssim,
    save_checkpoint, load_checkpoint, keep_last_n_checkpoints,
    save_metrics_to_csv, celeba_crop, save_image,
)


# ── Validation pass ───────────────────────────────────────────
def evaluate_validation(model, dataloader, device, criterion,
                        is_bichannel: bool = False):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    with torch.no_grad():
        for iin, ih, mean_b, std_b in dataloader:
            iin, ih = iin.to(device).float(), ih.to(device).float()
            if is_bichannel:
                out, _ = model(iin)
            else:
                out = model(iin)
            total_loss += criterion(out, ih).item()
            out_cpu = out.cpu()
            ih_cpu  = ih.cpu()
            for b in range(out_cpu.shape[0]):
                p = denormalize_per_image(
                    out_cpu[b], mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(
                    ih_cpu[b],  mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    return total_loss / len(dataloader), total_psnr / n, total_ssim / n


# ── Single test condition ─────────────────────────────────────
def evaluate_test_setting(model, dataloader, device, criterion,
                          is_bichannel: bool = False):
    model.eval()
    total_loss = total_psnr = total_ssim = n = 0
    total_am = total_amin = total_amax = 0
    with torch.no_grad():
        for batch_idx, (iin, ih, mean_b, std_b) in enumerate(dataloader):
            iin, ih = iin.to(device).float(), ih.to(device).float()
            if is_bichannel:
                out, alpha = model(iin)
                total_am   += alpha.mean().item()
                total_amin += alpha.min().item()
                total_amax += alpha.max().item()
            else:
                out = model(iin)
            total_loss += criterion(out, ih).item()
            out_cpu = out.cpu()
            ih_cpu  = ih.cpu()

            for b in range(out_cpu.shape[0]):
                p = denormalize_per_image(
                    out_cpu[b], mean_b[b].numpy(), std_b[b].numpy())
                t = denormalize_per_image(
                    ih_cpu[b],  mean_b[b].numpy(), std_b[b].numpy())
                total_psnr += compute_psnr(p, t)
                total_ssim += compute_ssim(p, t)
                n += 1
    nb = len(dataloader)
    if is_bichannel:
        print(f"  alpha mean={total_am/nb:.4f} "
              f"min={total_amin/nb:.4f} max={total_amax/nb:.4f}", flush=True)
    return total_loss / nb, total_psnr / n, total_ssim / n


# ── Sample image saving ───────────────────────────────────────
def save_sample_grid(model, test_files: List[str], device,
                     save_dir: Path, blur_type: str,
                     gaussian_sigma: float = None,
                     motion_length: int = None,
                     sample_idx: int = 0,
                     is_bichannel: bool = False) -> None:
    """Save raw/HR/LR/SR quad for one image + blur condition."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ds = FixedTestFaceDataset(test_files, blur_type=blur_type,
                              gaussian_sigma=gaussian_sigma,
                              motion_length=motion_length,
                              base_seed=SEED)
    model.eval()
    raw_path = test_files[sample_idx]
    iin, ih, mean_b, std_b = ds[sample_idx]

    # 0. Raw crop
    celeba_crop(Image.open(raw_path).convert("RGB")).resize(
        HR_SIZE, Image.BICUBIC).save(save_dir / "0_raw.png")
    # 1. HR
    save_image(denormalize_per_image(ih, mean_b.numpy(), std_b.numpy()),
               save_dir / "1_hr.png")
    # 2. LR (nearest upscale for visible pixelation)
    lr_arr = denormalize_per_image(iin, mean_b.numpy(), std_b.numpy())
    Image.fromarray(
        (lr_arr.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
    ).resize(HR_SIZE, Image.NEAREST).save(save_dir / "2_lr_input.png")
    # 3. SR
    with torch.no_grad():
        iin_b = iin.unsqueeze(0).to(device).float()
        if is_bichannel:
            out_b, alpha_b = model(iin_b)
            print(f"  Sample (α={alpha_b.mean().item():.3f}) → {save_dir}")
        else:
            out_b = model(iin_b)
            print(f"  Sample → {save_dir}")
    save_image(denormalize_per_image(out_b[0], mean_b.numpy(), std_b.numpy()),
               save_dir / "3_sr_output.png")


# ── Main training function ────────────────────────────────────
def train_model(model, train_loader, val_loader, device,
                is_bichannel: bool = False,
                resume_checkpoint: Path = None):
    """
    Unified training function for BasicCNN and BiChannelCNN.
    is_bichannel=True enables alpha logging and uses BiChannelCNN paths.
    """
    criterion  = nn.MSELoss()
    optimizer  = optim.AdamW(model.parameters(),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = WarmupCosineScheduler(optimizer,
                                       warmup_epochs=WARMUP_EPOCHS,
                                       t_max=LR_T_MAX, min_lr=MIN_LR)

    results_dir   = RESULTS_BI_DIR    if is_bichannel else RESULTS_BASIC_DIR
    ckpt_dir      = BI_CKPT_DIR       if is_bichannel else BASIC_CKPT_DIR
    ckpt_prefix   = "bichannel_epoch_" if is_bichannel else "basiccnn_epoch_"
    best_name     = "best_model.pth"   if is_bichannel else "best_model_basiccnn.pth"
    best_path     = results_dir / best_name
    csv_path      = results_dir / ("train_log_bichannel.csv"
                                   if is_bichannel else "train_log_basiccnn.csv")

    print(f"Optimizer: AdamW  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}", flush=True)
    print(f"LR schedule: {WARMUP_EPOCHS}-epoch warmup → cosine T_max={LR_T_MAX}", flush=True)
    print(f"Model: {'BiChannelCNN' if is_bichannel else 'BasicCNN'}", flush=True)

    start_epoch   = 0
    best_val_loss = float("inf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint and Path(resume_checkpoint).exists():
        start_epoch, best_val_loss = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = epoch_psnr = epoch_ssim = n = 0

        for batch_idx, (iin, ih, mean_b, std_b) in enumerate(train_loader):
            iin, ih = iin.to(device).float(), ih.to(device).float()
            optimizer.zero_grad()

            if is_bichannel:
                out, alpha = model(iin)
            else:
                out = model(iin)

            loss = criterion(out, ih)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                for b in range(out.shape[0]):
                    p = denormalize_per_image(out[b].detach(),
                                              mean_b[b].numpy(), std_b[b].numpy())
                    t = denormalize_per_image(ih[b],
                                              mean_b[b].numpy(), std_b[b].numpy())
                    epoch_psnr += compute_psnr(p, t)
                    epoch_ssim += compute_ssim(p, t)
                    n += 1

            if batch_idx == 0 and epoch == start_epoch:
                print(f"First batch — In:{iin.shape} Out:{out.shape} "
                      f"Target:{ih.shape}")

        # Alpha diagnostic (BiChannelCNN only)
        if is_bichannel and batch_idx == 0:
            print(f"[Epoch {epoch+1}] alpha mean={alpha.mean().item():.4f} "
                  f"min={alpha.min().item():.4f} max={alpha.max().item():.4f}", flush=True)

        train_loss = epoch_loss / len(train_loader)
        train_psnr = epoch_psnr / n
        train_ssim = epoch_ssim / n

        val_loss, val_psnr, val_ssim = evaluate_validation(
            model, val_loader, device, criterion, is_bichannel)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss:{train_loss:.4f} PSNR:{train_psnr:.2f} "
              f"SSIM:{train_ssim:.4f} | "
              f"Val Loss:{val_loss:.4f} PSNR:{val_psnr:.2f} "
              f"SSIM:{val_ssim:.4f} | LR:{lr:.1e}", flush=True)

        save_metrics_to_csv(csv_path,
            [epoch + 1, train_loss, train_psnr, train_ssim,
             val_loss, val_psnr, val_ssim, lr],
            ["epoch", "train_loss", "train_psnr", "train_ssim",
             "val_loss", "val_psnr", "val_ssim", "lr"])

        ckpt_path = ckpt_dir / f"{ckpt_prefix}{epoch+1:04d}.pth"
        save_checkpoint(model, optimizer, scheduler,
                        epoch + 1, val_loss, ckpt_path)
        keep_last_n_checkpoints(ckpt_dir, prefix=ckpt_prefix, n=KEEP_LAST_N)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler,
                            epoch + 1, val_loss, best_path)
            print(f"  → New best model (val_loss={best_val_loss:.4f})")

    return best_path


# ── Test pipeline ─────────────────────────────────────────────
def run_test_pipeline(test_files: List[str], best_model_path: Path,
                      device, is_bichannel: bool = False,
                      n_test: int = None):
    """
    Run all 6 blur conditions, print + save metrics and sample images.

    n_test : cap the number of test images (default: N_TEST from config).
             Matches original bichannel5_9.py which used N_TEST=20000
             but evaluated on a fixed subset for speed. Set this to
             match whatever N_TEST your original training script used.
    """
    from models import load_basiccnn_for_inference, load_bichannel_for_inference
    from config import N_TEST  # defined as 20000 in full run; override here

    if n_test is None:
        n_test = N_TEST
    test_files = test_files[:n_test]
    print(f"Test pipeline: {len(test_files)} images × 6 conditions", flush=True)

    if is_bichannel:
        model = load_bichannel_for_inference(best_model_path, device)
    else:
        model = load_basiccnn_for_inference(best_model_path, device)

    results_dir = RESULTS_BI_DIR if is_bichannel else RESULTS_BASIC_DIR
    criterion   = nn.MSELoss()
    results     = []
    lkw = dict(batch_size=BATCH_SIZE, shuffle=False,
               num_workers=NUM_WORKERS,
               pin_memory=False)   

    for sigma in [1, 3, 5]:
        tag = f"gaussian_sigma_{sigma}"
        print(f"\nEvaluating {tag} ...", flush=True)
        ds  = FixedTestFaceDataset(test_files, blur_type="gaussian",
                                   gaussian_sigma=sigma)
        l, p, s = evaluate_test_setting(
            model, DataLoader(ds, **lkw), device, criterion, is_bichannel)
        print(f"  {tag} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}",
              flush=True)
        results.append({"blur_type": tag, "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         results_dir / f"samples_{tag}",
                         blur_type="gaussian", gaussian_sigma=sigma,
                         is_bichannel=is_bichannel)

    for length in [2, 6, 9]:
        tag = f"motion_l_{length}"
        print(f"\nEvaluating {tag} ...", flush=True)
        ds  = FixedTestFaceDataset(test_files, blur_type="motion",
                                   motion_length=length)
        l, p, s = evaluate_test_setting(
            model, DataLoader(ds, **lkw), device, criterion, is_bichannel)
        print(f"  {tag} | Loss={l:.4f} | PSNR={p:.2f} | SSIM={s:.4f}",
              flush=True)
        results.append({"blur_type": tag, "loss": l, "psnr": p, "ssim": s})
        save_sample_grid(model, test_files, device,
                         results_dir / f"samples_{tag}",
                         blur_type="motion", motion_length=length,
                         is_bichannel=is_bichannel)

    model_name = "bichannel" if is_bichannel else "basiccnn"
    csv_path   = results_dir / f"test_metrics_{model_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["blur_type", "loss", "psnr", "ssim"])
        w.writeheader()
        w.writerows(results)
    print(f"\nTest metrics saved → {csv_path}", flush=True)
    return results
