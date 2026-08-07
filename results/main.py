"""
main.py
=======
Unified entry point for all training and evaluation modes.

Usage
-----
  # Train BasicCNN
  python main.py --mode train_basic --image-root /path/to/img_align_celeba

  # Resume BasicCNN training
  python main.py --mode train_basic --image-root ... --resume /path/to/ckpt.pth

  # Train BiChannelCNN (requires BasicCNN best checkpoint)
  python main.py --mode train_bichannel --image-root ...

  # Test BasicCNN (requires best_model_basiccnn.pth)
  python main.py --mode test_basic --image-root ...

  # Test BiChannelCNN (requires best_model.pth)
  python main.py --mode test_bichannel --image-root ...

  # Run classical methods (Bicubic, SC1, SC2, SFH)
  python main.py --mode classical --image-root ...

  # Specify a custom splits directory
  python main.py --mode train_basic --image-root ... --splits-dir /path/to/splits
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config as cfg
from utils import set_seed, make_splits, load_split, find_image_root
from models import (BasicCNN, BiChannelCNN,
                    load_basiccnn_into_bichannel,
                    load_basiccnn_for_inference,
                    load_bichannel_for_inference)
from dataset import TrainValFaceDataset
from trainer import train_model, run_test_pipeline
from classical import run_classical_pipeline


# ── Helpers ───────────────────────────────────────────────────
def get_splits(image_root: Path, splits_dir: Path):
    """Load existing splits or create new ones."""
    try:
        train = load_split("train", splits_dir)
        val   = load_split("val",   splits_dir)
        test  = load_split("test",  splits_dir)
        print(f"Loaded splits from {splits_dir}  "
              f"(train={len(train)}, val={len(val)}, test={len(test)})", flush=True)
        return train, val, test
    except FileNotFoundError:
        print(f"No splits found in {splits_dir} — creating new splits ...", flush=True)
        return make_splits(image_root, splits_dir=splits_dir)


def make_data_loaders(train_files, val_files):
    train_ds = TrainValFaceDataset(train_files)
    val_ds   = TrainValFaceDataset(val_files)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin)
    return train_loader, val_loader


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}", flush=True)
    return device


# ── Mode handlers ─────────────────────────────────────────────
def mode_train_basic(args, splits_dir):
    device = get_device()
    image_root   = find_image_root(args.image_root, cfg.IMAGE_DIR)
    train_f, val_f, test_f = get_splits(image_root, splits_dir)

    print("Creating DataLoaders...", flush=True)
    train_loader, val_loader = make_data_loaders(train_f, val_f)

    print("Checking directory permissions...", flush=True)
    cfg.RESULTS_BASIC_DIR.mkdir(parents=True, exist_ok=True) 

    print("Instantiating BasicCNN...", flush=True) 
    model = BasicCNN(init_weights=False).to(device)

    # Only initialize weights if we ARE NOT resuming
    if args.resume is None:
        print("Starting weight initialization on GPU...", flush=True)
        model._init_weights()
    else:
        print(f"Skipping initialization, loading checkpoint: {args.resume}", flush=True)

    print("Starting Trainer...", flush=True)
    best_path = train_model(model, train_loader, val_loader, device,
                            is_bichannel=False,
                            resume_checkpoint=args.resume)
    
    print(f"\nBest BasicCNN checkpoint: {best_path}")
    print("\nRunning test pipeline ...")
    results = run_test_pipeline(test_f, best_path, device, is_bichannel=False)
    _print_summary(results)


def mode_train_bichannel(args, splits_dir):
    device = get_device()
    image_root   = find_image_root(args.image_root, cfg.IMAGE_DIR)
    train_f, val_f, test_f = get_splits(image_root, splits_dir)
    train_loader, val_loader = make_data_loaders(train_f, val_f)

    cfg.RESULTS_BI_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Instantiating BiChannelCNN (no init)...", flush=True)
    model = BiChannelCNN(init_weights=False).to(device)

    if args.resume is None:
        model._init_weights()
        basic_ckpt = Path(args.basic_ckpt) if args.basic_ckpt else cfg.BASIC_BEST_CKPT
        if basic_ckpt.exists():
            print(f"Loading pretrained BasicCNN weights from {basic_ckpt}...", flush=True)
            model = load_basiccnn_into_bichannel(model, basic_ckpt, device)
        else:
            print(f"[Warning] BasicCNN checkpoint not found at {basic_ckpt}. "
                  "BiChannelCNN will train from scratch.")
    else:
        print(f"Skipping initialization/transfer, resuming from: {args.resume}", flush=True)

    print("Starting Trainer...", flush=True)
    best_path = train_model(model, train_loader, val_loader, device,
                            is_bichannel=True,
                            resume_checkpoint=args.resume)
    
    print(f"\nBest BiChannelCNN checkpoint: {best_path}")
    print("\nRunning test pipeline ...")
    results = run_test_pipeline(test_f, best_path, device, is_bichannel=True)
    _print_summary(results)

def mode_test_basic(args, splits_dir):
    device = get_device()
    image_root = find_image_root(args.image_root, cfg.IMAGE_DIR)
    test_f = load_split("test", splits_dir)
    print(f"Test split: {len(test_f)} images (will use first {cfg.N_TEST})", flush=True)

    model_path = Path(args.model_path) if args.model_path else cfg.BASIC_BEST_CKPT
    if not model_path.exists():
        print(f"ERROR: BasicCNN checkpoint not found at {model_path}", flush=True)
        return

    print(f"Testing BasicCNN from: {model_path}", flush=True)
    results = run_test_pipeline(test_f, model_path, device, is_bichannel=False)
    _print_summary(results)


def mode_test_bichannel(args, splits_dir):
    device = get_device()
    image_root = find_image_root(args.image_root, cfg.IMAGE_DIR)

    test_f = load_split("test", splits_dir)
    print(f"Loaded {len(test_f)} paths from test split.", flush=True) # DEBUG
    
    if len(test_f) == 0:
        print("ERROR: Test split is empty! Check your splits directory.", flush=True)
        return

    model_path = Path(args.model_path) if args.model_path else cfg.BI_BEST_CKPT
    if not model_path.exists():
        print(f"ERROR: BiChannelCNN checkpoint not found at {model_path}", flush=True)
        sys.exit(1)

    print(f"Testing BiChannelCNN from: {model_path}")
    results = run_test_pipeline(test_f, model_path, device, is_bichannel=True)
    _print_summary(results)


def mode_classical(args, splits_dir):
    image_root = find_image_root(args.image_root, cfg.IMAGE_DIR)
    train_f, _, test_f = get_splits(image_root, splits_dir)

    train_f = train_f[:cfg.CL_N_TRAIN]
    test_f  = test_f [:cfg.CL_N_TEST]
    print(f"Classical — using {len(train_f)} train, {len(test_f)} test images", flush=True)

    results = run_classical_pipeline(train_f, test_f)
    _print_summary(results, key="method")


def _print_summary(results, key="blur_type"):
    print("\n── Final Summary ──")
    for r in results:
        label = r.get(key, "") or r.get("blur_type", "")
        print(f"  {label:30s}  PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}")


# ── CLI ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Face Hallucination — unified entry point",
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--mode", required=True,
                   choices=["train_basic", "train_bichannel",
                            "test_basic", "test_bichannel", "classical"],
                   help=(
                       "train_basic     : train BasicCNN\n"
                       "train_bichannel : train BiChannelCNN (needs BasicCNN ckpt)\n"
                       "test_basic      : test BasicCNN on all 6 conditions\n"
                       "test_bichannel  : test BiChannelCNN on all 6 conditions\n"
                       "classical       : run Bicubic/SC1/SC2/SFH pipeline\n"))

    p.add_argument("--image-root",  default=None,
                   help="Path to img_align_celeba folder")
    p.add_argument("--splits-dir",  default=None,
                   help="Override splits directory (default: config.SPLITS_DIR)")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume training from")
    p.add_argument("--model-path",  default=None,
                   help="Checkpoint to use for test modes")
    p.add_argument("--basic-ckpt",  default=None,
                   help="BasicCNN checkpoint for BiChannelCNN pretrain transfer")
    return p.parse_args()


def main():
    args       = parse_args()
    splits_dir = Path(args.splits_dir) if args.splits_dir else cfg.SPLITS_DIR

    set_seed(cfg.SEED)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Splits dir: {splits_dir}", flush=True)

    dispatch = {
        "train_basic":     mode_train_basic,
        "train_bichannel": mode_train_bichannel,
        "test_basic":      mode_test_basic,
        "test_bichannel":  mode_test_bichannel,
        "classical":       mode_classical,
    }
    dispatch[args.mode](args, splits_dir)


if __name__ == "__main__":
    main()
