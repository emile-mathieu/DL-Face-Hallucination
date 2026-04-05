import os, sys
import csv
import cv2
import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader

from config import CONFIG
from data.dataset import FaceDataset, Classical_FaceDataset, FixedTestFaceDataset
from data.dataloader import (
    get_test_loader, 
    get_train_loader, 
    get_val_loader, 
    get_classical_train_dataset
)
from models.model import BasicCNN, BiChannelCNN
from training.train import run_train_pipeline
from training.test import evaluate_one_test_setting
from training.classical import classical_train, classical_evaluate_sigma
from utils.utils import(
    ensure_dir,
    save_bichannel_checkpoint,
    load_bichannel_checkpoint,
    save_rgb_image,
    psnr,
    ssim
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)



def run_test_dataset(
    bichannel_model,
    device,
    train_dir,
    test_dir,
    sigmas,
    classical_train_max_samples,
    test_max_samples,
    save_first_n,
):

    # ---------------------------------------------------------
    # PHASE 2: EVALUATION
    # ---------------------------------------------------------
    bichannel_model.eval()
    summary_rows = []

    for sigma in sigmas:
        print(f"\n--- Testing Sigma: {sigma} ---")
        
        # Combined container for stats
        totals = {
            "bichannel": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sc1": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sc2": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sfh": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        }

        # --- PART A: BICHACHENNEL EVALUATION (Uses DataLoader) ---
        bi_dataset = FixedTestFaceDataset(
            image_paths=test_image_paths,
            blur_type="gaussian",
            gaussian_sigma=sigma
        )
        bi_loader = DataLoader(bi_dataset, batch_size=1, shuffle=False)
        bi_loader = get_test_loader(
            image_dir=test_image_paths,
            
        )

        print(f"Running BiChannel Evaluation (N={len(bi_loader)})...")
        for idx, (Iin_norm, IH_norm, mean, std) in enumerate(bi_loader):
            Iin_norm = Iin_norm.to(device)
            ms = mean.view(-1, 3, 1, 1).to(device)
            ss = std.view(-1, 3, 1, 1).to(device)

            with torch.no_grad():
                raw_out, _ = bichannel_model(Iin_norm)
                # Handle model returning (pred, features) or just pred
                outputs = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
                
                eps = 1e-6
                # Denormalize Prediction
                pred_t = torch.atanh(torch.clamp(outputs, -1 + eps, 1 - eps)) * ss + ms
                pred_t = torch.clamp(pred_t, 0.0, 1.0)
                
                # Denormalize Ground Truth for fair comparison
                target_t = torch.atanh(torch.clamp(IH_norm.to(device), -1 + eps, 1 - eps)) * ss + ms
                target_t = torch.clamp(target_t, 0.0, 1.0)

            totals["bichannel"]["psnr"] += psnr(pred_t, target_t)
            totals["bichannel"]["ssim"] += ssim(pred_t, target_t)
            totals["bichannel"]["count"] += 1

            if idx < save_first_n:
                bi_np = pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                save_rgb_image(os.path.join(images_dir, f"sigma{sigma}_img{idx:03d}_bichannel.png"), bi_np)
                # Save Reference HR and LR for BiChannel part
                hr_ref = target_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                save_rgb_image(os.path.join(images_dir, f"sigma{sigma}_img{idx:03d}_hr.png"), hr_ref)

       
        # ---------------------------------------------------------
        # PHASE 3: AGGREGATE RESULTS FOR SIGMA
        # ---------------------------------------------------------
        for model_name, vals in totals.items():
            count = max(vals["count"], 1)
            avg_psnr = vals["psnr"] / count
            avg_ssim = vals["ssim"] / count

            summary_rows.append([sigma, model_name, avg_psnr, avg_ssim, count])
            print(f"Sigma={sigma} | {model_name:9} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | N: {count}")

    # Write Results to CSV
    eval_csv = os.path.join(results_root, "eval_metrics.csv")
    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "model", "avg_psnr", "avg_ssim", "num_images"])
        writer.writerows(summary_rows)

    print(f"\nEvaluation Complete. Results saved to: {eval_csv}")


def main(mode="bichannel"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths_cfg = CONFIG["paths"]
    sc1_cfg = CONFIG["sc1"]
    sc2_cfg = CONFIG["sc2"]
    sfh_cfg = CONFIG["sfh"]
    test_cfg = CONFIG["run_test_dataset"]

    classical_cfg = [sc1_cfg, sc2_cfg, sfh_cfg]

    train_path = resolve_path(paths_cfg["train_dir"])
    val_path = resolve_path(paths_cfg["val_dir"])
    test_path = resolve_path(paths_cfg["test_dir"])
    checkpoints_dir = resolve_path(paths_cfg["trained_models_dir"])
    results_root = resolve_path(paths_cfg["results_dir"])
    images_dir = os.path.join(results_root, "images")

    ensure_dir(checkpoints_dir)
    ensure_dir(results_root)
    ensure_dir(images_dir)

    hr_size = tuple(sc2_cfg["hr_size"])
    lr_size = tuple(sc2_cfg["lr_size"])



    # 1. Initialize Model
    if mode == "basic":
        model_cfg = CONFIG["basic"]
        model = BasicCNN().to(device)
    elif mode == "bichannel":
        model_cfg = CONFIG["bichannel"]
        model = BiChannelCNN().to(device)

    
    # 2. Initialize Datasets
    # Removed mean=mean, std=std because FaceDataset now calculates them internally per image
    train_loader = get_train_loader(
        train_path,
        batch_size=model_cfg["batch_size"],
        num_workers=model_cfg["num_workers"]
    )
    val_loader = get_val_loader(
        val_path, 
        batch_size=model_cfg["batch_size"], 
        num_workers=model_cfg["num_workers"]
    )


    # 3. Load checkpoint if it exists, if not train the model from scratch
    # Note: mean/std returned here might be None because we are using the new per-image logic,
    # but we keep the variables to avoid breaking the load function signature.
    
    model_ckpt_path = os.path.join(checkpoints_dir, model_cfg["checkpoint_name"])
    if model_cfg["load_if_exists"] and os.path.exists(model_ckpt_path):
        model, _, _ = load_bichannel_checkpoint(model, model_ckpt_path, device)
    else:
        print(f"Training {"BiChannel" if mode=="bichannel" else "Basic"} CNN with per-image normalization...")
        model = run_train_pipeline(
            model, 
            model_cfg, 
            train_loader, 
            val_loader, 
            device,
            model_ckpt_path
        )
        

    # 4. Prepare classical models

    classical_train_ds = get_classical_train_dataset(train_path, hr_size=hr_size)
    classical_models = classical_train(
        classical_train_ds,
        classical_cfg,
        checkpoints_dir,
        classical_train_max_samples=test_cfg["classical_train_max_samples"]
    )


    # 5. Evaluation both classical and CNN models

    totals_container = {
        "bichannel": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        "sc1": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        "sc2": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        "sfh": {"psnr": 0.0, "ssim": 0.0, "count": 0},
    }

    summary_rows = []

    for sigma in test_cfg["sigmas"]:
        print(f"\n--- Testing Sigma: {sigma} ---")

        save_cfg = {
            "save_first_n": test_cfg["save_first_n"],
            "images_dir": images_dir,
            "params": "sigma",
            "params_val": sigma
        }
        
        totals = totals_container.copy()

        test_loader = get_test_loader(test_path, blur_type="gaussian", gaussian_sigma=sigma, save_cfg=save_cfg)
        results = evaluate_one_test_setting(model, test_loader, device)
        totals["bichannel"]["psnr"] += results[1]
        totals["bichannel"]["ssim"] += results[2]
        totals["bichannel"]["count"] += results[3]

        classical_test_ds = get_classical_train_dataset(test_path, max_items=test_cfg["test_max_samples"],
                                                        blur_type="gaussian", gaussian_sigma=sigma)
        classical_results = classical_evaluate_sigma(classical_models, classical_test_ds, 
                                                     lr_size, hr_size, device, save_cfg)
        totals["bichannel"]["psnr"] += classical_results[1]
        totals["bichannel"]["ssim"] += classical_results[2]
        totals["bichannel"]["count"] += classical_results[3]

        for model_name, vals in totals.items():
            avg_psnr = vals["psnr"]
            avg_ssim = vals["ssim"]
            count = vals["count"]
            summary_rows.append(["sigma", sigma, model_name, avg_psnr, avg_ssim, count])
            print(f"Sigma={sigma} | {model_name:9} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | N: {count}")
    

    for length in test_cfg["lengths"]:
        print(f"\n--- Testing Motion Length: {length} ---")

        save_cfg = {
            "save_first_n": test_cfg["save_first_n"],
            "images_dir": images_dir,
            "params": "motion",
            "params_val": length
        }
        
        totals = totals_container.copy()

        test_loader = get_test_loader(test_path, blur_type="motion", motion_length=length, base_seed=42)
        results = evaluate_one_test_setting(model, test_loader, device, save_cfg=save_cfg)
        totals["bichannel"]["psnr"] += results[1]
        totals["bichannel"]["ssim"] += results[2]
        totals["bichannel"]["count"] += results[3]

        """
        ### NOT IMPLEMENTED YET ###
        classical_test_ds = get_classical_train_dataset(test_path, max_items=test_cfg["test_max_samples"],
                                                        blur_type="motion", motion_length=length, base_seed=42)
        
        classical_results = classical_evaluate_motion(classical_models, classical_test_ds, 
                                                      lr_size, hr_size, device, save_cfg=save_cfg)
        totals["bichannel"]["psnr"] += classical_results[1]
        totals["bichannel"]["ssim"] += classical_results[2]
        totals["bichannel"]["count"] += classical_results[3]
        """
        for model_name, vals in totals.items():
            avg_psnr = vals["psnr"]
            avg_ssim = vals["ssim"]
            count = vals["count"]
            summary_rows.append(["motion", length, model_name, avg_psnr, avg_ssim, count])
            print(f"Length={length} | {model_name:9} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | N: {count}")
    

    
    # Write Results to CSV
    eval_csv = os.path.join(results_root, "eval_metrics.csv")
    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["params", "params_val", "model", "avg_psnr", "avg_ssim", "num_images"])
        writer.writerows(summary_rows)

    print(f"\nEvaluation Complete. Results saved to: {eval_csv}")

    
    run_test_dataset(
        bichannel_model=model,
        # run_test_dataset may still need to handle per-image stats 
        # inside its own implementation using FixedTestFaceDataset
        device=device,
        train_dir=train_path,
        test_dir=test_path,
        sigmas=test_cfg["sigmas"],
        classical_train_max_samples=test_cfg["classical_train_max_samples"],
        test_max_samples=test_cfg["test_max_samples"],
        save_first_n=test_cfg["save_first_n"],
    )



    # 1. PREPARE IMAGE PATH LISTS
    test_image_paths = sorted([
        os.path.join(test_dir, f) for f in os.listdir(test_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if test_max_samples:
        test_image_paths = test_image_paths[:test_max_samples]


    # 4. Evaluation, both classical and the model
    print("\nRunning test dataset evaluation...")

    classical_train_ds = get_classical_train_dataset(train_path, hr_size=hr_size)
    total = run_classical_pipeline(
        classical_cfg,

    )
    total = run_test_pipeline()

    save_results(total)


if __name__ == "__main__":
    try:
        mode = sys.argv[1]
    except:
        mode = "bichannel"

    main(mode)
