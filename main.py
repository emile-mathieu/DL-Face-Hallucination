import os
import csv
import cv2
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from config import CONFIG
from data.dataset import FaceDataset, Classical_FaceDataset, compute_mean_std
from models.model import BiChannelCNN
from training.train import train
from reconstruction.reconstruction import (
    SR1_SC1_Yang_FeatureResidual,
    SR2_SC2_Kim_KRR,
    SR3_SFH,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def bicubic_to_hr(img, hr_size):
    return cv2.resize(img, hr_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)


def gaussian_blur_with_sigma(img, sigma):
    img = np.asarray(img, dtype=np.float32)
    out = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_lr_from_hr_sigma(hr_img, sigma, lr_size):
    blurred = gaussian_blur_with_sigma(hr_img, sigma)
    lr_img = cv2.resize(blurred, lr_size, interpolation=cv2.INTER_CUBIC)
    return np.clip(lr_img, 0.0, 1.0).astype(np.float32)


def save_rgb_image(path, img_float01):
    ensure_dir(os.path.dirname(path))
    img = np.clip(img_float01, 0.0, 1.0)
    img_u8 = (img * 255.0).round().astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def save_bichannel_checkpoint(model, mean, std, path):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
        },
        path,
    )
    print(f"Saved BiChannel checkpoint to: {path}")


def load_bichannel_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    print(f"Loaded BiChannel checkpoint from: {path}")
    return model, mean, std


def save_pickle_model(obj, path, label):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {label} model to: {path}")


def load_pickle_model(path, label):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded {label} model from: {path}")
    return obj


def collect_training_pairs(dataset, hr_size, max_samples=None):
    lr100_imgs_rgb = []
    hr100_imgs_rgb = []

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(total):
        il, ih = dataset[i]
        lr_hr_sized = bicubic_to_hr(il, hr_size=hr_size)

        lr100_imgs_rgb.append(lr_hr_sized.astype(np.float32))
        hr100_imgs_rgb.append(ih.astype(np.float32))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Collected training pair {i + 1}/{total}")

    return lr100_imgs_rgb, hr100_imgs_rgb


def build_sc1(sc1_cfg):
    return SR1_SC1_Yang_FeatureResidual(
        patch=sc1_cfg["patch"],
        stride=sc1_cfg["stride"],
        n_atoms=sc1_cfg["n_atoms"],
        lasso_alpha=sc1_cfg["lasso_alpha"],
        lasso_max_iter=sc1_cfg["lasso_max_iter"],
        lasso_tol=sc1_cfg["lasso_tol"],
        dl_alpha=sc1_cfg["dl_alpha"],
        dl_iter=sc1_cfg["dl_iter"],
        max_train_patches=sc1_cfg["max_train_patches"],
        per_image_patch_cap=sc1_cfg["per_image_patch_cap"],
        random_state=sc1_cfg["random_state"],
    )


def build_sc2(sc2_cfg):
    return SR2_SC2_Kim_KRR(
        patch_size=sc2_cfg["patch_size"],
        sigma_k=sc2_cfg["sigma_k"],
        lam=sc2_cfg["lam"],
        n_basis=sc2_cfg["n_basis"],
        n_train=sc2_cfg["n_train"],
        hr_size=tuple(sc2_cfg["hr_size"]),
        lr_size=tuple(sc2_cfg["lr_size"]),
        seed=sc2_cfg["seed"],
    )


def build_sfh(sfh_cfg):
    return SR3_SFH(
        max_exemplars=sfh_cfg["max_exemplars"],
        hr_size=tuple(sfh_cfg["hr_size"]),
        lr_size=tuple(sfh_cfg["lr_size"]),
        seed=sfh_cfg["seed"],
    )


def run_test_dataset(
    bichannel_model,
    mean,
    std,
    device,
    train_dir,
    test_dir,
    sigmas,
    classical_train_max_samples,
    test_max_samples,
    save_first_n,
):
    paths_cfg = CONFIG["paths"]
    sc1_cfg = CONFIG["sc1"]
    sc2_cfg = CONFIG["sc2"]
    sfh_cfg = CONFIG["sfh"]

    checkpoints_dir = resolve_path(paths_cfg["trained_models_dir"])
    results_root = resolve_path(paths_cfg["results_dir"])
    images_dir = os.path.join(results_root, "images")

    hr_size = tuple(sc2_cfg["hr_size"])
    lr_size = tuple(sc2_cfg["lr_size"])

    ensure_dir(checkpoints_dir)
    ensure_dir(results_root)
    ensure_dir(images_dir)

    print("Preparing datasets for test run...")
    classical_train_dataset = Classical_FaceDataset(train_dir, hr_size=hr_size)
    classical_test_dataset = Classical_FaceDataset(test_dir, hr_size=hr_size)

    bichannel_helper = FaceDataset(train_dir, mean=mean, std=std)

    total_test = (
        len(classical_test_dataset)
        if test_max_samples is None
        else min(len(classical_test_dataset), test_max_samples)
    )

    sc1_path = os.path.join(checkpoints_dir, sc1_cfg["checkpoint_name"])
    sc2_path = os.path.join(checkpoints_dir, sc2_cfg["checkpoint_name"])
    sfh_path = os.path.join(checkpoints_dir, sfh_cfg["checkpoint_name"])

    need_train_pairs = not (
        sc1_cfg["load_if_exists"] and os.path.exists(sc1_path)
        and sc2_cfg["load_if_exists"] and os.path.exists(sc2_path)
        and sfh_cfg["load_if_exists"] and os.path.exists(sfh_path)
    )

    train_lr100, train_hr100 = None, None
    if need_train_pairs:
        print("Collecting classical training pairs...")
        train_lr100, train_hr100 = collect_training_pairs(
            classical_train_dataset,
            hr_size=hr_size,
            max_samples=classical_train_max_samples,
        )

    if sc1_cfg["load_if_exists"] and os.path.exists(sc1_path):
        sc1 = load_pickle_model(sc1_path, "SC1")
    else:
        sc1 = build_sc1(sc1_cfg)
        print("Training SC1...")
        sc1.fit(train_lr100, train_hr100)
        if sc1_cfg["save_after_train"]:
            save_pickle_model(sc1, sc1_path, "SC1")

    if sc2_cfg["load_if_exists"] and os.path.exists(sc2_path):
        sc2 = load_pickle_model(sc2_path, "SC2")
    else:
        sc2 = build_sc2(sc2_cfg)
        print("Training SC2...")
        sc2.fit(train_lr100, train_hr100)
        if sc2_cfg["save_after_train"]:
            save_pickle_model(sc2, sc2_path, "SC2")

    if sfh_cfg["load_if_exists"] and os.path.exists(sfh_path):
        sfh = load_pickle_model(sfh_path, "SFH")
    else:
        sfh = build_sfh(sfh_cfg)
        print("Training SFH...")
        sfh.fit(train_lr100, train_hr100)
        if sfh_cfg["save_after_train"]:
            save_pickle_model(sfh, sfh_path, "SFH")

    bichannel_model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    summary_rows = []

    for sigma in sigmas:
        totals = {
            "bichannel": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sc1": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sc2": {"psnr": 0.0, "ssim": 0.0, "count": 0},
            "sfh": {"psnr": 0.0, "ssim": 0.0, "count": 0},
        }

        print(f"\n===== Testing sigma={sigma} =====")

        for idx in range(total_test):
            _, hr_img = classical_test_dataset[idx]
            hr_img = np.clip(hr_img, 0.0, 1.0).astype(np.float32)

            lr50 = make_lr_from_hr_sigma(hr_img, sigma=sigma, lr_size=lr_size)
            lr100 = bicubic_to_hr(lr50, hr_size=hr_size)

            bi_input = bichannel_helper.preprocess(lr50)
            bi_input_t = (
                torch.from_numpy(bi_input)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            with torch.no_grad():
                bi_pred = bichannel_model(bi_input_t).squeeze(0).cpu()
                bi_img = (
                    bichannel_helper.denormalize(bi_pred)
                    .permute(1, 2, 0)
                    .numpy()
                    .astype(np.float32)
                )
                bi_img = np.clip(bi_img, 0.0, 1.0)

            sc1_img = np.clip(sc1.predict(lr100), 0.0, 1.0).astype(np.float32)
            sc2_img = np.clip(sc2.predict(lr50), 0.0, 1.0).astype(np.float32)
            sfh_img = np.clip(sfh.predict(lr50), 0.0, 1.0).astype(np.float32)

            outputs = {
                "bichannel": bi_img,
                "sc1": sc1_img,
                "sc2": sc2_img,
                "sfh": sfh_img,
            }

            hr_tensor = (
                torch.from_numpy(hr_img)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            for model_name, pred_img in outputs.items():
                pred_tensor = (
                    torch.from_numpy(pred_img)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )

                img_psnr = psnr(pred_tensor, hr_tensor).item()
                img_ssim = ssim_metric(pred_tensor, hr_tensor).item()

                totals[model_name]["psnr"] += img_psnr
                totals[model_name]["ssim"] += img_ssim
                totals[model_name]["count"] += 1

                if idx < save_first_n:
                    filename = f"sigma{sigma}_img{idx:03d}_{model_name}.png"
                    save_rgb_image(os.path.join(images_dir, filename), pred_img)

            if idx < save_first_n:
                save_rgb_image(os.path.join(images_dir, f"sigma{sigma}_img{idx:03d}_hr.png"), hr_img)
                save_rgb_image(
                    os.path.join(images_dir, f"sigma{sigma}_img{idx:03d}_lr.png"),
                    bicubic_to_hr(lr50, hr_size),
                )

            if (idx + 1) % 10 == 0 or (idx + 1) == total_test:
                print(f"Sigma {sigma}: processed {idx + 1}/{total_test}")

        for model_name, vals in totals.items():
            count = max(vals["count"], 1)
            avg_psnr = vals["psnr"] / count
            avg_ssim = vals["ssim"] / count

            summary_rows.append([sigma, model_name, avg_psnr, avg_ssim, count])

            print(
                f"Sigma={sigma} | {model_name} -> "
                f"PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f} | N={count}"
            )

    eval_csv = os.path.join(results_root, "eval_metrics.csv")
    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "model", "avg_psnr", "avg_ssim", "num_images"])
        writer.writerows(summary_rows)

    print("\nFinished run_test_dataset()")
    print(f"Evaluation CSV: {eval_csv}")
    print(f"Saved images folder: {images_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths_cfg = CONFIG["paths"]
    bi_cfg = CONFIG["bichannel"]
    test_cfg = CONFIG["run_test_dataset"]
    sc2_cfg = CONFIG["sc2"]

    train_path = resolve_path(paths_cfg["train_dir"])
    val_path = resolve_path(paths_cfg["val_dir"])
    test_path = resolve_path(paths_cfg["test_dir"])
    checkpoints_dir = resolve_path(paths_cfg["trained_models_dir"])
    results_root = resolve_path(paths_cfg["results_dir"])

    ensure_dir(checkpoints_dir)
    ensure_dir(results_root)
    ensure_dir(os.path.join(results_root, "images"))

    hr_size = tuple(sc2_cfg["hr_size"])
    bichannel_ckpt_path = os.path.join(checkpoints_dir, bi_cfg["checkpoint_name"])

    model = BiChannelCNN().to(device)

    mean = None
    std = None

    if bi_cfg["load_if_exists"] and os.path.exists(bichannel_ckpt_path):
        model, mean, std = load_bichannel_checkpoint(model, bichannel_ckpt_path, device)

    if mean is None or std is None:
        print("Computing train-set mean/std...")
        mean, std = compute_mean_std(train_path, image_size=hr_size)
        print(f"Mean: {mean}")
        print(f"Std: {std}")

    train_dataset = FaceDataset(train_path, mean=mean, std=std)
    val_dataset = FaceDataset(val_path, mean=mean, std=std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bi_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bi_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    if not (bi_cfg["load_if_exists"] and os.path.exists(bichannel_ckpt_path)):
        print("Training BiChannel CNN...")

        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=bi_cfg["lr"],
            momentum=0.9,
            weight_decay=bi_cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=bi_cfg["patience"],
            min_lr=bi_cfg["min_lr"],
        )

        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            bi_cfg["num_epochs"],
            scheduler,
        )

        if bi_cfg["save_after_train"]:
            save_bichannel_checkpoint(model, mean, std, bichannel_ckpt_path)

    print("\nRunning test dataset evaluation...")
    run_test_dataset(
        bichannel_model=model,
        mean=mean,
        std=std,
        device=device,
        train_dir=train_path,
        test_dir=test_path,
        sigmas=test_cfg["sigmas"],
        classical_train_max_samples=test_cfg["classical_train_max_samples"],
        test_max_samples=test_cfg["test_max_samples"],
        save_first_n=test_cfg["save_first_n"],
    )


if __name__ == "__main__":
    main()