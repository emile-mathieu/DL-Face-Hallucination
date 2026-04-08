import os
import numpy as np
import torch
import cv2
from reconstruction.reconstruction import (
    SR1_SC1_Yang_FeatureResidual,
    SR2_SC2_Kim_KRR,
    SR3_SFH,
)
from utils.utils import(
    ensure_dir,
    load_pickle_model,
    save_pickle_model,
    psnr,
    ssim,
    save_rgb_image
)



# Classical models builder

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



# Helper methods

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



# Train the classical models

def classical_train(
        classical_train_ds,
        classical_cfg,
        checkpoints_dir,
        classical_train_max_samples,
    ):

    sc1_cfg, sc2_cfg, sfh_cfg = classical_cfg
    hr_size = tuple(sc2_cfg["hr_size"])
    lr_size = tuple(sc2_cfg["lr_size"])

    ensure_dir(checkpoints_dir)


    # ---------------------------------------------------------
    # PHASE 1: CLASSICAL MODEL PREPARATION (SC1, SC2, SFH)
    # ---------------------------------------------------------
    sc1_path = os.path.join(checkpoints_dir, sc1_cfg["checkpoint_name"])
    sc2_path = os.path.join(checkpoints_dir, sc2_cfg["checkpoint_name"])
    sfh_path = os.path.join(checkpoints_dir, sfh_cfg["checkpoint_name"])

    # Check if we need to train the classical models
    need_train = not (os.path.exists(sc1_path) and os.path.exists(sc2_path) and os.path.exists(sfh_path))

    if need_train:
        print("Collecting training pairs for Classical Models...")
        train_lr100, train_hr100 = collect_training_pairs(
            classical_train_ds,
            hr_size=hr_size,
            max_samples=classical_train_max_samples,
        )

    # SC1 Build/Load
    if sc1_cfg["load_if_exists"] and os.path.exists(sc1_path):
        sc1 = load_pickle_model(sc1_path, "SC1")
    else:
        sc1 = build_sc1(sc1_cfg)
        sc1.fit(train_lr100, train_hr100)
        if sc1_cfg["save_after_train"]: save_pickle_model(sc1, sc1_path, "SC1")

    # SC2 Build/Load
    if sc2_cfg["load_if_exists"] and os.path.exists(sc2_path):
        sc2 = load_pickle_model(sc2_path, "SC2")
    else:
        sc2 = build_sc2(sc2_cfg)
        sc2.fit(train_lr100, train_hr100)
        if sc2_cfg["save_after_train"]: save_pickle_model(sc2, sc2_path, "SC2")

    # SFH Build/Load
    if sfh_cfg["load_if_exists"] and os.path.exists(sfh_path):
        sfh = load_pickle_model(sfh_path, "SFH")
    else:
        sfh = build_sfh(sfh_cfg)
        sfh.fit(train_lr100, train_hr100)
        if sfh_cfg["save_after_train"]: save_pickle_model(sfh, sfh_path, "SFH")

    return sc1, sc2, sfh



# Evaluate classical methods for sigma

def classical_evaluate_sigma(
        classical_models,
        classical_test_ds,
        lr_size,
        hr_size,
        device,
        save_cfg
    ):

    sc1, sc2, sfh = classical_models
    num_classical = len(classical_test_ds)

    try:
        save_first_n = save_cfg["save_first_n"]
        images_dir = save_cfg["images_dir"]
        params = save_cfg["params"]
        sigma = params_val = save_cfg["params_val"]
    except:
        save_first_n = 0
        sigma = 2
    
    total_psnr = 0
    total_ssim = 0
    counts = 0

    print(f"Running Classical Evaluation (N={num_classical})...")
    for idx in range(num_classical):
        _, hr_img = classical_test_ds[idx]
        hr_img = np.clip(hr_img, 0.0, 1.0).astype(np.float32)

        # Generate synthetic inputs for classical models
        lr50 = make_lr_from_hr_sigma(hr_img, sigma=sigma, lr_size=lr_size)
        lr100 = bicubic_to_hr(lr50, hr_size=hr_size)

        preds = {
            "sc1": np.clip(sc1.predict(lr100), 0.0, 1.0),
            "sc2": np.clip(sc2.predict(lr50), 0.0, 1.0),
            "sfh": np.clip(sfh.predict(lr50), 0.0, 1.0),
        }

        hr_t = torch.from_numpy(hr_img).permute(2,0,1).unsqueeze(0).to(device)

        for name, img_np in preds.items():
            p_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
            total_psnr += psnr(p_t, hr_t)
            total_ssim += ssim(p_t, hr_t)
            counts += 1

            if idx < save_first_n:
                save_rgb_image(os.path.join(images_dir, f"{params}{params_val}_img{idx:03d}_{name}.png"), img_np)


    return total_psnr/counts, total_ssim/counts, counts


def classical_evaluate_motion(
        classical_models,
        classical_test_ds,
        lr_size,
        hr_size,
        device,
        save_cfg
    ):
    pass