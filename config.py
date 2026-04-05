import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "paths": {
        "root_dir": ROOT_DIR,
        "train_dir": os.path.join(ROOT_DIR, "data", "train"),
        "val_dir": os.path.join(ROOT_DIR, "data", "val"),
        "test_dir": os.path.join(ROOT_DIR, "data", "test"),
        "outputs_root": os.path.join(ROOT_DIR, "outputs"),
        "results_dir": os.path.join(ROOT_DIR, "results"),
        "trained_models_dir": os.path.join(ROOT_DIR, "checkpoints"),
    },

    "bichannel": {
        "batch_size": 200,
        "num_epochs": 10,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 5e-4,
        "patience": 5,
        "checkpoint_name": "bichannel.pth",
        "load_if_exists": True,
        "save_after_train": True,
        "num_workers": 4
    },

    "sc1": {
        "patch": 5,
        "stride": 2,
        "n_atoms": 256,
        "lasso_alpha": 0.001,
        "lasso_max_iter": 5000,
        "lasso_tol": 1e-4,
        "dl_alpha": 1.0,
        "dl_iter": 100,
        "max_train_patches": 60000,
        "per_image_patch_cap": 1500,
        "random_state": 42,
        "checkpoint_name": "sc1.pkl",
        "load_if_exists": True,
        "save_after_train": True,
    },

    "sc2": {
        "patch_size": 5,
        "sigma_k": 0.05,
        "lam": 5e-8,
        "n_basis": 300,
        "n_train": 10000,
        "hr_size": (100, 100),
        "lr_size": (50, 50),
        "seed": 42,
        "checkpoint_name": "sc2.pkl",
        "load_if_exists": True,
        "save_after_train": True,
    },

    "sfh": {
        "max_exemplars": 500,
        "hr_size": (100, 100),
        "lr_size": (50, 50),
        "seed": 42,
        "checkpoint_name": "sfh.pkl",
        "load_if_exists": True,
        "save_after_train": True,
    },

    "run_test_dataset": {
        "sigmas": (1, 3, 5),
        "classical_train_max_samples": 300,
        "test_max_samples": 500,
        "save_first_n": 5,
        "hr_size": (100, 100),
        "lr_size": (50, 50),
    },
}