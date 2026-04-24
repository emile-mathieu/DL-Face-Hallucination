"""Centralized configuration for julia_workings Python scripts."""

import os
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
	"""Load simple KEY=VALUE pairs from a .env file into process env.

	Existing environment variables are preserved.
	"""
	if not dotenv_path.exists():
		return

	for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key:
			os.environ.setdefault(key, value)


_THIS_DIR = Path(__file__).resolve().parent
_load_dotenv(_THIS_DIR / ".env")


# Shared environment-driven paths used by both bash and Python.
COMMON_WORKDIR = Path(os.getenv("FH_WORKDIR", "/home/msds/tans0444"))
COMMON_IMAGE_ROOT = Path(
	os.getenv("FH_IMAGE_ROOT", str(COMMON_WORKDIR / "celeba" / "img_align_celeba"))
)
COMMON_CHECKPOINT_DIR = Path(
	os.getenv(
		"FH_CHECKPOINT_DIR",
		str(COMMON_WORKDIR / "results_bichannel" / "checkpoints"),
	)
)

_BASIC_WORKDIR = COMMON_WORKDIR
_BICHANNEL_WORKDIR = COMMON_WORKDIR
_CLASSICAL_WORKDIR = COMMON_WORKDIR


BASIC_CONFIG = {
	"WORKDIR": _BASIC_WORKDIR,
	"DEFAULT_IMAGE_ROOTS": [COMMON_IMAGE_ROOT],
	"RESULTS_DIR": _BASIC_WORKDIR / "results_basiccnn",
	"SPLITS_DIR": _BASIC_WORKDIR / "splits_basiccnn",
	"CHECKPOINT_DIR": _BASIC_WORKDIR / "results_basiccnn" / "checkpoints",
	"SEED": 42,
	"MAX_IMAGES": 100_000,
	"TRAIN_RATIO": 0.6,
	"VAL_RATIO": 0.2,
	"TEST_RATIO": 0.2,
	"HR_SIZE": (100, 100),
	"TRAIN_LR_INPUT_SIZE": (48, 48),
	"TEST_FIXED_LR_SIZE": (50, 50),
	"TEST_NN_INPUT_SIZE": (48, 48),
	"CELEBA_CROP": True,
	"CROP_FRAC": 0.60,
	"METRIC_CHANNEL": "y",
	"BATCH_SIZE": 32,
	"NUM_EPOCHS": 600,
	"LEARNING_RATE": 5e-4,
	"MIN_LR": 1e-6,
	"WEIGHT_DECAY": 1e-3,
	"GRAD_CLIP": 1.0,
	"NUM_WORKERS": 2,
	"WARMUP_EPOCHS": 10,
	"LR_T_MAX": 590,
	"KEEP_LAST_N": 3,
}


BICHANNEL_CONFIG = {
	"WORKDIR": _BICHANNEL_WORKDIR,
	"DEFAULT_IMAGE_ROOTS": [COMMON_IMAGE_ROOT],
	"RESULTS_DIR": _BICHANNEL_WORKDIR / "results_bichannel",
	"SPLITS_DIR": _BICHANNEL_WORKDIR / "splits_bichannel",
	"CHECKPOINT_DIR": COMMON_CHECKPOINT_DIR,
	"SEED": 42,
	"MAX_IMAGES": 100_000,
	"TRAIN_RATIO": 0.6,
	"VAL_RATIO": 0.2,
	"TEST_RATIO": 0.2,
	"HR_SIZE": (100, 100),
	"TRAIN_LR_INPUT_SIZE": (48, 48),
	"TEST_FIXED_LR_SIZE": (50, 50),
	"TEST_NN_INPUT_SIZE": (48, 48),
	"CELEBA_CROP": True,
	"CROP_FRAC": 0.6,
	"METRIC_CHANNEL": "y",
	"BATCH_SIZE": 32,
	"NUM_EPOCHS": 600,
	"LEARNING_RATE": 5e-4,
	"MIN_LR": 1e-6,
	"WEIGHT_DECAY": 1e-3,
	"GRAD_CLIP": 1.0,
	"MOMENTUM": 0.9,
	"PATIENCE": 10,
	"NUM_WORKERS": 2,
	"WARMUP_EPOCHS": 10,
	"LR_T_MAX": 590,
	"KEEP_LAST_N": 3,
}


CLASSICAL_CONFIG = {
	"WORKDIR": _CLASSICAL_WORKDIR,
	"SPLITS_DIR": _CLASSICAL_WORKDIR / "splits_basiccnn",
	"RESULTS_DIR": _CLASSICAL_WORKDIR / "results_classical",
	"IMAGE_DIR": COMMON_IMAGE_ROOT,
	"SEED": 42,
	"HR_SIZE": (100, 100),
	"TEST_LR_SIZE": (50, 50),
	"CELEBA_CROP": True,
	"CROP_FRAC": 0.6,
	"max_images": 100,
	"N_TRAIN": 200,
	"N_TEST": 50,
	"USE_SPLITS": True,
	"SC1_PATCH_SIZE": 5,
	"SC1_DICT_SIZE": 256,
	"SC1_LAMBDA": 0.1,
	"SC1_N_TRAIN": 200,
	"SC1_ISTA_ITERS": 30,
	"SC2_PATCH_SIZE": 5,
	"SC2_SIGMA_K": 0.12,
	"SC2_LAMBDA_REG": 0.05,
	"SC2_N_BASIS": 256,
	"SC2_N_TRAIN": 200,
	"SFH_MAX_EXEMPLARS": 25,
}
