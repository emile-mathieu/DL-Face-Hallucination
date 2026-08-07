"""
config.py
=========
All shared configuration constants for the face hallucination project.
Edit WORKDIR and paths here — nothing else needs changing to run on a
different cluster or local machine.
"""

from pathlib import Path

# ── Cluster paths ─────────────────────────────────────────────
WORKDIR             = Path("/home/msds/tans0444") #change this to your own workdir
IMAGE_DIR           = WORKDIR / "celeba" / "img_align_celeba"
SPLITS_DIR          = WORKDIR / "splits"
RESULTS_BASIC_DIR   = WORKDIR / "results_basiccnn"
RESULTS_BI_DIR      = WORKDIR / "results_bichannel"
RESULTS_CL_DIR      = WORKDIR / "results_classical"
BASIC_CKPT_DIR      = RESULTS_BASIC_DIR / "checkpoints"
BI_CKPT_DIR         = RESULTS_BI_DIR    / "checkpoints"
BASIC_BEST_CKPT     = RESULTS_BASIC_DIR / "best_model_basiccnn.pth"
BI_BEST_CKPT        = RESULTS_BI_DIR    / "best_model.pth"

# ── Dataset ───────────────────────────────────────────────────
SEED        = 42
MAX_IMAGES  = 100_000
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2
N_TEST      = 20000       # number of test images used in evaluation
                       # matches original bichannel5_9.py / basic5_9.py (N_TEST=50)
                       # increase to e.g. 1000 or 20000 for a full evaluation run

# ── Image sizes ───────────────────────────────────────────────
HR_SIZE             = (100, 100)   # ground-truth HR output
TRAIN_LR_INPUT_SIZE = (48, 48)     # network input (paper Table 1)
TEST_FIXED_LR_SIZE  = (50, 50)     # LR size after test degradation
TEST_NN_INPUT_SIZE  = (48, 48)     # resize test LR → network input

# ── CelebA crop ───────────────────────────────────────────────
CELEBA_CROP      = True
CROP_FRAC        = 0.60
CROP_TOP_OFFSET  = 0.15   # top = (h-side)//2 + int(side * 0.15)

# ── Metrics ───────────────────────────────────────────────────
METRIC_CHANNEL = "y"   # "y" = luminance (paper); "rgb" = full colour

# ── Deep model hyperparameters ────────────────────────────────
NUM_WORKERS   = 2 

BATCH_SIZE    = 32
NUM_EPOCHS    = 600
LEARNING_RATE = 5e-4
MIN_LR        = 1e-6
WEIGHT_DECAY  = 1e-3
GRAD_CLIP     = 1.0     
WARMUP_EPOCHS = 10
LR_T_MAX      = 590    # NUM_EPOCHS - WARMUP_EPOCHS
KEEP_LAST_N   = 3

# ── Classical model hyperparameters ──────────────────────────
CL_SEED           = 42
CL_N_TRAIN        = 200
CL_N_TEST         = 50
CL_USE_SPLITS     = True

SC1_PATCH_SIZE    = 5
SC1_DICT_SIZE     = 256
SC1_LAMBDA        = 0.1
SC1_N_TRAIN       = 200
SC1_ISTA_ITERS    = 30

SC2_PATCH_SIZE    = 5
SC2_SIGMA_K       = 0.12
SC2_LAMBDA_REG    = 0.05
SC2_N_BASIS       = 256
SC2_N_TRAIN       = 200

SFH_MAX_EXEMPLARS = 25

# ── Test blur conditions (shared by all methods) ──────────────
TEST_CONDITIONS = [
    dict(blur_type="gaussian", gaussian_sigma=1,  motion_length=None),
    dict(blur_type="gaussian", gaussian_sigma=3,  motion_length=None),
    dict(blur_type="gaussian", gaussian_sigma=5,  motion_length=None),
    dict(blur_type="motion",   gaussian_sigma=None, motion_length=2),
    dict(blur_type="motion",   gaussian_sigma=None, motion_length=6),
    dict(blur_type="motion",   gaussian_sigma=None, motion_length=9),
]
