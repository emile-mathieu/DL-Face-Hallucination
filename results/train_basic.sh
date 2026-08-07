#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=train_basic
#SBATCH --time=6:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# ============================================================
# train_basic.sh
# Trains BasicCNN (600 epochs, AdamW + warmup-cosine).
# Expected runtime: ~48 hours
#
# Submit:  sbatch train_basic.sh
# Resume:  set RESUME_CKPT below, then resubmit.
# ============================================================

set -eo pipefail

module load cuda/12.8.0
module load anaconda

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate DL2 # Adjust if your environment name differs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ─────────────────────────────────────────────────────
WORKDIR=/home/msds/tans0444 # Adjust if your code is in a different directory
IMAGE_ROOT=/home/msds/tans0444/celeba/img_align_celeba  #Adjust if your images are stored elsewhere
CKPT_DIR=/home/msds/tans0444/results_basiccnn/checkpoints # Adjust if you want checkpoints saved to a different location

# Leave empty to start from scratch; set to a checkpoint path to resume.
RESUME_CKPT=""

cd "$WORKDIR"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Mode      : train_basic"
echo "========================================"

python -c "import torch; \
    print('PyTorch:', torch.__version__); \
    print('CUDA:',    torch.cuda.is_available()); \
    print('GPU:',     torch.cuda.get_device_name(0) \
          if torch.cuda.is_available() else 'N/A')"

# ── Auto-detect latest valid checkpoint if RESUME_CKPT not set ──
if [ -z "$RESUME_CKPT" ] && [ -d "$CKPT_DIR" ]; then
    echo "Searching for latest valid checkpoint in $CKPT_DIR ..."
    for ckpt in $(find "$CKPT_DIR" -maxdepth 1 -name 'basiccnn_epoch_*.pth' | sort -r); do
        if python - "$ckpt" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(ckpt, dict) or "model_state" not in ckpt:
    sys.exit(1)
PY
        then
            RESUME_CKPT="$ckpt"
            echo "Found checkpoint: $RESUME_CKPT"
            break
        fi
    done
fi

# ── Build command ─────────────────────────────────────────────
CMD="python main.py --mode train_basic --image-root $IMAGE_ROOT"
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    CMD="$CMD --resume $RESUME_CKPT"
else
    echo "Starting from scratch."
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"
