#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=train_bichannel
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# ============================================================
# train_bichannel.sh
# Trains BiChannelCNN (600 epochs, AdamW + warmup-cosine).
# Loads BasicCNN pretrained weights before training starts.
# MUST run train_basic.sh first.
# Duration: 48 hours 
#
# Submit:  sbatch train_bichannel.sh
# Resume:  set RESUME_CKPT below, then resubmit.
# ============================================================

set -eo pipefail

module load cuda/12.8.0
module load anaconda

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate DL2 #Adjust to your own environment name if different

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ─────────────────────────────────────────────────────
WORKDIR=/home/msds/tans0444 # Adjust to your own working directory if different
IMAGE_ROOT=/home/msds/tans0444/celeba/img_align_celeba # Adjust to your own image root if different
BASIC_CKPT=/home/msds/tans0444/results_basiccnn/best_model_basiccnn.pth # Adjust to your own BasicCNN checkpoint path if different
BI_CKPT_DIR=/home/msds/tans0444/results_bichannel/checkpoints # Adjust to your own BiChannelCNN checkpoint directory if different

# Leave empty to start from scratch; set to a checkpoint path to resume.
RESUME_CKPT=""

cd "$WORKDIR"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Mode      : train_bichannel"
echo "========================================"

python -c "import torch; \
    print('PyTorch:', torch.__version__); \
    print('CUDA:',    torch.cuda.is_available()); \
    print('GPU:',     torch.cuda.get_device_name(0) \
          if torch.cuda.is_available() else 'N/A')"

# ── Check BasicCNN checkpoint ─────────────────────────────────
if [ ! -f "$BASIC_CKPT" ]; then
    echo "WARNING: BasicCNN checkpoint not found at $BASIC_CKPT"
    echo "         BiChannelCNN will train from scratch (not recommended)."
fi

# ── Auto-detect latest valid bichannel checkpoint ─────────────
if [ -z "$RESUME_CKPT" ] && [ -d "$BI_CKPT_DIR" ]; then
    echo "Searching for latest valid checkpoint in $BI_CKPT_DIR ..."
    for ckpt in $(find "$BI_CKPT_DIR" -maxdepth 1 -name 'bichannel_epoch_*.pth' | sort -r); do
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
CMD="python main.py --mode train_bichannel \
     --image-root $IMAGE_ROOT \
     --basic-ckpt $BASIC_CKPT"

if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    CMD="$CMD --resume $RESUME_CKPT"
else
    echo "Starting from scratch (with BasicCNN pretrain transfer)."
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"
