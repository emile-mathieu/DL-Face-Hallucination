#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=test_models
#SBATCH --time=01:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# ============================================================
# test_models.sh
# Tests either BasicCNN or BiChannelCNN on all 6 blur conditions.
# Runtime: ~5-10 minutes (inference only).
#
# Submit to test BasicCNN:
#   sbatch test_models.sh basic
#
# Submit to test BiChannelCNN:
#   sbatch test_models.sh bichannel
#
# Optionally override the checkpoint:
#   sbatch test_models.sh basic /path/to/custom.pth
# ============================================================

set -eo pipefail

module load cuda/12.8.0
module load anaconda

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate DL2 # replace with your actual environment name

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Arguments ─────────────────────────────────────────────────
# $1 = "basic" or "bichannel" (default: bichannel)
# $2 = optional path to checkpoint (default: best checkpoint for the model)
MODEL=${1:-bichannel}
CUSTOM_CKPT=${2:-""}

WORKDIR=/home/msds/tans0444 # replace with your actual working directory
IMAGE_ROOT=/home/msds/tans0444/celeba/img_align_celeba # replace with your actual image directory

cd "$WORKDIR"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Model     : $MODEL"
echo "========================================"

python -c "import torch; \
    print('PyTorch:', torch.__version__); \
    print('CUDA:',    torch.cuda.is_available()); \
    print('GPU:',     torch.cuda.get_device_name(0) \
          if torch.cuda.is_available() else 'N/A')"

# ── Validate model argument ───────────────────────────────────
if [ "$MODEL" != "basic" ] && [ "$MODEL" != "bichannel" ]; then
    echo "ERROR: First argument must be 'basic' or 'bichannel'"
    echo "Usage: sbatch test_models.sh [basic|bichannel] [optional_ckpt_path]"
    exit 1
fi

# ── Build command ─────────────────────────────────────────────
if [ "$MODEL" == "basic" ]; then
    MODE="test_basic"
    DEFAULT_CKPT=/home/msds/tans0444/results_basiccnn/best_model_basiccnn.pth
else
    MODE="test_bichannel"
    DEFAULT_CKPT=/home/msds/tans0444/results_bichannel/best_model.pth
fi

CKPT="${CUSTOM_CKPT:-$DEFAULT_CKPT}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    exit 1
fi

echo "Checkpoint: $CKPT"
CMD="python main.py --mode $MODE --image-root $IMAGE_ROOT --model-path $CKPT"
echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"
