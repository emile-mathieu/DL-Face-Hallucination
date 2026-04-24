#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=bichannel5
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail

: "${CUDA_MODULE:=cuda/12.8.0}"
: "${ANACONDA_MODULE:=anaconda}"
: "${CONDA_ENV:=DL2}"
: "${FH_WORKDIR:=/home/msds/tans0444}"
: "${FH_IMAGE_ROOT:=/home/msds/tans0444/celeba/img_align_celeba}"
: "${FH_CHECKPOINT_DIR:=/home/msds/tans0444/checkpoints}"

module load "$CUDA_MODULE"
module load "$ANACONDA_MODULE"

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR="$FH_WORKDIR"
SCRIPT_NAME=bichannel5.py
IMAGE_ROOT="$FH_IMAGE_ROOT"
CHECKPOINT_DIR="$FH_CHECKPOINT_DIR"

cd "$WORKDIR"

echo "PWD=$(pwd)"
echo "HOSTNAME=$(hostname)"
echo "START TIME=$(date)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "ERROR: $SCRIPT_NAME not found in $WORKDIR"
    exit 1
fi

LATEST_VALID_CKPT=""

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Searching for latest valid checkpoint in $CHECKPOINT_DIR"

    for ckpt in $(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'bichannel_epoch_*.pth' | sort -r); do
        echo "Checking checkpoint: $ckpt"
        if python - "$ckpt" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("Missing model_state")
    print(f"VALID: {path}")
except Exception as e:
    print(f"INVALID: {path} -> {e}")
    sys.exit(1)
PY
        then
            LATEST_VALID_CKPT="$ckpt"
            break
        fi
    done
fi

if [ -n "$LATEST_VALID_CKPT" ]; then
    echo "Resuming from latest valid checkpoint: $LATEST_VALID_CKPT"
    python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT" --resume "$LATEST_VALID_CKPT"
else
    echo "No valid checkpoint found. Starting from scratch."
    python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT"
fi

echo "END TIME=$(date)"