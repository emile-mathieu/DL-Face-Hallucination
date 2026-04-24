#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=basic5
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail

ENV_FILE="$SLURM_SUBMIT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
	set -a
	source "$ENV_FILE"
	set +a
fi

: "${CUDA_MODULE:=cuda/12.8.0}"
: "${ANACONDA_MODULE:=anaconda}"
: "${CONDA_ENV:=DL2}"
: "${FH_WORKDIR:=/home/msds/tans0444}"
: "${FH_IMAGE_ROOT:=${FH_WORKDIR}/celeba/img_align_celeba}"
: "${FH_BASIC_CHECKPOINT:=}"

module load "$CUDA_MODULE"
module load "$ANACONDA_MODULE"

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate "$CONDA_ENV" #change your environment 

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR="$FH_WORKDIR" #change to your own directory
SCRIPT_NAME=BasicCNN.py
IMAGE_ROOT="$FH_IMAGE_ROOT" #change to your own directory 
BASIC_CHECKPOINT="$FH_BASIC_CHECKPOINT"

LATEST_VALID_CKPT=""
if [ -d "$BASIC_CHECKPOINT" ]; then
    echo "Searching for latest valid checkpoint in $BASIC_CHECKPOINT"

    for ckpt in $(find "$BASIC_CHECKPOINT/checkpoints" -name 'basiccnn_epoch_*.pth' | sort -r); do
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

cd "$WORKDIR"

echo "PWD=$(pwd)"
echo "HOSTNAME=$(hostname)"
echo "START TIME=$(date)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

if [ -n "$LATEST_VALID_CKPT" ]; then
	if [ ! -f "$LATEST_VALID_CKPT" ]; then
		echo "ERROR: checkpoint not found: $LATEST_VALID_CKPT"
		exit 1
	fi
	echo "Resuming BasicCNN from checkpoint: $LATEST_VALID_CKPT"
	python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT" --resume "$LATEST_VALID_CKPT"
else
	python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT"
fi

echo "END TIME=$(date)"
