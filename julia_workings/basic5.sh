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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
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

cd "$WORKDIR"

echo "PWD=$(pwd)"
echo "HOSTNAME=$(hostname)"
echo "START TIME=$(date)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT"

echo "END TIME=$(date)"
