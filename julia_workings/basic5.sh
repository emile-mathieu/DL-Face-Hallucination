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

module load cuda/12.8.0
module load anaconda

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate DL2 #change your environment 

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/home/msds/tans0444 #change to your own directory
SCRIPT_NAME=basic5.py
IMAGE_ROOT=/home/msds/tans0444/celeba/img_align_celeba #change to your own directory 

cd "$WORKDIR"

echo "PWD=$(pwd)"
echo "HOSTNAME=$(hostname)"
echo "START TIME=$(date)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python "$SCRIPT_NAME" --image-root "$IMAGE_ROOT"

echo "END TIME=$(date)"
