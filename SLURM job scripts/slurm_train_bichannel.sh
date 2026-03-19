#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=run8fh_train_bichannel
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train Bi-channel CNN on CelebA (mixed Gaussian + motion blur, paper-like).
# Requires: CelebA2.py in working dir; CelebA at DATA_ROOT.

WORKDIR=/home/msds/tans0444
DATA_ROOT=/home/msds/tans0444
SAVE_DIR=/home/msds/tans0444/checkpoints

cd "${WORKDIR}"
echo "PWD=$(pwd)"
nvidia-smi || true

mkdir -p logs "${SAVE_DIR}"

python CelebA2.py train \
  --data_root "${DATA_ROOT}" \
  --model bichannel \
  --blur_type mixed \
  --batch_size 200 \
  --epochs 5000 \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --device cuda \
  --save_dir "${SAVE_DIR}"
