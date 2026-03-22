#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=train_basic
#SBATCH --time=06:00:00
#SBATCH --output=../logs/output4_%x_%j.out
#SBATCH --error=../logs/error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train Basic CNN (no fusion) on CelebA for ablation.

parent_dir="$(dirname "$(pwd)")"
echo "$parent_dir"

WORKDIR=$parent_dir
DATA_ROOT=$parent_dir
SAVE_DIR=$parent_dir/checkpoints

cd "${WORKDIR}"
echo "PWD=$(pwd)"
nvidia-smi || true

mkdir -p logs "${SAVE_DIR}"

python CelebA2.py train \
  --data_root "${DATA_ROOT}" \
  --model basic \
  --blur_type mixed \
  --batch_size 200 \
  --epochs 25 \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --device cuda \
  --save_dir "${SAVE_DIR}"
