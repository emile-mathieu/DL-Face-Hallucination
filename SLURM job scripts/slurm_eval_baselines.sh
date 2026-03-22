#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=eval_baselines
#SBATCH --time=06:00:00
#SBATCH --output=../logs/output4_%x_%j.out
#SBATCH --error=../logs/error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train and evaluate SC1, SC2, SFH on CelebA; print Table-2-style PSNR/SSIM.
# Uses CPU (sklearn/scipy); no GPU required. Optional: add --gres=gpu:1 if using face_alignment with CUDA for SFH.

parent_dir="$(dirname "$(pwd)")"
echo "$parent_dir"

WORKDIR=$parent_dir
DATA_ROOT=$parent_dir
SAVE_DIR=$parent_dir/checkpoints

cd "${WORKDIR}"
echo "PWD=$(pwd)"

mkdir -p logs

python CelebA2.py eval_baselines \
  --data_root "${DATA_ROOT}" \
  --seed 42 \
  --train_hr_limit 1200 \
  --val_hr_limit 0 \
  --test_hr_limit 300 \
  --train_pairs 600 \
  --sr1_atoms 256 \
  --sr2_anchors 8000 \
  --sr2_knn 64 \
  --metric_color_space y \
