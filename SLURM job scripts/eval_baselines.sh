#!/bin/sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=eval_baselines
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

# Train and evaluate SC1, SC2, SFH on CelebA; print Table-2-style PSNR/SSIM.
# Uses CPU (sklearn/scipy); no GPU required. Optional: add --gres=gpu:1 if using face_alignment with CUDA for SFH.

DATA_ROOT=/home/msds/tans0444/
mkdir -p logs

python CelebA1.py eval_baselines \
  --data_root "$DATA_ROOT" \
  --seed 42 \
  --train_hr_limit 1200 \
  --test_hr_limit 300 \
  --train_pairs 600 \
  --sr1_atoms 256 \
  --sr2_anchors 8000 \
  --sr2_knn 64
