#!/bin/sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=train_basic
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

# Train Basic CNN (no fusion) on CelebA for ablation.

DATA_ROOT=/home/msds/tans0444/
SAVE_DIR=checkpoints
mkdir -p logs $SAVE_DIR

python CelebA1.py train \
  --data_root "$DATA_ROOT" \
  --model basic \
  --blur_type mixed \
  --batch_size 200 \
  --epochs 5000 \
  --lr 1e-5 \
  --save_dir "$SAVE_DIR"
