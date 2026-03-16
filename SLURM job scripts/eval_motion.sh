#!/bin/sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=eval_motion
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

# Evaluate a trained model under motion blur (l=2,6,9).
# Set CHECKPOINT to your saved .pt file (e.g. checkpoints/bichannel_mixed.pt). Change model accordinly

DATA_ROOT=/home/msds/tans0444/
CHECKPOINT=checkpoints/bichannel_mixed.pt

mkdir -p logs

python CelebA1.py eval_motion \
  --data_root "$DATA_ROOT" \
  --model bichannel \ 
  --checkpoint "$CHECKPOINT" \
  --batch_size 64
