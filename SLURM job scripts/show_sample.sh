#!/bin/sh
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=show_sample
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

# Save one sample HR image and its Gaussian/motion blurred versions.

DATA_ROOT=/home/msds/tans0444/
OUTPUT_DIR=sample_images
mkdir -p logs $OUTPUT_DIR

python CelebA1.py show_sample \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --sigma 3.0 \
  --length 6.0
