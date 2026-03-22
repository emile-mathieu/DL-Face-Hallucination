#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=show_sample
#SBATCH --time=06:00:00
#SBATCH --output=../logs/output4_%x_%j.out
#SBATCH --error=../logs/error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Save one sample HR image, degraded images, and method outputs.

parent_dir="$(dirname "$(pwd)")"
echo "$parent_dir"

WORKDIR=$parent_dir
DATA_ROOT=$parent_dir
SAVE_DIR=$parent_dir/checkpoints
OUTPUT_DIR=/home/msds/tans0444/sample_images
BICHANNEL_CKPT="${SAVE_DIR}/bichannel_mixed.pt"
BASIC_CKPT="${SAVE_DIR}/basic_mixed.pt"

cd "${WORKDIR}"
echo "PWD=$(pwd)"
echo "BICHANNEL_CKPT=${BICHANNEL_CKPT}"
echo "BASIC_CKPT=${BASIC_CKPT}"

mkdir -p logs "${OUTPUT_DIR}"

python CelebA2.py show_sample \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --sigma 3.0 \
  --length 6.0 \
  --bichannel_checkpoint "${BICHANNEL_CKPT}" \
  --basic_checkpoint "${BASIC_CKPT}" \
  --seed 42
