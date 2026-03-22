#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=eval_motion
#SBATCH --time=06:00:00
#SBATCH --output=../logs/output4_%x_%j.out
#SBATCH --error=../logs/error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Evaluate a trained model under motion blur (l=2,6,9).
# Set CHECKPOINT to your saved .pt file (e.g. checkpoints/bichannel_mixed.pt or checkpoints/basic_mixed.pt). Change model accordinly
# Can change --metric_color_space to rgb or y for different metrics (e.g. PSNR in RGB vs Y channel). Note that SFH is only computed in Y channel, so use --metric_color_space y for that.

parent_dir="$(dirname "$(pwd)")"
echo "$parent_dir"

WORKDIR=$parent_dir
SAVE_DIR=$parent_dir/checkpoints
CKPT="${SAVE_DIR}/basic_mixed.pt"

cd "${WORKDIR}"
echo "PWD=$(pwd)"
echo "CKPT=${CKPT}"

test -f "${CKPT}" || { echo "Checkpoint not found: ${CKPT}"; ls -lah "${SAVE_DIR}"; exit 1; }

python CelebA2.py eval_motion \
  --data_root /home/msds/tans0444 \
  --model basic \
  --checkpoint "${CKPT}" \
  --device cuda \
  --batch_size 32 \
  --metric_color_space rgb \
  --seed 42