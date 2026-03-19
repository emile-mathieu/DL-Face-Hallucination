#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=run8fh_eval_gaussian
#SBATCH --time=06:00:00
#SBATCH --output=output4_%x_%j.out
#SBATCH --error=error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Evaluate a trained model under Gaussian blur (sigma=1,3,5).
# can switch between bichannel and basic by changing --model and checkpoint path accordingly
# can change --metric_color_space to rgb or y for different metrics (e.g. PSNR in RGB vs Y channel)

WORKDIR=/home/msds/tans0444
DATA_ROOT=/home/msds/tans0444
SAVE_DIR=/home/msds/tans0444/checkpoints
CKPT="${SAVE_DIR}/basic_mixed.pt"

cd "${WORKDIR}"
echo "PWD=$(pwd)"
echo "CKPT=${CKPT}"

mkdir -p logs

test -f "${CKPT}" || { echo "Checkpoint not found: ${CKPT}"; ls -lah "${SAVE_DIR}"; exit 1; }

python CelebA2.py eval_gaussian \
  --data_root "${DATA_ROOT}" \
  --model basic \
  --checkpoint "${CKPT}" \
  --device cuda \
  --batch_size 64 \
  --metric_color_space rgb \
  --seed 42
