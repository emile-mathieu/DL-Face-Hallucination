#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=run8fh_eval_gaussian
#SBATCH --time=06:00:00
#SBATCH --output=../logs/output4_%x_%j.out
#SBATCH --error=../logs/error4_%x_%j.err

module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Legacy wrapper name kept for compatibility.
# Uses main.py evaluation (includes Gaussian and motion outputs).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PY="${WORKDIR}/main.py"

if [[ ! -f "${MAIN_PY}" ]]; then
  echo "ERROR: main.py not found at ${MAIN_PY}"
  echo "DEBUG: SCRIPT_DIR=${SCRIPT_DIR}"
  echo "DEBUG: WORKDIR=${WORKDIR}"
  exit 1
fi

cd "${WORKDIR}"
echo "PWD=$(pwd)"
echo "Using entrypoint: ${MAIN_PY}"

mkdir -p "${WORKDIR}/logs"

python "${MAIN_PY}" --mode basic --task all
