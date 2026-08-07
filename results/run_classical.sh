#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=classical
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# ============================================================
# run_classical.sh
# Runs Bicubic, SC1, SC2, and SFH on all 6 blur conditions.
# Pure CPU job — no GPU needed (gres=gpu:0).
# Expected runtime: 15 minutes
#
# Key settings in config.py that affect runtime:
#   CL_N_TRAIN   = 200   (training images for SC1/SC2/SFH)
#   CL_N_TEST    = 50    (test images per condition)
#   SC1_DICT_SIZE = 256  
#   SC2_N_BASIS   = 256  
#
# Submit: sbatch run_classical.sh
# ============================================================

set -eo pipefail

module load anaconda

eval "$(conda shell.bash hook)"
export QT_XCB_GL_INTEGRATION=""
conda activate DL2 # replace with your actual conda environment name

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OPENCV_FORK_PARAMS=1 

WORKDIR=/home/msds/tans0444 # replace with your actual working directory
IMAGE_ROOT=/home/msds/tans0444/celeba/img_align_celeba # replace with the actual path to the CelebA images on your system

cd "$WORKDIR"

echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Mode      : classical"
echo "========================================"

python -c "import cv2, numpy, sklearn; \
    print('cv2:', cv2.__version__); \
    print('numpy:', numpy.__version__); \
    print('sklearn:', sklearn.__version__)"

echo ""
python main.py --mode classical --image-root "$IMAGE_ROOT"

echo ""
echo "========================================"
echo "Finished : $(date)"
echo "========================================"
