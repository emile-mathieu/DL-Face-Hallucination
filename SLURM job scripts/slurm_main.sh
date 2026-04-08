#!/bin/bash
#SBATCH --job-name=main_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --partition=MGPU-TC2

#SBATCH --output=../logs/output_%x_%j.out
#SBATCH --error=../logs/error_%x_%j.err

# -------------------------
# Environment setup
# -------------------------
module load anaconda
eval "$(conda shell.bash hook)"
conda activate DL2

echo "=============================="
echo "SLURM JOB STARTED"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Working dir (before cd): $(pwd)"
echo "=============================="

# -------------------------
# Navigate to project root
# -------------------------
cd ..

echo "Working dir (after cd): $(pwd)"

# -------------------------
# Check GPU
# -------------------------
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python

# (Optional) sometimes not available on compute nodes
nvidia-smi || echo "nvidia-smi not available"

# -------------------------
# Run main pipeline
# -------------------------
echo "Running main.py..."
python -u main.py bichannel

# -------------------------
# Done
# -------------------------
echo "=============================="
echo "JOB FINISHED"
echo "Date: $(date)"
echo "=============================="