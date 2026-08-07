Experiments Guide

# Main Entry Point
The experimental pipeline starts from `experiments/main.py`.

# Model Modes
- basic
- bichannel

# Task Modes
- "all": run the CNN path and the classical path
- "train-cnn": run only the CNN training/loading path
- "classical-only": run only the classical training/loading and evaluation path

# Run Commands
# Local run examples
python experiments/main.py --mode basic --task train-cnn
python experiments/main.py --mode bichannel --task train-cnn
python experiments/main.py --mode bichannel --task classical-only
python experiments/main.py --mode bichannel --task all


### SLURM run
Core scripts are in experiments/SLURM job scripts/:
- slurm_train_basic.sh
- slurm_train_bichannel.sh
- slurm_eval_baselines.sh
- slurm_eval_gaussian.sh
- slurm_eval_motion.sh
- slurm_submit_all_three.sh

Recommended single-command submission:
bash "experiments/SLURM job scripts/slurm_submit_all_three.sh"

## Data Requirements
The pipeline expects split folders:
- experiments/data/train
- experiments/data/val
- experiments/data/test
