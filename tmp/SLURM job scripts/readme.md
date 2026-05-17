Core SLURM scripts (cleaned)

1. slurm_train_basic.sh
- Train Basic CNN.

2. slurm_train_bichannel.sh
- Train BiChannel CNN.

3. slurm_eval_baselines.sh
- Train/evaluate classical methods (SC1, SC2, SFH).

4. slurm_eval_gaussian.sh
- Evaluate trained CNN checkpoint on Gaussian settings.

5. slurm_eval_motion.sh
- Evaluate trained CNN checkpoint on motion settings.

6. slurm_submit_all_three.sh
- Orchestrator script: submits
	- Basic training
	- BiChannel training
	- Classical baseline evaluation (after both training jobs succeed)

Recommended usage

- Single-command core pipeline:
	- bash "SLURM job scripts/slurm_submit_all_three.sh"

- Optional standalone evaluation jobs:
	- sbatch "SLURM job scripts/slurm_eval_gaussian.sh"
	- sbatch "SLURM job scripts/slurm_eval_motion.sh"