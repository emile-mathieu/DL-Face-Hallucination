# Project Structure

```text
DL-Face-Hallucination/
├── readme.md
├── Project-Structure.md
├── requirements.txt
├── experiments/
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── dataloader.py
│   │   └── dataset.py
│   ├── models/
│   │   └── model.py
│   ├── reconstruction/
│   │   ├── bichannel5.9.py
│   │   ├── bichannel5.sh
│   │   ├── reconstruction.py
│   │   └── test.py
│   ├── splits_bichannel/
│   │   ├── split_summary.json
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   ├── training/
│   │   ├── classical.py
│   │   ├── inference.py
│   │   ├── test.py
│   │   └── train.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── utils.py
│   └── SLURM job scripts/
│       ├── readme.md
│       ├── slurm_eval_baselines.sh
│       ├── slurm_eval_gaussian.sh
│       ├── slurm_eval_motion.sh
│       ├── slurm_submit_all_three.sh
│       ├── slurm_train_basic.sh
│       └── slurm_train_bichannel.sh
└── results/
	├── classical.py
	├── config.py
	├── dataset.py
	├── environment.yml
	├── main.py
	├── models.py
	├── readme.txt
	├── run_classical.sh
	├── setup.sh
	├── setup_data.py
	├── test_models.sh
	├── train_basic.sh
	├── train_bichannel.sh
	├── trainer.py
	└── utils.py
```

## Overview
The repository has two layers: `results/` and `experiments/`.
## Main Folders
### `results/`
- `main.py` is the submission entry point.
- `config.py` contains the configuration settings.
- `dataset.py`, `trainer.py`, `models.py`, `classical.py`, and `utils.py` make up the main submission workflow.
- `setup.sh`, `setup_data.py`, and the shell scripts support setup and runs.
- `environment.yml` defines the conda environment.

### `experiments/`
- `main.py` is the experimental entry point.
- `config.py` contains the configuration settings.
- `data/`, `models/`, `training/`, and `reconstruction/` contain main bulding blocks for the experimental workflow.
- `SLURM job scripts/` contains cluster run scripts.

### More information
See the README files in each folder for run commands and setup notes.
