# Project Structure

```text
Deep-Learning-Face-Hallucination/
├── requirements.txt
├── readme.md
├── Project-Structure.md
├── common_config.py
├── BasicCNN.py
├── basiccnn.sh
├── BiChannelCNN.py
├── bichannel.sh
├── ClassicalCNN.py
│
├── setup/
│   ├── environment.yml
│   ├── setup.sh
│   └── setup_data.py
│
├── experiments/
│   ├── config.py
│   ├── main.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── dataloader.py
│   │
│   ├── models/
│   │   └── model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── inference.py
│   │   └── classical.py
│   │
│   ├── reconstruction/
│   │   ├── reconstruction.py
│   │   ├── test.py
│   │   └── bichannel5.9.py
│   │   └── bichannel5.sh
│   │
│   ├── utils/
│   │   ├── utils.py
│   │   └── logger.py
│   │
│   ├── splits_bichannel/
│   │   ├── split_summary.json
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   │
│   └── SLURM job scripts/
│       ├── readme.md
│       ├── slurm_train_basic.sh
│       ├── slurm_train_bichannel.sh
│       ├── slurm_eval_baselines.sh
│       ├── slurm_eval_gaussian.sh
│       ├── slurm_eval_motion.sh
│       └── slurm_submit_all_three.sh
│
├── checkpoints/                # generated at runtime
└── results/                    # generated at runtime
```

## Core Modules

### `experiments/main.py`
Unified pipeline entry point.
- Selects mode: `basic` or `bichannel`
- Selects task: `all`, `train-cnn`, `classical-only`
- Coordinates training, evaluation, checkpoint loading, and CSV export

### `experiments/config.py`
Centralized config for:
- data/checkpoint/result paths
- Basic/BiChannel hyperparameters
- scheduler settings
- SC1/SC2/SFH settings
- test settings (sigma, motion length, sample limits)

### `experiments/data/`
- `dataset.py`: face dataset loading, degradation (gaussian/motion), per-image normalization
- `dataloader.py`: train/val/test/classical dataloader builders

### `experiments/models/`
- `model.py`: `BasicCNN` and `BiChannelCNN`

### `experiments/training/`
- `train.py`: CNN training loop, warmup-cosine scheduler, checkpoint rolling
- `test.py`: CNN test-time evaluation helpers
- `inference.py`: utility functions for sample output inference
- `classical.py`: build/load/train/eval orchestration for SC1/SC2/SFH

### `experiments/reconstruction/`
- `reconstruction.py`: classical super-resolution methods and shared routines
- `test.py`: reconstruction testing helpers

### `experiments/utils/`
- `utils.py`: metrics, checkpoint IO, pickle model IO, image saving
- `logger.py`: CSV metric logging helper

## Pipeline Overview

### CNN path
1. Build dataloaders from `experiments/data/`
2. Build model from `experiments/models/model.py`
3. Train/load via `experiments/training/train.py`
4. Evaluate via `experiments/training/test.py`
5. Save model and metrics

### Classical path
1. Build/load SC1, SC2, SFH via `experiments/training/classical.py`
2. Use implementations from `experiments/reconstruction/reconstruction.py`
3. Evaluate on configured test settings
4. Save metrics and images

## Input / Output Shapes
- Input (LR): `(B, 3, 48, 48)`
- Output (HR): `(B, 3, 100, 100)`
