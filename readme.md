# AI6103 Group Project: Learning Face Hallucination in the Wild

This repository contains our organized implementation for face hallucination (single-image face super-resolution), based on the AAAI 2015 paper.

## Reference Paper
**Learning Face Hallucination in the Wild**  
Erjin Zhou, Haoqiang Fan, Zhimin Cao, Yuning Jiang, Qi Yin  
Proceedings of AAAI 2015  
https://ojs.aaai.org/index.php/AAAI/article/view/9795

## Implemented Methods
1. Bicubic baseline
2. Basic CNN
3. BiChannel CNN
4. SC1 classical baseline
5. SC2 classical baseline
6. SFH classical baseline

## Repository Entry Point
The main entry point is `main.py`.

### Model Modes
- `basic`
- `bichannel`

### Task Modes
- `all`: run CNN path (train/load + evaluate) and classical path
- `train-cnn`: run only selected CNN training/load path
- `classical-only`: run only classical model training/load + evaluation

## Run Commands
### Local run examples
```bash
python main.py --mode basic --task train-cnn
python main.py --mode bichannel --task train-cnn
python main.py --mode bichannel --task classical-only
python main.py --mode bichannel --task all
```

### SLURM run
Core scripts are in `SLURM job scripts/`:
- `slurm_train_basic.sh`
- `slurm_train_bichannel.sh`
- `slurm_eval_baselines.sh`
- `slurm_submit_all_three.sh`

Recommended single-command submission:
```bash
bash "SLURM job scripts/slurm_submit_all_three.sh"
```

## Data Requirements
The pipeline expects split folders:
- `data/train`
- `data/val`
- `data/test`

Prepare data with:
```bash
python setup/setup_data.py
```

## Configuration
All runtime settings are centralized in `config.py`, including:
- paths
- model hyperparameters
- scheduler settings
- classical method settings
- test/evaluation settings

## Output Artifacts
- CNN checkpoints: `checkpoints/basic.pth`, `checkpoints/bichannel.pth`
- Classical checkpoints: `checkpoints/sc1.pkl`, `checkpoints/sc2.pkl`, `checkpoints/sfh.pkl`
- Metrics CSV: `results/eval_metrics.csv`
- Images: `results/images/`

## High-Level Pipeline
1. Build/load data from `data/`
2. Build selected model from `models/model.py`
3. Train/evaluate CNN via `training/train.py` and `training/test.py`
4. Train/evaluate classical models via `training/classical.py` and `reconstruction/reconstruction.py`
5. Save checkpoints and metrics to `checkpoints/` and `results/`
