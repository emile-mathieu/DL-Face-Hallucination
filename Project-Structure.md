# 📁 Project Structure

```project/
│
├── data/
│   ├── dataset.py        # Loads images, creates LR/HR pairs, preprocessing
│   └── dataloader.py     # Builds PyTorch DataLoader (batching, shuffle)
│
├── models/
│   └── model.py          # Bi-Channel CNN + Basic CNN architectures
│
├── training/
│   ├── train.py          # Training loop (MSE, PSNR, SSIM, CSV logging, best model saving)
│   ├── evaluate.py       # Evaluation loop (PSNR, SSIM, CSV logging)
│   └── inference.py      # Runs model on single image + saves LR/SR/HR outputs
│
├── utils/
│   └── logger.py         # CSV logging utility (train + eval metrics)
│
├── results/
│   ├── train_metrics.csv # Training metrics per epoch (PSNR, SSIM, loss)
│   ├── eval_metrics.csv  # Evaluation results
│   ├── best_model.pth    # Best model checkpoint (based on PSNR)
│   └── images/           # Saved inference outputs (lr.png, sr.png, hr.png)
│
└── main.py               # Entry point: train → save → evaluate
```
---
# 🔄 Pipeline Overview
1. High-Res Image (IH)
2. Blur + Downsample
3. Low-Res Image (IL)
4. Resize → (48×48)
5. Model (Bi-Channel CNN)
6. High-Res Output (100×100)
---

# 🧩 Components

### 📦 data/
- **dataset.py**
  - Loads images
  - Generates low-resolution inputs
  - Applies preprocessing (resize + normalize)

- **dataloader.py**
  - Creates batches using PyTorch `DataLoader`

---

### 🧠 models/
- **model.py**
  - CNN feature extractor
  - Reconstruction branch
  - Alpha (fusion weight) branch
  - Combines:
    ```
    Output = α * Upsampled Input + (1 - α) * Reconstruction
    ```

---

### 🏋️ training/
- **train.py**
  - Handles training loop
  - Computes loss (MSE)
  - Performs backpropagation

- **evaluate.py**
  - Runs evaluation on validation set
  - Computes metrics (PSNR, SSIM)

- **inference.py**
  - Runs model on single image
  - Saves generated outputs

---
### 🛠 utils/
- **logger.py**
  - Utility for logging metrics to CSV files

### 🚀 main.py
- Entry point of the project
- Runs training, validation and testing depending on configuration

---

# 📐 Input / Output

| Type | Shape |
|------|------|
| Input (LR) | (B, 3, 48, 48) |
| Output (HR) | (B, 3, 100, 100) |

---

# ⚙️ Key Idea

The model combines:
- **Raw image information**
- **Learned deep features**