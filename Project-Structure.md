# 📁 Project Structure

```project/
│
├── data/
│   ├── dataset.py        # Loads images and creates (LR, HR) pairs FINISHED
│   └── dataloader.py     # Wraps dataset into PyTorch DataLoader FINISHED
│
├── models/
│   └── model.py          # Bi-Channel CNN architecture FINISHED
│
├── training/
│   ├── train.py          # Training loop (forward, loss, backprop) (NOT FINISHED!)
│   └── test.py           # Evaluation / inference FINISHED (NOT FINISHED!)
│
└── main.py               # Entry point (run training or testing) TODO
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

- **test.py**
  - Runs inference on new images
  - Evaluates model performance

---

### 🚀 main.py
- Entry point of the project
- Runs training or testing depending on configuration

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