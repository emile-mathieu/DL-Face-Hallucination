# AI6103 Group Project: Learning Face Hallucination in the Wild

This repository contains the code for our project on learning face hallucination in the wild. The project aims to enhance low-resolution facial images by generating high-resolution versions using deep learning techniques.

## Reference Paper
Our work contains the reimplementation and experimental investigation of the AAAI paper:  

**"Learning Face Hallucination in the Wild"**  
**Authors:** Erjin Zhou, Haoqiang Fan, Zhimin Cao, Yuning Jiang, Qi Yin  
*Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence* (AAAI 2015)  
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/9795

# face_hallucination_bichannel_celebA.py — Overview & Paper Consistency

## What the script does (brief)

The script reproduces **“Learning Face Hallucination in the Wild”** (Zhou et al., AAAI 2015) on CelebA. It implements **6 methods** from the paper’s Table 2:

| # | Method | Role in script |
|---|--------|-----------------|
| 1 | **Bicubic** | No learning: LR 50→100 bicubic upsampling; used as baseline. |
| 2 | **SC1** (Yang et al. 2008; 2010) | Coupled dictionary learning on Y-channel patches; trained in `eval_baselines`. |
| 3 | **SC2** (Kim & Kwon 2010) | Anchor K-NN + ridge regression on Y-channel; trained in `eval_baselines`. |
| 4 | **SFH** (Yang, Liu, Yang 2013) | Landmark-guided mask + screened-Poisson refinement; uses SC1; optional `face_alignment`. |
| 5 | **Basic CNN** | Paper’s ablation: same backbone, single branch, no fusion; train with `train --model basic`. |
| 6 | **Bi-channel CNN** | Paper’s main method: two branches (I_rec + α), fusion α·↑Iin + (1−α)·I_rec; train with `train --model bichannel`. |

**Data:** CelebA via torchvision; 60/20/20 train/val/test; HR 100×100; degradation = Gaussian (σ∈[0,7]) or motion (length∈[0,11], θ∈[−π,π]) then downsample (e.g. 100→50). For CNNs, LR is resized to 48×48 and normalized (per-image mean/std + tanh).

**Evaluation:** PSNR and SSIM (Wang et al. 2004) under Gaussian σ=1,3,5 and motion l=2,6,9; Table-2-style output; paper reference table with all 6 methods.

**Commands:**

| Command | Purpose |
|---------|---------|
| `train` | Train Bi-channel or Basic CNN (blur: gaussian/motion/mixed). |
| `eval_gaussian` | Eval a trained CNN under σ=1,3,5 (needs `--checkpoint`). |
| `eval_motion` | Eval a trained CNN under l=2,6,9 (needs `--checkpoint`). |
| `print_paper_table` | Print full Table 2 (all 6 methods) from the paper; no data/GPU. |
| `eval_baselines` | Train SC1, SC2 (SFH uses SC1); validate all 4 baselines on val set; eval on test (4 baselines, or all 6 if `--bichannel_checkpoint` and `--basic_checkpoint` are given). |
| `show_sample` | Save one HR image and its Gaussian/motion blurred versions. |

---

## Consistency with the paper

| Paper | Script |
|-------|--------|
| **Degradation (Eq. 1)** IL = ↓(IH ⊗ G) | Gaussian or motion blur, then downsample (e.g. 100→50). |
| **Blur ranges** | σ∈[0,7], motion length ∈[0,11], θ∈[−π,π]; downscale factor 2–5. |
| **Network (Table 1)** | 3 conv+pool (32, 64, 128 filters); Bi-channel: two FC groups (I_rec and α). |
| **Input/output** | CNN input 48×48 RGB; output 100×100 RGB. |
| **Init & training** | Weights N(0, 0.001), biases 0; SGD momentum 0.9, lr=1e-5; lr decay on val plateau; batch 200, 5000 cycles (means roughtly 8-9 epochs) |
| **Fusion (Eq. 10)** | α·↑Iin + (1−α)·Irec, α∈[0,1]. |
| **Table 2** | Bicubic, SC1, SC2, SFH, Basic CNN, Bi-channel CNN; σ=1,3,5 and l=2,6,9; PSNR/SSIM. |
| **Split** | 60% train, 20% val, 20% test. |

**Who gets trained / validated / evaluated**

- **Bicubic:** No training. Validated and evaluated (same pipeline; prediction = bicubic-upsampled LR).
- **SC1, SC2:** Trained in `eval_baselines`; validated on val set; evaluated on test set.
- **SFH:** No separate training (uses trained SC1). Validated and evaluated on val/test.
- **Basic CNN, Bi-channel CNN:** Trained via `train` (with validation during training). Evaluated via `eval_gaussian`/`eval_motion`, or in `eval_baselines` with `--bichannel_checkpoint` and/or `--basic_checkpoint` for one table with all 6.

**Caveats:** Exact paper numbers need the same data and seeds; this script uses CelebA and the above setup to produce comparable Table-2-style results. Bicubic, SC1, SC2, SFH, Basic CNN, and Bi-channel CNN are implemented to match the paper’s setup.
