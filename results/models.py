"""
models.py
=========
Model definitions:
  - BasicCNN          (paper-exact architecture, AAAI 2015)
  - BiChannelCNN      (paper-exact architecture, AAAI 2015)
  - WarmupCosineScheduler
  - load_basiccnn_into_bichannel()
  - load_model_for_inference()  (for both models)

Architecture notes
------------------
Both models use paper-exact tanh activations and MaxPool throughout.
BasicCNN weights are directly compatible with BiChannelCNN's conv and
reconstruction branch — this is required for the pretrain transfer.

Initialisation
--------------
  Conv layers  : orthogonal_(gain=0.6)   — preserves gradient magnitude
  FC layers    : Normal(0, 1/sqrt(fan_in)) — keeps tanh in linear regime
Both differ from the paper's N(0,0.001) which causes gradient vanishing.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import WARMUP_EPOCHS, LR_T_MAX, MIN_LR


# ── BasicCNN ──────────────────────────────────────────────────
class BasicCNN(nn.Module):
    """
    Paper-exact BasicCNN (AAAI 2015, Table 1).
      Input : (N, 3, 48, 48)  tanh-normalised LR
      Output: (N, 3, 100, 100) tanh-normalised HR prediction
    """
    def __init__(self, init_weights=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32,  kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 2000)
        self.fc2   = nn.Linear(2000, 3 * 100 * 100)
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        print(f"  [Init] Initializing weights on {next(self.parameters()).device}...", flush=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        print("  [Init] Done.", flush=True)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = torch.tanh(self.fc1(x.flatten(start_dim=1)))
        x = torch.tanh(self.fc2(x))
        return x.view(-1, 3, 100, 100)


# ── BiChannelCNN ──────────────────────────────────────────────
class BiChannelCNN(nn.Module):
    """
    Paper-exact BiChannelCNN (AAAI 2015, Fig. 3 & Table 1).
      Input : (N, 3, 48, 48)  tanh-normalised LR
      Output: (N, 3, 100, 100) fused HR prediction, (N, 1, 1, 1) alpha

    Fusion: output = alpha * bicubic_upsample(input) + (1-alpha) * I_rec
    Alpha  = 0.5 * tanh(fc2_2(...)) + 0.5  →  alpha ∈ (0, 1)
    """
    def __init__(self, init_weights=False):
        super().__init__()
        # Shared encoder (weights loaded from BasicCNN pretrain)
        self.conv1 = nn.Conv2d(3,  32,  kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        # Reconstruction branch (loaded from BasicCNN)
        self.fc1_1 = nn.Linear(128 * 4 * 4, 2000)
        self.fc2_1 = nn.Linear(2000, 3 * 100 * 100)
        # Alpha branch (trained from scratch)
        self.fc1_2 = nn.Linear(128 * 4 * 4, 100)
        self.fc2_2 = nn.Linear(100, 1)
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        print(f"  [Init] Initializing weights on {next(self.parameters()).device}...", flush=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        print("  [Init] Done.", flush=True)

    def forward(self, x):
        inp = x
        # Shared encoder
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        f = torch.flatten(x, start_dim=1)
        # Reconstruction branch (Eq. 11)
        i_rec = torch.tanh(self.fc1_1(f))
        i_rec = torch.tanh(self.fc2_1(i_rec))
        i_rec = i_rec.view(-1, 3, 100, 100)
        # Alpha branch (Eq. 12)
        alpha = torch.tanh(self.fc1_2(f))
        alpha = (0.5 * torch.tanh(self.fc2_2(alpha)) + 0.5).view(-1, 1, 1, 1)
        # Fusion (Eq. 10)
        i_up = F.interpolate(inp, size=(100, 100),
                             mode="bicubic", align_corners=False)
        return alpha * i_up + (1 - alpha) * i_rec, alpha


# ── Pretrain transfer ─────────────────────────────────────────
def load_basiccnn_into_bichannel(
    bichannel: BiChannelCNN,
    basic_ckpt_path: Path,
    device,
) -> BiChannelCNN:
    """
    Copy BasicCNN encoder + reconstruction branch into BiChannelCNN.
    Mapping: conv1/2/3 → conv1/2/3, fc1 → fc1_1, fc2 → fc2_1.
    Alpha branch (fc1_2, fc2_2) keeps its fresh init.
    """
    ckpt  = torch.load(basic_ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) else ckpt

    mapping = {
        "conv1.weight": "conv1.weight", "conv1.bias": "conv1.bias",
        "conv2.weight": "conv2.weight", "conv2.bias": "conv2.bias",
        "conv3.weight": "conv3.weight", "conv3.bias": "conv3.bias",
        "fc1.weight":   "fc1_1.weight", "fc1.bias":   "fc1_1.bias",
        "fc2.weight":   "fc2_1.weight", "fc2.bias":   "fc2_1.bias",
    }
    bi_state = bichannel.state_dict()
    transferred = 0
    for basic_key, bi_key in mapping.items():
        if basic_key in state and bi_key in bi_state:
            bi_state[bi_key] = state[basic_key]
            transferred += 1
    bichannel.load_state_dict(bi_state)
    print(f"[BiChannelCNN] Transferred {transferred} tensors "
          f"from BasicCNN checkpoint: {basic_ckpt_path}")
    return bichannel


# ── Inference loaders ─────────────────────────────────────────
def load_basiccnn_for_inference(model_path: Path, device) -> BasicCNN:
    model = BasicCNN(init_weights=False).to(device)
    print(f"Loading BasicCNN weights from {model_path}...", flush=True)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    return model

def load_bichannel_for_inference(model_path: Path, device) -> BiChannelCNN:
    model = BiChannelCNN(init_weights=False).to(device)
    print(f"Loading weights from {model_path}...", flush=True)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval() 
    return model


# ── LR schedule ───────────────────────────────────────────────
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup (epochs 0..warmup-1) then cosine annealing.
    Replaces ReduceLROnPlateau which permanently collapses LR on any
    unlucky plateau epoch, effectively ending training by epoch ~45.
    """
    def __init__(self, optimizer, warmup_epochs: int = WARMUP_EPOCHS,
                 t_max: int = LR_T_MAX, min_lr: float = MIN_LR,
                 last_epoch: int = -1):
        self.warmup  = warmup_epochs
        self.t_max   = t_max
        self.min_lr  = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            alpha = (e + 1) / max(self.warmup, 1)
            return [self.min_lr + alpha * (base - self.min_lr)
                    for base in self.base_lrs]
        progress = min((e - self.warmup) / max(self.t_max, 1), 1.0)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + cosine * (base - self.min_lr)
                for base in self.base_lrs]
