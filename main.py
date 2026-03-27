import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import FaceDataset
from torch.utils.data import DataLoader

from models.model import BiChannelCNN
from training.train import train
from training.test import evaluate


def main():
    # -------------------------
    # 1. Config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "data/train"
    val_path = "data/val"

    batch_size = 32
    num_epochs = 10
    lr = 1e-3

    # -------------------------
    # 2. Data
    # -------------------------
    train_dataset = FaceDataset(train_path)
    val_dataset = FaceDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 3. Model
    # -------------------------
    model = BiChannelCNN().to(device)

    # -------------------------
    # 4. Loss + Optimizer
    # -------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # 5. Train
    # -------------------------
    print("Starting training...\n")
    train(optimizer, criterion, model, train_loader, device, num_epochs)

    # -------------------------
    # 6. Evaluate
    # -------------------------
    print("\nEvaluating on validation set...\n")
    evaluate(model, val_loader, device)


# Entry point
if __name__ == "__main__":
    main()