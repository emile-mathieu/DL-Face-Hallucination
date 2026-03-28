import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import FaceDataset, compute_mean_std
from torch.utils.data import DataLoader

from models.model import BiChannelCNN
from training.train import train
from training.test import evaluate

# Main entry point, becareful (plz) with this file as it runs the whole pipeline (training + eval).
def main():
    # -------------------------
    # 1. Config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "data/train"
    val_path = "data/val"

    batch_size = 32
    num_epochs = 20
    lr = 1e-5
    min_lr = 1e-6
    weight_decay = 0.0005
    patience = 5

    # -------------------------
    # 2. Compute mean/std from training set
    # -------------------------
    mean, std = compute_mean_std(train_path, image_size=(100, 100))
    print("Training mean:", mean)
    print("Training std :", std)

    # -------------------------
    # 3. Data
    # -------------------------
    train_dataset = FaceDataset(train_path, mean=mean, std=std)
    val_dataset = FaceDataset(val_path, mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 4. Model
    # -------------------------
    model = BiChannelCNN().to(device)

    # -------------------------
    # 5. Loss + Optimizer + Scheduler
    # -------------------------
    criterion = nn.MSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,      # 1e-5 -> 1e-6
        patience=patience,
        min_lr=min_lr
    )

    # -------------------------
    # 6. Train
    # -------------------------
    print("Starting training...\n")
    train(
        optimizer=optimizer,
        criterion=criterion,
        model=model,
        dataloader=train_loader,
        device=device,
        dataset=train_dataset,
        num_epochs=num_epochs,
        scheduler=scheduler
    )

    # -------------------------
    # 7. Validate 
    # -------------------------
    print("\nEvaluating on validation set...\n")
    evaluate(model, val_loader, device, val_dataset)


# Entry point
if __name__ == "__main__":
    main()
