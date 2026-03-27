import torch
from data.dataset import FaceDataset
from torch.utils.data import DataLoader

# Preapre our DataLoaders for training, validation, and testing. 
# Made different DataLoader settings (e.g., batch size, shuffling) for each training stage.
def get_test_loader(data_path):
    dataset = FaceDataset(image_dir=data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,   # one image at a time
        shuffle=False
    )
    return dataloader

def get_val_loader(data_path, batch_size=32):
    dataset = FaceDataset(image_dir=data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # no shuffling for validation
        pin_memory=True
    )
    return dataloader

def get_dataloader(data_path, batch_size=32):
    dataset = FaceDataset(image_dir=data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,      
        pin_memory=True  # This is for faster GPU transfer, allows tensors to be allocated in page-locked memory
    )
    return dataloader