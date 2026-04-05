from data.dataset import FaceDataset
from torch.utils.data import DataLoader

# Prepare our DataLoaders for training, validation, and testing. 
# Made different DataLoader settings (e.g., batch size, shuffling) for each training stage.
def get_test_loader(data_path, max_items=None,
                    blur_type=None, gaussian_sigma=None, motion_length=None):
    
    dataset = FaceDataset(
        image_dir=data_path,
        max_items=max_items,
        blur_type=blur_type,
        gaussian_sigma=gaussian_sigma,
        motion_length=motion_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1, #one image at a time
        num_workers=4,
        shuffle=False
    )
    return dataloader


def get_val_loader(data_path, batch_size=32, num_workers=4):
    dataset = FaceDataset(image_dir=data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False, # no shuffling for validation
        pin_memory=True
    )
    return dataloader


def get_train_loader(data_path, is_classical=False, batch_size=32, num_workers=4):
    dataset = FaceDataset(image_dir=data_path, is_classical=is_classical)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True # This is for faster GPU transfer, allows tensors to be allocated in page-locked memory
    )
    return dataloader

# Classical dataset
def get_classical_train_dataset(data_path, hr_size=(100,100)):
    dataset = FaceDataset(
        image_dir=data_path, 
        is_classical=True, 
        classical_hr_size=hr_size
    )
    return dataset

