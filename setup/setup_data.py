import os
import random
import shutil
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent
ROOT_DIR = SETUP_DIR.parent

CELEBA_DIR = ROOT_DIR / "CelebA" / "img_align_celeba"

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR   = ROOT_DIR / "data" / "val"
TEST_DIR  = ROOT_DIR / "data" / "test"

# Configuration portion
TOTAL_IMAGES = 100000   # <-- change this (N)
TRAIN_RATIO = 0.6
VAL_RATIO = (1 - TRAIN_RATIO) / 2
TEST_RATIO = (1 - TRAIN_RATIO) / 2

SEED = 42


# Ensure that celeba folder exists
def check_celeba():
    if not CELEBA_DIR.exists():
        raise FileNotFoundError(
            f"\n[ERROR] CelebA not found at:\n{CELEBA_DIR}\n\n"
            "Expected structure:\n"
            "parent_dir/celeba/img_align_celeba/*.jpg\n"
        )

# if there are already files in it.
def already_prepared():
    return (
        TRAIN_DIR.exists()
        and VAL_DIR.exists()
        and TEST_DIR.exists()
        and len(list(TRAIN_DIR.glob("*"))) > 0
    )


# populate files
def prepare_split():
    print("Preparing dataset split...")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in CELEBA_DIR.glob("*")
        if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]

    if len(images) == 0:
        raise ValueError("No images found in CelebA folder.")

    print(f"Total available images: {len(images)}")

    random.seed(SEED)

    if TOTAL_IMAGES > len(images):
        selected_images = images
    else:
        selected_images = random.sample(images, TOTAL_IMAGES)

    print(f"Using {len(selected_images)} images")

    random.shuffle(selected_images)

    n_total = len(selected_images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val

    # retrieve the images into the 
    train_imgs = selected_images[:n_train]
    val_imgs   = selected_images[n_train:n_train + n_val]
    test_imgs  = selected_images[n_train + n_val:]

    print(f"[INFO] Train: {len(train_imgs)}")
    print(f"[INFO] Val  : {len(val_imgs)}")
    print(f"[INFO] Test : {len(test_imgs)}")

    def copy_images(img_list, target_dir):
        for img in img_list:
            shutil.copy(img, target_dir / img.name)

    print("Copying files... (this may take time)")

    copy_images(train_imgs, TRAIN_DIR)
    copy_images(val_imgs, VAL_DIR)
    copy_images(test_imgs, TEST_DIR)

    print("Dataset prepared successfully.")

def clear_existing_split():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
            
def main(overwrite):
    check_celeba()

    if overwrite:
        print("Overwrite enabled. Removing existing dataset split...")
        clear_existing_split()

    if already_prepared():
        print("Dataset already exists. Skipping split.")
        return

    prepare_split()


if __name__ == "__main__":
    # change this to True, if you want to replace your current data folders
    main(overwrite=True)