import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils import BATCH_SIZE, LOG_RESOLUTION, NUM_WORKERS, DATASET, RANDOM_SEED, VAL_SPLIT

"""
dataset.py
Dataset loading and preprocessing
Author: Tyreece Paul
"""

# Hyperparameters
DATASET_DIR = DATASET
IMAGE_SIZE = 2 ** LOG_RESOLUTION  

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1] for GAN
])

# Data augmentation and normalization for validation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def get_dataloaders(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """
    Create training and validation DataLoaders with optimizations.
    Args:
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data to use for validation.
    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
    """

    # Load full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)

    # Compute lengths for train/val split
    val_len = int(len(full_dataset) * val_split)
    train_len = len(full_dataset) - val_len

    # Fix seed for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)

    # Replace validation transforms
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    return train_loader, val_loader