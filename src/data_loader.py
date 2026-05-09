"""
Data loading utilities for CIFAR-10.

This module does three simple jobs:
1) Download/load the CIFAR-10 dataset.
2) Apply basic image transforms (tensor conversion + normalization).
3) Return PyTorch DataLoader objects for training and testing.

The code is intentionally written in a beginner-friendly style with clear names
and explanations.
"""

from typing import Tuple

import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Class names in CIFAR-10, in label-index order (0 to 9).
CIFAR10_CLASSES: Tuple[str, ...] = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_data_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, Tuple[str, ...]]:
    """
    Create train/test DataLoaders for CIFAR-10.

    Args:
        batch_size: Number of images per batch.
        data_dir: Folder where CIFAR-10 is stored (or downloaded to).
        num_workers: Number of background worker processes used by DataLoader.

    Returns:
        A tuple of:
        - train_loader: DataLoader for training data (shuffled each epoch).
        - test_loader: DataLoader for test data (not shuffled).
        - classes: Tuple of CIFAR-10 class names.

    Notes:
        - `ToTensor()` converts pixel values from [0, 255] to [0.0, 1.0].
        - `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` shifts each channel
          from [0.0, 1.0] approximately to [-1.0, 1.0].
    """
    # Step 1: Define how every image should be transformed.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Step 2: Build dataset objects.
    # `download=True` means the dataset is downloaded automatically if missing.
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Step 3: Wrap datasets in DataLoaders to iterate in mini-batches.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data to reduce order bias.
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep test order stable for evaluation.
        num_workers=num_workers,
    )

    return train_loader, test_loader, CIFAR10_CLASSES


if __name__ == "__main__":
    # Small smoke test to show that the loader works.
    train_loader, _, classes = get_data_loaders(batch_size=4)
    images, labels = next(iter(train_loader))

    label_names = [classes[label_idx] for label_idx in labels.tolist()]
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Labels in this batch: {label_names}")