"""
CIFAR-10 Data Loading Utilities.

Downloads, transforms, and creates PyTorch DataLoaders for CIFAR-10 dataset batches.
"""

from typing import Tuple
import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_CLASSES: Tuple[str, ...] = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

def get_data_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    shuffle_test: bool = False
) -> Tuple[DataLoader, DataLoader, Tuple[str, ...]]:
    """
    Constructs train and test DataLoaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): Mini-batch size for DataLoader instances. Defaults to 128.
        data_dir (str): Root directory path to store/download CIFAR-10 data. Defaults to "./data".
        num_workers (int): Number of background subprocesses for data loading. Defaults to 2.
        shuffle_test (bool): Whether to shuffle test dataset order. Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, Tuple[str, ...]]: A 3-tuple containing:
            - train_loader: Training DataLoader instance (shuffled).
            - test_loader: Evaluation DataLoader instance.
            - classes: Tuple of class name strings.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
    )

    return train_loader, test_loader, CIFAR10_CLASSES