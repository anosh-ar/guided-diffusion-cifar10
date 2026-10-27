"""CIFAR-10 data pipeline."""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = len(CIFAR10_CLASSES)
IMAGE_SIZE = 32
CHANNELS = 3

# ToTensor scales to [0, 1]; Normalize(0.5, 0.5) maps that to [-1, 1], the range the diffusion process works in
_to_model_range = [transforms.ToTensor(), transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)]
TRAIN_TRANSFORM = transforms.Compose([transforms.RandomHorizontalFlip(), *_to_model_range])
EVAL_TRANSFORM = transforms.Compose(_to_model_range)


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map images from ``[-1, 1]`` back to ``[0, 1]``."""
    return ((x + 1) / 2).clamp(0, 1)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4, val_fraction: float = 0.1,
                    seed: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test loaders for CIFAR-10.

    The official training set is split into train and validation with a fixed seed, so the split is identical
    across runs. Augmentation (horizontal flip) is only applied to the training split.
    """
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=TRAIN_TRANSFORM)
    eval_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=EVAL_TRANSFORM)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=EVAL_TRANSFORM)

    n_val = int(len(train_full) * val_fraction)
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_set = Subset(train_full, perm[n_val:])
    val_set = Subset(eval_full, perm[:n_val])

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
