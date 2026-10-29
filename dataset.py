# -*- coding: utf-8 -*-
"""Dataset utilities (ImageFolder-style)."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import config


def get_transforms(train: bool = True):
    """Return preprocessing transforms. Image processing code is excluded
    from this repository on purpose, so only standard augmentation is used."""
    normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def _make_loader(root: str, train: bool) -> Dataset:
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Please set SAONET_DATA_ROOT or edit config.DATA_ROOT. "
            "The tungsten ore X-ray dataset is proprietary and not included."
        )
    return datasets.ImageFolder(root, transform=get_transforms(train))


def get_loaders(data_root: str = None, batch_size: int = None, num_workers: int = None):
    """Create train / val / test DataLoaders.

    Expected layout:
        data_root/train/<class>/img.jpg
        data_root/val/<class>/img.jpg
        data_root/test/<class>/img.jpg
    """
    data_root = data_root or config.DATA_ROOT
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    train_set = _make_loader(os.path.join(data_root, "train"), True)
    val_set = _make_loader(os.path.join(data_root, "val"), False)
    test_set = _make_loader(os.path.join(data_root, "test"), False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_set.classes)


# A lightweight in-memory dataset used for unit smoke tests.
class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 16, num_classes: int = None):
        self.num_samples = num_samples
        self.num_classes = num_classes or config.NUM_CLASSES

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, config.IMG_SIZE, config.IMG_SIZE)
        label = idx % self.num_classes
        return img, label
