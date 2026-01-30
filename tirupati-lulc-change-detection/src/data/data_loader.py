"""PyTorch data loaders for LULC data."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, List


class LULCDataset(Dataset):
    """Dataset for LULC classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        transform=None
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()
        
        # Load label
        with rasterio.open(self.label_paths[idx]) as src:
            label = src.read(1)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label


def create_dataloaders(
    train_images: List[str],
    train_labels: List[str],
    val_images: List[str],
    val_labels: List[str],
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = LULCDataset(train_images, train_labels)
    val_dataset = LULCDataset(val_images, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
