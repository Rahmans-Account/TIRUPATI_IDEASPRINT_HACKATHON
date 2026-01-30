"""Model utility functions."""

import torch
import numpy as np
from typing import Tuple, Optional


def get_device() -> torch.device:
    """Get available device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_rotations: int = 4
) -> torch.Tensor:
    """
    Predict with Test Time Augmentation.
    
    Args:
        model: PyTorch model
        image: Input image tensor
        num_rotations: Number of rotations (0, 90, 180, 270)
        
    Returns:
        Averaged predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = model(image)
        predictions.append(pred)
        
        # Horizontal flip
        pred_hflip = model(torch.flip(image, [-1]))
        predictions.append(torch.flip(pred_hflip, [-1]))
        
        # Vertical flip
        pred_vflip = model(torch.flip(image, [-2]))
        predictions.append(torch.flip(pred_vflip, [-2]))
        
        # Rotations
        for k in range(1, num_rotations):
            rotated = torch.rot90(image, k, [-2, -1])
            pred_rot = model(rotated)
            predictions.append(torch.rot90(pred_rot, -k, [-2, -1]))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
