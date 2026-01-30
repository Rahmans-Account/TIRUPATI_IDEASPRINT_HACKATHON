"""Training script for U-Net model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.models.unet import UNet
from src.training.losses import CombinedLoss
from src.training.metrics import MetricsTracker
from src.utils.logger import default_logger as logger


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int = 5
) -> dict:
    """Train for one epoch."""
    model.train()
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        
        # Update metrics
        metrics_tracker.update(predictions, labels, loss.item())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    epoch_metrics = metrics_tracker.compute(num_classes)
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 5
) -> dict:
    """Validate for one epoch."""
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics_tracker.update(predictions, labels, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    epoch_metrics = metrics_tracker.compute(num_classes)
    return epoch_metrics


def train_unet(
    model: UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    save_dir: str = 'models/unet'
):
    """
    Complete training loop for U-Net.
    
    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        save_dir: Directory to save models
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Loss function
    class_weights = torch.tensor(config.get('class_weights', [1.0]*5)).to(device)
    criterion = CombinedLoss(class_weights=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.get('scheduler_patience', 10),
        factor=config.get('scheduler_factor', 0.5)
    )
    
    # Training loop
    num_epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 20)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        logger.info(f"
Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Scheduler step
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), save_path / 'best_model.pth')
            logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save training history
    import json
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    return history
