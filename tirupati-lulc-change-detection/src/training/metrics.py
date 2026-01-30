"""Evaluation metrics for LULC classification."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple


def calculate_iou(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[int, float]:
    """Calculate Intersection over Union for each class."""
    iou_per_class = {}
    
    for cls in range(num_classes):
        pred_mask = predictions == cls
        target_mask = targets == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            iou_per_class[cls] = float('nan')
        else:
            iou_per_class[cls] = intersection / union
    
    return iou_per_class


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 5,
    class_names: list = None
) -> Dict:
    """Calculate all evaluation metrics."""
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Overall accuracy
    accuracy = accuracy_score(target_flat, pred_flat)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        target_flat,
        pred_flat,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    
    # Mean metrics
    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_f1 = f1.mean()
    
    # IoU
    iou = calculate_iou(predictions, targets, num_classes)
    mean_iou = np.nanmean(list(iou.values()))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(
        target_flat,
        pred_flat,
        labels=list(range(num_classes))
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'mean_precision': float(mean_precision),
        'mean_recall': float(mean_recall),
        'mean_f1': float(mean_f1),
        'mean_iou': float(mean_iou),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_iou': iou,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions, targets, loss=None):
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if loss is not None:
            self.losses.append(float(loss))
    
    def compute(self, num_classes=5):
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        metrics = calculate_metrics(predictions, targets, num_classes)
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics
