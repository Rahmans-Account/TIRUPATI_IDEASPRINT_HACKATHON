"""Build and analyze transition matrix."""

import numpy as np
import pandas as pd
from typing import Dict, List


def build_transition_matrix(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    num_classes: int = 5,
    class_names: List[str] = None
) -> pd.DataFrame:
    """
    Build transition matrix showing class-to-class changes.
    
    Args:
        classification_t1: Classification at time 1
        classification_t2: Classification at time 2
        num_classes: Number of classes
        class_names: Names of classes
        
    Returns:
        DataFrame with transition matrix
    """
    # Initialize matrix
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Count transitions
    for i in range(num_classes):
        for j in range(num_classes):
            mask_t1 = (classification_t1 == i)
            mask_t2 = (classification_t2 == j)
            matrix[i, j] = np.logical_and(mask_t1, mask_t2).sum()
    
    # Create DataFrame
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    df = pd.DataFrame(
        matrix,
        index=[f"{name}_T1" for name in class_names],
        columns=[f"{name}_T2" for name in class_names]
    )
    
    return df


def calculate_transition_percentages(
    transition_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Calculate percentage transitions."""
    total = transition_matrix.sum().sum()
    return (transition_matrix / total * 100).round(2)


def get_major_transitions(
    transition_matrix: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Get top N transitions by pixel count."""
    # Exclude diagonal (no change)
    matrix = transition_matrix.copy()
    np.fill_diagonal(matrix.values, 0)
    
    # Get top transitions
    stacked = matrix.stack().sort_values(ascending=False).head(top_n)
    
    return pd.DataFrame({
        'From': [idx[0] for idx in stacked.index],
        'To': [idx[1] for idx in stacked.index],
        'Pixels': stacked.values
    })
