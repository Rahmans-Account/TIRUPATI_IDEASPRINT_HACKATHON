"""Change detection algorithms."""

import numpy as np
from typing import Tuple, Dict


def detect_changes(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    confidence_t1: np.ndarray = None,
    confidence_t2: np.ndarray = None,
    min_confidence: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect changes between two classification maps.
    
    Args:
        classification_t1: Classification map at time 1
        classification_t2: Classification map at time 2
        confidence_t1: Confidence scores at time 1
        confidence_t2: Confidence scores at time 2
        min_confidence: Minimum confidence threshold
        
    Returns:
        Tuple of (change_map, confidence_map)
    """
    # Binary change map
    change_map = (classification_t1 != classification_t2).astype(np.uint8)
    
    # Confidence map
    if confidence_t1 is not None and confidence_t2 is not None:
        confidence_map = np.minimum(confidence_t1, confidence_t2)
        
        # Mask low confidence pixels
        change_map[confidence_map < min_confidence] = 0
    else:
        confidence_map = np.ones_like(change_map, dtype=np.float32)
    
    return change_map, confidence_map


def create_transition_map(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    num_classes: int = 5
) -> np.ndarray:
    """
    Create transition map encoding class transitions.
    
    Args:
        classification_t1: Classification at time 1
        classification_t2: Classification at time 2
        num_classes: Number of classes
        
    Returns:
        Transition map where value = t1_class * num_classes + t2_class
    """
    return classification_t1 * num_classes + classification_t2


def analyze_transitions(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    num_classes: int = 5,
    pixel_area: float = 900.0  # 30m x 30m for Landsat
) -> Dict:
    """
    Analyze transitions between classifications.
    
    Args:
        classification_t1: Classification at time 1
        classification_t2: Classification at time 2
        num_classes: Number of classes
        pixel_area: Area of each pixel in square meters
        
    Returns:
        Dictionary with transition statistics
    """
    # Change detection
    change_map, _ = detect_changes(classification_t1, classification_t2)
    
    # Calculate total changed area
    total_changed_pixels = change_map.sum()
    total_changed_area = total_changed_pixels * pixel_area
    
    # Percentage of area changed
    total_pixels = classification_t1.size
    change_percentage = (total_changed_pixels / total_pixels) * 100
    
    results = {
        'total_changed_pixels': int(total_changed_pixels),
        'total_changed_area_sqm': float(total_changed_area),
        'total_changed_area_sqkm': float(total_changed_area / 1_000_000),
        'change_percentage': float(change_percentage)
    }
    
    return results
