"""Calculate texture features using GLCM."""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List, Tuple


def calculate_glcm_features(
    image: np.ndarray,
    distances: List[int] = [1],
    angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
    levels: int = 256,
    symmetric: bool = True,
    normed: bool = True
) -> dict:
    """
    Calculate GLCM (Gray Level Co-occurrence Matrix) texture features.
    
    Args:
        image: Input grayscale image
        distances: List of pixel pair distance offsets
        angles: List of pixel pair angles in radians
        levels: Number of gray levels
        symmetric: Whether to create symmetric matrix
        normed: Whether to normalize the matrix
        
    Returns:
        Dictionary of texture features
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed
    )
    
    # Calculate properties
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'asm': graycoprops(glcm, 'ASM').mean()
    }
    
    return features


def calculate_texture_features_multiband(
    image: np.ndarray,
    bands_to_use: List[int] = None
) -> np.ndarray:
    """
    Calculate texture features for multi-band image.
    
    Args:
        image: Multi-band image (channels, height, width)
        bands_to_use: Indices of bands to use
        
    Returns:
        Stacked texture features
    """
    if bands_to_use is None:
        bands_to_use = range(image.shape[0])
    
    texture_stack = []
    
    for band_idx in bands_to_use:
        band = image[band_idx]
        features = calculate_glcm_features(band)
        
        # Create feature maps (simplified - use same value across image)
        for feature_name, feature_value in features.items():
            feature_map = np.full_like(band, feature_value, dtype=np.float32)
            texture_stack.append(feature_map)
    
    return np.stack(texture_stack)
