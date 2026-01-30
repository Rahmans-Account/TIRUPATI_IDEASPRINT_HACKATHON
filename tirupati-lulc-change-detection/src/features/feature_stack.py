"""Combine spectral bands and derived features."""

import numpy as np
from typing import List, Dict, Optional
from src.features.spectral_indices import calculate_all_indices
from src.features.texture_features import calculate_texture_features_multiband


def normalize_bands(bands: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize band values.
    
    Args:
        bands: Input bands (channels, height, width)
        method: Normalization method ('minmax', 'zscore', 'percentile')
        
    Returns:
        Normalized bands
    """
    normalized = bands.copy().astype(np.float32)
    
    if method == 'minmax':
        for i in range(bands.shape[0]):
            band = bands[i]
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                normalized[i] = (band - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        for i in range(bands.shape[0]):
            band = bands[i]
            mean, std = band.mean(), band.std()
            if std > 0:
                normalized[i] = (band - mean) / std
    
    elif method == 'percentile':
        for i in range(bands.shape[0]):
            band = bands[i]
            p2, p98 = np.percentile(band, [2, 98])
            if p98 > p2:
                normalized[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    return normalized


def create_feature_stack(
    bands: Dict[str, np.ndarray],
    include_indices: bool = True,
    include_texture: bool = False,
    normalize: bool = True,
    normalization_method: str = 'minmax'
) -> np.ndarray:
    """
    Create complete feature stack from spectral bands.
    
    Args:
        bands: Dictionary of spectral bands
        include_indices: Whether to include spectral indices
        include_texture: Whether to include texture features
        normalize: Whether to normalize features
        normalization_method: Method for normalization
        
    Returns:
        Feature stack (channels, height, width)
    """
    features = []
    
    # Add original bands
    band_stack = np.stack([bands[key] for key in sorted(bands.keys())])
    features.append(band_stack)
    
    # Add spectral indices
    if include_indices:
        indices = calculate_all_indices(bands)
        if indices:
            index_stack = np.stack([indices[key] for key in sorted(indices.keys())])
            features.append(index_stack)
    
    # Add texture features
    if include_texture:
        texture_stack = calculate_texture_features_multiband(band_stack)
        features.append(texture_stack)
    
    # Concatenate all features
    feature_stack = np.concatenate(features, axis=0)
    
    # Normalize if requested
    if normalize:
        feature_stack = normalize_bands(feature_stack, method=normalization_method)
    
    return feature_stack


def extract_features_from_raster(
    raster_path: str,
    band_names: List[str],
    **kwargs
) -> np.ndarray:
    """
    Extract features from raster file.
    
    Args:
        raster_path: Path to raster file
        band_names: Names of bands in order
        **kwargs: Additional arguments for create_feature_stack
        
    Returns:
        Feature stack
    """
    import rasterio
    
    with rasterio.open(raster_path) as src:
        bands_array = src.read()
    
    # Create bands dictionary
    bands = {name: bands_array[i] for i, name in enumerate(band_names)}
    
    # Create feature stack
    features = create_feature_stack(bands, **kwargs)
    
    return features
