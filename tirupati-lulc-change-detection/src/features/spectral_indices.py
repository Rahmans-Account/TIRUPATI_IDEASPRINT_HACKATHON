"""Calculate spectral indices from satellite imagery."""

import numpy as np
from typing import Dict


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red: Red band array
        nir: Near-infrared band array
        
    Returns:
        NDVI array
    """
    return (nir - red) / (nir + red + 1e-8)


def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Built-up Index.
    
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Args:
        swir: Short-wave infrared band array
        nir: Near-infrared band array
        
    Returns:
        NDBI array
    """
    return (swir - nir) / (swir + nir + 1e-8)


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        green: Green band array
        nir: Near-infrared band array
        
    Returns:
        NDWI array
    """
    return (green - nir) / (green + nir + 1e-8)


def calculate_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate Modified Normalized Difference Water Index.
    
    MNDWI = (Green - SWIR) / (Green + SWIR)
    
    Args:
        green: Green band array
        swir: Short-wave infrared band array
        
    Returns:
        MNDWI array
    """
    return (green - swir) / (green + swir + 1e-8)


def calculate_evi(
    red: np.ndarray,
    nir: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0
) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index.
    
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    
    Args:
        red: Red band array
        nir: Near-infrared band array
        blue: Blue band array
        G: Gain factor
        C1: Coefficient for aerosol resistance
        C2: Coefficient for aerosol resistance
        L: Canopy background adjustment
        
    Returns:
        EVI array
    """
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L
    return numerator / (denominator + 1e-8)


def calculate_savi(
    red: np.ndarray,
    nir: np.ndarray,
    L: float = 0.5
) -> np.ndarray:
    """
    Calculate Soil Adjusted Vegetation Index.
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    Args:
        red: Red band array
        nir: Near-infrared band array
        L: Soil brightness correction factor
        
    Returns:
        SAVI array
    """
    return ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)


def calculate_all_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate all spectral indices.
    
    Args:
        bands: Dictionary with band names as keys and arrays as values
               Expected keys: 'blue', 'green', 'red', 'nir', 'swir'
    
    Returns:
        Dictionary of calculated indices
    """
    indices = {}
    
    if 'red' in bands and 'nir' in bands:
        indices['ndvi'] = calculate_ndvi(bands['red'], bands['nir'])
        indices['savi'] = calculate_savi(bands['red'], bands['nir'])
    
    if 'swir' in bands and 'nir' in bands:
        indices['ndbi'] = calculate_ndbi(bands['swir'], bands['nir'])
    
    if 'green' in bands and 'nir' in bands:
        indices['ndwi'] = calculate_ndwi(bands['green'], bands['nir'])
    
    if 'green' in bands and 'swir' in bands:
        indices['mndwi'] = calculate_mndwi(bands['green'], bands['swir'])
    
    if 'blue' in bands and 'red' in bands and 'nir' in bands:
        indices['evi'] = calculate_evi(bands['red'], bands['nir'], bands['blue'])
    
    return indices
