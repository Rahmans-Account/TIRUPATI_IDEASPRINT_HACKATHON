"""Data preprocessing functions."""

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger


def clip_raster_to_boundary(raster_path: str, boundary_path: str, output_path: str) -> str:
    """
    Clip raster to boundary shapefile.
    
    Args:
        raster_path: Path to input raster
        boundary_path: Path to boundary shapefile
        output_path: Path to save clipped raster
        
    Returns:
        Path to clipped raster
    """
    logger.info(f"Clipping {raster_path} to boundary...")
    
    # Read boundary
    boundary = gpd.read_file(boundary_path)
    
    # Read raster
    with rasterio.open(raster_path) as src:
        # Reproject boundary to match raster CRS
        if boundary.crs != src.crs:
            logger.info(f"Reprojecting boundary from {boundary.crs} to {src.crs}")
            boundary = boundary.to_crs(src.crs)
        
        # Clip raster
        out_image, out_transform = rio_mask(src, boundary.geometry, crop=True)
        out_meta = src.meta.copy()
        
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Save clipped raster
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    logger.success(f"Clipped raster saved to {output_path}")
    return output_path


def load_landsat_bands(image_path: str, bands: list = None) -> Tuple[np.ndarray, dict]:
    """Load Landsat bands."""
    with rasterio.open(image_path) as src:
        if bands:
            data = src.read(bands)
        else:
            data = src.read()
        meta = src.meta.copy()
    return data, meta


def cloud_mask_landsat(image: np.ndarray, qa_band: np.ndarray) -> np.ndarray:
    """Apply cloud mask to Landsat image."""
    # Simplified cloud masking
    cloud_mask = (qa_band & (1 << 3)) == 0  # Clear condition
    masked_image = image.copy()
    masked_image[:, ~cloud_mask] = 0
    return masked_image


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize image values."""
    normalized = image.astype(np.float32)
    
    if method == 'minmax':
        for i in range(image.shape[0]):
            band = image[i]
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                normalized[i] = (band - min_val) / (max_val - min_val)
    
    return normalized
