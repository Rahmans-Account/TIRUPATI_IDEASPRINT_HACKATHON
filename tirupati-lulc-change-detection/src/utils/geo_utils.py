"""Geospatial utility functions."""

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, box
from typing import Tuple, List, Optional
from pathlib import Path


def read_raster(file_path: str) -> Tuple[np.ndarray, dict]:
    """Read raster file and return data with metadata."""
    with rasterio.open(file_path) as src:
        data = src.read()
        meta = src.meta.copy()
    return data, meta


def write_raster(
    data: np.ndarray,
    output_path: str,
    meta: dict,
    nodata: Optional[float] = None
) -> None:
    """Write data to raster file."""
    meta.update({
        'count': data.shape[0] if len(data.shape) == 3 else 1,
        'dtype': data.dtype
    })
    
    if nodata is not None:
        meta['nodata'] = nodata
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        if len(data.shape) == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def clip_raster_to_boundary(
    raster_path: str,
    boundary_path: str,
    output_path: str,
    crop: bool = True
) -> Tuple[np.ndarray, dict]:
    """Clip raster to boundary shapefile."""
    boundary = gpd.read_file(boundary_path)
    
    with rasterio.open(raster_path) as src:
        # Reproject boundary to match raster CRS if needed
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)
        
        clipped, transform = mask(src, boundary.geometry, crop=crop)
        meta = src.meta.copy()
        
        meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': transform
        })
    
    write_raster(clipped, output_path, meta)
    
    return clipped, meta


def reproject_raster(
    src_path: str,
    dst_path: str,
    dst_crs: str = 'EPSG:32644',
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """Reproject raster to different CRS."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )


def calculate_area_statistics(
    classified_raster: str,
    pixel_area: float,
    class_names: List[str]
) -> dict:
    """Calculate area statistics for each class."""
    data, _ = read_raster(classified_raster)
    
    if len(data.shape) == 3:
        data = data[0]
    
    stats = {}
    total_pixels = data.size
    
    for class_idx, class_name in enumerate(class_names):
        class_pixels = np.sum(data == class_idx)
        class_area_sqm = class_pixels * pixel_area
        class_area_sqkm = class_area_sqm / 1_000_000
        percentage = (class_pixels / total_pixels) * 100
        
        stats[class_name] = {
            'pixels': int(class_pixels),
            'area_sqm': float(class_area_sqm),
            'area_sqkm': float(class_area_sqkm),
            'percentage': float(percentage)
        }
    
    return stats


def create_tiles(
    raster_path: str,
    output_dir: str,
    tile_size: int = 256,
    overlap: int = 32
) -> List[str]:
    """Split raster into tiles for processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tile_paths = []
    
    with rasterio.open(raster_path) as src:
        height, width = src.shape
        
        for i in range(0, height, tile_size - overlap):
            for j in range(0, width, tile_size - overlap):
                window = rasterio.windows.Window(
                    j, i,
                    min(tile_size, width - j),
                    min(tile_size, height - i)
                )
                
                tile_data = src.read(window=window)
                
                tile_meta = src.meta.copy()
                tile_meta.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)
                })
                
                tile_path = output_path / f"tile_{i}_{j}.tif"
                write_raster(tile_data, str(tile_path), tile_meta)
                tile_paths.append(str(tile_path))
    
    return tile_paths
