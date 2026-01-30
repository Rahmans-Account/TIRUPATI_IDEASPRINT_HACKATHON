"""Preprocess all satellite data."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_utils import get_full_config
from src.utils.logger import default_logger as logger
from src.data.preprocessing import load_landsat_bands, normalize_image
from src.features.spectral_indices import calculate_all_indices
from src.utils.geo_utils import clip_raster_to_boundary, create_tiles, write_raster


def _extract_band_dict(image_data):
    """Map multiband Landsat array to named bands used by indices."""
    if image_data.ndim != 3 or image_data.shape[0] < 5:
        return {}

    # Assumes Landsat stack ordered as B2, B3, B4, B5, B6, B7
    return {
        "blue": image_data[0],
        "green": image_data[1],
        "red": image_data[2],
        "nir": image_data[3],
        "swir": image_data[4]
    }


def preprocess_pipeline(config, clip_only: bool = False, years: list[int] | None = None):
    """Run preprocessing pipeline.

    Args:
        clip_only: If True, only clip rasters to boundary (skips indices and tiling).
    """
    
    logger.info("Starting preprocessing pipeline...")
    
    # Get paths
    raw_dir = Path(config.data.raw.landsat)
    boundary_path = Path(config.data.shapefiles.boundary)
    clipped_dir = Path(config.data.processed.clipped)
    indices_dir = Path(config.data.processed.indices)
    tiles_dir = Path(config.data.processed.tiles) / "train"
    
    # Create output directories
    clipped_dir.mkdir(parents=True, exist_ok=True)
    if not clip_only:
        indices_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    year_list = years or [config.time_periods.t1.year, config.time_periods.t2.year]
    for year_dir in year_list:
        year_path = raw_dir / str(year_dir)
        
        if not year_path.exists():
            logger.warning(f"Year directory not found: {year_path}")
            continue
        
        logger.info(f"Processing year {year_dir}...")
        
        # Find satellite images
        image_files = list(year_path.glob("*.TIF"))
        
        if not image_files:
            logger.warning(f"No images found in {year_path}")
            continue
        
        for image_path in image_files:
            logger.info(f"Processing {image_path.name}...")
            
            # Step 1: Clip to boundary
            clipped_path = clipped_dir / f"{image_path.stem}_clipped.tif"
            if not clipped_path.exists():
                try:
                    clip_raster_to_boundary(
                        str(image_path),
                        str(boundary_path),
                        str(clipped_path)
                    )
                    logger.info(f"Clipped to boundary: {clipped_path.name}")
                except Exception as e:
                    logger.error(f"Error clipping: {e}")
                    continue
            
            if clip_only:
                continue

            # Step 2: Calculate indices
            logger.info("Calculating spectral indices...")
            try:
                image_data, meta = load_landsat_bands(str(clipped_path))
                image_data = normalize_image(image_data, method=config.preprocessing.normalization_method)

                band_dict = _extract_band_dict(image_data)
                if not band_dict:
                    logger.warning("Unable to map bands for indices (expected 5+ bands).")
                else:
                    indices = calculate_all_indices(band_dict)
                    for name, index_data in indices.items():
                        index_path = indices_dir / f"{image_path.stem}_{name}.tif"
                        index_meta = meta.copy()
                        write_raster(index_data.astype("float32"), str(index_path), index_meta, nodata=-9999)
                        logger.info(f"Saved index: {index_path.name}")
            except Exception as e:
                logger.error(f"Error calculating indices: {e}")
            
            # Step 3: Create tiles
            if config.preprocessing.tile_size:
                logger.info("Creating tiles...")
                try:
                    tile_paths = create_tiles(
                        str(clipped_path),
                        str(tiles_dir / image_path.stem),
                        tile_size=config.preprocessing.tile_size,
                        overlap=config.preprocessing.tile_overlap
                    )
                    logger.info(f"Created {len(tile_paths)} tiles")
                except Exception as e:
                    logger.error(f"Error creating tiles: {e}")
    
    logger.info("Preprocessing completed!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess satellite data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--clip-only', action='store_true',
                       help='Only clip rasters to boundary (fast mode)')
    parser.add_argument('--years', nargs=2, type=int,
                       help='Two years to process (e.g., 2015 2023)')
    args = parser.parse_args()
    
    config = get_full_config()
    years = list(args.years) if args.years else None
    preprocess_pipeline(config, clip_only=args.clip_only, years=years)


if __name__ == '__main__':
    main()
