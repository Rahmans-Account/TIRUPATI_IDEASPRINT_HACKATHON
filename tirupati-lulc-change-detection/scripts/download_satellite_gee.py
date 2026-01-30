"""
Download Landsat satellite imagery for Tirupati using Google Earth Engine.

Prerequisites:
1. Sign up for Google Earth Engine: https://earthengine.google.com/
2. Install earthengine-api: pip install earthengine-api
3. Authenticate: earthengine authenticate
"""

import ee
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_utils import get_full_config
from src.utils.logger import default_logger as logger
import geopandas as gpd


def initialize_earth_engine():
    """Initialize Google Earth Engine with your project."""
    try:
        # Initialize with your GCP project
        ee.Initialize(project='gen-lang-client-0197533066')
        logger.info("Earth Engine initialized successfully with project: gen-lang-client-0197533066")
    except Exception as e:
        logger.error(f"Earth Engine initialization failed: {e}")
        logger.info("Make sure Earth Engine API is enabled in your GCP project:")
        logger.info("https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
        sys.exit(1)


def load_study_area(boundary_path: str):
    """Load study area boundary and convert to Earth Engine geometry."""
    gdf = gpd.read_file(boundary_path)
    
    # Reproject to WGS84 if needed
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Create EE geometry
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
    
    logger.info(f"Study area bounds: {bounds}")
    return region, gdf


def get_landsat_composite(region, start_date: str, end_date: str, max_cloud: int = 20):
    """
    Get cloud-masked Landsat composite for a time period.
    
    Args:
        region: Earth Engine geometry
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud: Maximum cloud cover percentage
        
    Returns:
        Earth Engine image with bands
    """
    
    # Landsat 8/9 Collection 2 Surface Reflectance
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
    
    logger.info(f"Found {collection.size().getInfo()} images for {start_date} to {end_date}")
    
    def mask_clouds(image):
        """Apply cloud mask using QA_PIXEL band."""
        qa = image.select('QA_PIXEL')
        # Bit 3: Cloud
        # Bit 4: Cloud shadow
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return image.updateMask(cloud_mask)
    
    def scale_bands(image):
        """Apply scaling factors for surface reflectance."""
        optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical, None, True).addBands(thermal, None, True)
    
    # Apply cloud masking and scaling
    composite = collection \
        .map(mask_clouds) \
        .map(scale_bands) \
        .median() \
        .clip(region)
    
    # Select bands: Blue, Green, Red, NIR, SWIR1, SWIR2
    composite = composite.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    
    return composite


def export_to_drive(image, region, description: str, folder: str = 'LULC_Tirupati'):
    """
    Export image to Google Drive.
    
    Args:
        image: Earth Engine image
        region: Export region
        description: Export description
        folder: Google Drive folder name
    """
    
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=30,  # 30m resolution for Landsat
        maxPixels=1e13,
        crs='EPSG:32644',  # UTM Zone 44N for Tirupati
        fileFormat='GeoTIFF'
    )
    
    task.start()
    logger.info(f"Export task started: {description}")
    logger.info(f"Check status at: https://code.earthengine.google.com/tasks")
    
    return task


def download_satellite_data(config, years: list = [2018, 2024]):
    """
    Download Landsat data for multiple years.
    
    Args:
        config: Configuration object
        years: List of years to download
    """
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Load study area
    boundary_path = Path(config.data.shapefiles.boundary)
    if not boundary_path.exists():
        logger.error(f"Boundary shapefile not found: {boundary_path}")
        return
    
    region, gdf = load_study_area(str(boundary_path))
    
    # Download data for each year
    tasks = []
    
    for year in years:
        # Use dry season months (Jan-Mar) for better clarity
        start_date = f"{year}-01-01"
        end_date = f"{year}-03-31"
        
        logger.info(f"Processing year {year}...")
        
        try:
            composite = get_landsat_composite(region, start_date, end_date)
            
            # Export to Google Drive
            description = f"Tirupati_Landsat_{year}"
            task = export_to_drive(composite, region.getInfo()['coordinates'], description)
            tasks.append((year, task))
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Export tasks submitted successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Go to: https://code.earthengine.google.com/tasks")
    logger.info("2. Wait for tasks to complete (5-15 minutes)")
    logger.info("3. Download files from Google Drive folder: LULC_Tirupati")
    logger.info("4. Place downloaded files in:")
    for year, _ in tasks:
        logger.info(f"   - data/raw/landsat/{year}/")
    logger.info("\nThen run: python scripts/preprocess_all.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Landsat data using Google Earth Engine')
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2024],
                       help='Years to download (default: 2018 2024)')
    args = parser.parse_args()
    
    config = get_full_config()
    download_satellite_data(config, years=args.years)


if __name__ == '__main__':
    main()
