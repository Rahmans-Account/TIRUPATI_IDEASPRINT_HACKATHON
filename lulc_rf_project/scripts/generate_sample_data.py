#!/usr/bin/env python3
"""
Generate synthetic sample data for testing the LULC pipeline
Creates fake satellite imagery and corresponding labels
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import argparse


def generate_synthetic_image(width=1000, height=1000, n_bands=7):
    """Generate synthetic satellite-like imagery"""
    
    # Create realistic-looking spectral bands
    image = np.zeros((n_bands, height, width), dtype=np.float32)
    
    # Band 1-3 (Visible): Random values with some structure
    for band in range(3):
        image[band] = np.random.normal(0.3, 0.1, (height, width))
    
    # Band 4 (NIR): Higher values for vegetation
    image[3] = np.random.normal(0.6, 0.15, (height, width))
    
    # Band 5-6 (SWIR): Similar to visible
    for band in range(4, 6):
        image[band] = np.random.normal(0.25, 0.08, (height, width))
    
    # Add some spatial structure (Gaussian smoothing)
    from scipy.ndimage import gaussian_filter
    for band in range(n_bands):
        image[band] = gaussian_filter(image[band], sigma=3)
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image


def generate_synthetic_labels(width=1000, height=1000):
    """Generate synthetic LULC labels with realistic patterns"""
    
    labels = np.zeros((height, width), dtype=np.uint8)
    
    # Create regions for different classes
    # Water (0): Bottom left
    labels[int(height*0.7):, :int(width*0.3)] = 0
    
    # Vegetation (1): Top half with some noise
    labels[:int(height*0.5), :] = 1
    
    # Urban (2): Center
    labels[int(height*0.4):int(height*0.7), int(width*0.3):int(width*0.7)] = 2
    
    # Barren (3): Right side
    labels[:, int(width*0.7):] = 3
    
    # Agriculture (4): Bottom right
    labels[int(height*0.7):, int(width*0.5):int(width*0.7)] = 4
    
    # Add some noise/mixing
    noise = np.random.random((height, width))
    labels[noise > 0.9] = np.random.randint(0, 5, size=np.sum(noise > 0.9))
    
    # Smooth boundaries slightly
    from scipy.ndimage import median_filter
    labels = median_filter(labels, size=3)
    
    return labels


def save_geotiff(data, output_path, is_label=False):
    """Save data as GeoTIFF with georeferencing"""
    
    if is_label:
        # Single band for labels
        height, width = data.shape
        count = 1
        dtype = 'uint8'
    else:
        # Multi-band for imagery
        count, height, width = data.shape
        dtype = 'float32'
    
    # Create transform (fake coordinates)
    transform = from_bounds(0, 0, width, height, width, height)
    
    # Create metadata
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': dtype,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        if is_label:
            dst.write(data, 1)
        else:
            for i in range(count):
                dst.write(data[i], i + 1)
    
    print(f"✓ Saved: {output_path}")


def generate_sample_dataset(output_dir, n_images=3, size=1000):
    """Generate complete sample dataset"""
    
    output_path = Path(output_dir)
    
    print(f"Generating {n_images} sample images...")
    
    for i in range(n_images):
        print(f"\nGenerating sample {i+1}/{n_images}...")
        
        # Generate image
        image = generate_synthetic_image(size, size)
        image_file = output_path / 'raw' / f'sample_image_{i+1}.tif'
        save_geotiff(image, image_file, is_label=False)
        
        # Generate labels
        labels = generate_synthetic_labels(size, size)
        label_file = output_path / 'training' / 'labels' / f'sample_image_{i+1}_label.tif'
        save_geotiff(labels, label_file, is_label=True)
    
    print(f"\n✅ Sample dataset created in {output_path}")
    print(f"   Images: {output_path / 'raw'}")
    print(f"   Labels: {output_path / 'training' / 'labels'}")
    print("\nNext steps:")
    print("1. Extract features: python scripts/extract_features.py --input data/raw --labels data/training/labels --output data/training")
    print("2. Train model: python scripts/train_random_forest.py --features data/training/features.npy --labels data/training/labels.npy")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic sample data')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--n-images', type=int, default=3, help='Number of images to generate')
    parser.add_argument('--size', type=int, default=1000, help='Image size (width and height)')
    
    args = parser.parse_args()
    
    generate_sample_dataset(args.output, args.n_images, args.size)


if __name__ == '__main__':
    main()
