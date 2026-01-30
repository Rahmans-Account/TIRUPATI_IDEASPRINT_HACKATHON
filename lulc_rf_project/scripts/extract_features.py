#!/usr/bin/env python3
"""
Feature Extraction Script for LULC Classification
Extracts spectral bands and calculates vegetation/urban indices
"""


# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import rasterio
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
import sys


class FeatureExtractor:
    """Extract features from satellite imagery for LULC classification"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.band_map = self.config['band_mapping']
        
    def calculate_ndvi(self, nir, red):
        """Normalized Difference Vegetation Index"""
        return np.where(
            (nir + red) == 0,
            0,
            (nir - red) / (nir + red)
        )
    
    def calculate_ndbi(self, swir, nir):
        """Normalized Difference Built-up Index"""
        return np.where(
            (swir + nir) == 0,
            0,
            (swir - nir) / (swir + nir)
        )
    
    def calculate_ndwi(self, green, nir):
        """Normalized Difference Water Index"""
        return np.where(
            (green + nir) == 0,
            0,
            (green - nir) / (green + nir)
        )
    
    def calculate_savi(self, nir, red, L=0.5):
        """Soil Adjusted Vegetation Index"""
        return np.where(
            (nir + red + L) == 0,
            0,
            ((nir - red) / (nir + red + L)) * (1 + L)
        )
    
    def calculate_evi(self, nir, red, blue):
        """Enhanced Vegetation Index"""
        denominator = nir + 6 * red - 7.5 * blue + 1
        return np.where(
            denominator == 0,
            0,
            2.5 * ((nir - red) / denominator)
        )
    
    def extract_features_from_image(self, image_path):
        """Extract all features from a single image"""
        try:
            with rasterio.open(image_path) as src:
                # Read bands
                bands = {}
                for i in range(1, min(src.count + 1, 8)):
                    bands[f'B{i}'] = src.read(i).astype(np.float32)
                
                # Get specific bands for indices
                blue = bands.get('B1', bands['B2'])  # Adjust based on sensor
                green = bands.get('B2', bands['B3'])
                red = bands.get('B3', bands['B4'])
                nir = bands.get('B4', bands['B5'])
                swir1 = bands.get('B5', bands['B6'])
                
                # Calculate indices
                ndvi = self.calculate_ndvi(nir, red)
                ndbi = self.calculate_ndbi(swir1, nir)
                ndwi = self.calculate_ndwi(green, nir)
                savi = self.calculate_savi(nir, red)
                evi = self.calculate_evi(nir, red, blue)
                
                # Stack all features
                feature_list = list(bands.values()) + [ndvi, ndbi, ndwi, savi, evi]
                features = np.stack(feature_list, axis=-1)
                
                return features, src.meta
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None
    
    def extract_training_data(self, image_path, label_path):
        """Extract features and labels for training"""
        print(f"Processing: {image_path}")
        
        # Extract features
        features, meta = self.extract_features_from_image(image_path)
        if features is None:
            return None, None
        
        # Load labels
        try:
            with rasterio.open(label_path) as src:
                labels = src.read(1)
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")
            return None, None
        
        # Reshape to (n_pixels, n_features)
        h, w, n_features = features.shape
        X = features.reshape(-1, n_features)
        y = labels.flatten()
        
        # Remove invalid pixels (e.g., no-data values)
        valid_mask = ~np.isnan(X).any(axis=1) & (y >= 0) & (y < 5)
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def process_directory(self, input_dir, label_dir, output_dir):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        label_path = Path(label_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        
        all_features = []
        all_labels = []
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in tqdm(image_files, desc="Extracting features"):
            # Find corresponding label file
            label_file = label_path / f"{img_file.stem}_label.tif"
            if not label_file.exists():
                label_file = label_path / f"{img_file.stem}.tif"
            
            if not label_file.exists():
                print(f"Warning: No label file found for {img_file.name}")
                continue
            
            X, y = self.extract_training_data(img_file, label_file)
            
            if X is not None and y is not None:
                all_features.append(X)
                all_labels.append(y)
        
        if all_features:
            # Concatenate all samples
            X_combined = np.vstack(all_features)
            y_combined = np.hstack(all_labels)
            
            # Save to disk
            np.save(output_path / 'features.npy', X_combined)
            np.save(output_path / 'labels.npy', y_combined)
            
            print(f"\n✅ Feature extraction complete!")
            print(f"   Total samples: {X_combined.shape[0]:,}")
            print(f"   Features per sample: {X_combined.shape[1]}")
            print(f"   Class distribution:")
            for class_id in range(5):
                count = np.sum(y_combined == class_id)
                percentage = (count / len(y_combined)) * 100
                class_name = self.config['classes'][class_id]['name']
                print(f"     {class_name} (Class {class_id}): {count:,} ({percentage:.2f}%)")
            
            return X_combined, y_combined
        else:
            print("❌ No valid training data extracted")
            return None, None


def main():
    parser = argparse.ArgumentParser(description='Extract features from satellite imagery')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--labels', type=str, required=True, help='Directory with label masks')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.config)
    extractor.process_directory(args.input, args.labels, args.output)


if __name__ == '__main__':
    main()
