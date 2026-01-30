#!/usr/bin/env python3
"""
Random Forest Inference Script for LULC Classification
Predicts land cover classes for new satellite imagery
"""


# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm

from scripts.extract_features import FeatureExtractor


class LULCPredictor:
    """Run inference on satellite imagery using trained Random Forest model"""
    
    def __init__(self, model_path, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Features: {self.model.n_features_in_}")
        print(f"  Classes: {self.model.n_classes_}")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config_path)
        
        self.classes = self.config['classes']
    
    def predict_image(self, image_path, output_path=None, chunk_size=1000):
        """
        Predict LULC classes for an entire image
        Uses chunked processing for memory efficiency
        """
        print(f"\n🔍 Processing image: {image_path}")
        
        # Extract features
        print("Extracting features...")
        features, meta = self.feature_extractor.extract_features_from_image(image_path)
        
        if features is None:
            print("❌ Failed to extract features")
            return None
        
        h, w, n_features = features.shape
        print(f"✓ Image shape: {h}x{w}, Features: {n_features}")
        
        # Reshape for prediction
        X = features.reshape(-1, n_features)
        
        # Handle NaN values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        print(f"Predicting {X_valid.shape[0]:,} valid pixels...")
        
        # Predict in chunks to manage memory
        predictions = np.zeros(X.shape[0], dtype=np.uint8)
        
        for i in tqdm(range(0, X_valid.shape[0], chunk_size), desc="Predicting"):
            chunk = X_valid[i:i+chunk_size]
            chunk_pred = self.model.predict(chunk)
            
            # Map back to original indices
            valid_indices = np.where(valid_mask)[0]
            predictions[valid_indices[i:i+chunk_size]] = chunk_pred
        
        # Reshape to image dimensions
        prediction_map = predictions.reshape(h, w)
        
        # Save output
        if output_path:
            self._save_prediction(prediction_map, output_path, meta)
        
        # Print class distribution
        self._print_statistics(prediction_map)
        
        return prediction_map
    
    def predict_with_probability(self, image_path, output_path=None):
        """Predict with class probabilities"""
        print(f"\n🔍 Processing image with probabilities: {image_path}")
        
        # Extract features
        features, meta = self.feature_extractor.extract_features_from_image(image_path)
        
        if features is None:
            return None, None
        
        h, w, n_features = features.shape
        X = features.reshape(-1, n_features)
        
        # Handle NaN values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        print(f"Predicting with probabilities...")
        
        # Predict classes and probabilities
        predictions = np.zeros(X.shape[0], dtype=np.uint8)
        probabilities = np.zeros((X.shape[0], self.model.n_classes_), dtype=np.float32)
        
        valid_indices = np.where(valid_mask)[0]
        
        predictions[valid_indices] = self.model.predict(X_valid)
        probabilities[valid_indices] = self.model.predict_proba(X_valid)
        
        # Reshape
        prediction_map = predictions.reshape(h, w)
        probability_maps = probabilities.reshape(h, w, -1)
        
        # Save outputs
        if output_path:
            self._save_prediction(prediction_map, output_path, meta)
            
            # Save probability maps
            prob_path = Path(output_path).parent / f"{Path(output_path).stem}_probabilities.tif"
            self._save_probabilities(probability_maps, prob_path, meta)
        
        return prediction_map, probability_maps
    
    def _save_prediction(self, prediction_map, output_path, meta):
        """Save prediction map as GeoTIFF"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        meta.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(prediction_map, 1)
            
            # Add color map
            colormap = {i: tuple(self.classes[i]['color']) 
                       for i in range(len(self.classes))}
            dst.write_colormap(1, colormap)
        
        print(f"✓ Prediction saved to {output_file}")
    
    def _save_probabilities(self, probability_maps, output_path, meta):
        """Save probability maps as multi-band GeoTIFF"""
        n_classes = probability_maps.shape[2]
        
        meta.update({
            'driver': 'GTiff',
            'count': n_classes,
            'dtype': 'float32',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(n_classes):
                dst.write(probability_maps[:, :, i], i + 1)
                dst.set_band_description(i + 1, self.classes[i]['name'])
        
        print(f"✓ Probabilities saved to {output_path}")
    
    def _print_statistics(self, prediction_map):
        """Print classification statistics"""
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        
        total_pixels = prediction_map.size
        
        for class_id, class_info in sorted(self.classes.items()):
            count = np.sum(prediction_map == class_id)
            percentage = (count / total_pixels) * 100
            area_km2 = count * 0.0009  # Assuming 30m pixels
            
            print(f"{class_info['name']:15s}: {count:10,} pixels ({percentage:5.2f}%) ~{area_km2:,.1f} km²")
        
        print("="*60)
    
    def batch_predict(self, input_dir, output_dir):
        """Process multiple images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        
        print(f"\n📁 Processing {len(image_files)} images...")
        
        for img_file in image_files:
            output_file = output_path / f"{img_file.stem}_classified.tif"
            try:
                self.predict_image(img_file, output_file)
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
        
        print(f"\n✓ Batch processing complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run LULC inference with Random Forest')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output path or directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    parser.add_argument('--probabilities', action='store_true', help='Save probability maps')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = LULCPredictor(args.model, args.config)
    
    # Run prediction
    if args.batch:
        predictor.batch_predict(args.input, args.output)
    else:
        if args.probabilities:
            predictor.predict_with_probability(args.input, args.output)
        else:
            predictor.predict_image(args.input, args.output)
    
    print("\n✅ Inference complete!")


if __name__ == '__main__':
    main()
