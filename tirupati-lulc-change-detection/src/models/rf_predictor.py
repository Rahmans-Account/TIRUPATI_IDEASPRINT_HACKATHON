# Random Forest inference for LULC (from lulc_rf_project)
import numpy as np
import rasterio
import joblib
from pathlib import Path
from src.features.feature_extractor import FeatureExtractor

class LULCPredictor:
    def __init__(self, model_path, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = joblib.load(model_path)
        self.feature_extractor = FeatureExtractor(config_path)
        self.classes = self.config['classes']
    def predict_image(self, image_path, output_path=None):
        # Example: load image, extract features, predict, save output
        with rasterio.open(image_path) as src:
            img = src.read()
            profile = src.profile
        n_bands, height, width = img.shape
        X = img.reshape(n_bands, -1).T
        pred = self.model.predict(X)
        pred_img = pred.reshape(height, width).astype(np.uint8)
        if output_path:
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(pred_img, 1)
        return pred_img
