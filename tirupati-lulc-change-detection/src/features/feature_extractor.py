# Feature extraction for LULC classification (from lulc_rf_project)
import numpy as np
import rasterio
from pathlib import Path
import yaml

class FeatureExtractor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.band_map = self.config['band_mapping'] if 'band_mapping' in self.config else {}

    def calculate_ndvi(self, nir, red):
        return np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
    def calculate_ndbi(self, swir, nir):
        return np.where((swir + nir) == 0, 0, (swir - nir) / (swir + nir))
    def calculate_ndwi(self, green, nir):
        return np.where((green + nir) == 0, 0, (green - nir) / (green + nir))
    def calculate_savi(self, nir, red, L=0.5):
        return np.where((nir + red + L) == 0, 0, ((nir - red) / (nir + red + L)) * (1 + L))
    def calculate_evi(self, nir, red, blue):
        denominator = nir + 6 * red - 7.5 * blue + 1
        return np.where(denominator == 0, 0, 2.5 * (nir - red) / denominator)
    # ... (other methods as needed)
