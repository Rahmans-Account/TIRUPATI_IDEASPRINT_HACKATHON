import numpy as np
import joblib
from pathlib import Path
import rasterio
from src.features.spectral_indices import calculate_all_indices

input_raster = Path('data/processed/clipped/Tirupati_Landsat_2024_clipped.tif')
model_path = Path('models/random_forest/model.pkl')
output_raster = Path('data/results/classifications/lulc_2024_rf.tif')

print(f'Random Forest classification saved to {output_raster}')
from src.models.rf_predictor import LULCPredictor

input_raster = 'data/processed/clipped/Tirupati_Landsat_2024_clipped.tif'
model_path = 'models/random_forest/model.pkl'
output_raster = 'data/results/classifications/lulc_2024_rf.tif'

predictor = LULCPredictor(model_path, 'config/config.yaml')
predictor.predict_image(input_raster, output_raster)
print(f'Random Forest classification saved to {output_raster}')
