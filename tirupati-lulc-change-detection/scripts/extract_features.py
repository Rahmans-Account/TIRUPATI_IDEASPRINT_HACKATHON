# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import rasterio
from pathlib import Path
from src.features.feature_extractor import FeatureExtractor
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, 'config', 'config.yaml')

# Paths (update as needed)
image_path = os.path.join(project_root, 'data', 'processed', 'clipped', 'Tirupati_Landsat_2018.tif')
label_path = os.path.join(project_root, 'data', 'training', 'labels', 'Tirupati_Landsat_2018_aligned.tif')
features_out = os.path.join(project_root, 'data', 'training', 'features.npy')
labels_out = os.path.join(project_root, 'data', 'training', 'labels.npy')

# Extract features from image
extractor = FeatureExtractor(config_path)
with rasterio.open(image_path) as img_src:
    img = img_src.read()
    # Example: use all bands as features
    n_bands, height, width = img.shape
    X = img.reshape(n_bands, -1).T

# Load labels (must be same shape as image)
with rasterio.open(label_path) as lbl_src:
    y = lbl_src.read(1).flatten()

# Optionally, filter out nodata pixels
mask = (y >= 0)
X = X[mask]
y = y[mask]

np.save(features_out, X)
np.save(labels_out, y)
print(f'Features saved to {features_out}, labels saved to {labels_out}')
