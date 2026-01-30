
import sys
import numpy as np
from pathlib import Path
# Ensure project root is in sys.path for src imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))
from src.models.rf_trainer import LULCTrainer

# Paths
project_root = Path(__file__).resolve().parent.parent
features_path = project_root / 'data' / 'training' / 'features.npy'
labels_path = project_root / 'data' / 'training' / 'labels.npy'
val_features_path = project_root / 'data' / 'training' / 'val_features.npy'
val_labels_path = project_root / 'data' / 'training' / 'val_labels.npy'
config_path = project_root / 'config' / 'config.yaml'

# Load data
trainer = LULCTrainer(str(config_path))
X, y = trainer.load_data(str(features_path), str(labels_path))

# Split and save validation set
X_train, X_val, y_train, y_val = trainer.prepare_data(X, y)
np.save(val_features_path, X_val)
np.save(val_labels_path, y_val)
print(f"Validation features saved to {val_features_path}")
print(f"Validation labels saved to {val_labels_path}")
