import sys
import os
from pathlib import Path
import traceback
# Ensure project root is in sys.path for src imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from src.models.rf_trainer import LULCTrainer

# Use absolute paths for reliability
features_path = os.path.join(project_root, 'data', 'training', 'features.npy')
labels_path = os.path.join(project_root, 'data', 'training', 'labels.npy')
model_path = os.path.join(project_root, 'models', 'random_forest', 'model.pkl')
config_path = os.path.join(project_root, 'config', 'config.yaml')

try:
    trainer = LULCTrainer(config_path)
    print('Loading data...')
    X, y = trainer.load_data(features_path, labels_path)
    print(f'Loaded features: {X.shape}, labels: {y.shape}')
    X_train, X_val, y_train, y_val = trainer.prepare_data(X, y)
    print('Training model...')
    trainer.train(X_train, y_train)
    print('Saving model...')
    trainer.save(model_path)
    print(f'Model trained and saved to {model_path}')
except Exception as e:
    print('Error during training:')
    traceback.print_exc()
    sys.exit(1)
