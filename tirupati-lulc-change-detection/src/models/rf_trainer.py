# Random Forest training for LULC (from lulc_rf_project)

import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LULCTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_params = self.config['model']
        self.classes = self.config['classes']
        self.model = None
    def load_data(self, features_path, labels_path):
        X = np.load(features_path)
        y = np.load(labels_path)
        # Ensure labels are integer class indices for classification
        y = np.round(y).astype(np.int32)
        return X, y
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        # Check if stratify is possible
        unique, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            print("[WARNING] Some classes have less than 2 samples. Proceeding without stratification.")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    def train(self, X_train, y_train):
        # Always ensure y_train is integer class indices
        y_train = np.round(y_train).astype(np.int32)
        if X_train.shape[0] > 100000:
            print(f"[WARNING] Training on {X_train.shape[0]:,} samples. Consider subsampling for speed.")
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
    def save(self, model_path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
