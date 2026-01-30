"""Random Forest classifier for LULC classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Tuple, Optional


class LULCRandomForest:
    """Random Forest model for LULC classification."""
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 30,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = 'balanced'
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight
        )
    
    def prepare_data(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image data for Random Forest.
        
        Args:
            image: Image array (channels, height, width)
            
        Returns:
            Reshaped array (n_pixels, n_features)
        """
        n_channels, height, width = image.shape
        return image.reshape(n_channels, -1).T
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> dict:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score
        }
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            
            y_pred = self.model.predict(X_val)
            print("Validation Classification Report:")
            print(classification_report(y_val, y_pred))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def predict_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict LULC classes for entire image.
        
        Args:
            image: Image array (channels, height, width)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        n_channels, height, width = image.shape
        
        # Prepare data
        X = self.prepare_data(image)
        
        # Predict
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Reshape to image dimensions
        pred_image = predictions.reshape(height, width)
        prob_image = probabilities.reshape(height, width, -1)
        
        return pred_image, prob_image
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LULCRandomForest':
        """Load model from file."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
