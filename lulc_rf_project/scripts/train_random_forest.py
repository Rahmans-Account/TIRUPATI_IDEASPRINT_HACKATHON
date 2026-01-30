#!/usr/bin/env python3
"""
Random Forest Training Script for LULC Classification
Includes cross-validation, hyperparameter tuning, and model persistence
"""


# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json


class LULCTrainer:
    """Train Random Forest model for LULC classification"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_params = self.config['model']
        self.classes = self.config['classes']
        self.model = None
        self.training_history = {}
        
    def load_data(self, features_path, labels_path):
        """Load preprocessed features and labels"""
        print("Loading training data...")
        X = np.load(features_path)
        y = np.load(labels_path)
        
        print(f"✓ Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
        print(f"✓ Class distribution:")
        for class_id, class_info in self.classes.items():
            count = np.sum(y == class_id)
            percentage = (count / len(y)) * 100
            print(f"  {class_info['name']}: {count:,} ({percentage:.2f}%)")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and validation sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"✓ Training samples: {X_train.shape[0]:,}")
        print(f"✓ Validation samples: {X_val.shape[0]:,}")
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train, y_train, tune_hyperparameters=False):
        """Train Random Forest classifier"""
        
        if tune_hyperparameters:
            print("\n🔧 Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            print("\n🌲 Training Random Forest model...")
            self.model = RandomForestClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
        
        print("✓ Model training complete!")
        
        return self.model
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Perform grid search for hyperparameter tuning"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [20, 30, 40],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(
            random_state=self.model_params['random_state'],
            n_jobs=self.model_params['n_jobs']
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1, verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.training_history['best_params'] = grid_search.best_params_
        self.training_history['best_cv_score'] = grid_search.best_score_
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, X_val, y_val, save_plots=True, output_dir='results'):
        """Evaluate model performance on validation set"""
        print("\n📊 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        
        # Calculate metrics
        report = classification_report(
            y_val, y_pred,
            target_names=[self.classes[i]['name'] for i in sorted(self.classes.keys())],
            output_dict=True
        )
        
        # Print results
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            y_val, y_pred,
            target_names=[self.classes[i]['name'] for i in sorted(self.classes.keys())]
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        # Store in history
        self.training_history['accuracy'] = report['accuracy']
        self.training_history['report'] = report
        self.training_history['confusion_matrix'] = cm.tolist()
        
        if save_plots:
            self._save_evaluation_plots(cm, output_dir)
        
        return report, cm
    
    def _save_evaluation_plots(self, cm, output_dir):
        """Save confusion matrix and feature importance plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        class_names = [self.classes[i]['name'] for i in sorted(self.classes.keys())]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix to {output_path / 'confusion_matrix.png'}")
        
        # Feature Importance
        importances = self.model.feature_importances_
        feature_names = self._get_feature_names()
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Feature Importances', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved feature importance to {output_path / 'feature_importance.png'}")
    
    def _get_feature_names(self):
        """Generate feature names"""
        bands = self.config['features']['bands']
        indices = self.config['features']['indices']
        return bands + indices
    
    def save_model(self, output_path='models/random_forest/model.pkl'):
        """Save trained model to disk"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_file)
        print(f"\n✓ Model saved to {output_file}")
        
        # Save training history
        history_file = output_file.parent / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"✓ Training history saved to {history_file}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier',
            'n_features': self.model.n_features_in_,
            'n_classes': len(self.classes),
            'class_names': {k: v['name'] for k, v in self.classes.items()}
        }
        
        metadata_file = output_file.parent / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file}")
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(
            self.model, X, y,
            cv=cv, n_jobs=-1, scoring='accuracy'
        )
        
        print(f"✓ Cross-validation scores: {scores}")
        print(f"✓ Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        self.training_history['cv_scores'] = scores.tolist()
        self.training_history['cv_mean'] = scores.mean()
        self.training_history['cv_std'] = scores.std()
        
        return scores


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest for LULC classification')
    parser.add_argument('--features', type=str, required=True, help='Path to features.npy')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels.npy')
    parser.add_argument('--output', type=str, default='models/random_forest/model.pkl',
                        help='Output path for trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--cv', action='store_true',
                        help='Perform cross-validation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LULCTrainer(args.config)
    
    # Load data
    X, y = trainer.load_data(args.features, args.labels)
    
    # Split data
    X_train, X_val, y_train, y_val = trainer.prepare_data(X, y)
    
    # Train model
    trainer.train_model(X_train, y_train, tune_hyperparameters=args.tune)
    
    # Cross-validation (optional)
    if args.cv:
        trainer.cross_validate(X_train, y_train)
    
    # Evaluate
    trainer.evaluate_model(X_val, y_val)
    
    # Save model
    trainer.save_model(args.output)
    
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
