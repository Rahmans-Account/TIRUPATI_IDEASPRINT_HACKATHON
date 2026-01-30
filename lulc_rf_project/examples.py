#!/usr/bin/env python3
"""
Example Usage Script
Demonstrates how to use the LULC classification system programmatically
"""

import numpy as np
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.extract_features import FeatureExtractor
from scripts.train_random_forest import LULCTrainer
from scripts.run_rf_inference import LULCPredictor
from scripts.evaluate_random_forest import LULCEvaluator


def example_1_feature_extraction():
    """Example: Extract features from satellite imagery"""
    print("\n" + "="*70)
    print("Example 1: Feature Extraction")
    print("="*70)
    
    extractor = FeatureExtractor('config/config.yaml')
    
    # Process a single image
    image_path = 'data/raw/sample_image_1.tif'
    label_path = 'data/training/labels/sample_image_1_label.tif'
    
    if Path(image_path).exists() and Path(label_path).exists():
        X, y = extractor.extract_training_data(image_path, label_path)
        print(f"✓ Extracted {X.shape[0]:,} samples with {X.shape[1]} features")
    else:
        print("⚠️  Sample data not found. Run generate_sample_data.py first.")


def example_2_model_training():
    """Example: Train a Random Forest model"""
    print("\n" + "="*70)
    print("Example 2: Model Training")
    print("="*70)
    
    # Check if training data exists
    features_path = 'data/training/features.npy'
    labels_path = 'data/training/labels.npy'
    
    if not Path(features_path).exists():
        print("⚠️  Training data not found. Run feature extraction first.")
        return
    
    # Load data
    trainer = LULCTrainer('config/config.yaml')
    X, y = trainer.load_data(features_path, labels_path)
    
    # Split data
    X_train, X_val, y_train, y_val = trainer.prepare_data(X, y, test_size=0.2)
    
    # Train model
    model = trainer.train_model(X_train, y_train)
    
    # Evaluate
    report, cm = trainer.evaluate_model(X_val, y_val, save_plots=False)
    
    print(f"✓ Model accuracy: {report['accuracy']:.4f}")


def example_3_inference():
    """Example: Run inference on new imagery"""
    print("\n" + "="*70)
    print("Example 3: Inference")
    print("="*70)
    
    model_path = 'models/random_forest/model.pkl'
    
    if not Path(model_path).exists():
        print("⚠️  Model not found. Train the model first.")
        return
    
    # Initialize predictor
    predictor = LULCPredictor(model_path, 'config/config.yaml')
    
    # Run prediction
    test_image = 'data/raw/sample_image_1.tif'
    
    if Path(test_image).exists():
        output_path = 'results/example_classification.tif'
        prediction_map = predictor.predict_image(test_image, output_path)
        print(f"✓ Classification saved to {output_path}")
    else:
        print("⚠️  Test image not found.")


def example_4_evaluation():
    """Example: Evaluate model performance"""
    print("\n" + "="*70)
    print("Example 4: Model Evaluation")
    print("="*70)
    
    model_path = 'models/random_forest/model.pkl'
    
    if not Path(model_path).exists():
        print("⚠️  Model not found. Train the model first.")
        return
    
    # Initialize evaluator
    evaluator = LULCEvaluator(model_path, 'config/config.yaml')
    
    # Load validation data
    val_features = 'data/training/features.npy'
    val_labels = 'data/training/labels.npy'
    
    if Path(val_features).exists():
        X_val, y_val = evaluator.load_validation_data(val_features, val_labels)
        
        # Evaluate
        results, cm, y_pred, y_proba = evaluator.evaluate(X_val, y_val)
        
        print(f"✓ Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"✓ Cohen's Kappa: {results['kappa']:.4f}")
    else:
        print("⚠️  Validation data not found.")


def example_5_custom_workflow():
    """Example: Custom workflow with in-memory processing"""
    print("\n" + "="*70)
    print("Example 5: Custom Workflow")
    print("="*70)
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 10000
    n_features = 12
    
    # Generate features (simulating NDVI, NDBI, etc.)
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (5 classes)
    y = np.random.randint(0, 5, n_samples)
    
    print(f"Created synthetic dataset: {n_samples:,} samples, {n_features} features")
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Model trained on {len(X_train):,} samples")
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    # Feature importance
    importances = model.feature_importances_
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    print("\nTop 5 Most Important Features:")
    indices = np.argsort(importances)[::-1][:5]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")


def main():
    """Run all examples"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🌍 LULC Classification Examples 🌍               ║
    ║                                                               ║
    ║           Programmatic Usage Demonstrations                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    example_1_feature_extraction()
    example_2_model_training()
    example_3_inference()
    example_4_evaluation()
    example_5_custom_workflow()
    
    print("\n" + "="*70)
    print("📚 More Examples:")
    print("="*70)
    print("- Jupyter Notebook: notebooks/exploration.ipynb")
    print("- Quick Start: python quickstart.py")
    print("- Full Documentation: README.md and SETUP_GUIDE.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
