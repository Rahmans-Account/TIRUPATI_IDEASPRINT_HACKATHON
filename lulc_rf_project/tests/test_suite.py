#!/usr/bin/env python3
"""
Test Suite for LULC Classification System
Run tests to verify all components work correctly
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_features import FeatureExtractor
from scripts.train_random_forest import LULCTrainer
from scripts.run_rf_inference import LULCPredictor
from scripts.evaluate_random_forest import LULCEvaluator


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction module"""
    
    def setUp(self):
        self.extractor = FeatureExtractor()
    
    def test_ndvi_calculation(self):
        """Test NDVI calculation"""
        nir = np.array([0.8, 0.7, 0.6])
        red = np.array([0.2, 0.3, 0.4])
        ndvi = self.extractor.calculate_ndvi(nir, red)
        
        expected = np.array([0.6, 0.4, 0.2])
        np.testing.assert_array_almost_equal(ndvi, expected)
    
    def test_ndbi_calculation(self):
        """Test NDBI calculation"""
        swir = np.array([0.6, 0.5, 0.4])
        nir = np.array([0.2, 0.3, 0.4])
        ndbi = self.extractor.calculate_ndbi(swir, nir)
        
        expected = np.array([0.5, 0.25, 0.0])
        np.testing.assert_array_almost_equal(ndbi, expected)
    
    def test_ndwi_calculation(self):
        """Test NDWI calculation"""
        green = np.array([0.5, 0.4, 0.3])
        nir = np.array([0.3, 0.4, 0.5])
        ndwi = self.extractor.calculate_ndwi(green, nir)
        
        expected = np.array([0.25, 0.0, -0.25])
        np.testing.assert_array_almost_equal(ndwi, expected)


class TestModelTraining(unittest.TestCase):
    """Test model training module"""
    
    def setUp(self):
        self.trainer = LULCTrainer()
        
        # Create synthetic data
        np.random.seed(42)
        self.X = np.random.randn(1000, 12)
        self.y = np.random.randint(0, 5, 1000)
    
    def test_data_preparation(self):
        """Test data splitting"""
        X_train, X_val, y_train, y_val = self.trainer.prepare_data(
            self.X, self.y, test_size=0.2
        )
        
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_val), 200)
        self.assertEqual(X_train.shape[1], 12)
    
    def test_model_training(self):
        """Test model can be trained"""
        X_train, X_val, y_train, y_val = self.trainer.prepare_data(
            self.X, self.y, test_size=0.2
        )
        
        model = self.trainer.train_model(X_train, y_train)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        
        # Test prediction
        predictions = model.predict(X_val)
        self.assertEqual(len(predictions), len(y_val))


class TestInference(unittest.TestCase):
    """Test inference module"""
    
    def setUp(self):
        # Create a temporary model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train on dummy data
        X = np.random.randn(100, 12)
        y = np.random.randint(0, 5, 100)
        model.fit(X, y)
        
        # Save to temp file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / 'test_model.pkl'
        
        import joblib
        joblib.dump(model, self.model_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_model_loading(self):
        """Test model can be loaded"""
        predictor = LULCPredictor(str(self.model_path))
        
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.model.n_features_in_, 12)


class TestEvaluation(unittest.TestCase):
    """Test evaluation module"""
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly"""
        # Perfect predictions
        y_true = np.array([0, 1, 2, 3, 4] * 20)
        y_pred = y_true.copy()
        y_proba = np.eye(5)[y_true]
        
        # Create evaluator with temp model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 12)
        model.fit(X, y_true)
        
        temp_dir = tempfile.mkdtemp()
        model_path = Path(temp_dir) / 'test_model.pkl'
        
        import joblib
        joblib.dump(model, model_path)
        
        evaluator = LULCEvaluator(str(model_path))
        results = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        # Perfect accuracy
        self.assertEqual(results['overall_accuracy'], 1.0)
        self.assertEqual(results['kappa'], 1.0)
        
        shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 RUNNING TEST SUITE")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestInference))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
