#!/usr/bin/env python3
"""
Quick Start Script - Automated Pipeline Execution
Runs the complete LULC classification pipeline with sample data
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")
    return result


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'scikit-learn', 'rasterio', 'matplotlib', 
        'seaborn', 'pandas', 'yaml', 'tqdm', 'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✓ All dependencies installed!")


def main():
    parser = argparse.ArgumentParser(description='Quick start pipeline for LULC classification')
    parser.add_argument('--skip-data-gen', action='store_true', 
                        help='Skip synthetic data generation')
    parser.add_argument('--n-images', type=int, default=3,
                        help='Number of synthetic images to generate')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning (slower)')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🌍 LULC Random Forest Classification Pipeline 🌍       ║
    ║                                                               ║
    ║           Quick Start - Automated Execution                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    check_dependencies()
    
    # Paths
    project_root = Path(__file__).parent
    scripts_dir = project_root / 'scripts'
    data_dir = project_root / 'data'
    
    # Step 1: Generate sample data
    if not args.skip_data_gen:
        run_command(
            [sys.executable, str(scripts_dir / 'generate_sample_data.py'),
             '--output', str(data_dir),
             '--n-images', str(args.n_images),
             '--size', '1000'],
            'Generating synthetic sample data'
        )
    else:
        print("\nℹ️  Skipping data generation (using existing data)")
    
    # Step 2: Extract features
    run_command(
        [sys.executable, str(scripts_dir / 'extract_features.py'),
         '--input', str(data_dir / 'raw'),
         '--labels', str(data_dir / 'training' / 'labels'),
         '--output', str(data_dir / 'training')],
        'Extracting features from satellite imagery'
    )
    
    # Step 3: Train model
    train_cmd = [
        sys.executable, str(scripts_dir / 'train_random_forest.py'),
        '--features', str(data_dir / 'training' / 'features.npy'),
        '--labels', str(data_dir / 'training' / 'labels.npy'),
        '--output', str(project_root / 'models' / 'random_forest' / 'model.pkl')
    ]
    
    if args.tune:
        train_cmd.append('--tune')
    
    run_command(train_cmd, 'Training Random Forest model')
    
    # Step 4: Run inference
    test_image = list((data_dir / 'raw').glob('*.tif'))[0]
    run_command(
        [sys.executable, str(scripts_dir / 'run_rf_inference.py'),
         '--model', str(project_root / 'models' / 'random_forest' / 'model.pkl'),
         '--input', str(test_image),
         '--output', str(project_root / 'results' / 'test_classification.tif')],
        'Running inference on test image'
    )
    
    # Step 5: Evaluate model
    run_command(
        [sys.executable, str(scripts_dir / 'evaluate_random_forest.py'),
         '--model', str(project_root / 'models' / 'random_forest' / 'model.pkl'),
         '--features', str(data_dir / 'training' / 'features.npy'),
         '--labels', str(data_dir / 'training' / 'labels.npy'),
         '--output', str(project_root / 'results')],
        'Evaluating model performance'
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print("\n📁 Results saved to:")
    print(f"   Model: models/random_forest/model.pkl")
    print(f"   Metrics: results/evaluation_results.json")
    print(f"   Plots: results/*.png")
    print(f"   Classification: results/test_classification.tif")
    print("\n📊 Next steps:")
    print("   1. View dashboard: cd frontend && python -m http.server 8000")
    print("   2. Open browser to: http://localhost:8000/dashboard.html")
    print("   3. Or explore with Jupyter: jupyter notebook notebooks/exploration.ipynb")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
