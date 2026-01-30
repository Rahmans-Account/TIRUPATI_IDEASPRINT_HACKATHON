# 🚀 LULC Random Forest - Quick Reference Card

## Installation (One-time)
```bash
cd lulc_rf_project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (5 Minutes)
```bash
# 1. Generate test data
python scripts/generate_sample_data.py --output data --n-images 3

# 2. Run complete pipeline
python quickstart.py --n-images 3

# 3. View dashboard
cd frontend && python -m http.server 8000
# Open: http://localhost:8000/dashboard.html
```

## Manual Pipeline

### Step 1: Extract Features
```bash
python scripts/extract_features.py \
    --input data/raw \
    --labels data/training/labels \
    --output data/training
```

### Step 2: Train Model
```bash
# Basic
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy

# With tuning
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --tune
```

### Step 3: Run Inference
```bash
# Single image
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw/image.tif \
    --output results/classified.tif

# Batch
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw \
    --output results \
    --batch
```

### Step 4: Evaluate
```bash
python scripts/evaluate_random_forest.py \
    --model models/random_forest/model.pkl \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output results
```

### Step 5: Visualize
```bash
python scripts/visualize_results.py \
    --classification results/classified.tif \
    --original data/raw/image.tif \
    --output-dir results/visualizations
```

## File Locations

### Input Data
- Images: `data/raw/*.tif`
- Labels: `data/training/labels/*_label.tif`

### Outputs
- Model: `models/random_forest/model.pkl`
- Metrics: `results/evaluation_results.json`
- Plots: `results/*.png`
- Classifications: `results/*.tif`

## Common Tasks

### Check Model Status
```bash
python -c "import joblib; m = joblib.load('models/random_forest/model.pkl'); print(f'Features: {m.n_features_in_}, Classes: {m.n_classes_}')"
```

### View Training Metrics
```bash
cat models/random_forest/training_history.json
```

### View Evaluation Results
```bash
cat results/evaluation_results.json
```

### Run Tests
```bash
python tests/test_suite.py
```

### View Examples
```bash
python examples.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Configuration

Edit `config/config.yaml` to change:
- Model parameters (n_estimators, max_depth, etc.)
- Class definitions and colors
- Feature bands and indices
- File paths

## LULC Classes

| ID | Name | Color | Use Case |
|----|------|-------|----------|
| 0 | Water | Blue | Rivers, lakes, oceans |
| 1 | Vegetation | Green | Forests, grasslands |
| 2 | Urban | Red | Buildings, roads |
| 3 | Barren | Yellow | Desert, bare soil |
| 4 | Agriculture | Cyan | Croplands, farms |

## Key Features

### Spectral Indices
- **NDVI**: (NIR - Red) / (NIR + Red) - Vegetation health
- **NDBI**: (SWIR - NIR) / (SWIR + NIR) - Built-up areas
- **NDWI**: (Green - NIR) / (Green + NIR) - Water bodies
- **SAVI**: Soil-adjusted vegetation
- **EVI**: Enhanced vegetation

## Troubleshooting

### Out of Memory
```bash
# Sample data
python -c "import numpy as np; X = np.load('data/training/features.npy'); y = np.load('data/training/labels.npy'); idx = np.random.choice(len(X), 100000, False); np.save('data/training/features_small.npy', X[idx]); np.save('data/training/labels_small.npy', y[idx])"
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Check Installation
```bash
python -c "import sklearn, rasterio, yaml; print('✓ OK')"
```

## Performance Tips

1. **Use all CPU cores**: Set `n_jobs: -1` in config
2. **Reduce data**: Sample pixels if dataset is huge
3. **Tune parameters**: Use `--tune` flag for training
4. **Batch processing**: Use `--batch` for multiple images

## Need Help?

1. Read `SETUP_GUIDE.md` for detailed instructions
2. Check `FILE_INDEX.md` for file descriptions
3. Run `python examples.py` for usage examples
4. Examine `results/` directory for outputs

## Project Structure
```
lulc_rf_project/
├── scripts/          # Python scripts
├── config/          # Configuration
├── frontend/        # Dashboard
├── notebooks/       # Jupyter notebooks
├── data/           # Data (generated)
├── models/         # Trained models (generated)
└── results/        # Outputs (generated)
```

---
**Version 1.0** | Made with ❤️ for remote sensing
