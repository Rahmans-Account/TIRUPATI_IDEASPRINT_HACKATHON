# 📋 Project File Index

## Core Documentation
- **README.md** - Project overview, features, and quick start
- **SETUP_GUIDE.md** - Comprehensive setup and usage instructions
- **LICENSE** - MIT License
- **.gitignore** - Git ignore patterns

## Configuration
- **config/config.yaml** - Main configuration file (model params, classes, features)

## Python Scripts

### Core Pipeline Scripts (`scripts/`)
1. **extract_features.py** - Extract spectral features and indices from satellite imagery
2. **train_random_forest.py** - Train Random Forest classifier with cross-validation
3. **run_rf_inference.py** - Run predictions on new imagery
4. **evaluate_random_forest.py** - Comprehensive model evaluation with metrics
5. **visualize_results.py** - Create publication-quality visualizations

### Utility Scripts
6. **generate_sample_data.py** - Generate synthetic test data
7. **quickstart.py** - Automated pipeline execution
8. **examples.py** - Example usage demonstrations

## Frontend
- **frontend/dashboard.html** - Interactive web dashboard for model monitoring

## Notebooks
- **notebooks/exploration.ipynb** - Interactive Jupyter notebook for exploration

## Tests
- **tests/test_suite.py** - Comprehensive test suite

## Dependencies
- **requirements.txt** - Python package dependencies

## Directory Structure

```
lulc_rf_project/
│
├── README.md                      # Main documentation
├── SETUP_GUIDE.md                 # Detailed setup guide
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── quickstart.py                  # Quick start script
├── examples.py                    # Usage examples
├── .gitignore                     # Git ignore
│
├── config/
│   └── config.yaml               # Configuration
│
├── scripts/
│   ├── extract_features.py       # Feature extraction
│   ├── train_random_forest.py    # Model training
│   ├── run_rf_inference.py       # Inference
│   ├── evaluate_random_forest.py # Evaluation
│   ├── visualize_results.py      # Visualization
│   └── generate_sample_data.py   # Sample data generator
│
├── frontend/
│   └── dashboard.html            # Web dashboard
│
├── notebooks/
│   └── exploration.ipynb         # Jupyter notebook
│
├── tests/
│   └── test_suite.py             # Tests
│
├── data/                         # Data directory (created on first run)
│   ├── raw/                      # Raw satellite images
│   ├── training/                 # Training data
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   └── labels/              # Label masks
│   ├── validation/              # Validation data
│   └── processed/               # Processed data
│
├── models/                       # Models directory (created on first run)
│   └── random_forest/
│       ├── model.pkl            # Trained model
│       ├── training_history.json
│       └── model_metadata.json
│
└── results/                      # Results directory (created on first run)
    ├── evaluation_results.json
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── visualizations/
```

## Quick Command Reference

### Setup
```bash
pip install -r requirements.txt
```

### Generate Sample Data
```bash
python scripts/generate_sample_data.py --output data --n-images 5
```

### Extract Features
```bash
python scripts/extract_features.py \
    --input data/raw \
    --labels data/training/labels \
    --output data/training
```

### Train Model
```bash
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output models/random_forest/model.pkl
```

### Run Inference
```bash
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw/test_image.tif \
    --output results/classified.tif
```

### Evaluate Model
```bash
python scripts/evaluate_random_forest.py \
    --model models/random_forest/model.pkl \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output results
```

### View Dashboard
```bash
cd frontend
python -m http.server 8000
# Open: http://localhost:8000/dashboard.html
```

### Quick Start (All-in-One)
```bash
python quickstart.py --n-images 5
```

### Run Tests
```bash
python tests/test_suite.py
```

### View Examples
```bash
python examples.py
```

## Feature Summary

### Spectral Bands
- B1: Blue
- B2: Green
- B3: Red
- B4: NIR (Near-Infrared)
- B5: SWIR1 (Shortwave Infrared 1)
- B6: SWIR2 (Shortwave Infrared 2)
- B7: Additional band

### Spectral Indices
- **NDVI** - Normalized Difference Vegetation Index
- **NDBI** - Normalized Difference Built-up Index
- **NDWI** - Normalized Difference Water Index
- **SAVI** - Soil Adjusted Vegetation Index
- **EVI** - Enhanced Vegetation Index

### LULC Classes
0. Water (Blue)
1. Vegetation (Green)
2. Urban (Red)
3. Barren (Yellow)
4. Agriculture (Cyan)

## Model Parameters (Default)

```yaml
n_estimators: 100
max_depth: 30
min_samples_split: 10
min_samples_leaf: 5
random_state: 42
n_jobs: -1
class_weight: balanced
```

## Output Formats

### Training Output
- `model.pkl` - Trained Random Forest model (joblib)
- `training_history.json` - Training metrics and history
- `model_metadata.json` - Model information
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Feature importance plot

### Inference Output
- `*.tif` - Classified GeoTIFF with color map
- `*_probabilities.tif` - Class probability maps (optional)

### Evaluation Output
- `evaluation_results.json` - Complete metrics
- `confusion_matrix.png` - Confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `per_class_metrics.png` - Per-class performance chart
- `feature_importance.png` - Feature importance ranking

## Customization

### Add New Classes
Edit `config/config.yaml`:
```yaml
classes:
  5:
    name: "New Class"
    color: [128, 128, 128]
```

### Add New Features
Edit `scripts/extract_features.py`:
```python
def calculate_custom_index(self, band1, band2):
    return (band1 - band2) / (band1 + band2 + 1e-10)
```

### Adjust Model Parameters
Edit `config/config.yaml`:
```yaml
model:
  n_estimators: 200
  max_depth: 40
```

## Support and Troubleshooting

Common issues and solutions are documented in SETUP_GUIDE.md

For more help:
1. Check SETUP_GUIDE.md troubleshooting section
2. Review examples.py for usage patterns
3. Run test_suite.py to verify installation
4. Examine log output for error messages

## Citation

If you use this code in research, please cite appropriately.

## Version

Current Version: 1.0.0
Date: 2024
