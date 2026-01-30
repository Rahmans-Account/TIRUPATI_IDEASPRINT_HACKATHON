# 🚀 Complete Setup and Usage Guide

## Installation Steps

### 1. Clone or Extract the Project
```bash
cd lulc_rf_project
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n lulc python=3.9
conda activate lulc
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start with Sample Data

### Generate Test Data
```bash
python scripts/generate_sample_data.py --output data --n-images 5 --size 1000
```

This creates:
- 5 synthetic satellite images in `data/raw/`
- Corresponding label masks in `data/training/labels/`

### Complete Pipeline Execution

#### Step 1: Extract Features
```bash
python scripts/extract_features.py \
    --input data/raw \
    --labels data/training/labels \
    --output data/training
```

**What it does:**
- Reads satellite images and label masks
- Calculates spectral indices (NDVI, NDBI, NDWI, SAVI, EVI)
- Extracts feature vectors for each pixel
- Saves as NumPy arrays

**Output:**
- `data/training/features.npy` - Feature matrix (n_samples × n_features)
- `data/training/labels.npy` - Label vector (n_samples,)

#### Step 2: Train Model
```bash
# Basic training
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output models/random_forest/model.pkl

# With hyperparameter tuning
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output models/random_forest/model.pkl \
    --tune

# With cross-validation
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --output models/random_forest/model.pkl \
    --cv
```

**What it does:**
- Loads training data
- Splits into train/validation sets
- Trains Random Forest classifier
- Evaluates on validation set
- Saves model and metrics

**Output:**
- `models/random_forest/model.pkl` - Trained model
- `models/random_forest/training_history.json` - Training metrics
- `models/random_forest/model_metadata.json` - Model information
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/feature_importance.png` - Feature importance plot

#### Step 3: Run Inference
```bash
# Single image prediction
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw/sample_image_1.tif \
    --output results/classified_image.tif

# Batch prediction
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw \
    --output results/predictions \
    --batch

# With probability maps
python scripts/run_rf_inference.py \
    --model models/random_forest/model.pkl \
    --input data/raw/sample_image_1.tif \
    --output results/classified_image.tif \
    --probabilities
```

**What it does:**
- Loads trained model
- Extracts features from new imagery
- Predicts LULC classes
- Saves classified map as GeoTIFF

**Output:**
- Classified LULC map
- (Optional) Probability maps for each class
- Classification statistics

#### Step 4: Evaluate Model
```bash
python scripts/evaluate_random_forest.py \
    --model models/random_forest/model.pkl \
    --features data/validation/features.npy \
    --labels data/validation/labels.npy \
    --output results
```

**What it does:**
- Loads validation dataset
- Makes predictions
- Calculates comprehensive metrics
- Generates visualization plots

**Output:**
- `results/evaluation_results.json` - All metrics in JSON
- `results/confusion_matrix.png` - Confusion matrix
- `results/confusion_matrix_normalized.png` - Normalized confusion matrix
- `results/per_class_metrics.png` - Per-class performance chart
- `results/feature_importance.png` - Feature importance ranking

#### Step 5: View Dashboard
```bash
# Start simple HTTP server
cd frontend
python -m http.server 8000

# Open browser to:
# http://localhost:8000/dashboard.html
```

## Working with Real Data

### Data Preparation

#### 1. Organize Your Files
```
data/
├── raw/                    # Your satellite images (.tif)
│   ├── scene_001.tif
│   ├── scene_002.tif
│   └── ...
└── training/
    └── labels/            # Corresponding label masks
        ├── scene_001_label.tif
        ├── scene_002_label.tif
        └── ...
```

#### 2. Label File Format
- Single-band GeoTIFF
- Pixel values: 0 (Water), 1 (Vegetation), 2 (Urban), 3 (Barren), 4 (Agriculture)
- Same dimensions as input image
- Same coordinate system

#### 3. Image Requirements
- Multi-band GeoTIFF (at least 6-7 bands)
- Typical band order: Blue, Green, Red, NIR, SWIR1, SWIR2
- Georeferenced (with CRS)
- Reflectance values (0-1 or 0-10000)

### Customization

#### Modify Classes
Edit `config/config.yaml`:
```yaml
classes:
  0:
    name: "Water"
    color: [0, 0, 255]
  1:
    name: "Forest"
    color: [0, 100, 0]
  # Add more classes...
```

#### Adjust Model Parameters
Edit `config/config.yaml`:
```yaml
model:
  n_estimators: 200
  max_depth: 40
  min_samples_split: 5
  # Other RF parameters...
```

#### Add Custom Features
Edit `scripts/extract_features.py` and add your index calculation:
```python
def calculate_custom_index(self, band1, band2):
    """Your custom spectral index"""
    return (band1 - band2) / (band1 + band2)
```

## Troubleshooting

### Common Issues

#### 1. Memory Error During Training
**Problem:** Dataset too large for RAM
**Solution:**
```bash
# Sample your data
python -c "
import numpy as np
X = np.load('data/training/features.npy')
y = np.load('data/training/labels.npy')
indices = np.random.choice(len(X), 500000, replace=False)
np.save('data/training/features_sampled.npy', X[indices])
np.save('data/training/labels_sampled.npy', y[indices])
"
```

#### 2. Feature Extraction Fails
**Problem:** Band mapping doesn't match your sensor
**Solution:** Update `config/config.yaml` with correct band indices

#### 3. Poor Classification Results
**Solutions:**
- Increase training data
- Balance class samples
- Tune hyperparameters with `--tune` flag
- Check if features are informative

#### 4. Slow Inference
**Solutions:**
- Reduce image size (spatial resolution)
- Use fewer features
- Process in smaller chunks

## Advanced Usage

### Cross-Validation
```bash
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --cv
```

### Hyperparameter Tuning
```bash
python scripts/train_random_forest.py \
    --features data/training/features.npy \
    --labels data/training/labels.npy \
    --tune
```

### Export Model to Other Formats
```python
import joblib
from sklearn.tree import export_text

# Load model
model = joblib.load('models/random_forest/model.pkl')

# Export first tree as text
tree_rules = export_text(model.estimators_[0], feature_names=feature_names)
with open('results/tree_rules.txt', 'w') as f:
    f.write(tree_rules)
```

### Integration with GIS Software

#### QGIS
1. Load classified GeoTIFF in QGIS
2. Apply color ramp matching class colors
3. Calculate area statistics per class

#### ArcGIS
1. Import classified raster
2. Use Zonal Statistics for area calculation
3. Export to shapefile for vector analysis

## Performance Optimization

### For Large Datasets

#### 1. Parallel Processing
Modify `config.yaml`:
```yaml
model:
  n_jobs: -1  # Use all CPU cores
```

#### 2. Chunked Processing
Inference automatically uses chunked processing for memory efficiency.

#### 3. Feature Selection
Remove less important features after analyzing feature importance:
```python
# Keep only top N features
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:]  # Top 10
X_reduced = X[:, top_indices]
```

## Validation Best Practices

1. **Use separate validation data** - Don't evaluate on training data
2. **Stratified sampling** - Maintain class proportions in splits
3. **Cross-validation** - For robust performance estimates
4. **Check confusion matrix** - Identify problematic class pairs
5. **Visual inspection** - Always look at classified maps

## Citation

If you use this code in your research, please cite:

```
@software{lulc_rf_classifier,
  title = {LULC Random Forest Classifier},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/lulc_rf_project}
}
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review example outputs in `results/`
- Examine log files for error messages

## Next Steps

1. ✅ Generate or prepare your training data
2. ✅ Extract features
3. ✅ Train model
4. ✅ Evaluate performance
5. ✅ Run inference on new imagery
6. ✅ Analyze results in dashboard

Good luck with your LULC classification! 🌍
