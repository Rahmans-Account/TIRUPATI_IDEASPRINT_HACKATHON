# 🌍 LULC Random Forest Classifier

A comprehensive Land Use Land Cover (LULC) classification system using Random Forest machine learning.

## 🎯 Features

- **Multi-class Classification**: 5 LULC classes (Water, Vegetation, Urban, Barren, Agriculture)
- **Feature Extraction**: Automated spectral indices calculation (NDVI, NDBI, NDWI, etc.)
- **Model Training**: Random Forest with hyperparameter tuning
- **Inference Pipeline**: Batch prediction on new satellite imagery
- **Evaluation Suite**: Comprehensive accuracy metrics and visualization
- **Interactive Dashboard**: Web-based UI for model monitoring and results

## 📁 Project Structure

```
lulc_rf_project/
├── data/
│   ├── training/          # Training features and labels
│   ├── validation/        # Validation dataset
│   ├── processed/         # Processed indices
│   └── raw/              # Raw satellite imagery
├── models/
│   └── random_forest/    # Trained models
├── scripts/
│   ├── extract_features.py
│   ├── train_random_forest.py
│   ├── run_rf_inference.py
│   └── evaluate_random_forest.py
├── frontend/
│   └── dashboard.html    # Interactive web dashboard
├── config/
│   └── config.yaml       # Configuration settings
└── notebooks/            # Jupyter notebooks for exploration

```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your satellite imagery in `data/raw/` and create labeled masks in `data/training/labels/`.

### 3. Extract Features

```bash
python scripts/extract_features.py --input data/raw/ --output data/training/
```

### 4. Train Model

```bash
python scripts/train_random_forest.py --features data/training/features.npy --labels data/training/labels.npy
```

### 5. Run Inference

```bash
python scripts/run_rf_inference.py --model models/random_forest/model.pkl --input data/raw/test_image.tif
```

### 6. Evaluate Performance

```bash
python scripts/evaluate_random_forest.py --model models/random_forest/model.pkl --validation data/validation/
```

### 7. Launch Dashboard

```bash
python -m http.server 8000
# Open browser to http://localhost:8000/frontend/dashboard.html
```

## 📊 LULC Classes

| Class ID | Class Name | Color Code |
|----------|------------|------------|
| 0 | Water | Blue |
| 1 | Vegetation | Green |
| 2 | Urban | Red |
| 3 | Barren | Yellow |
| 4 | Agriculture | Cyan |

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Number of estimators
- Class names and colors
- Feature bands
- Output paths

## 📈 Model Performance

The system provides:
- Overall Accuracy
- Per-class Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Rankings

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License
