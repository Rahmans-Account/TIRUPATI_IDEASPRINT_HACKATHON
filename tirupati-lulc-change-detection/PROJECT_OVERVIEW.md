# Tirupati LULC Change Detection Project: Complete Overview & Documentation

## Project Summary
This project is an end-to-end Land Use/Land Cover (LULC) classification and change detection pipeline for the Tirupati region. It combines geospatial data processing, machine learning (Random Forest), and interactive analytics dashboards to provide insights into land cover changes over time.

---

## Tech Stack

### Backend/Data Science
- **Python 3.x**: Core language for data processing, modeling, and automation
- **scikit-learn**: Random Forest model for LULC classification
- **NumPy**: Efficient numerical operations
- **joblib**: Model serialization
- **Pandas**: Data manipulation (if used in scripts)
- **Custom Scripts**: For feature extraction, alignment, training, evaluation, and analytics

### Frontend
- **Next.js (React)**: Interactive dashboard and visualization
- **Tailwind CSS**: Styling (if used)
- **JavaScript/TypeScript**: Frontend logic

### Data
- **Satellite Imagery**: Raw and processed geospatial data
- **Shapefiles**: Region boundaries
- **NumPy Arrays**: Features and labels for ML
- **YAML/JSON**: Configuration files

---

## Project Structure

- `tirupati-lulc-change-detection/`
  - `scripts/`: All backend scripts for data processing, training, evaluation, etc.
  - `data/`: Contains raw, processed, results, shapefiles, and training data
  - `models/`: Trained model files (e.g., Random Forest)
  - `config/`: YAML config files for paths, model, and pipeline settings
  - `dashboard/` & `frontend/`: Next.js dashboard app for analytics and visualization
  - `src/`: Python package for modular code (data, features, models, utils, etc.)
  - `tests/`: Test suites for backend code

---

## End-to-End Process Flow

1. **Data Preparation**
   - Align satellite images and labels using `align_label_to_image.py`.
   - Extract features from imagery with `extract_features.py`.
   - Output: `features.npy`, `labels.npy` in `data/processed/`.

2. **Train/Validation Split**
   - Use `split_and_save_validation.py` to split features/labels into training and validation sets.
   - Output: `val_features.npy`, `val_labels.npy` for validation.

3. **Model Training**
   - Run `train_random_forest.py` to train a Random Forest classifier on the training data.
   - Model saved to `models/random_forest/model.pkl`.

4. **Model Evaluation & Analytics**
   - Run `evaluate_random_forest.py` to compute accuracy, confusion matrix, and classification report on validation data.
   - Analytics results are saved or displayed for dashboard use.

5. **Frontend Dashboard**
   - Next.js dashboard visualizes results, analytics, and LULC change maps.
   - Interactive pages for overview, maps, change detection, analytics, and export.

---

## How to Start the Application

### 1. Backend Setup
- Ensure Python 3.x is installed.
- (Recommended) Create and activate a virtual environment:
  ```sh
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  source .venv/bin/activate  # Linux/Mac
  ```
- Install dependencies:
  ```sh
  pip install -r tirupati-lulc-change-detection/requirements.txt
  ```

### 2. Data Processing Pipeline
- Align and extract features:
  ```sh
  python tirupati-lulc-change-detection/scripts/align_label_to_image.py
  python tirupati-lulc-change-detection/scripts/extract_features.py
  ```
- Split validation data:
  ```sh
  python tirupati-lulc-change-detection/scripts/split_and_save_validation.py
  ```
- Train the model:
  ```sh
  python tirupati-lulc-change-detection/scripts/train_random_forest.py
  ```
- Evaluate the model:
  ```sh
  python tirupati-lulc-change-detection/scripts/evaluate_random_forest.py
  ```

### 3. Frontend Setup
- Go to the frontend directory:
  ```sh
  cd tirupati-lulc-change-detection/frontend
  ```
- Install frontend dependencies:
  ```sh
  npm install
  ```
- Build and start the dashboard:
  ```sh
  npm run build
  npm start
  ```
- Open your browser at [http://localhost:3000](http://localhost:3000)

---

## How Data Analytics Are Formed
- **Model Evaluation**: The evaluation script computes accuracy, confusion matrix, and classification report using scikit-learn.
- **Analytics Output**: Results are saved or made available for the dashboard to visualize class distributions, accuracy, and change maps.
- **Frontend Visualization**: The dashboard fetches results and displays them as charts, maps, and tables for user exploration.

---

## Technical & Non-Technical Notes
- **Reproducibility**: All scripts are modular and can be run independently.
- **Extensibility**: You can swap the Random Forest model for other classifiers with minimal changes.
- **Data Requirements**: Ensure all required data files are present in the correct folders.
- **Troubleshooting**: Check logs and script outputs for errors; ensure all dependencies are installed.

---

## Quick Reference
- **Train pipeline**: Run all backend scripts in order (align, extract, split, train, evaluate)
- **View analytics**: Start the frontend and open the dashboard
- **Export results**: Use the dashboard's export page

---

## Contact & Support
For questions or issues, refer to the README files or contact the project maintainer.

---

# System Architecture

```mermaid
flowchart TD
    A[Raw Satellite Data (Landsat/Sentinel)] --> B[Preprocessing & Clipping<br>scripts/preprocess_all.py, src/data/preprocessing.py]
    B --> C[Feature Extraction<br>scripts/extract_features.py, src/features/feature_extractor.py, src/features/spectral_indices.py]
    C --> D[Label Alignment<br>scripts/align_label_to_image.py, scripts/check_alignment.py]
    D --> E[Training Data Preparation<br>scripts/split_and_save_validation.py]
    E --> F[Model Training<br>scripts/train_random_forest.py, src/models/rf_trainer.py]
    F --> G[Model Inference<br>scripts/run_rf_inference.py, src/models/rf_predictor.py]
    G --> H[Change Detection & Analytics<br>scripts/run_inference.py, src/change_detection/detector.py, src/change_detection/transition_matrix.py]
    H --> I[Visualization Generation<br>scripts/generate_visualizations.py, scripts/generate_enhanced_visuals.py, src/visualization/maps.py, src/visualization/charts.py]
    I --> J[Export for Frontend<br>scripts/export_visuals.py]
    J --> K[Frontend Dashboard<br>frontend/ (Next.js), dashboard/app.py (Streamlit)]
```

---

# Technical Project Flow & File Functionality

## Data Acquisition & Preprocessing
- **scripts/download_satellite_gee.py**: Downloads satellite imagery from Google Earth Engine.
- **scripts/preprocess_all.py**: Orchestrates preprocessing (clipping, normalization, tiling) using src/data/preprocessing.py and geo_utils.
- **src/data/preprocessing.py**: Functions for clipping rasters, loading bands, and normalization.
- **src/utils/geo_utils.py**: Geospatial utilities for raster/vector operations.

## Label Alignment
- **scripts/check_alignment.py**: Checks if label and image rasters are aligned (shape, CRS, transform).
- **scripts/align_label_to_image.py**: Reprojects/resamples label raster to match image raster.

## Feature Extraction
- **scripts/extract_features.py**: Loads imagery and aligned labels, extracts features, saves as .npy arrays.
- **src/features/feature_extractor.py**: Extracts spectral and index features (NDVI, NDBI, NDWI, etc.).
- **src/features/spectral_indices.py**: Implements spectral index calculations.

## Training/Validation Split
- **scripts/split_and_save_validation.py**: Splits features/labels into training and validation sets, saves .npy arrays.
- **src/models/rf_trainer.py**: LULCTrainer class for loading, splitting, and preparing data.

## Model Training
- **scripts/train_random_forest.py**: Loads data, trains Random Forest, saves model.
- **src/models/rf_trainer.py**: Handles model instantiation, training, and saving.

## Model Inference
- **scripts/run_rf_inference.py**: Loads trained model, predicts on new imagery, saves classified raster.
- **src/models/rf_predictor.py**: LULCPredictor class for inference and raster output.

## Change Detection & Analytics
- **scripts/run_inference.py**: Orchestrates change detection, transition analysis, and statistics.
- **src/change_detection/detector.py**: Implements change detection logic.
- **src/change_detection/transition_matrix.py**: Builds transition matrices for analytics.

## Visualization & Export
- **scripts/generate_visualizations.py**: Generates maps, charts, and analytics plots.
- **scripts/generate_enhanced_visuals.py**: Enhanced, dashboard-ready visualizations.
- **src/visualization/maps.py**: Map plotting utilities (classification, change, transitions).
- **src/visualization/charts.py**: Statistical and comparison charts.
- **scripts/export_visuals.py**: Exports visual assets for frontend use.

## Frontend & Dashboard
- **frontend/**: Next.js app for interactive analytics and visualization.
- **dashboard/app.py**: Streamlit dashboard for quick exploration.

## Utilities & Config
- **src/utils/config_utils.py**: Loads and merges YAML config files.
- **src/utils/logger.py**: Logging setup for scripts and modules.
- **config/config.yaml**: Main configuration (model, classes, paths, study area).

---

This section provides a technical, file-by-file breakdown of the project flow, showing how each component fits into the LULC change detection pipeline from raw data to analytics dashboard.
