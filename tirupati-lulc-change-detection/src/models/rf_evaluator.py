# Evaluate Random Forest model for LULC (from lulc_rf_project)
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_rf(model_path, features_path, labels_path):
    X = np.load(features_path)
    y = np.load(labels_path)
    clf = joblib.load(model_path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    print(f'Validation Accuracy: {acc:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(report)
