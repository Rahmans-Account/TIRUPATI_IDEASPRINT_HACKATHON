
import sys
import numpy as np
from pathlib import Path


import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure project root is in sys.path for src imports (if needed)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

# Paths for validation features and labels
features_path = project_root / 'data' / 'training' / 'val_features.npy'
labels_path = project_root / 'data' / 'training' / 'val_labels.npy'
model_path = project_root / 'models' / 'random_forest' / 'model.pkl'

# Load validation data
X_val = np.load(features_path)
y_val = np.load(labels_path)

# Load trained model
clf = joblib.load(model_path)

# Predict
y_pred = clf.predict(X_val)

# Evaluate
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f'Validation Accuracy: {acc:.4f}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(report)
