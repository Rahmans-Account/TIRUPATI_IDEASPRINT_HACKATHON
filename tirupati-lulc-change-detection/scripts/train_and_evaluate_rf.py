
# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
import subprocess
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

import sys

def run_script(script_name):
	script_path = project_root / 'scripts' / script_name
	result = subprocess.run([sys.executable, str(script_path)], cwd=project_root, check=True)
	return result

# Step 1: Extract features for training
print('Extracting features...')
run_script('extract_features.py')

# Step 2: Train Random Forest
print('Training Random Forest...')
run_script('train_random_forest.py')

# Step 3: Evaluate model
print('Evaluating model...')
run_script('evaluate_random_forest.py')

# Step 4: Run inference on new image
print('Running inference on new image...')
run_script('run_rf_inference.py')

print('Pipeline complete!')
