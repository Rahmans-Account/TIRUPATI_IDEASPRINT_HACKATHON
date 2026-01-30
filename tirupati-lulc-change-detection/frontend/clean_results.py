# Clean Results Script for Hackathon Demo
# This script deletes all static result images and CSVs from the frontend so the dashboard always starts empty.
# Run this before your demo or before running the model pipeline.

import shutil
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'public', 'results')

# List of subfolders to clean
SUBFOLDERS = ['maps', 'charts', 'interactive']

# List of top-level files to remove
TOP_LEVEL_FILES = [
    'lulc_2018.png', 'lulc_2024.png', 'change_map.png', 'change_confidence.png',
    'transition_map.png', 'confidence_2018.png', 'confidence_2024.png',
    'change_statistics.csv', 'transition_matrix.csv', 'visualization_gallery.html'
]

def clean_results():
    # Remove top-level files
    for fname in TOP_LEVEL_FILES:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")
    # Remove all files in subfolders
    for sub in SUBFOLDERS:
        subdir = os.path.join(RESULTS_DIR, sub)
        if os.path.exists(subdir):
            for f in os.listdir(subdir):
                fpath = os.path.join(subdir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    print(f"Deleted {fpath}")

if __name__ == "__main__":
    clean_results()
    print("All static results cleaned. Ready for a fresh run!")
