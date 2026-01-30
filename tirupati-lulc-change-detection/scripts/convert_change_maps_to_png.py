import os
import rasterio
import numpy as np
from PIL import Image

# Mapping of input .tif files to output .png files
file_map = {
    'data/results/change_detection/change_map.tif': 'frontend/public/results/maps/change_map.png',
    'data/results/change_detection/change_confidence.tif': 'frontend/public/results/maps/change_confidence.png',
    'data/results/change_detection/transition_map.tif': 'frontend/public/results/maps/transition_map.png',
}

def tif_to_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        # Normalize for visualization if needed
        arr = arr.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (np.ptp(arr) if np.ptp(arr) > 0 else 1)).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(png_path)
        print(f"Saved: {png_path}")

if __name__ == "__main__":
    for tif, png in file_map.items():
        if os.path.exists(tif):
            os.makedirs(os.path.dirname(png), exist_ok=True)
            tif_to_png(tif, png)
        else:
            print(f"Missing: {tif}")
