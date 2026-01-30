import os

# Use absolute paths for reliability
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(project_root, 'data', 'processed', 'clipped', 'Tirupati_Landsat_2018.tif')
label_path = os.path.join(project_root, 'data', 'training', 'labels', 'Tirupati_Landsat_2018.tif')
output_label_path = os.path.join(project_root, 'data', 'training', 'labels', 'Tirupati_Landsat_2018_aligned.tif')

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

with rasterio.open(image_path) as img_src:
    img_profile = img_src.profile.copy()
    img_shape = img_src.read(1).shape
    img_transform = img_src.transform
    img_crs = img_src.crs

with rasterio.open(label_path) as lbl_src:
    label = lbl_src.read(1)
    lbl_profile = lbl_src.profile.copy()
    lbl_transform = lbl_src.transform
    lbl_crs = lbl_src.crs

    # Prepare output array
    aligned_label = np.zeros(img_shape, dtype=label.dtype)

    reproject(
        source=label,
        destination=aligned_label,
        src_transform=lbl_transform,
        src_crs=lbl_crs,
        dst_transform=img_transform,
        dst_crs=img_crs,
        resampling=Resampling.nearest
    )

    # Update profile for output
    out_profile = img_profile.copy()
    out_profile.update(dtype=label.dtype, count=1)

    with rasterio.open(output_label_path, 'w', **out_profile) as dst:
        dst.write(aligned_label, 1)

print(f'Aligned label saved to {output_label_path}')
