import rasterio

image_path = 'data/processed/clipped/Tirupati_Landsat_2018_clipped.tif'
label_path = 'data/training/labels/Tirupati_Landsat_2018.tif'

with rasterio.open(image_path) as img_src:
    img_shape = img_src.read(1).shape
    img_transform = img_src.transform
    img_crs = img_src.crs
    print(f"Image shape: {img_shape}")
    print(f"Image CRS: {img_crs}")
    print(f"Image transform: {img_transform}")

with rasterio.open(label_path) as lbl_src:
    lbl_shape = lbl_src.read(1).shape
    lbl_transform = lbl_src.transform
    lbl_crs = lbl_src.crs
    print(f"Label shape: {lbl_shape}")
    print(f"Label CRS: {lbl_crs}")
    print(f"Label transform: {lbl_transform}")

if img_shape != lbl_shape or img_transform != lbl_transform or img_crs != lbl_crs:
    print("\nMismatch detected!\n")
    print("Best fix: Reproject and resample label to match image.")
    print("You can use rasterio.warp.reproject or gdalwarp for this task.")
else:
    print("\nImage and label are aligned.")
