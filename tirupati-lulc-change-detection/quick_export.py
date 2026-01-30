"""Quick export script to generate dashboard visuals."""
import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap


def _save_png(data: np.ndarray, output_path: str, title: str, cmap, vmin=None, vmax=None, dpi: int = 120):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def export_results(fast: bool = False) -> None:
    output_dir = Path("frontend/public/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = 100 if fast else 150

    # LULC color map
    colors = ["#228B22", "#0000FF", "#FFFF00", "#8B4513", "#FF0000"]
    lulc_cmap = ListedColormap(colors)

    # Generate LULC maps
    for year in [2018, 2024]:
        lulc_path = Path(f"data/results/classifications/lulc_{year}.tif")
        if not lulc_path.exists():
            continue
        data = rasterio.open(lulc_path).read(1)
        _save_png(
            data,
            str(output_dir / f"lulc_{year}.png"),
            f"LULC {year}",
            lulc_cmap,
            vmin=0,
            vmax=4,
            dpi=dpi,
        )
        print(f"✓ Exported lulc_{year}.png")

    # Change/transition maps
    change_outputs = [
        ("data/results/change_detection/change_map.tif", "change_map.png", "Change Map", "Reds"),
        ("data/results/change_detection/transition_map.tif", "transition_map.png", "Transition Map", "viridis"),
        ("data/results/change_detection/change_confidence.tif", "change_confidence.png", "Change Confidence", "magma"),
    ]

    for src_path, out_name, title, cmap_name in change_outputs:
        src = Path(src_path)
        if not src.exists():
            continue
        data = rasterio.open(src).read(1)
        _save_png(
            data,
            str(output_dir / out_name),
            title,
            cmap_name,
            dpi=dpi,
        )
        print(f"✓ Exported {out_name}")

    # Confidence maps
    for year in [2018, 2024]:
        conf_path = Path(f"data/results/classifications/confidence_{year}.tif")
        if not conf_path.exists():
            continue
        data = rasterio.open(conf_path).read(1)
        _save_png(
            data,
            str(output_dir / f"confidence_{year}.png"),
            f"Confidence {year}",
            "magma",
            vmin=0,
            vmax=1,
            dpi=dpi,
        )
        print(f"✓ Exported confidence_{year}.png")

    # Copy CSVs
    for csv_name in ["transition_matrix.csv", "change_statistics.csv"]:
        src = Path("data/results/statistics") / csv_name
        if src.exists():
            shutil.copy(src, output_dir / csv_name)
    print("✓ Copied CSV files")

    print("\n✅ Export complete! Files ready in frontend/public/results/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard visuals")
    parser.add_argument("--fast", action="store_true", help="Lower DPI for faster export")
    args = parser.parse_args()

    export_results(fast=args.fast)


if __name__ == "__main__":
    main()
