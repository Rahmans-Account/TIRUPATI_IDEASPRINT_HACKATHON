"""Export visualization assets for the Next.js frontend."""

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_utils import get_full_config
from src.utils.logger import default_logger as logger


LULC_COLORS = {
    0: "#228B22",  # Forest
    1: "#0000FF",  # Water
    2: "#FFFF00",  # Agriculture
    3: "#8B4513",  # Barren
    4: "#FF0000",  # Built-up
}


def _save_class_map(array: np.ndarray, output_path: Path, title: str) -> None:
    cmap = plt.matplotlib.colors.ListedColormap(list(LULC_COLORS.values()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(array, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_gray_map(array: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(array, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _read_single_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def export_visuals(config, frontend_public: Path) -> None:
    results_dir = Path(config.data.results.classifications)
    change_dir = Path(config.data.results.change_detection)
    stats_dir = Path(config.data.results.statistics)

    maps_dir = Path(config.data.results.visualizations) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    frontend_public.mkdir(parents=True, exist_ok=True)

    for year in [config.time_periods.t1.year, config.time_periods.t2.year]:
        lulc_path = results_dir / f"lulc_{year}.tif"
        if lulc_path.exists():
            data = _read_single_band(lulc_path)
            png_path = maps_dir / f"lulc_{year}.png"
            _save_class_map(data, png_path, f"LULC {year}")
            frontend_copy = frontend_public / f"lulc_{year}.png"
            frontend_copy.write_bytes(png_path.read_bytes())
            logger.info(f"Exported {png_path.name}")

        conf_path = results_dir / f"confidence_{year}.tif"
        if conf_path.exists():
            conf = _read_single_band(conf_path)
            png_path = maps_dir / f"confidence_{year}.png"
            _save_gray_map(conf, png_path, f"Confidence {year}")
            frontend_copy = frontend_public / f"confidence_{year}.png"
            frontend_copy.write_bytes(png_path.read_bytes())

    change_map_path = change_dir / "change_map.tif"
    if change_map_path.exists():
        change = _read_single_band(change_map_path)
        png_path = maps_dir / "change_map.png"
        _save_gray_map(change, png_path, "Change Map")
        frontend_copy = frontend_public / "change_map.png"
        frontend_copy.write_bytes(png_path.read_bytes())

    transition_map_path = change_dir / "transition_map.tif"
    if transition_map_path.exists():
        transition = _read_single_band(transition_map_path)
        png_path = maps_dir / "transition_map.png"
        _save_gray_map(transition, png_path, "Transition Map")
        frontend_copy = frontend_public / "transition_map.png"
        frontend_copy.write_bytes(png_path.read_bytes())

    change_conf_path = change_dir / "change_confidence.tif"
    if change_conf_path.exists():
        change_conf = _read_single_band(change_conf_path)
        png_path = maps_dir / "change_confidence.png"
        _save_gray_map(change_conf, png_path, "Change Confidence")
        frontend_copy = frontend_public / "change_confidence.png"
        frontend_copy.write_bytes(png_path.read_bytes())

    for csv_name in ["transition_matrix.csv", "change_statistics.csv"]:
        csv_path = stats_dir / csv_name
        if csv_path.exists():
            frontend_copy = frontend_public / csv_name
            frontend_copy.write_bytes(csv_path.read_bytes())

    logger.info("Export completed.")


def main():
    parser = argparse.ArgumentParser(description="Export visuals for frontend")
    parser.add_argument(
        "--frontend-dir",
        type=str,
        default="frontend/public/results",
        help="Frontend public results directory",
    )
    args = parser.parse_args()

    config = get_full_config()
    frontend_public = Path(args.frontend_dir)
    export_visuals(config, frontend_public)


if __name__ == "__main__":
    main()
