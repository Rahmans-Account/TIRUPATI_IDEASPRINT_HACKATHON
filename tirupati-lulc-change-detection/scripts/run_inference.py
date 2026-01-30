"""Run inference on satellite imagery."""

import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from src.change_detection.detector import detect_changes, analyze_transitions, create_transition_map
from src.change_detection.transition_matrix import build_transition_matrix
from src.utils.config_utils import get_full_config
from src.utils.logger import default_logger as logger
from src.utils.geo_utils import read_raster, write_raster
from src.features.spectral_indices import calculate_all_indices
import pandas as pd


def load_model(model_type: str, config):
    """Load trained model."""
    if model_type == 'baseline':
        return None
    else:
        logger.warning(f"Model type '{model_type}' not supported without PyTorch. Using baseline classifier.")
        return None


def _extract_band_dict(image_data):
    """Map multiband Landsat array to named bands used by indices."""
    if image_data.ndim != 3 or image_data.shape[0] < 5:
        return {}

    return {
        "blue": image_data[0],
        "green": image_data[1],
        "red": image_data[2],
        "nir": image_data[3],
        "swir": image_data[4]
    }


def _baseline_classify(image_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rule-based baseline classification using spectral indices."""
    band_dict = _extract_band_dict(image_data)
    indices = calculate_all_indices(band_dict) if band_dict else {}

    ndvi = indices.get("ndvi")
    ndwi = indices.get("ndwi")
    mndwi = indices.get("mndwi")
    ndbi = indices.get("ndbi")

    if ndvi is None or ndwi is None or ndbi is None:
        raise ValueError("Missing indices for baseline classification.")

    water_mask = (ndwi > 0.3) | ((mndwi is not None) & (mndwi > 0.3))
    forest_mask = ndvi > 0.6
    agriculture_mask = (ndvi > 0.3) & (ndvi <= 0.6)
    builtup_mask = (ndbi > 0.2) & (ndvi < 0.3)
    barren_mask = ~(water_mask | forest_mask | agriculture_mask | builtup_mask)

    classification = np.zeros(ndvi.shape, dtype=np.uint8)
    classification[water_mask] = 1
    classification[agriculture_mask] = 2
    classification[barren_mask] = 3
    classification[builtup_mask] = 4
    classification[forest_mask] = 0

    confidence = np.zeros_like(ndvi, dtype=np.float32) + 0.5
    water_conf = np.clip((np.maximum(ndwi, mndwi if mndwi is not None else ndwi)[water_mask] - 0.3) / 0.4, 0, 1)
    confidence[water_mask] = water_conf
    confidence[forest_mask] = np.clip((ndvi[forest_mask] - 0.6) / 0.4, 0, 1)
    confidence[agriculture_mask] = np.clip((ndvi[agriculture_mask] - 0.3) / 0.3, 0, 1)
    confidence[builtup_mask] = np.clip((ndbi[builtup_mask] - 0.2) / 0.3, 0, 1)
    confidence[barren_mask] = 0.4

    confidence = np.clip(confidence + 0.2, 0, 1)
    return classification, confidence


def run_inference(config, model_type='unet', detect_change=True, years: list[int] | None = None):
    """Run complete inference pipeline."""
    
    logger.info("Starting inference...")
    
    # Load model
    model = load_model(model_type, config)
    if model is None:
        logger.info("Using baseline rule-based classifier")
    else:
        logger.info(f"Loaded {model_type} model")
    
    # Get paths
    clipped_dir = Path(config.data.processed.clipped)
    results_dir = Path(config.data.results.classifications)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference for each year
    classifications = {}
    confidence_maps = {}
    
    year_list = years or [config.time_periods.t1.year, config.time_periods.t2.year]
    for year in year_list:
        logger.info(f"Processing year {year}...")
        
        # Find image (simplified - adjust based on your naming)
        image_files = list(clipped_dir.glob(f"*{year}*clipped.tif"))
        
        if not image_files:
            logger.warning(f"No images found for year {year}")
            continue
        
        image_path = image_files[0]
        logger.info(f"Processing {image_path.name}")
        
        # Load image
        image_data, meta = read_raster(str(image_path))
        
        # Run prediction
        logger.info("Running prediction...")
        prediction = None
        confidence = None

        if model is None:
            prediction, confidence = _baseline_classify(image_data)
        elif model_type == 'random_forest':
            prediction, prob_image = model.predict_image(image_data)
            confidence = np.max(prob_image, axis=-1)
        else:
            try:
                input_tensor = torch.from_numpy(image_data).unsqueeze(0).float()
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = np.argmax(probs, axis=0).astype(np.uint8)
                confidence = np.max(probs, axis=0).astype(np.float32)
            except Exception as e:
                logger.warning(f"U-Net inference failed ({e}). Falling back to baseline.")
                prediction, confidence = _baseline_classify(image_data)

        # Save classification and confidence
        output_path = results_dir / f"lulc_{year}.tif"
        write_raster(prediction, str(output_path), meta, nodata=255)
        logger.info(f"Saved classification: {output_path.name}")

        confidence_path = results_dir / f"confidence_{year}.tif"
        write_raster(confidence.astype("float32"), str(confidence_path), meta, nodata=-1)
        logger.info(f"Saved confidence: {confidence_path.name}")
        
        classifications[year] = prediction
        confidence_maps[year] = confidence
    
    # Change detection
    if detect_change and len(classifications) == 2:
        logger.info("Running change detection...")
        
        years = sorted(classifications.keys())
        class_t1 = classifications[years[0]]
        class_t2 = classifications[years[1]]
        
        # Detect changes
        conf_t1 = confidence_maps.get(years[0])
        conf_t2 = confidence_maps.get(years[1])
        change_map, confidence_map = detect_changes(class_t1, class_t2, conf_t1, conf_t2)

        transition_map = create_transition_map(class_t1, class_t2, num_classes=len(config.lulc_classes))
        
        # Build transition matrix
        class_names = [cls['name'] for cls in config.lulc_classes]
        transition_matrix = build_transition_matrix(
            class_t1, class_t2,
            num_classes=len(class_names),
            class_names=class_names
        )
        
        # Save results
        change_dir = Path(config.data.results.change_detection)
        change_dir.mkdir(parents=True, exist_ok=True)
        
        stats_dir = Path(config.data.results.statistics)
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rasters
        change_path = change_dir / 'change_map.tif'
        write_raster(change_map.astype("uint8"), str(change_path), meta, nodata=0)

        transition_path = change_dir / 'transition_map.tif'
        write_raster(transition_map.astype("uint8"), str(transition_path), meta, nodata=255)

        confidence_path = change_dir / 'change_confidence.tif'
        write_raster(confidence_map.astype("float32"), str(confidence_path), meta, nodata=-1)

        # Save transition matrix
        transition_matrix.to_csv(stats_dir / 'transition_matrix.csv')
        logger.info("Saved transition matrix")
        
        # Analyze transitions
        analysis = analyze_transitions(class_t1, class_t2)
        pd.DataFrame([analysis]).to_csv(stats_dir / 'change_statistics.csv', index=False)
        logger.info("Saved change statistics")
    
    logger.info("Inference completed!")


def main():
    parser = argparse.ArgumentParser(description='Run LULC inference')
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['unet', 'random_forest', 'baseline'],
                       help='Model to use for inference')
    parser.add_argument('--detect-changes', action='store_true',
                       help='Run change detection')
    parser.add_argument('--years', nargs=2, type=int,
                       help='Two years to process (e.g., 2015 2023)')
    args = parser.parse_args()
    
    config = get_full_config()
    years = list(args.years) if args.years else None
    run_inference(config, model_type=args.model, detect_change=args.detect_changes, years=years)


if __name__ == '__main__':
    main()
