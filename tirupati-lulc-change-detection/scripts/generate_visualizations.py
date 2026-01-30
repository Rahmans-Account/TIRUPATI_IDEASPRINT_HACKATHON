"""Generate all visualizations for LULC analysis."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.visualization.maps import (
    plot_classification_map,
    plot_side_by_side_comparison,
    plot_change_map,
    plot_transition_heatmap
)
from src.visualization.charts import (
    plot_area_comparison_bar,
    plot_change_percentage,
    plot_pie_charts_comparison,
    create_sankey_diagram,
    create_interactive_comparison_chart,
    plot_confusion_matrix
)
from src.utils.logger import default_logger as logger
from src.utils.config_utils import get_full_config


def generate_all_visualizations(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    change_map: np.ndarray,
    confidence_map: np.ndarray,
    transition_matrix: np.ndarray,
    area_stats_t1: dict,
    area_stats_t2: dict,
    output_dir: str = "data/results/visualizations"
):
    """
    Generate all visualization outputs.
    
    Args:
        classification_t1: Classification for time 1
        classification_t2: Classification for time 2
        change_map: Change detection map
        confidence_map: Confidence scores
        transition_matrix: Transition matrix
        area_stats_t1: Area statistics for time 1
        area_stats_t2: Area statistics for time 2
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    maps_dir = output_path / "maps"
    charts_dir = output_path / "charts"
    
    maps_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    # 1. Classification Maps
    logger.info("Creating classification maps...")
    plot_classification_map(
        classification_t1,
        title="LULC Classification - 2018",
        save_path=str(maps_dir / "lulc_2018.png")
    )
    
    plot_classification_map(
        classification_t2,
        title="LULC Classification - 2024",
        save_path=str(maps_dir / "lulc_2024.png")
    )
    
    # 2. Side-by-side Comparison
    logger.info("Creating side-by-side comparison...")
    plot_side_by_side_comparison(
        classification_t1,
        classification_t2,
        save_path=str(maps_dir / "comparison_2018_2024.png")
    )
    
    # 3. Change Detection Map
    logger.info("Creating change detection map...")
    plot_change_map(
        change_map,
        confidence_map,
        save_path=str(maps_dir / "change_detection.png")
    )
    
    # 4. Transition Heatmap
    logger.info("Creating transition matrix heatmap...")
    class_names = ['Forest', 'Water', 'Agriculture', 'Barren', 'Built-up']
    plot_transition_heatmap(
        transition_matrix,
        class_names,
        save_path=str(maps_dir / "transition_heatmap.png")
    )
    
    # 5. Area Comparison Bar Chart
    logger.info("Creating area comparison chart...")
    plot_area_comparison_bar(
        area_stats_t1,
        area_stats_t2,
        save_path=str(charts_dir / "area_comparison.png")
    )
    
    # 6. Percentage Change Chart
    logger.info("Creating percentage change chart...")
    plot_change_percentage(
        area_stats_t1,
        area_stats_t2,
        save_path=str(charts_dir / "percentage_change.png")
    )
    
    # 7. Pie Charts Comparison
    logger.info("Creating pie charts comparison...")
    plot_pie_charts_comparison(
        area_stats_t1,
        area_stats_t2,
        save_path=str(charts_dir / "pie_comparison.png")
    )
    
    # 8. Sankey Diagram
    logger.info("Creating Sankey diagram...")
    sankey_fig = create_sankey_diagram(
        transition_matrix,
        class_names,
        min_flow=100
    )
    sankey_fig.write_html(str(charts_dir / "sankey_transitions.html"))
    
    # 9. Interactive Comparison Chart
    logger.info("Creating interactive comparison chart...")
    interactive_fig = create_interactive_comparison_chart(
        area_stats_t1,
        area_stats_t2
    )
    interactive_fig.write_html(str(charts_dir / "interactive_comparison.html"))
    
    logger.info(f"All visualizations saved to: {output_path}")
    logger.info(f"  - Maps: {maps_dir}")
    logger.info(f"  - Charts: {charts_dir}")


def create_sample_data():
    """Create sample data for demonstration."""
    # Sample classifications
    np.random.seed(42)
    size = 500
    
    classification_t1 = np.random.randint(0, 5, (size, size))
    classification_t2 = classification_t1.copy()
    
    # Simulate some changes
    change_mask = np.random.random((size, size)) < 0.15
    classification_t2[change_mask] = np.random.randint(0, 5, change_mask.sum())
    
    # Change map
    change_map = (classification_t1 != classification_t2).astype(int)
    
    # Confidence map
    confidence_map = np.random.uniform(0.5, 1.0, (size, size))
    
    # Transition matrix
    transition_matrix = np.array([
        [8500, 50, 200, 100, 150],
        [20, 1800, 10, 30, 40],
        [150, 30, 6000, 500, 1320],
        [80, 40, 400, 2500, 980],
        [10, 5, 50, 20, 3915]
    ])
    
    # Area statistics
    class_names = ['Forest', 'Water Bodies', 'Agriculture', 'Barren Land', 'Built-up']
    area_stats_t1 = {
        'Forest': {'area_sqkm': 850, 'percentage': 34.0},
        'Water Bodies': {'area_sqkm': 180, 'percentage': 7.2},
        'Agriculture': {'area_sqkm': 600, 'percentage': 24.0},
        'Barren Land': {'area_sqkm': 250, 'percentage': 10.0},
        'Built-up': {'area_sqkm': 120, 'percentage': 4.8}
    }
    
    area_stats_t2 = {
        'Forest': {'area_sqkm': 800, 'percentage': 32.0},
        'Water Bodies': {'area_sqkm': 190, 'percentage': 7.6},
        'Agriculture': {'area_sqkm': 598, 'percentage': 23.9},
        'Barren Land': {'area_sqkm': 300, 'percentage': 12.0},
        'Built-up': {'area_sqkm': 400, 'percentage': 16.0}
    }
    
    return (classification_t1, classification_t2, change_map, confidence_map,
            transition_matrix, area_stats_t1, area_stats_t2)


def main():
    """Main function."""
    logger.info("Starting visualization generation...")
    
    # Create sample data (replace with actual data loading)
    logger.info("Loading data...")
    data = create_sample_data()
    
    # Generate all visualizations
    generate_all_visualizations(*data)
    
    logger.info("Visualization generation completed!")


if __name__ == '__main__':
    main()
