#!/usr/bin/env python3
"""
Visualization Utilities for LULC Results
Create publication-quality figures from classification results
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import argparse


class LULCVisualizer:
    """Create visualizations for LULC classification results"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['classes']
        self.class_names = [self.classes[i]['name'] for i in sorted(self.classes.keys())]
        self.class_colors = [tuple(np.array(self.classes[i]['color'])/255) 
                            for i in sorted(self.classes.keys())]
    
    def plot_classification_map(self, classification_path, output_path=None, title='LULC Classification'):
        """Create a styled classification map"""
        
        with rasterio.open(classification_path) as src:
            classification = src.read(1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.class_colors)
        
        # Plot
        im = ax.imshow(classification, cmap=cmap, vmin=0, vmax=len(self.classes)-1)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(len(self.classes)))
        cbar.set_ticklabels(self.class_names)
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, original_path, classification_path, output_path=None):
        """Plot side-by-side comparison of original and classified image"""
        
        # Read original image
        with rasterio.open(original_path) as src:
            # Try to create RGB composite
            if src.count >= 3:
                r = src.read(min(4, src.count))  # Try NIR or Red
                g = src.read(min(3, src.count))  # Red or Green
                b = src.read(min(2, src.count))  # Green or Blue
            else:
                r = g = b = src.read(1)
            
            rgb = np.dstack([r, g, b])
            rgb = (rgb - np.nanmin(rgb)) / (np.nanmax(rgb) - np.nanmin(rgb))
            rgb = np.nan_to_num(rgb, 0)
        
        # Read classification
        with rasterio.open(classification_path) as src:
            classification = src.read(1)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Original image
        axes[0].imshow(rgb)
        axes[0].set_title('Original Image (RGB)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Classification
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.class_colors)
        im = axes[1].imshow(classification, cmap=cmap, vmin=0, vmax=len(self.classes)-1)
        axes[1].set_title('LULC Classification', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_ticks(range(len(self.classes)))
        cbar.set_ticklabels(self.class_names)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_area_statistics(self, classification_path, output_path=None, pixel_size=30):
        """Plot area statistics for each class"""
        
        with rasterio.open(classification_path) as src:
            classification = src.read(1)
        
        # Calculate areas
        pixel_area = (pixel_size ** 2) / 1e6  # km²
        areas = []
        percentages = []
        
        for class_id in range(len(self.classes)):
            count = np.sum(classification == class_id)
            area = count * pixel_area
            percentage = (count / classification.size) * 100
            areas.append(area)
            percentages.append(percentage)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        bars = ax1.bar(self.class_names, areas, color=self.class_colors)
        ax1.set_title('Area by Land Cover Class', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Area (km²)', fontsize=13)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, area in zip(bars, areas):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{area:.1f}', ha='center', va='bottom', fontsize=11)
        
        # Pie chart
        ax2.pie(percentages, labels=self.class_names, autopct='%1.1f%%',
                colors=self.class_colors, startangle=90)
        ax2.set_title('Land Cover Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_legend(self, output_path='results/legend.png'):
        """Create a standalone legend"""
        
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.axis('off')
        
        # Create legend patches
        from matplotlib.patches import Patch
        patches = [Patch(color=color, label=name) 
                  for color, name in zip(self.class_colors, self.class_names)]
        
        legend = ax.legend(handles=patches, loc='center', fontsize=14,
                          title='LULC Classes', title_fontsize=16,
                          frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Legend saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize LULC classification results')
    parser.add_argument('--classification', type=str, required=True, 
                        help='Path to classified GeoTIFF')
    parser.add_argument('--original', type=str, 
                        help='Path to original image for comparison')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--pixel-size', type=int, default=30,
                        help='Pixel size in meters')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz = LULCVisualizer(args.config)
    
    print("Creating visualizations...")
    
    # Classification map
    viz.plot_classification_map(
        args.classification,
        output_path=output_dir / 'classification_map.png'
    )
    
    # Comparison (if original provided)
    if args.original:
        viz.plot_comparison(
            args.original,
            args.classification,
            output_path=output_dir / 'comparison.png'
        )
    
    # Area statistics
    viz.plot_area_statistics(
        args.classification,
        output_path=output_dir / 'area_statistics.png',
        pixel_size=args.pixel_size
    )
    
    # Legend
    viz.create_legend(output_path=output_dir / 'legend.png')
    
    print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
