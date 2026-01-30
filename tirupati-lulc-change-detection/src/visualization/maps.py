
"""Map visualization utilities for LULC data."""

# --- Headless backend for hackathon/automation ---
import matplotlib
matplotlib.use("Agg")  # ⭐ headless backend, no GUI popups
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Optional


class LULCColorScheme:
    """Standard color scheme for LULC classes."""
    
    COLORS = {
        0: '#228B22',  # Forest - Forest Green
        1: '#0000FF',  # Water - Blue
        2: '#FFFF00',  # Agriculture - Yellow
        3: '#8B4513',  # Barren - Saddle Brown
        4: '#FF0000'   # Built-up - Red
    }
    
    NAMES = {
        0: 'Forest',
        1: 'Water Bodies',
        2: 'Agriculture',
        3: 'Barren Land',
        4: 'Built-up'
    }
    
    @classmethod
    def get_cmap(cls, num_classes=5):
        """Get matplotlib colormap."""
        colors = [cls.COLORS[i] for i in range(num_classes)]
        return mcolors.ListedColormap(colors)
    
    @classmethod
    def get_legend_elements(cls, num_classes=5):
        """Get legend elements for matplotlib."""
        return [Patch(facecolor=cls.COLORS[i], label=cls.NAMES[i]) 
                for i in range(num_classes)]


def plot_classification_map(
    classification: np.ndarray,
    title: str = "LULC Classification",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show_legend: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot LULC classification map.
    
    Args:
        classification: Classification array
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        show_legend: Whether to show legend
        dpi: DPI for saving
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap
    cmap = LULCColorScheme.get_cmap()
    
    # Plot
    im = ax.imshow(classification, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    if show_legend:
        legend_elements = LULCColorScheme.get_legend_elements()
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1, 0.5), fontsize=12)
    
    # Add scale bar
    scalebar_length = classification.shape[1] // 10
    ax.plot([50, 50 + scalebar_length], [classification.shape[0] - 50] * 2,
           'k-', linewidth=3)
    ax.text(50 + scalebar_length/2, classification.shape[0] - 70,
           f'{scalebar_length * 30}m', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_side_by_side_comparison(
    classification_t1: np.ndarray,
    classification_t2: np.ndarray,
    year_t1: int = 2018,
    year_t2: int = 2024,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """Plot side-by-side comparison of two time periods."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    cmap = LULCColorScheme.get_cmap()
    
    # Plot T1
    axes[0].imshow(classification_t1, cmap=cmap, vmin=0, vmax=4)
    axes[0].set_title(f'LULC Classification - {year_t1}', 
                     fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Plot T2
    im = axes[1].imshow(classification_t2, cmap=cmap, vmin=0, vmax=4)
    axes[1].set_title(f'LULC Classification - {year_t2}', 
                     fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    # Add shared legend
    legend_elements = LULCColorScheme.get_legend_elements()
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(1.05, 0.5), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_change_map(
    change_map: np.ndarray,
    confidence_map: Optional[np.ndarray] = None,
    title: str = "Land Use Change Detection",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Plot change detection map."""
    if confidence_map is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Change map
        change_cmap = mcolors.ListedColormap(['lightgray', 'red'])
        axes[0].imshow(change_map, cmap=change_cmap)
        axes[0].set_title('Change Detection (Red = Change)', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Confidence map
        im = axes[1].imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title('Change Confidence', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', fontsize=12)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        change_cmap = mcolors.ListedColormap(['lightgray', 'red'])
        ax.imshow(change_map, cmap=change_cmap)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        legend_elements = [
            Patch(facecolor='lightgray', label='No Change'),
            Patch(facecolor='red', label='Change Detected')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    class_names: List[str],
    title: str = "LULC Transition Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Plot transition matrix as heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels([f"{name} (2024)" for name in class_names], fontsize=10)
    ax.set_yticklabels([f"{name} (2018)" for name in class_names], fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = transition_matrix[i, j]
            color = 'white' if value > transition_matrix.max() / 2 else 'black'
            ax.text(j, i, f'{int(value)}', ha="center", va="center", 
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('To (2024)', fontsize=12, fontweight='bold')
    ax.set_ylabel('From (2018)', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Pixels', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
