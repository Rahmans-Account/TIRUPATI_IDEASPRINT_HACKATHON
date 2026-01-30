"""Color utilities and palettes for LULC visualization."""

import numpy as np
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple


# Standard LULC color scheme
LULC_COLORS = {
    'forest': '#228B22',      # Forest Green
    'water': '#0000FF',       # Blue
    'agriculture': '#FFFF00', # Yellow
    'barren': '#8B4513',      # Saddle Brown
    'builtup': '#FF0000',     # Red
    'no_data': '#FFFFFF'      # White
}

# RGB values
LULC_RGB = {
    'forest': (34, 139, 34),
    'water': (0, 0, 255),
    'agriculture': (255, 255, 0),
    'barren': (139, 69, 19),
    'builtup': (255, 0, 0),
    'no_data': (255, 255, 255)
}

# Change detection colors
CHANGE_COLORS = {
    'no_change': '#D3D3D3',    # Light Gray
    'change': '#FF0000',        # Red
    'high_confidence': '#00FF00',  # Green
    'low_confidence': '#FFFF00'    # Yellow
}


def create_custom_cmap(colors: List[str], name: str = 'custom') -> mcolors.LinearSegmentedColormap:
    """Create custom colormap from list of colors."""
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


def get_confidence_colormap() -> mcolors.LinearSegmentedColormap:
    """Get colormap for confidence visualization (red to green)."""
    colors = ['#FF0000', '#FFFF00', '#00FF00']  # Red -> Yellow -> Green
    return create_custom_cmap(colors, 'confidence')


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def apply_transparency(hex_color: str, alpha: float = 0.5) -> str:
    """Apply transparency to hex color."""
    rgb = hex_to_rgb(hex_color)
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'


def get_transition_color(from_class: int, to_class: int) -> str:
    """Get color for specific transition."""
    class_colors = list(LULC_COLORS.values())[:5]
    
    if from_class == to_class:
        return '#D3D3D3'  # No change - gray
    else:
        # Blend source and target colors
        from_rgb = hex_to_rgb(class_colors[from_class])
        to_rgb = hex_to_rgb(class_colors[to_class])
        blend_rgb = tuple((f + t) // 2 for f, t in zip(from_rgb, to_rgb))
        return rgb_to_hex(blend_rgb)
