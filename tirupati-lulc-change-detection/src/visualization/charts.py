"""Statistical charts and plots for LULC analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional


def plot_area_comparison_bar(
    area_stats_t1: Dict,
    area_stats_t2: Dict,
    year_t1: int = 2018,
    year_t2: int = 2024,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot bar chart comparing area by class between two time periods."""
    classes = list(area_stats_t1.keys())
    area_t1 = [area_stats_t1[cls]['area_sqkm'] for cls in classes]
    area_t2 = [area_stats_t2[cls]['area_sqkm'] for cls in classes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, area_t1, width, label=str(year_t1), 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, area_t2, width, label=str(year_t2), 
                   color='coral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('LULC Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Area (km²)', fontsize=13, fontweight='bold')
    ax.set_title('Area Comparison by LULC Class', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_change_percentage(
    area_stats_t1: Dict,
    area_stats_t2: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot percentage change for each class."""
    classes = list(area_stats_t1.keys())
    pct_change = []
    
    for cls in classes:
        area_t1 = area_stats_t1[cls]['area_sqkm']
        area_t2 = area_stats_t2[cls]['area_sqkm']
        if area_t1 > 0:
            change = ((area_t2 - area_t1) / area_t1) * 100
        else:
            change = 100 if area_t2 > 0 else 0
        pct_change.append(change)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if x >= 0 else 'red' for x in pct_change]
    bars = ax.barh(classes, pct_change, color=colors, alpha=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, pct_change)):
        x_pos = val + (2 if val >= 0 else -2)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, i, f'{val:+.1f}%', 
               va='center', ha=ha, fontsize=11, fontweight='bold')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Percentage Change (%)', fontsize=13, fontweight='bold')
    ax.set_title('Percentage Change in LULC Area (2018-2024)', 
                fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pie_charts_comparison(
    area_stats_t1: Dict,
    area_stats_t2: Dict,
    year_t1: int = 2018,
    year_t2: int = 2024,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """Plot side-by-side pie charts for two time periods."""
    from src.visualization.maps import LULCColorScheme
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    classes = list(area_stats_t1.keys())
    colors = [LULCColorScheme.COLORS[i] for i in range(len(classes))]
    
    # T1 pie chart
    sizes_t1 = [area_stats_t1[cls]['percentage'] for cls in classes]
    axes[0].pie(sizes_t1, labels=classes, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[0].set_title(f'LULC Distribution - {year_t1}', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # T2 pie chart
    sizes_t2 = [area_stats_t2[cls]['percentage'] for cls in classes]
    axes[1].pie(sizes_t2, labels=classes, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title(f'LULC Distribution - {year_t2}', 
                     fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_sankey_diagram(
    transition_matrix: np.ndarray,
    class_names: List[str],
    min_flow: int = 100,
    title: str = "LULC Transition Flows"
) -> go.Figure:
    """Create interactive Sankey diagram for transitions."""
    from src.visualization.maps import LULCColorScheme
    
    sources = []
    targets = []
    values = []
    colors = []
    
    n_classes = len(class_names)
    
    for i in range(n_classes):
        for j in range(n_classes):
            if transition_matrix[i, j] >= min_flow:
                sources.append(i)
                targets.append(j + n_classes)
                values.append(int(transition_matrix[i, j]))
                
                rgba_color = LULCColorScheme.COLORS[i]
                colors.append(f'rgba({int(rgba_color[1:3], 16)}, '
                            f'{int(rgba_color[3:5], 16)}, '
                            f'{int(rgba_color[5:7], 16)}, 0.4)')
    
    node_labels = [f"{name} (2018)" for name in class_names] + \
                  [f"{name} (2024)" for name in class_names]
    
    node_colors = [LULCColorScheme.COLORS[i % n_classes] for i in range(2 * n_classes)]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family='Arial Black')),
        font=dict(size=12),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_interactive_comparison_chart(
    area_stats_t1: Dict,
    area_stats_t2: Dict,
    year_t1: int = 2018,
    year_t2: int = 2024
) -> go.Figure:
    """Create interactive comparison chart with multiple views."""
    from src.visualization.maps import LULCColorScheme
    
    classes = list(area_stats_t1.keys())
    area_t1 = [area_stats_t1[cls]['area_sqkm'] for cls in classes]
    area_t2 = [area_stats_t2[cls]['area_sqkm'] for cls in classes]
    pct_t1 = [area_stats_t1[cls]['percentage'] for cls in classes]
    pct_t2 = [area_stats_t2[cls]['percentage'] for cls in classes]
    colors = [LULCColorScheme.COLORS[i] for i in range(len(classes))]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Area Comparison (km²)', 'Percentage Distribution'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    fig.add_trace(
        go.Bar(name=str(year_t1), x=classes, y=area_t1, marker_color=colors,
              opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name=str(year_t2), x=classes, y=area_t2, marker_color=colors,
              opacity=1.0),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=classes, values=pct_t2, marker_colors=colors,
              textposition='inside', textinfo='percent+label'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="LULC Comparison Dashboard",
        font=dict(size=12)
    )
    
    return fig


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot confusion matrix for model evaluation."""
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized' if normalize else 'Count'},
               ax=ax)
    
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
