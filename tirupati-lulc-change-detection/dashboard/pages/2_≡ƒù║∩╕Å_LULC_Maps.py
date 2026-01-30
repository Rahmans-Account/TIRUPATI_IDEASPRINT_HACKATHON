"""LULC Maps visualization page."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.title("🗺️ LULC Classification Maps")

# Year selection
col1, col2 = st.columns(2)

with col1:
    year1 = st.selectbox("Select First Time Period", [2018, 2024], index=0)

with col2:
    year2 = st.selectbox("Select Second Time Period", [2018, 2024], index=1)

st.markdown("---")

# Map display
col1, col2 = st.columns(2)

# Define colors for LULC classes
lulc_colors = {
    0: '#228B22',  # Forest - Green
    1: '#0000FF',  # Water - Blue
    2: '#FFFF00',  # Agriculture - Yellow
    3: '#8B4513',  # Barren - Brown
    4: '#FF0000'   # Built-up - Red
}

class_names = {
    0: 'Forest',
    1: 'Water Bodies',
    2: 'Agriculture',
    3: 'Barren Land',
    4: 'Built-up'
}

with col1:
    st.subheader(f"LULC Map - {year1}")
    
    # Create sample classification map (replace with actual data)
    sample_map = np.random.randint(0, 5, (100, 100))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.matplotlib.colors.ListedColormap(list(lulc_colors.values()))
    im = ax.imshow(sample_map, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(f"LULC Classification - {year1}")
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(list(class_names.values()))
    
    st.pyplot(fig)

with col2:
    st.subheader(f"LULC Map - {year2}")
    
    # Create sample classification map
    sample_map2 = np.random.randint(0, 5, (100, 100))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sample_map2, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(f"LULC Classification - {year2}")
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_ticklabels(list(class_names.values()))
    
    st.pyplot(fig)

# Legend
st.markdown("---")
st.subheader("📋 LULC Class Legend")

legend_cols = st.columns(5)
for i, (code, name) in enumerate(class_names.items()):
    with legend_cols[i]:
        st.markdown(f"**{name}**")
        st.color_picker("", lulc_colors[code], disabled=True, key=f"color_{code}")
