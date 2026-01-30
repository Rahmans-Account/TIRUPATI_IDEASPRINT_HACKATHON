"""Change detection visualization page."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("🔄 Change Detection Analysis")

# Confidence threshold
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Minimum confidence level for displaying changes"
)

st.markdown("---")

# Change map display
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Change Detection Map")
    
    # Create sample change map
    change_map = np.random.randint(0, 2, (100, 100))
    confidence = np.random.uniform(0, 1, (100, 100))
    
    # Mask low confidence
    masked_change = change_map.copy()
    masked_change[confidence < confidence_threshold] = 0
    
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.matplotlib.colors.ListedColormap(['lightgray', 'red'])
    im = ax.imshow(masked_change, cmap=cmap)
    ax.set_title("Change Detection (Red = Change, Gray = No Change)")
    ax.axis('off')
    
    st.pyplot(fig)

with col2:
    st.subheader("Change Statistics")
    
    # Calculate statistics
    total_pixels = masked_change.size
    changed_pixels = (masked_change == 1).sum()
    unchanged_pixels = (masked_change == 0).sum()
    change_percentage = (changed_pixels / total_pixels) * 100
    
    st.metric("Total Pixels", f"{total_pixels:,}")
    st.metric("Changed Pixels", f"{changed_pixels:,}", f"{change_percentage:.2f}%")
    st.metric("Unchanged Pixels", f"{unchanged_pixels:,}")
    st.metric("Avg Confidence", f"{confidence.mean():.3f}")

# Change hotspots
st.markdown("---")
st.subheader("🔥 Change Hotspots")

st.markdown("""
Areas with highest concentration of land use changes:
""")

hotspot_data = pd.DataFrame({
    'Region': ['Urban Core', 'Northern Suburbs', 'Eastern Belt', 'Western Hills', 'Southern Plains'],
    'Change %': [45.2, 32.1, 28.5, 15.3, 12.8],
    'Dominant Transition': [
        'Agriculture → Built-up',
        'Barren → Built-up',
        'Agriculture → Built-up',
        'Forest → Barren',
        'Agriculture → Barren'
    ]
})

st.dataframe(hotspot_data, use_container_width=True)
