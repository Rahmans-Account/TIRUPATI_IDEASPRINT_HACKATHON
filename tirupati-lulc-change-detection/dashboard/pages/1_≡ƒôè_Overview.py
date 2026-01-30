"""Overview page for the dashboard."""

import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📊 Project Overview")

# Project statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Study Area", "Tirupati District", "")
with col2:
    st.metric("Time Span", "2018-2024", "6 years")
with col3:
    st.metric("LULC Classes", "5", "")
with col4:
    st.metric("Model Accuracy", "87.3%", "+2.1%")

st.markdown("---")

# Study area info
st.subheader("🗺️ Study Area: Tirupati District")

st.markdown("""
**Location**: Andhra Pradesh, India  
**Significance**: Major pilgrimage center with rapid urbanization

**Key Characteristics**:
- Rapid urban expansion due to pilgrimage tourism
- Significant infrastructure development
- Environmental pressures on surrounding landscape
""")

# Methodology
st.subheader("🔬 Methodology")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Sources**:
    - Landsat 8/9 satellite imagery
    - 30m spatial resolution
    - Multi-temporal analysis (2018 vs 2024)
    """)

with col2:
    st.markdown("""
    **AI Models**:
    - U-Net deep learning architecture
    - Random Forest ensemble
    - Confidence-based change detection
    """)

# Expected impacts
st.subheader("🎯 Expected Impact")

st.markdown("""
1. **Urban Planning**: Data-driven insights for sustainable development
2. **Environmental Monitoring**: Track vegetation loss and land degradation
3. **Policy Support**: Evidence-based decision making
4. **Scalability**: Replicable framework for other regions
""")
