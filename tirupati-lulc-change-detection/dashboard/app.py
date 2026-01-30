"""Main Streamlit dashboard for LULC Change Detection."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Tirupati LULC Change Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🌍 Tirupati LULC Change Detection System")
st.markdown("### AI-Powered Land Use Land Cover Analysis")

st.markdown("""
---

## Welcome to the LULC Change Detection Dashboard

This interactive dashboard provides comprehensive analysis of land use and land cover changes 
in Tirupati District between 2018 and 2024.

### 📊 Key Features:

- **LULC Maps**: View side-by-side classification maps
- **Change Detection**: Interactive change analysis with confidence scores
- **Analytics**: Detailed statistics and transition matrices
- **Export Tools**: Download maps, data, and reports

### 🎯 Project Overview:

This system uses AI-powered remote sensing to:
1. Classify satellite imagery into 5 land cover types
2. Detect pixel-level changes between time periods
3. Quantify transitions with confidence estimates
4. Provide actionable insights for urban planning

### 📈 LULC Classes:

- 🟢 **Forest**: Dense vegetation and tree cover
- 🔵 **Water Bodies**: Rivers, lakes, and ponds
- 🟡 **Agriculture**: Cropland and farmland
- 🟤 **Barren Land**: Exposed soil and rocky areas
- 🔴 **Built-up**: Urban areas and infrastructure

---

### 🚀 Getting Started:

Use the sidebar to navigate between different sections of the dashboard.

""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("""
Select a page from above to:
- View LULC maps
- Analyze changes
- Explore statistics
- Export results
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Project**: Tirupati LULC Change Detection  
**Technology**: AI + Remote Sensing  
**Data Source**: Landsat 8/9  
**Version**: 1.0.0
""")
