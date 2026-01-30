#!/usr/bin/env python3
"""
Complete project runner - executes full pipeline with all concepts:
1. Data preprocessing
2. LULC classification
3. Change detection
4. Enhanced visualization generation
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Projects\hack\tirupati-lulc-change-detection")

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"▶ {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed!")
        return False
    
    print(f"\n✅ {description} complete!\n")
    return True

def main():
    print_section("🚀 COMPLETE PROJECT EXECUTION")
    
    python = sys.executable
    
    # Step 1: Preprocessing
    print_section("STEP 1: DATA PREPROCESSING")
    print("""
    This step will:
    ✓ Load satellite imagery (Landsat 2018 & 2024)
    ✓ Clip to Tirupati boundary
    ✓ Normalize spectral data
    ✓ Prepare for classification
    """)
    
    if not run_command(
        [python, "scripts/preprocess_all.py", "--clip-only"],
        "Data Preprocessing"
    ):
        return
    
    # Step 2: Classification
    print_section("STEP 2: LULC CLASSIFICATION & CHANGE DETECTION")
    print("""
    This step will:
    ✓ Run baseline rule-based classifier
    ✓ Generate confidence maps
    ✓ Detect land cover changes
    ✓ Create transition matrix
    """)
    
    if not run_command(
        [python, "scripts/run_inference.py", "--model", "baseline", "--detect-changes"],
        "LULC Classification & Change Detection"
    ):
        return
    
    # Step 3: Enhanced Visualizations
    print_section("STEP 3: ENHANCED VISUALIZATION GENERATION")
    print("""
    This step will create:
    📍 MAPS (5 visualizations):
       • LULC Classification 2018
       • LULC Classification 2024
       • Side-by-Side Comparison
       • Change Detection Enhanced
       • Transition Heatmap
    
    📊 CHARTS (3 visualizations):
       • Area Comparison Bar Chart
       • Percentage Change Chart
       • Pie Charts Comparison
    
    🎯 INTERACTIVE (2 visualizations):
       • Sankey Diagram (Plotly)
       • Interactive Comparison Dashboard
    """)
    
    run_command(
        [python, "scripts/generate_enhanced_visuals.py"],
        "Enhanced Visualization Generation"
    )
    
    # Success summary
    print_section("✨ PROJECT EXECUTION COMPLETE")
    print("""
    🎉 All pipeline stages executed successfully!
    
    📊 WHAT YOU CAN NOW ACCESS:
    
    1. 🌐 FRONTEND DASHBOARD
       URL: http://localhost:3000
       Pages:
       • Overview (/) - Dashboard home
       • LULC Maps (/lulc) - Classifications
       • Change (/change) - Change analysis
       • Analytics (/analytics) - Statistics
       • Gallery (/gallery) - ✨ Full visualization gallery
       • Export (/export) - Download results
       • Upload (/upload) - Future scope
    
    2. 📁 GENERATED FILES
       Location: frontend/public/results/
       • maps/ - 5 PNG visualizations
       • charts/ - 3 statistical charts
       • interactive/ - 2 HTML dashboards
       • CSV files - Transition matrix & statistics
    
    3. 📝 AVAILABLE DATA
       • LULC classifications (2018 & 2024)
       • Change detection maps
       • Confidence scores
       • Transition matrix (5x5)
       • Area statistics by class
    
    4. 🎨 VISUALIZATION TYPES
       • Geospatial maps with legends
       • Bar charts with comparisons
       • Heatmaps for transitions
       • Interactive Sankey diagrams
       • Plotly dashboards with hover data
    
    ═══════════════════════════════════════════════════════════════════
    
    NEXT STEPS:
    
    1. Visit http://localhost:3000/gallery to view all visualizations
    2. Explore different tabs: Maps → Charts → Interactive
    3. Download visualizations for presentations
    4. Check Analytics page for detailed statistics
    5. View LULC Maps for side-by-side comparison
    
    ═══════════════════════════════════════════════════════════════════
    
    MONITORING:
    
    • Frontend: http://localhost:3000 (running in terminal)
    • Logs: logs/lulc_detection.log
    • Results: data/results/ (all outputs)
    • Frontend: frontend/public/results/ (web-accessible)
    
    ═══════════════════════════════════════════════════════════════════
    
    🎯 KEY FEATURES IMPLEMENTED:
    
    ✓ Multi-year LULC classification (2018 → 2024)
    ✓ Pixel-level change detection
    ✓ Class transition analysis
    ✓ Professional cartography
    ✓ Statistical analysis
    ✓ Interactive dashboards
    ✓ Web-based gallery
    ✓ Download capabilities
    
    ═══════════════════════════════════════════════════════════════════
    """)

if __name__ == "__main__":
    main()
