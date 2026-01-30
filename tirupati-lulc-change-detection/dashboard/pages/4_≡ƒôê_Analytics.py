"""Analytics and statistics page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("📈 LULC Analytics & Statistics")

# Transition Matrix
st.subheader("Transition Matrix")
st.markdown("Pixel counts showing class-to-class transitions")

# Sample transition matrix
class_names = ['Forest', 'Water', 'Agriculture', 'Barren', 'Built-up']
transition_data = np.array([
    [8500, 50, 200, 100, 150],
    [20, 1800, 10, 30, 40],
    [150, 30, 6000, 500, 1320],
    [80, 40, 400, 2500, 980],
    [10, 5, 50, 20, 3915]
])

transition_df = pd.DataFrame(
    transition_data,
    index=[f"{name} (2018)" for name in class_names],
    columns=[f"{name} (2024)" for name in class_names]
)

st.dataframe(transition_df, use_container_width=True)

# Visualizations
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Area Change by Class")
    
    # Sample data
    area_change = pd.DataFrame({
        'Class': class_names,
        '2018': [850, 180, 600, 250, 120],
        '2024': [800, 190, 598, 300, 400]
    })
    
    fig = go.Figure(data=[
        go.Bar(name='2018', x=area_change['Class'], y=area_change['2018']),
        go.Bar(name='2024', x=area_change['Class'], y=area_change['2024'])
    ])
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Class Distribution 2024")
    
    fig = px.pie(
        values=area_change['2024'],
        names=area_change['Class'],
        title="LULC Class Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Major transitions
st.markdown("---")
st.subheader("Top 10 Major Transitions")

major_transitions = pd.DataFrame({
    'From': ['Agriculture', 'Agriculture', 'Barren', 'Forest', 'Agriculture'],
    'To': ['Built-up', 'Barren', 'Built-up', 'Barren', 'Forest'],
    'Pixels': [1320, 500, 980, 200, 200],
    'Area (km²)': [1.188, 0.450, 0.882, 0.180, 0.180]
})

st.dataframe(major_transitions, use_container_width=True)
