from setuptools import setup, find_packages

setup(
    name="tirupati-lulc-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.0.1",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.8",
)
