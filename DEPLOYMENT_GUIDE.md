# Streamlit Community Cloud Deployment Guide

## Quick Setup for Streamlit Community Cloud

### 1. Prepare Files for Deployment

When uploading to Streamlit Community Cloud, use these files:

**requirements.txt** (copy from `deployment_requirements.txt`):
```
streamlit>=1.28.0
folium>=0.14.0
streamlit-folium>=0.20.0
opencv-python-headless>=4.8.0
matplotlib>=3.5.0
plotly>=5.0.0
numpy>=1.21.0,<2.0.0
pandas>=1.5.0
pillow>=9.0.0
scikit-learn>=1.3.0
```

**packages.txt** (copy from `deployment_packages.txt`):
```
libgl1-mesa-glx
libglib2.0-0
ffmpeg
```

### 2. Key Changes Made for Deployment

- ✅ Removed TensorFlow (caused platform compatibility issues)
- ✅ Switched to `opencv-python-headless` (Streamlit Cloud compatible)
- ✅ Used only scikit-learn for ML (lightweight and compatible)
- ✅ Added proper system packages for OpenCV support
- ✅ Constrained NumPy version for stability

### 3. Deployment Steps

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file as `app.py`
5. The app will automatically use the requirements.txt and packages.txt files

### 4. Features Maintained

- ✅ Real-time waste detection using scikit-learn models
- ✅ Interactive maps with Folium
- ✅ Full location database with 25+ water bodies
- ✅ Professional space-themed UI
- ✅ Comprehensive analytics and reporting
- ✅ Image processing with OpenCV
- ✅ Export capabilities (JSON, KML)

### 5. Performance Notes

- The app uses Random Forest and Gradient Boosting classifiers
- All machine learning is handled by scikit-learn (no TensorFlow required)
- Synthetic training data ensures consistent performance
- Location-specific analysis provides realistic results

The deployment is now optimized for Streamlit Community Cloud's environment and should work without the TensorFlow compatibility issues you encountered.