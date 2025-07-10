# AquaDoc - Water Body Waste Detection System

## Overview

AquaDoc (also branded as AquaGuard) is an AI/ML-powered water body waste detection system that analyzes satellite imagery to monitor and detect waste pollution in water bodies worldwide. The application provides an interactive web interface for selecting locations, processing satellite images, and visualizing waste detection results with comprehensive analytics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with custom CSS styling
- **UI Theme**: Professional dark theme with blue/cyan gradients and space-inspired styling
- **Typography**: Custom Google Fonts (Orbitron for headers, JetBrains Mono for body text)
- **Interactive Components**: Folium maps for geospatial visualization, Plotly for analytics charts
- **Navigation**: Session state-based multi-page navigation system

### Backend Architecture
- **Core Language**: Python 3.x
- **Image Processing**: OpenCV and PIL for computer vision operations
- **Data Processing**: NumPy for numerical computations, Pandas for data manipulation
- **Visualization**: Matplotlib and Plotly for charts and graphs

### Processing Pipeline
The system follows a three-stage processing pipeline:
1. **Image Preprocessing**: Gaussian blur, bilateral filtering, and CLAHE enhancement
2. **Waste Detection**: Contour-based analysis with machine learning classification
3. **Results Visualization**: Statistical analysis and interactive map overlays

## Key Components

### 1. Image Processing Module (`utils/image_processor.py`)
- **Purpose**: Preprocesses satellite images for optimal waste detection
- **Techniques**: 
  - Gaussian blur for noise reduction
  - Bilateral filtering for edge preservation
  - CLAHE for contrast enhancement
  - Color space conversions (RGB/BGR handling)

### 2. Waste Detection Engine (`utils/waste_detector.py`)
- **Purpose**: AI-powered waste identification in water bodies
- **Methods**:
  - Contour analysis with area and shape filtering
  - Confidence scoring system (high/medium/low thresholds)
  - Adjustable sensitivity parameters
  - Classification of waste types and sizes

### 3. Geospatial Utilities (`utils/map_utils.py`)
- **Purpose**: Interactive mapping and location management
- **Features**:
  - Folium-based map generation
  - Satellite imagery overlay support
  - Custom markers and popups
  - Multi-layer map controls

### 4. Data Management
- **Water Bodies Database** (`data/water_bodies.py`): Pre-configured locations with coordinates and metadata
- **Sample Image Generator** (`data/sample_images.py`): Synthetic satellite image creation for demonstration

## Data Flow

1. **Location Selection**: User selects from 25+ pre-configured water bodies
2. **Image Acquisition**: System generates or retrieves satellite imagery
3. **Preprocessing**: Image enhancement through OpenCV pipeline
4. **Detection**: AI analysis identifies potential waste regions
5. **Classification**: Detected regions are scored and categorized
6. **Visualization**: Results displayed on interactive maps with statistical overlays
7. **Analytics**: Comprehensive reporting with pollution metrics

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **Folium**: Interactive map generation
- **Plotly**: Data visualization and charting
- **NumPy/Pandas**: Numerical computing and data manipulation
- **Pillow (PIL)**: Image handling and manipulation

### System Dependencies
- **libgl1**: OpenGL library for computer vision operations
- **libglib2.0-0**: Core GLib library for system operations

### Visualization Stack
- **Matplotlib**: Static plotting and chart generation
- **Streamlit-Folium**: Integration between Streamlit and Folium maps
- **Google Fonts**: Custom typography (Orbitron, JetBrains Mono)

## Deployment Strategy

### Platform Configuration
- **Target Platform**: Replit cloud environment
- **Package Management**: Requirements specified in `packages.txt`
- **Entry Point**: `app.py` serves as main Streamlit application
- **Asset Structure**: Modular organization with separate utils and data directories

### Scalability Considerations
- **Image Processing**: Optimized OpenCV operations for performance
- **Memory Management**: Efficient NumPy array handling for large satellite images
- **User Interface**: Responsive design with sidebar navigation and status indicators
- **State Management**: Session-based navigation for multi-page workflows

### Performance Optimizations
- **Caching Strategy**: Streamlit's built-in caching for expensive operations
- **Image Compression**: PIL-based optimization for web display
- **Lazy Loading**: On-demand processing of satellite imagery
- **Modular Architecture**: Separated concerns for maintainability and testing

## Recent Changes

### Migration and ML Enhancement (July 10, 2025)
- **Migration Complete**: Successfully migrated from Replit Agent to standard Replit environment
- **Advanced ML Models**: Implemented real machine learning models using Random Forest and Gradient Boosting
- **Location-Specific Analysis**: Added LocationAnalyzer for realistic, location-based waste detection
- **Enhanced Results**: Improved analysis accuracy with location-specific pollution profiles
- **Real-World Data**: Analysis now reflects actual water body characteristics and pollution patterns