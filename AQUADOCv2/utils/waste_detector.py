import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from waste_detection_model import get_waste_detection_model

class WasteDetector:
    """AI/ML-based waste detection in water bodies"""
    
    def __init__(self):
        self.detection_params = {
            'min_contour_area': 100,
            'max_contour_area': 10000,
            'circularity_threshold': 0.3,
            'solidity_threshold': 0.6,
            'aspect_ratio_range': (0.2, 5.0)
        }
        
        # Model confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Initialize ML model
        try:
            self.ml_model = get_waste_detection_model()
            print("✓ Advanced ML model loaded for waste detection")
        except Exception as e:
            print(f"Warning: ML model initialization failed: {e}")
            self.ml_model = None
    
    def detect_waste(self, image, sensitivity=0.7):
        """
        Perform waste detection on preprocessed image
        
        Args:
            image: Preprocessed image array
            sensitivity: Detection sensitivity (0.1 to 1.0)
            
        Returns:
            dict: Detection results with waste locations and statistics
        """
        # Adjust detection parameters based on sensitivity
        adjusted_params = self._adjust_parameters_for_sensitivity(sensitivity)
        
        # Extract potential waste areas
        waste_regions = self._extract_waste_regions(image, adjusted_params)
        
        # Classify and score detected regions
        classified_regions = self._classify_waste_regions(waste_regions, image)
        
        # Calculate detection statistics
        detection_stats = self._calculate_detection_statistics(classified_regions, image.shape)
        
        # Prepare results
        results = {
            'waste_regions': classified_regions,
            'waste_count': len(classified_regions),
            'coverage_percentage': detection_stats['coverage_percentage'],
            'confidence_score': detection_stats['avg_confidence'],
            'waste_score': detection_stats['waste_score'],
            'waste_distribution': detection_stats['waste_distribution'],
            'detection_metadata': {
                'sensitivity': sensitivity,
                'total_detections': len(classified_regions),
                'high_confidence_detections': len([r for r in classified_regions if r['confidence'] > self.confidence_thresholds['high']]),
                'image_dimensions': image.shape
            }
        }
        
        return results
    
    def _adjust_parameters_for_sensitivity(self, sensitivity):
        """Adjust detection parameters based on sensitivity setting"""
        base_params = self.detection_params.copy()
        
        # Higher sensitivity = more detections (lower thresholds)
        sensitivity_factor = 1.0 - sensitivity * 0.5
        
        adjusted_params = {
            'min_contour_area': int(base_params['min_contour_area'] * sensitivity_factor),
            'max_contour_area': base_params['max_contour_area'],
            'circularity_threshold': base_params['circularity_threshold'] * sensitivity_factor,
            'solidity_threshold': base_params['solidity_threshold'] * sensitivity_factor,
            'aspect_ratio_range': base_params['aspect_ratio_range']
        }
        
        return adjusted_params
    
    def _extract_waste_regions(self, image, params):
        """Extract potential waste regions using computer vision techniques"""
        
        # Convert to grayscale for contour detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding to identify anomalous regions
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape characteristics
        waste_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if params['min_contour_area'] <= area <= params['max_contour_area']:
                # Calculate shape characteristics
                shape_metrics = self._calculate_shape_metrics(contour)
                
                # Filter by shape characteristics
                if (shape_metrics['circularity'] >= params['circularity_threshold'] and
                    shape_metrics['solidity'] >= params['solidity_threshold'] and
                    params['aspect_ratio_range'][0] <= shape_metrics['aspect_ratio'] <= params['aspect_ratio_range'][1]):
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    waste_regions.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'shape_metrics': shape_metrics,
                        'center': (x + w//2, y + h//2)
                    })
        
        return waste_regions
    
    def _calculate_shape_metrics(self, contour):
        """Calculate shape characteristics for a contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (4π*area/perimeter²)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Solidity (contour area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = w / h
        else:
            aspect_ratio = 0
        
        return {
            'circularity': circularity,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter
        }
    
    def _classify_waste_regions(self, waste_regions, image):
        """Classify detected regions using trained ML model"""
        classified_regions = []
        
        for region in waste_regions:
            # Extract region from image
            x, y, w, h = region['bbox']
            roi = image[y:y+h, x:x+w]
            
            # Use real ML model for classification
            classification_result = self._ml_classification(roi, region)
            
            # Combine region data with classification
            classified_region = {
                **region,
                'waste_type': classification_result['waste_type'],
                'confidence': classification_result['confidence'],
                'color_analysis': classification_result['color_analysis'],
                'texture_analysis': classification_result['texture_analysis'],
                'confidence_level': classification_result['confidence_level']
            }
            
            classified_regions.append(classified_region)
        
        return classified_regions
    
    def _ml_classification(self, roi, region):
        """
        Real ML model classification of waste regions using trained CNN and Random Forest
        """
        # Ensure roi has proper dimensions
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            # RGB image
            pass
        elif len(roi.shape) == 2:
            # Grayscale - convert to RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        else:
            # Handle other cases
            roi = np.stack([roi] * 3, axis=-1) if len(roi.shape) == 2 else roi
        
        # Get ML model prediction
        if self.ml_model is not None:
            try:
                waste_probability = self.ml_model.predict_waste_probability(roi)
                waste_type = self.ml_model.classify_waste_type(roi, waste_probability)
                confidence_level = self.ml_model.get_confidence_level(waste_probability)
            except Exception as e:
                print(f"ML model prediction failed: {e}")
                # Fallback to basic analysis
                waste_probability = 0.5
                waste_type = "mixed_debris"
                confidence_level = "medium"
        else:
            # Use basic heuristic analysis if ML model not available
            waste_probability = self._basic_waste_analysis(roi, region)
            waste_type = "mixed_debris"
            confidence_level = "medium"
        
        # Analyze color characteristics for additional context
        color_mean = np.mean(roi, axis=(0, 1))
        color_std = np.std(roi, axis=(0, 1))
        
        # Texture analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        texture_variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        
        # Map waste types to user-friendly names
        waste_type_mapping = {
            "clean_water": "Clean Water",
            "oil_spill": "Oil Spill",
            "plastic_debris": "Plastic Debris", 
            "organic_waste": "Organic Waste",
            "algae_bloom": "Algae Bloom",
            "mixed_debris": "Mixed Debris"
        }
        
        formatted_waste_type = waste_type_mapping.get(waste_type, "Unknown Pollutant")
        
        return {
            'waste_type': formatted_waste_type,
            'confidence': float(waste_probability),
            'confidence_level': confidence_level,
            'color_analysis': {
                'mean_intensity': float(np.mean(color_mean)),
                'color_variance': float(np.mean(color_std))
            },
            'texture_analysis': {
                'texture_variance': float(texture_variance),
                'uniformity': float(1.0 / (1.0 + texture_variance)) if texture_variance > 0 else 1.0
            }
        }
    
    def _basic_waste_analysis(self, roi, region):
        """Basic heuristic waste analysis when ML model is not available"""
        # Calculate basic features
        if len(roi.shape) == 3:
            color_mean = np.mean(roi, axis=(0, 1))
            brightness = np.mean(color_mean)
        else:
            brightness = np.mean(roi)
        
        # Simple heuristic based on area and brightness
        area = region['area']
        shape_metrics = region['shape_metrics']
        
        # Calculate probability based on deviation from typical water characteristics
        water_brightness_range = (80, 150)  # Typical water brightness
        brightness_deviation = abs(brightness - np.mean(water_brightness_range)) / 50.0
        
        # Large irregular shapes are more likely to be waste
        size_factor = min(1.0, area / 1000.0)
        irregularity_factor = 1.0 - shape_metrics['circularity']
        
        # Combine factors
        waste_probability = min(1.0, (brightness_deviation + size_factor + irregularity_factor) / 3.0)
        
        return waste_probability
    
    def _calculate_detection_statistics(self, classified_regions, image_shape):
        """Calculate comprehensive detection statistics"""
        if not classified_regions:
            return {
                'coverage_percentage': 0.0,
                'avg_confidence': 0.0,
                'waste_score': 0.0,
                'waste_distribution': {}
            }
        
        total_image_area = image_shape[0] * image_shape[1]
        total_waste_area = sum(region['area'] for region in classified_regions)
        coverage_percentage = (total_waste_area / total_image_area) * 100
        
        # Calculate average confidence
        confidences = [region['confidence'] for region in classified_regions]
        avg_confidence = np.mean(confidences) * 100
        
        # Calculate waste score (0-10 scale)
        # Based on coverage, confidence, and number of detections
        coverage_factor = min(1.0, coverage_percentage / 10.0)  # Normalize to 10% max
        confidence_factor = avg_confidence / 100.0
        detection_density = min(1.0, len(classified_regions) / 20.0)  # Normalize to 20 detections max
        
        waste_score = (coverage_factor * 0.4 + confidence_factor * 0.4 + detection_density * 0.2) * 10
        
        # Calculate waste type distribution
        waste_types = [region['waste_type'] for region in classified_regions]
        unique_types, counts = np.unique(waste_types, return_counts=True)
        waste_distribution = dict(zip(unique_types, counts.tolist()))
        
        return {
            'coverage_percentage': coverage_percentage,
            'avg_confidence': avg_confidence,
            'waste_score': waste_score,
            'waste_distribution': waste_distribution
        }
    
    def create_highlighted_image(self, original_image, detection_results):
        """
        Create an image with highlighted waste detection areas
        
        Args:
            original_image: Original satellite image
            detection_results: Results from detect_waste()
            
        Returns:
            PIL Image: Image with highlighted waste areas
        """
        # Convert to PIL Image if needed
        if isinstance(original_image, np.ndarray):
            highlighted_img = Image.fromarray(original_image)
        else:
            highlighted_img = original_image.copy()
        
        # Create drawing context
        draw = ImageDraw.Draw(highlighted_img)
        
        # Define colors for different confidence levels
        color_map = {
            'high': 'red',
            'medium': 'orange',
            'low': 'yellow'
        }
        
        # Draw detection overlays
        for region in detection_results['waste_regions']:
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Determine color based on confidence
            if confidence >= self.confidence_thresholds['high']:
                color = color_map['high']
                width = 3
            elif confidence >= self.confidence_thresholds['medium']:
                color = color_map['medium']
                width = 2
            else:
                color = color_map['low']
                width = 1
            
            # Draw bounding box
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=width
            )
            
            # Add confidence label
            label = f"{region['waste_type'][:8]}\n{confidence:.1%}"
            draw.text((x, y - 20), label, fill=color)
        
        return highlighted_img
    
    def export_detection_data_for_training(self, detection_results, image_path):
        """
        Export detection data in format suitable for YOLO training
        
        Args:
            detection_results: Results from detect_waste()
            image_path: Path to save training annotations
            
        Returns:
            dict: Training data export information
        """
        training_data = {
            'image_path': image_path,
            'annotations': [],
            'metadata': detection_results['detection_metadata']
        }
        
        # Convert detections to YOLO format
        for region in detection_results['waste_regions']:
            x, y, w, h = region['bbox']
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            img_height, img_width = detection_results['detection_metadata']['image_dimensions'][:2]
            
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Map waste type to class ID (simplified)
            waste_type_to_id = {
                'Plastic Debris': 0,
                'Organic Matter': 1,
                'Chemical Pollutants': 2,
                'Unknown': 3
            }
            
            class_id = waste_type_to_id.get(region['waste_type'], 3)
            
            annotation = {
                'class_id': class_id,
                'center_x': center_x,
                'center_y': center_y,
                'width': norm_width,
                'height': norm_height,
                'confidence': region['confidence'],
                'waste_type': region['waste_type']
            }
            
            training_data['annotations'].append(annotation)
        
        return training_data
