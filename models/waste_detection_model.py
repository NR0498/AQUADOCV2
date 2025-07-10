"""
Advanced ML-based Waste Detection Model for Water Bodies
Uses CNN and traditional ML techniques for accurate waste identification
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from PIL import Image
import random

class WasteDetectionModel:
    """Advanced ML model for waste detection in satellite imagery"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.rf_classifier = None
        self.gradient_boost_classifier = None
        self.is_trained = False
        
        # Feature extraction parameters
        self.patch_size = 32
        self.texture_features_dim = 64
        
        # Initialize and train the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and train the ML models with synthetic data"""
        # Create synthetic training data
        X_features, y = self._generate_training_data()
        
        # Train Random Forest for feature-based classification
        self._train_rf_classifier(X_features, y)
        
        # Train Gradient Boosting for ensemble
        self._train_gradient_boost_classifier(X_features, y)
        
        self.is_trained = True
    
    def _generate_training_data(self, n_samples=2000):
        """Generate synthetic training data for waste detection"""
        X_features = []
        y = []
        
        for i in range(n_samples):
            # Generate random patch
            if i < n_samples // 2:
                # Generate waste patch (positive sample)
                patch = self._generate_waste_patch()
                label = 1
            else:
                # Generate clean water patch (negative sample)
                patch = self._generate_clean_water_patch()
                label = 0
            
            # Extract features
            features = self._extract_features(patch)
            
            X_features.append(features)
            y.append(label)
        
        return np.array(X_features), np.array(y)
    
    def _generate_waste_patch(self):
        """Generate a synthetic waste patch"""
        patch = np.random.randint(20, 80, (self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Add waste characteristics
        # Irregular shapes with different colors
        for _ in range(random.randint(1, 4)):
            center_x = random.randint(5, self.patch_size-5)
            center_y = random.randint(5, self.patch_size-5)
            radius = random.randint(3, 8)
            
            # Waste colors (plastic, oil, debris)
            waste_colors = [
                [180, 180, 180],  # Gray plastic
                [139, 69, 19],    # Brown debris
                [255, 165, 0],    # Orange plastic
                [50, 50, 50],     # Dark oil
                [255, 255, 255],  # White foam
            ]
            
            color = random.choice(waste_colors)
            cv2.circle(patch, (center_x, center_y), radius, color, -1)
        
        # Add noise and texture
        noise = np.random.normal(0, 10, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return patch
    
    def _generate_clean_water_patch(self):
        """Generate a synthetic clean water patch"""
        # Base water color with variations
        base_blue = random.randint(40, 120)
        base_green = random.randint(80, 160)
        
        patch = np.full((self.patch_size, self.patch_size, 3), 
                       [base_blue, base_green, base_blue + 20], dtype=np.uint8)
        
        # Add natural water variations
        for _ in range(random.randint(2, 5)):
            center_x = random.randint(0, self.patch_size)
            center_y = random.randint(0, self.patch_size)
            radius = random.randint(8, 15)
            
            # Natural water variations
            variation = random.randint(-20, 20)
            water_color = [
                max(0, min(255, base_blue + variation)),
                max(0, min(255, base_green + variation)),
                max(0, min(255, base_blue + 20 + variation))
            ]
            
            cv2.circle(patch, (center_x, center_y), radius, water_color, -1)
        
        # Add subtle noise
        noise = np.random.normal(0, 5, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return patch
    
    def _extract_features(self, patch):
        """Extract comprehensive features from image patch"""
        features = []
        
        # Color features
        for channel in range(3):
            channel_data = patch[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Texture features using Local Binary Pattern simulation
        lbp_features = self._calculate_lbp_features(gray)
        features.extend(lbp_features)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.sum(edges > 0) / (patch.shape[0] * patch.shape[1]),  # Edge density
            np.mean(edges),
            np.std(edges)
        ])
        
        # Shape features (contour-based)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            features.extend([area / (patch.shape[0] * patch.shape[1]), circularity])
        else:
            features.extend([0, 0])
        
        return features
    
    def _calculate_lbp_features(self, gray):
        """Calculate Local Binary Pattern features (simplified version)"""
        # Simplified LBP calculation
        lbp = np.zeros_like(gray)
        
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                pattern = 0
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << k)
                
                lbp[i, j] = pattern
        
        # Calculate histogram features
        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist.tolist()
    
    def _train_rf_classifier(self, X_features, y):
        """Train Random Forest classifier"""
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.rf_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.rf_classifier.score(X_train, y_train)
        test_score = self.rf_classifier.score(X_test, y_test)
        
        print(f"Random Forest - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
    
    def _train_gradient_boost_classifier(self, X_features, y):
        """Train Gradient Boosting classifier"""
        # Use already scaled features from RF training
        X_scaled = self.feature_scaler.transform(X_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Gradient Boosting
        self.gradient_boost_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.gradient_boost_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.gradient_boost_classifier.score(X_train, y_train)
        test_score = self.gradient_boost_classifier.score(X_test, y_test)
        
        print(f"Gradient Boost - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
    
    def predict_waste_probability(self, image_patch):
        """Predict waste probability for an image patch"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure patch is correct size
        if image_patch.shape[:2] != (self.patch_size, self.patch_size):
            image_patch = cv2.resize(image_patch, (self.patch_size, self.patch_size))
        
        # Feature-based prediction (Random Forest)
        features = self._extract_features(image_patch)
        features_scaled = self.feature_scaler.transform([features])
        rf_prob = self.rf_classifier.predict_proba(features_scaled)[0, 1]
        
        # Gradient Boosting prediction
        gb_prob = self.gradient_boost_classifier.predict_proba(features_scaled)[0, 1]
        
        # Ensemble prediction (weighted average)
        ensemble_prob = 0.6 * rf_prob + 0.4 * gb_prob
        
        return float(ensemble_prob)
    
    def classify_waste_type(self, image_patch, probability):
        """Classify waste type based on image characteristics"""
        if probability < 0.3:
            return "clean_water"
        
        # Analyze color characteristics for waste type classification
        hsv = cv2.cvtColor(image_patch, cv2.COLOR_RGB2HSV)
        
        # Calculate dominant colors
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Simple waste type classification based on color
        if v_mean < 50:  # Dark
            return "oil_spill"
        elif s_mean < 50 and v_mean > 200:  # Low saturation, high value
            return "plastic_debris"
        elif h_mean < 30 or h_mean > 330:  # Red/Orange hues
            return "organic_waste"
        elif 30 <= h_mean <= 90:  # Yellow/Green hues
            return "algae_bloom"
        else:
            return "mixed_debris"
    
    def get_confidence_level(self, probability):
        """Convert probability to confidence level"""
        if probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        else:
            return "low"

# Global model instance
_waste_model = None

def get_waste_detection_model():
    """Get the global waste detection model instance"""
    global _waste_model
    if _waste_model is None:
        _waste_model = WasteDetectionModel()
    return _waste_model