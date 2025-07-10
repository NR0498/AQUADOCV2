import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    """Handle image preprocessing operations for waste detection"""
    
    def __init__(self):
        self.preprocessing_params = {
            'gaussian_blur_kernel': (5, 5),
            'bilateral_filter_d': 9,
            'bilateral_filter_sigma_color': 75,
            'bilateral_filter_sigma_space': 75,
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': (8, 8)
        }
    
    def preprocess_image(self, image):
        """
        Apply comprehensive preprocessing to satellite image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array: Preprocessed image
        """
        # Convert PIL to numpy if necessary
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to BGR for OpenCV processing
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Apply preprocessing pipeline
        processed_img = self._apply_preprocessing_pipeline(img_bgr)
        
        # Convert back to RGB
        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        return processed_img
    
    def _apply_preprocessing_pipeline(self, image):
        """Apply the complete preprocessing pipeline"""
        
        # 1. Noise reduction using bilateral filter
        denoised = cv2.bilateralFilter(
            image,
            self.preprocessing_params['bilateral_filter_d'],
            self.preprocessing_params['bilateral_filter_sigma_color'],
            self.preprocessing_params['bilateral_filter_sigma_space']
        )
        
        # 2. Contrast enhancement using CLAHE
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(
                clipLimit=self.preprocessing_params['clahe_clip_limit'],
                tileGridSize=self.preprocessing_params['clahe_tile_grid_size']
            )
            
            enhanced = np.zeros_like(denoised)
            for i in range(3):
                enhanced[:, :, i] = clahe.apply(denoised[:, :, i])
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.preprocessing_params['clahe_clip_limit'],
                tileGridSize=self.preprocessing_params['clahe_tile_grid_size']
            )
            enhanced = clahe.apply(denoised)
        
        # 3. Edge-preserving smoothing
        smoothed = cv2.edgePreservingFilter(enhanced, flags=2, sigma_s=50, sigma_r=0.4)
        
        # 4. Sharpening
        sharpened = self._apply_sharpening(smoothed)
        
        return sharpened
    
    def _apply_sharpening(self, image):
        """Apply unsharp masking for image sharpening"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def extract_water_mask(self, image):
        """
        Extract water areas from the image using color and texture analysis
        
        Args:
            image: Preprocessed image array
            
        Returns:
            numpy array: Binary mask of water areas
        """
        # Convert to HSV for better water detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define water color ranges in HSV
        # Water typically appears in blue/cyan ranges
        lower_water1 = np.array([100, 50, 50])
        upper_water1 = np.array([130, 255, 255])
        
        lower_water2 = np.array([0, 0, 50])
        upper_water2 = np.array([10, 50, 200])
        
        # Create masks for water detection
        mask1 = cv2.inRange(hsv, lower_water1, upper_water1)
        mask2 = cv2.inRange(hsv, lower_water2, upper_water2)
        
        # Combine masks
        water_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        return water_mask
    
    def enhance_multispectral_bands(self, image):
        """
        Simulate multispectral band enhancement for better waste detection
        
        Args:
            image: RGB image array
            
        Returns:
            numpy array: Enhanced multispectral representation
        """
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance L channel (lightness)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Simulate NIR (Near Infrared) band enhancement
        # This helps differentiate between water and debris
        nir_simulation = self._simulate_nir_band(enhanced)
        
        # Combine RGB with simulated NIR
        multispectral = np.dstack([enhanced, nir_simulation])
        
        return multispectral
    
    def _simulate_nir_band(self, rgb_image):
        """
        Simulate Near Infrared band from RGB image
        This is a simplified simulation for demonstration
        """
        # NIR typically shows vegetation as bright and water as dark
        # Convert to grayscale and apply inverse transformation
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Enhance vegetation (simulate high NIR reflectance)
        # and suppress water (low NIR reflectance)
        nir_sim = 255 - gray
        
        # Apply additional processing to enhance the simulation
        nir_sim = cv2.GaussianBlur(nir_sim, (3, 3), 0)
        
        return nir_sim
    
    def calculate_image_statistics(self, original, processed):
        """
        Calculate statistical measures for image quality assessment
        
        Args:
            original: Original image array
            processed: Processed image array
            
        Returns:
            dict: Image quality statistics
        """
        stats = {}
        
        # Convert to grayscale for statistics
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Calculate basic statistics
        stats['original_mean'] = np.mean(orig_gray)
        stats['original_std'] = np.std(orig_gray)
        stats['processed_mean'] = np.mean(proc_gray)
        stats['processed_std'] = np.std(proc_gray)
        
        # Calculate contrast enhancement ratio
        stats['contrast_enhancement'] = stats['processed_std'] / stats['original_std']
        
        # Calculate signal-to-noise ratio improvement (simplified)
        stats['snr_improvement'] = stats['processed_mean'] / stats['original_mean']
        
        return stats
