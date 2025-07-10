"""
Sample image generator for water body analysis
Creates synthetic satellite-like images for demonstration
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import cv2

class SampleImages:
    """Generate sample satellite images for water body analysis"""
    
    def __init__(self):
        self.image_size = (800, 600)
        self.water_color_ranges = {
            'clean': [(30, 100, 150), (50, 120, 180)],
            'moderate': [(40, 90, 120), (60, 110, 150)],
            'polluted': [(60, 80, 100), (80, 100, 120)]
        }
        
    def get_sample_image(self, location_name):
        """
        Generate or retrieve a sample satellite image for a given location
        
        Args:
            location_name: Name of the water body
            
        Returns:
            PIL.Image: Generated satellite image
        """
        # Determine pollution level based on location characteristics
        pollution_level = self._determine_pollution_level(location_name)
        
        # Generate base satellite image
        image = self._generate_base_satellite_image(pollution_level)
        
        # Add realistic features
        image = self._add_water_body_features(image, location_name)
        
        # Add waste elements based on pollution level
        image = self._add_waste_elements(image, pollution_level)
        
        # Apply final processing for realism
        image = self._apply_satellite_effects(image)
        
        return image
    
    def _determine_pollution_level(self, location_name):
        """Determine pollution level based on location characteristics"""
        # Simulate pollution levels based on location type
        urban_indicators = ['salt', 'great salt', 'salton', 'mono']
        pristine_indicators = ['baikal', 'crater', 'tahoe', 'bled', 'como']
        
        location_lower = location_name.lower()
        
        if any(indicator in location_lower for indicator in urban_indicators):
            return 'polluted'
        elif any(indicator in location_lower for indicator in pristine_indicators):
            return 'clean'
        else:
            return 'moderate'
    
    def _generate_base_satellite_image(self, pollution_level):
        """Generate base satellite image with water and land"""
        width, height = self.image_size
        
        # Create base image
        image = Image.new('RGB', (width, height), (34, 139, 34))  # Forest green base
        draw = ImageDraw.Draw(image)
        
        # Generate water body shape
        water_shape = self._generate_water_shape(width, height)
        
        # Select water color based on pollution level
        water_colors = self.water_color_ranges[pollution_level]
        
        # Fill water area with gradient
        for polygon in water_shape:
            # Create gradient effect
            base_color = random.choice(water_colors)
            color_variation = tuple(
                max(0, min(255, base_color[i] + random.randint(-20, 20)))
                for i in range(3)
            )
            draw.polygon(polygon, fill=color_variation)
        
        # Add land features around water
        self._add_land_features(image, draw, water_shape)
        
        return image
    
    def _generate_water_shape(self, width, height):
        """Generate realistic water body shapes"""
        # Create main water body
        center_x, center_y = width // 2, height // 2
        
        # Generate irregular water boundary
        water_shapes = []
        
        # Main water body
        main_points = []
        num_points = 12
        base_radius = min(width, height) * 0.3
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            radius_variation = random.uniform(0.7, 1.3)
            radius = base_radius * radius_variation
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            main_points.append((int(x), int(y)))
        
        water_shapes.append(main_points)
        
        # Add smaller water bodies/inlets
        for _ in range(random.randint(1, 3)):
            inlet_points = []
            inlet_center_x = random.randint(width // 4, 3 * width // 4)
            inlet_center_y = random.randint(height // 4, 3 * height // 4)
            inlet_radius = random.randint(30, 80)
            
            for i in range(6):
                angle = (2 * np.pi * i) / 6
                x = inlet_center_x + inlet_radius * np.cos(angle)
                y = inlet_center_y + inlet_radius * np.sin(angle)
                inlet_points.append((int(x), int(y)))
            
            water_shapes.append(inlet_points)
        
        return water_shapes
    
    def _add_land_features(self, image, draw, water_shapes):
        """Add realistic land features around water"""
        width, height = self.image_size
        
        # Add vegetation patches
        for _ in range(random.randint(8, 15)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(20, 60)
            
            # Vegetation colors
            veg_color = (
                random.randint(20, 60),
                random.randint(80, 140),
                random.randint(20, 60)
            )
            
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=veg_color
            )
        
        # Add shoreline details
        for _ in range(random.randint(5, 10)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            sand_color = (
                random.randint(180, 220),
                random.randint(170, 210),
                random.randint(120, 160)
            )
            
            draw.ellipse(
                [x - 15, y - 15, x + 15, y + 15],
                fill=sand_color
            )
    
    def _add_waste_elements(self, image, pollution_level):
        """Add waste elements based on pollution level"""
        if pollution_level == 'clean':
            return image
        
        # Convert to numpy for easier manipulation
        img_array = np.array(image)
        
        # Define waste characteristics
        waste_configs = {
            'moderate': {'count': random.randint(3, 8), 'size_range': (5, 15), 'opacity': 0.6},
            'polluted': {'count': random.randint(8, 20), 'size_range': (8, 25), 'opacity': 0.8}
        }
        
        config = waste_configs.get(pollution_level, waste_configs['moderate'])
        
        # Add waste patches
        for _ in range(config['count']):
            self._add_single_waste_patch(img_array, config)
        
        # Add floating debris
        for _ in range(config['count'] // 2):
            self._add_floating_debris(img_array, config)
        
        return Image.fromarray(img_array)
    
    def _add_single_waste_patch(self, img_array, config):
        """Add a single waste patch to the image"""
        height, width = img_array.shape[:2]
        
        # Random location
        x = random.randint(0, width - config['size_range'][1])
        y = random.randint(0, height - config['size_range'][1])
        
        # Random size
        size = random.randint(*config['size_range'])
        
        # Waste colors (plastics, oils, etc.)
        waste_colors = [
            [120, 80, 60],   # Brown (organic waste)
            [80, 80, 80],    # Gray (plastic)
            [60, 40, 20],    # Dark brown (oil)
            [100, 100, 100], # Light gray (foam)
            [140, 100, 80]   # Tan (mixed debris)
        ]
        
        waste_color = random.choice(waste_colors)
        
        # Create circular waste patch
        y_coords, x_coords = np.ogrid[:height, :width]
        center_x, center_y = x + size // 2, y + size // 2
        
        # Create mask for circular patch
        mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (size // 2)**2
        
        # Apply waste color with some transparency effect
        for i in range(3):
            img_array[mask, i] = (
                img_array[mask, i] * (1 - config['opacity']) + 
                waste_color[i] * config['opacity']
            ).astype(np.uint8)
    
    def _add_floating_debris(self, img_array, config):
        """Add small floating debris elements"""
        height, width = img_array.shape[:2]
        
        # Multiple small debris pieces
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, width - 5)
            y = random.randint(0, height - 5)
            
            debris_size = random.randint(2, 4)
            debris_color = [
                random.randint(50, 100),
                random.randint(40, 80),
                random.randint(30, 70)
            ]
            
            # Small rectangular debris
            img_array[y:y+debris_size, x:x+debris_size] = debris_color
    
    def _apply_satellite_effects(self, image):
        """Apply effects to make image look like satellite imagery"""
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Add slight blur to simulate atmospheric effects
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Add noise to simulate sensor noise
        noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Adjust contrast and brightness for satellite look
        img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)
        
        # Convert back to PIL
        processed_image = Image.fromarray(img_array)
        
        # Apply slight sharpening
        processed_image = processed_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return processed_image
    

    
    def _add_water_body_features(self, image, location_name):
        """Add specific features based on water body type"""
        # Add context-specific features
        location_lower = location_name.lower()
        
        if 'salt' in location_lower:
            # Add salt formations
            self._add_salt_formations(image)
        elif 'crater' in location_lower:
            # Add volcanic features
            self._add_volcanic_features(image)
        elif any(indicator in location_lower for indicator in ['alpine', 'mountain', 'tahoe']):
            # Add mountain/alpine features
            self._add_alpine_features(image)
        
        return image
    
    def _add_salt_formations(self, image):
        """Add salt crystal formations for salt lakes"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add white/light colored salt deposits
        for _ in range(random.randint(3, 7)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(10, 30)
            
            salt_color = (
                random.randint(220, 255),
                random.randint(220, 255),
                random.randint(220, 255)
            )
            
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=salt_color
            )
    
    def _add_volcanic_features(self, image):
        """Add volcanic rock features for crater lakes"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add dark volcanic rock
        for _ in range(random.randint(4, 8)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(15, 40)
            
            volcanic_color = (
                random.randint(40, 80),
                random.randint(40, 80),
                random.randint(40, 80)
            )
            
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=volcanic_color
            )
    
    def _add_alpine_features(self, image):
        """Add alpine/mountain features"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Add snow patches or rocky areas
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(20, 50)
            
            # Snow or light rock color
            alpine_color = (
                random.randint(180, 255),
                random.randint(180, 255),
                random.randint(180, 255)
            )
            
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=alpine_color
            )
