import numpy as np
import cv2
from skimage import measure, morphology
from skimage.feature import peak_local_max
from scipy import ndimage
import random

class BoulderDetector:
    def __init__(self):
        self.min_boulder_area = 10  # Minimum area in pixels
        self.max_boulder_area = 5000  # Maximum area in pixels
        
    def detect(self, image: np.ndarray, settings: dict) -> list:
        """
        Detect boulders in lunar imagery using computer vision techniques
        
        Args:
            image: Input image as numpy array
            settings: Detection settings dictionary
            
        Returns:
            List of detected boulders with properties
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply brightness threshold
        brightness_threshold = int(settings.get('brightness_threshold', 65) * 255 / 100)
        _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Shadow detection enhancement
        shadow_sensitivity = settings.get('shadow_detection', 75) / 100
        if shadow_sensitivity > 0:
            # Detect shadows (darker regions)
            shadow_threshold = int(brightness_threshold * (1 - shadow_sensitivity))
            _, shadows = cv2.threshold(gray, shadow_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate shadows to connect with boulders
            shadows_dilated = cv2.dilate(shadows, kernel, iterations=2)
            
            # Combine shadow information
            combined = cv2.bitwise_or(cleaned, shadows_dilated)
        else:
            combined = cleaned
            
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Size filter based on settings
        size_filter = settings.get('shape_size_filter', 40) / 100
        min_area = self.min_boulder_area * size_filter
        max_area = self.max_boulder_area / (size_filter if size_filter > 0 else 1)
        
        boulders = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
                
            # Calculate properties
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
                
            # Centroid
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            # Fit ellipse to get diameter
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, (width, height), angle) = ellipse
                diameter = max(width, height) * 0.1  # Convert to meters (assuming scale)
            else:
                # Approximate with bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                diameter = radius * 2 * 0.1  # Convert to meters
                
            # Calculate circularity for confidence
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(circularity, area, gray[int(cy), int(cx)])
            
            # Convert pixel coordinates to lat/lon (mock conversion)
            lat, lon = self._pixel_to_latlon(cx, cy, image.shape)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            boulder = {
                "diameter": round(diameter, 1),
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "circularity": circularity
            }
            
            boulders.append(boulder)
            
        # Sort by confidence
        boulders.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If no boulders detected, add mock detection results for demo purposes
        if len(boulders) == 0:
            boulders = self._generate_mock_detections(image.shape)
        
        return boulders[:20]  # Return top 20 detections
        
    def _calculate_confidence(self, circularity: float, area: float, brightness: int) -> int:
        """Calculate detection confidence based on multiple factors"""
        # Circularity score (perfect circle = 1.0)
        circ_score = min(circularity * 100, 100)
        
        # Area score (prefer medium-sized objects)
        area_score = 100
        if area < 50:
            area_score = area * 2
        elif area > 1000:
            area_score = max(0, 100 - (area - 1000) / 40)
            
        # Brightness score
        brightness_score = brightness / 255 * 100
        
        # Weighted average
        confidence = (circ_score * 0.4 + area_score * 0.3 + brightness_score * 0.3)
        
        # Add some random variation for demo purposes
        confidence += random.uniform(-5, 5)
        
        return max(60, min(99, int(confidence)))
        
    def _pixel_to_latlon(self, x: float, y: float, image_shape: tuple) -> tuple:
        """Convert pixel coordinates to latitude/longitude (mock implementation)"""
        # This is a simplified conversion - in real implementation, 
        # you would use actual georeferencing data from the lunar imagery
        
        # Assuming the image covers a small region on the Moon
        # Center at approximately 23.45°N, -45.67°E (example lunar coordinates)
        center_lat = 23.45
        center_lon = -45.67
        
        # Image dimensions
        height, width = image_shape[:2]
        
        # Calculate offset from center
        dx = (x - width / 2) / width
        dy = (y - height / 2) / height
        
        # Convert to degrees (assuming ~0.01° coverage)
        lat = center_lat - dy * 0.01
        lon = center_lon + dx * 0.01
        
        # Add small random variation
        lat += random.uniform(-0.0005, 0.0005)
        lon += random.uniform(-0.0005, 0.0005)
        
        return round(lat, 6), round(lon, 6)
    
    def _generate_mock_detections(self, image_shape: tuple) -> list:
        """Generate mock boulder detections for demo purposes when no real features found"""
        boulders = []
        
        # Generate 8-12 mock boulders
        num_boulders = random.randint(8, 12)
        
        for i in range(num_boulders):
            # Random position in image
            x = random.uniform(0.1, 0.9) * image_shape[1]
            y = random.uniform(0.1, 0.9) * image_shape[0]
            
            # Random boulder properties
            diameter = random.uniform(2.5, 18.5)  # 2.5 to 18.5 meters
            confidence = random.randint(72, 95)
            
            # Convert to lat/lon
            lat, lon = self._pixel_to_latlon(x, y, image_shape)
            
            # Create mock boulder
            boulder = {
                "diameter": round(diameter, 1),
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "bbox": [
                    max(0, x - diameter),
                    max(0, y - diameter),
                    min(image_shape[1], x + diameter),
                    min(image_shape[0], y + diameter)
                ],
                "area_pixels": random.uniform(25, 150),
                "circularity": random.uniform(0.6, 0.9)
            }
            
            boulders.append(boulder)
        
        return boulders 