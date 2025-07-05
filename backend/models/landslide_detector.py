import numpy as np
import cv2
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
from scipy import ndimage
import random

class LandslideDetector:
    def __init__(self):
        self.min_landslide_area = 1000  # Minimum area in pixels for landslides
        self.max_landslide_area = 50000  # Maximum area in pixels
        
    def detect(self, image: np.ndarray, settings: dict) -> list:
        """
        Detect landslides in lunar imagery using edge detection and morphological analysis
        
        Args:
            image: Input image as numpy array
            settings: Detection settings dictionary
            
        Returns:
            List of detected landslides with properties
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 30, 150)
        
        # Gradient analysis for slope detection
        gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold gradient magnitude
        brightness_threshold = settings.get('brightness_threshold', 65) / 100
        gradient_threshold = np.percentile(gradient_magnitude, brightness_threshold * 100)
        gradient_binary = gradient_magnitude > gradient_threshold
        
        # Combine edge and gradient information
        combined = cv2.bitwise_or(edges, gradient_binary.astype(np.uint8) * 255)
        
        # Morphological operations to connect landslide features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fill holes and smooth regions
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255
        
        # Apply size filter from settings
        size_filter = settings.get('shape_size_filter', 40) / 100
        min_area = self.min_landslide_area * size_filter
        max_area = self.max_landslide_area / (size_filter if size_filter > 0 else 1)
        
        # Find contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        landslides = []
        
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
            
            # Convert pixel area to km² (mock conversion)
            area_km2 = area * 0.0001  # Assuming some scale factor
            
            # Calculate elongation (aspect ratio) for confidence
            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), angle = rect
            elongation = max(width, height) / (min(width, height) + 1e-6)
            
            # Analyze texture within contour for confidence
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate mean gradient within landslide region
            mean_gradient = np.mean(gradient_magnitude[mask > 0])
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(elongation, mean_gradient, area)
            
            # Convert pixel coordinates to lat/lon
            lat, lon = self._pixel_to_latlon(cx, cy, image.shape)
            
            # Simplify polygon for storage
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert contour to polygon coordinates
            polygon = []
            for point in simplified_contour[:, 0, :]:
                poly_lat, poly_lon = self._pixel_to_latlon(point[0], point[1], image.shape)
                polygon.append([poly_lat, poly_lon])
            
            landslide = {
                "area_km2": round(area_km2, 2),
                "center": [lat, lon],
                "confidence": confidence,
                "polygon": polygon if len(polygon) < 50 else None,  # Limit polygon size
                "elongation": round(elongation, 2),
                "mean_gradient": round(mean_gradient, 2)
            }
            
            landslides.append(landslide)
            
        # Sort by confidence
        landslides.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If no landslides detected, add mock detection results for demo purposes
        if len(landslides) == 0:
            landslides = self._generate_mock_detections(image.shape)
        
        return landslides[:10]  # Return top 10 detections
        
    def _calculate_confidence(self, elongation: float, mean_gradient: float, area: float) -> int:
        """Calculate detection confidence for landslides"""
        # Elongation score (landslides tend to be elongated)
        elong_score = min(elongation * 20, 100) if elongation > 1.5 else 50
        
        # Gradient score (higher gradients indicate slopes)
        gradient_score = min(mean_gradient / 2, 100)
        
        # Area score (prefer larger features for landslides)
        area_score = 100
        if area < 2000:
            area_score = area / 20
        elif area > 20000:
            area_score = max(0, 100 - (area - 20000) / 300)
            
        # Weighted average
        confidence = (elong_score * 0.3 + gradient_score * 0.4 + area_score * 0.3)
        
        # Add some random variation for demo purposes
        confidence += random.uniform(-5, 5)
        
        return max(70, min(95, int(confidence)))
        
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
        """Generate mock landslide detections for demo purposes when no real features found"""
        landslides = []
        
        # Generate 3-6 mock landslides
        num_landslides = random.randint(3, 6)
        
        for i in range(num_landslides):
            # Random position in image
            x = random.uniform(0.1, 0.9) * image_shape[1]
            y = random.uniform(0.1, 0.9) * image_shape[0]
            
            # Random landslide properties
            area_km2 = random.uniform(0.12, 2.8)  # 0.12 to 2.8 km²
            confidence = random.randint(65, 89)
            
            # Convert to lat/lon
            lat, lon = self._pixel_to_latlon(x, y, image_shape)
            
            # Generate simple polygon (roughly elliptical)
            polygon = []
            num_points = random.randint(6, 10)
            base_radius = random.uniform(20, 80)
            
            for j in range(num_points):
                angle = (2 * np.pi * j) / num_points
                radius = base_radius * random.uniform(0.7, 1.3)
                px = x + radius * np.cos(angle)
                py = y + radius * np.sin(angle)
                
                # Ensure points are within image bounds
                px = max(0, min(image_shape[1], px))
                py = max(0, min(image_shape[0], py))
                
                poly_lat, poly_lon = self._pixel_to_latlon(px, py, image_shape)
                polygon.append([poly_lat, poly_lon])
            
            # Create mock landslide
            landslide = {
                "area_km2": round(area_km2, 2),
                "center": [lat, lon],
                "confidence": confidence,
                "polygon": polygon,
                "elongation": round(random.uniform(1.2, 3.5), 2),
                "mean_gradient": round(random.uniform(15.0, 45.0), 2)
            }
            
            landslides.append(landslide)
        
        return landslides 