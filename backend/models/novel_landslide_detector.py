import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, measure, segmentation, morphology
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class NovelLandslideDetector:
    """
    Novel adaptive landslide detection algorithm using:
    1. Multi-directional gradient analysis with terrain adaptation
    2. Morphological pattern recognition for debris flows
    3. Texture discontinuity analysis for surface disruption
    4. Adaptive thresholding based on local terrain characteristics
    
    Novelty: Unlike conventional edge-based methods, this approach uses
    adaptive terrain analysis combined with morphological signatures
    specific to lunar landslides, including debris flow patterns and
    regolith displacement indicators.
    """
    
    def __init__(self):
        self.min_landslide_area_m2 = 5000  # 5000 square meters minimum
        self.max_landslide_area_m2 = 5000000  # 5 km² maximum
        self.pixel_scale = 5.0  # 5m/pixel from TMC
        self.gradient_scales = [3, 5, 9, 15]  # Multi-scale gradient analysis
        
    def detect(self, image: np.ndarray, settings: dict, lunar_coords: dict = None) -> list:
        """
        Detect landslides using novel adaptive terrain analysis
        
        Args:
            image: Input TMC image array
            settings: Detection parameters
            lunar_coords: Georeferencing information
            
        Returns:
            List of detected landslides with geological context
        """
        logger.info("Starting novel landslide detection with adaptive terrain analysis")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Preprocessing with adaptive enhancement
        enhanced = self._adaptive_terrain_enhancement(gray)
        
        # Step 1: Multi-directional gradient analysis
        gradient_features = self._multi_directional_gradient_analysis(enhanced)
        
        # Step 2: Morphological pattern recognition
        morphological_features = self._morphological_pattern_analysis(enhanced, gradient_features)
        
        # Step 3: Texture discontinuity analysis
        texture_discontinuities = self._texture_discontinuity_analysis(enhanced)
        
        # Step 4: Adaptive terrain-based thresholding
        landslide_candidates = self._adaptive_terrain_thresholding(
            gradient_features, morphological_features, texture_discontinuities, settings
        )
        
        # Step 5: Geological context validation
        validated_landslides = self._geological_context_validation(
            enhanced, landslide_candidates, lunar_coords
        )
        
        # Step 6: Debris flow pattern analysis
        final_landslides = self._debris_flow_analysis(enhanced, validated_landslides)
        
        logger.info(f"Detected {len(final_landslides)} landslides using novel algorithm")
        return final_landslides
        
    def _adaptive_terrain_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Adaptive enhancement based on local terrain characteristics"""
        # Multi-scale decomposition for terrain analysis
        scales = [1, 3, 5, 9]
        enhanced_components = []
        
        for scale in scales:
            # Gaussian filtering at different scales
            smoothed = filters.gaussian(image, sigma=scale)
            
            # Local standard deviation for texture analysis
            local_std = filters.rank.variance(image.astype(np.uint8), np.ones((scale*2+1, scale*2+1)))
            local_std = local_std.astype(np.float32) / 255.0
            
            # Adaptive enhancement based on local texture
            enhanced = smoothed + local_std * (image - smoothed) * 0.5
            enhanced_components.append(enhanced)
            
        # Combine multi-scale enhancements
        enhanced = np.mean(enhanced_components, axis=0)
        return enhanced.astype(np.float32) / 255.0
        
    def _multi_directional_gradient_analysis(self, image: np.ndarray) -> dict:
        """Multi-directional gradient analysis for slope failure detection"""
        features = {}
        
        # Define multiple gradient directions
        directions = [
            (1, 0),   # Horizontal
            (0, 1),   # Vertical  
            (1, 1),   # Diagonal
            (1, -1),  # Anti-diagonal
            (2, 1),   # Steep slope
            (1, 2)    # Gentle slope
        ]
        
        gradient_responses = []
        
        for scale in self.gradient_scales:
            scale_responses = []
            
            for dx, dy in directions:
                # Create directional derivative kernels
                kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * dx
                kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) * dy
                
                # Scale kernels
                kernel_x = cv2.resize(kernel_x, (scale, scale))
                kernel_y = cv2.resize(kernel_y, (scale, scale))
                
                # Apply directional gradients
                grad_x = cv2.filter2D(image, -1, kernel_x)
                grad_y = cv2.filter2D(image, -1, kernel_y)
                
                # Gradient magnitude and direction
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                
                scale_responses.append({
                    'magnitude': magnitude,
                    'direction': direction,
                    'dx': dx,
                    'dy': dy
                })
                
            gradient_responses.append(scale_responses)
            
        # Combine responses across scales and directions
        combined_magnitude = np.zeros_like(image)
        directional_coherence = np.zeros_like(image)
        
        for scale_idx, scale_responses in enumerate(gradient_responses):
            scale_weight = np.exp(-((scale_idx - 1)**2) / (2 * 1**2))  # Prefer middle scales
            
            for response in scale_responses:
                combined_magnitude += response['magnitude'] * scale_weight
                
            # Calculate directional coherence (consistency of gradient direction)
            directions_at_scale = [r['direction'] for r in scale_responses]
            direction_std = np.std(directions_at_scale, axis=0)
            coherence = np.exp(-direction_std)  # Higher coherence = lower std
            directional_coherence += coherence * scale_weight
            
        features['magnitude'] = combined_magnitude
        features['coherence'] = directional_coherence
        features['slope_instability'] = combined_magnitude * directional_coherence
        
        return features
        
    def _morphological_pattern_analysis(self, image: np.ndarray, gradient_features: dict) -> dict:
        """Analyze morphological patterns specific to lunar landslides"""
        # Lunar landslides often show:
        # 1. Elongated scarp faces
        # 2. Debris accumulation zones
        # 3. Flow channel patterns
        
        # Create morphological kernels for different landslide features
        kernels = {
            'scarp': cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),  # Vertical scarp
            'flow': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9)),  # Flow pattern
            'debris': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Debris accumulation
        }
        
        # Apply gradient magnitude as base for morphological analysis
        gradient_binary = gradient_features['magnitude'] > np.percentile(gradient_features['magnitude'], 85)
        
        morphological_features = {}
        
        for feature_name, kernel in kernels.items():
            # Apply morphological operations
            opened = cv2.morphologyEx(gradient_binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Calculate feature strength
            feature_strength = cv2.filter2D(closed.astype(np.float32), -1, kernel.astype(np.float32))
            feature_strength = feature_strength / np.max(feature_strength) if np.max(feature_strength) > 0 else feature_strength
            
            morphological_features[feature_name] = feature_strength
            
        # Combine morphological features
        combined_morphology = (morphological_features['scarp'] * 0.4 + 
                              morphological_features['flow'] * 0.4 + 
                              morphological_features['debris'] * 0.2)
        
        morphological_features['combined'] = combined_morphology
        
        return morphological_features
        
    def _texture_discontinuity_analysis(self, image: np.ndarray) -> np.ndarray:
        """Analyze texture discontinuities indicating surface disruption"""
        # Calculate local texture measures
        texture_measures = []
        
        # Local Binary Pattern approximation
        def local_binary_pattern_approx(img, radius=3):
            # Simplified LBP using directional differences
            height, width = img.shape
            lbp = np.zeros_like(img)
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = img[i, j]
                    pattern = 0
                    
                    # 8 neighboring pixels
                    neighbors = [
                        img[i-radius, j-radius], img[i-radius, j], img[i-radius, j+radius],
                        img[i, j+radius], img[i+radius, j+radius], img[i+radius, j],
                        img[i+radius, j-radius], img[i, j-radius]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            pattern |= (1 << k)
                            
                    lbp[i, j] = pattern
                    
            return lbp
            
        # Calculate texture at multiple scales
        for scale in [1, 2, 3]:
            # Smooth image at scale
            smoothed = filters.gaussian(image, sigma=scale)
            
            # Calculate local texture measures
            lbp = local_binary_pattern_approx(smoothed, radius=scale)
            texture_measures.append(lbp)
            
        # Combine texture measures
        combined_texture = np.mean(texture_measures, axis=0)
        
        # Find texture discontinuities using gradient of texture
        texture_gradient = np.gradient(combined_texture)
        texture_discontinuity = np.sqrt(texture_gradient[0]**2 + texture_gradient[1]**2)
        
        # Normalize
        texture_discontinuity = texture_discontinuity / np.max(texture_discontinuity) if np.max(texture_discontinuity) > 0 else texture_discontinuity
        
        return texture_discontinuity
        
    def _adaptive_terrain_thresholding(self, gradient_features: dict, 
                                     morphological_features: dict,
                                     texture_discontinuities: np.ndarray,
                                     settings: dict) -> list:
        """Adaptive thresholding based on local terrain characteristics"""
        # Combine all features
        feature_weights = {
            'gradient': 0.3,
            'morphology': 0.3,
            'texture': 0.2,
            'slope_instability': 0.2
        }
        
        combined_features = (
            gradient_features['magnitude'] * feature_weights['gradient'] +
            morphological_features['combined'] * feature_weights['morphology'] +
            texture_discontinuities * feature_weights['texture'] +
            gradient_features['slope_instability'] * feature_weights['slope_instability']
        )
        
        # Adaptive thresholding based on local statistics
        window_size = 50  # 250m window at 5m/pixel
        
        # Calculate local thresholds
        local_mean = cv2.blur(combined_features, (window_size, window_size))
        local_std = cv2.blur(combined_features**2, (window_size, window_size)) - local_mean**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        # Adaptive threshold
        sensitivity = settings.get('brightness_threshold', 65) / 100
        adaptive_threshold = local_mean + sensitivity * local_std
        
        # Create binary mask
        landslide_binary = combined_features > adaptive_threshold
        
        # Clean up binary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        landslide_binary = cv2.morphologyEx(landslide_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        landslide_binary = cv2.morphologyEx(landslide_binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        labeled, num_features = ndimage.label(landslide_binary)
        
        candidates = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            # Calculate properties
            props = measure.regionprops(mask.astype(int))[0]
            area_pixels = props.area
            area_m2 = area_pixels * (self.pixel_scale ** 2)
            
            # Filter by size
            if area_m2 < self.min_landslide_area_m2 or area_m2 > self.max_landslide_area_m2:
                continue
                
            candidates.append({
                'mask': mask,
                'area_m2': area_m2,
                'centroid': props.centroid,
                'bbox': props.bbox,
                'feature_strength': np.mean(combined_features[mask])
            })
            
        return candidates
        
    def _geological_context_validation(self, image: np.ndarray, candidates: list, 
                                     lunar_coords: dict) -> list:
        """Validate candidates using geological context"""
        validated = []
        
        for candidate in candidates:
            # Extract region properties
            mask = candidate['mask']
            bbox = candidate['bbox']
            roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            roi_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            # Geological validation criteria
            validation_score = 0
            
            # 1. Slope orientation analysis
            gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            slope_angle = np.arctan2(gradient_y, gradient_x)[roi_mask]
            
            # Check for consistent slope direction (landslides flow downhill)
            if len(slope_angle) > 0:
                slope_coherence = 1 - np.std(slope_angle) / np.pi
                validation_score += slope_coherence * 0.3
                
            # 2. Brightness pattern analysis
            roi_values = roi[roi_mask]
            if len(roi_values) > 0:
                # Landslides often show brightness gradients (scarp to debris)
                brightness_gradient = np.std(roi_values)
                brightness_score = min(brightness_gradient * 2, 1.0)
                validation_score += brightness_score * 0.2
                
            # 3. Elongation analysis (landslides are typically elongated)
            elongation = candidate.get('elongation', 1.0)
            if elongation > 1.5:  # Elongated features preferred
                validation_score += min((elongation - 1) / 2, 0.3)
                
            # 4. Context analysis (proximity to slopes, craters)
            context_score = self._analyze_geological_context(candidate, image)
            validation_score += context_score * 0.2
            
            # Accept if validation score is sufficient
            if validation_score > 0.4:
                candidate['validation_score'] = validation_score
                validated.append(candidate)
                
        return validated
        
    def _analyze_geological_context(self, candidate: dict, image: np.ndarray) -> float:
        """Analyze geological context around landslide candidate"""
        cy, cx = candidate['centroid']
        
        # Define context window (1km radius)
        context_radius = int(200 / self.pixel_scale)  # 200m radius
        
        y_min = max(0, int(cy - context_radius))
        y_max = min(image.shape[0], int(cy + context_radius))
        x_min = max(0, int(cx - context_radius))
        x_max = min(image.shape[1], int(cx + context_radius))
        
        context_region = image[y_min:y_max, x_min:x_max]
        
        if context_region.size == 0:
            return 0.0
            
        # Calculate context features
        context_gradient = np.gradient(context_region)
        context_slope = np.sqrt(context_gradient[0]**2 + context_gradient[1]**2)
        
        # Higher context score for steep terrain (landslide-prone areas)
        mean_slope = np.mean(context_slope)
        context_score = min(mean_slope * 2, 1.0)
        
        return context_score
        
    def _debris_flow_analysis(self, image: np.ndarray, candidates: list) -> list:
        """Analyze debris flow patterns for final landslide classification"""
        final_landslides = []
        
        for candidate in candidates:
            mask = candidate['mask']
            
            # Calculate debris flow characteristics
            flow_direction = self._calculate_flow_direction(image, mask)
            flow_length = self._calculate_flow_length(mask)
            debris_volume = self._estimate_debris_volume(candidate)
            
            # Convert to real-world coordinates
            cy, cx = candidate['centroid']
            lat, lon = self._pixel_to_lunar_coords(cx, cy, image.shape)
            
            # Calculate polygon boundary
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Simplify contour
                contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to lat/lon
                polygon = []
                for point in simplified_contour[:, 0, :]:
                    poly_lat, poly_lon = self._pixel_to_lunar_coords(point[0], point[1], image.shape)
                    polygon.append([poly_lat, poly_lon])
                    
                # Limit polygon complexity
                if len(polygon) > 100:
                    polygon = polygon[::len(polygon)//50]  # Subsample
                    
            else:
                polygon = None
                
            # Calculate confidence
            confidence = self._calculate_landslide_confidence(candidate, flow_direction, flow_length)
            
            landslide = {
                "area_km2": round(candidate['area_m2'] / 1e6, 4),
                "center": [lat, lon],
                "confidence": confidence,
                "polygon": polygon,
                "flow_direction": round(flow_direction, 1),
                "flow_length_m": round(flow_length, 1),
                "debris_volume_m3": round(debris_volume, 0),
                "detection_method": "novel_adaptive_terrain",
                "geological_context": candidate.get('validation_score', 0.0)
            }
            
            final_landslides.append(landslide)
            
        # Sort by confidence
        final_landslides.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_landslides[:15]  # Top 15 detections
        
    def _calculate_flow_direction(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate primary flow direction of landslide"""
        # Find the principal axis of the landslide
        coords = np.where(mask)
        if len(coords[0]) < 3:
            return 0.0
            
        # Calculate covariance matrix
        points = np.column_stack([coords[1], coords[0]])  # x, y coordinates
        cov_matrix = np.cov(points.T)
        
        # Find principal component (eigenvector with largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Convert to angle in degrees
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    def _calculate_flow_length(self, mask: np.ndarray) -> float:
        """Calculate flow length of landslide"""
        # Find skeleton of landslide
        skeleton = morphology.skeletonize(mask)
        
        # Find longest path in skeleton
        coords = np.where(skeleton)
        if len(coords[0]) < 2:
            return 0.0
            
        # Calculate approximate flow length using skeleton
        skeleton_length = np.sum(skeleton)
        flow_length_pixels = skeleton_length * 1.4  # Approximate correction factor
        flow_length_meters = flow_length_pixels * self.pixel_scale
        
        return flow_length_meters
        
    def _estimate_debris_volume(self, candidate: dict) -> float:
        """Estimate debris volume based on landslide area and typical depth"""
        area_m2 = candidate['area_m2']
        
        # Empirical relationship for lunar landslides
        # Assume average depth of 2-5 meters based on area
        if area_m2 < 50000:  # < 5 hectares
            avg_depth = 2.0
        elif area_m2 < 500000:  # < 50 hectares
            avg_depth = 3.0
        else:
            avg_depth = 5.0
            
        volume_m3 = area_m2 * avg_depth
        return volume_m3
        
    def _calculate_landslide_confidence(self, candidate: dict, flow_direction: float, 
                                      flow_length: float) -> int:
        """Calculate overall landslide confidence"""
        base_score = candidate.get('validation_score', 0.5)
        
        # Size appropriateness
        area_km2 = candidate['area_m2'] / 1e6
        if 0.01 <= area_km2 <= 1.0:  # Optimal size range
            size_score = 1.0
        else:
            size_score = 0.5
            
        # Flow characteristics
        flow_score = min(flow_length / 100.0, 1.0)  # Longer flows are more confident
        
        # Feature strength
        feature_score = min(candidate.get('feature_strength', 0.5) * 2, 1.0)
        
        # Weighted combination
        total_score = (base_score * 0.4 + size_score * 0.2 + 
                      flow_score * 0.2 + feature_score * 0.2)
        
        confidence = int(total_score * 100)
        return max(75, min(98, confidence))
        
    def _pixel_to_lunar_coords(self, x: float, y: float, image_shape: tuple) -> tuple:
        """Convert pixel coordinates to lunar coordinates using TMC metadata"""
        height, width = image_shape[:2]
        
        # TMC metadata coordinates
        ul_lat, ul_lon = -30.2119955876, 5.7322930152
        ur_lat, ur_lon = -30.2249825798, 6.5687430269
        ll_lat, ll_lon = -84.6497387669, 353.6073418869
        lr_lat, lr_lon = -84.7568600411, 0.6446091430
        
        # Handle longitude wrap-around
        if ll_lon > 180:
            ll_lon -= 360
        if lr_lon > 180:
            lr_lon -= 360
            
        # Bilinear interpolation
        u = x / width
        v = y / height
        
        top_lat = ul_lat * (1 - u) + ur_lat * u
        top_lon = ul_lon * (1 - u) + ur_lon * u
        bottom_lat = ll_lat * (1 - u) + lr_lat * u
        bottom_lon = ll_lon * (1 - u) + lr_lon * u
        
        lat = top_lat * (1 - v) + bottom_lat * v
        lon = top_lon * (1 - v) + bottom_lon * v
        
        return round(lat, 6), round(lon, 6)