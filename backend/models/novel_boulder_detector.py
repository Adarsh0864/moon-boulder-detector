import numpy as np
import cv2
import scipy.io
from scipy import ndimage
from skimage import filters, segmentation, measure, restoration
import pywt
from sklearn.cluster import DBSCAN
import random
import logging

logger = logging.getLogger(__name__)

class NovelBoulderDetector:
    """
    Novel multi-scale boulder detection algorithm using:
    1. Wavelet-based multi-scale decomposition
    2. Illumination-shadow coupling analysis 
    3. Geometric constraint validation
    4. Spectral signature classification
    
    Novelty: Unlike conventional methods that rely solely on brightness/shadow,
    this approach uses wavelet coefficient patterns to identify circular features
    at multiple scales, coupled with shadow-illumination pairing for validation.
    """
    
    def __init__(self):
        self.min_diameter_m = 0.5  # Minimum boulder diameter in meters
        self.max_diameter_m = 50.0  # Maximum boulder diameter
        self.pixel_scale = 5.0  # 5m/pixel from TMC data
        self.wavelet_scales = [2, 4, 8, 16]  # Multiple detection scales
        
    def detect(self, image: np.ndarray, settings: dict, lunar_coords: dict = None) -> list:
        """
        Detect boulders using novel multi-scale wavelet approach
        
        Args:
            image: Input TMC image array
            settings: Detection parameters
            lunar_coords: Georeferencing information from TMC metadata
            
        Returns:
            List of detected boulders with enhanced properties
        """
        logger.info("Starting novel boulder detection with multi-scale wavelet analysis")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Normalize image for consistent processing
        gray = self._normalize_image(gray)
        
        # Step 1: Multi-scale wavelet decomposition
        wavelet_features = self._wavelet_multiscale_analysis(gray)
        
        # Step 2: Shadow-illumination coupling analysis
        shadow_illumination_map = self._shadow_illumination_coupling(gray, settings)
        
        # Step 3: Geometric constraint validation
        candidate_regions = self._identify_candidate_regions(wavelet_features, shadow_illumination_map)
        
        # Step 4: Boulder validation and classification
        validated_boulders = self._validate_boulder_candidates(
            gray, candidate_regions, settings, lunar_coords
        )
        
        # Step 5: Statistical clustering for noise reduction
        final_boulders = self._cluster_and_refine(validated_boulders)
        
        logger.info(f"Detected {len(final_boulders)} boulders using novel algorithm")
        return final_boulders
        
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using adaptive histogram equalization"""
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(image.astype(np.uint8))
        return normalized.astype(np.float32) / 255.0
        
    def _wavelet_multiscale_analysis(self, image: np.ndarray) -> dict:
        """
        Novel multi-scale wavelet decomposition for circular feature detection
        
        Uses Daubechies wavelets to capture circular patterns at different scales
        """
        features = {}
        
        for scale in self.wavelet_scales:
            # Wavelet decomposition
            coeffs = pywt.swt2(image, 'db4', level=3, trim_approx=True)
            
            # Extract detail coefficients (horizontal, vertical, diagonal)
            cH, cV, cD = coeffs[0][1], coeffs[1][1], coeffs[2][1]
            
            # Calculate circular feature strength
            circular_strength = np.sqrt(cH**2 + cV**2 + cD**2)
            
            # Apply Gaussian smoothing based on scale
            sigma = scale / 4.0
            smoothed = filters.gaussian(circular_strength, sigma=sigma)
            
            # Find local maxima that could be boulder centers
            local_maxima = filters.rank.maximum(smoothed, np.ones((scale, scale)))
            boulder_candidates = smoothed == local_maxima
            
            features[f'scale_{scale}'] = {
                'coefficients': circular_strength,
                'candidates': boulder_candidates,
                'strength_map': smoothed
            }
            
        return features
        
    def _shadow_illumination_coupling(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """
        Novel shadow-illumination coupling analysis
        
        Analyzes the spatial relationship between bright (illuminated) and 
        dark (shadow) regions to identify boulder signatures
        """
        # Calculate gradient for slope estimation
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Estimate illumination direction from dominant gradient
        illumination_angle = np.arctan2(np.mean(gradient_y), np.mean(gradient_x))
        
        # Create directional kernels for shadow detection
        shadow_kernel = self._create_directional_kernel(illumination_angle + np.pi, 7)
        illumination_kernel = self._create_directional_kernel(illumination_angle, 7)
        
        # Apply directional filtering
        shadow_response = cv2.filter2D(image, -1, shadow_kernel)
        illumination_response = cv2.filter2D(image, -1, illumination_kernel)
        
        # Calculate shadow-illumination coupling strength
        coupling_strength = illumination_response - shadow_response
        
        # Normalize and threshold
        coupling_normalized = (coupling_strength - coupling_strength.min()) / \
                            (coupling_strength.max() - coupling_strength.min())
        
        threshold = settings.get('brightness_threshold', 65) / 100
        coupling_binary = coupling_normalized > threshold
        
        return coupling_binary.astype(np.float32)
        
    def _create_directional_kernel(self, angle: float, size: int) -> np.ndarray:
        """Create directional kernel for shadow/illumination detection"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                # Calculate direction from center
                dy, dx = i - center, j - center
                if dx == 0 and dy == 0:
                    continue
                    
                pixel_angle = np.arctan2(dy, dx)
                angle_diff = abs(pixel_angle - angle)
                
                # Weight based on angle similarity and distance
                if angle_diff < np.pi/4:  # Within 45 degrees
                    distance = np.sqrt(dx**2 + dy**2)
                    weight = np.exp(-distance/2) * np.cos(angle_diff)
                    kernel[i, j] = weight
                    
        return kernel / np.sum(np.abs(kernel))  # Normalize
        
    def _identify_candidate_regions(self, wavelet_features: dict, 
                                  shadow_illumination_map: np.ndarray) -> list:
        """Combine wavelet and shadow analysis to identify boulder candidates"""
        candidates = []
        
        # Combine evidence from all scales
        combined_evidence = np.zeros_like(shadow_illumination_map)
        
        for scale_name, features in wavelet_features.items():
            scale = int(scale_name.split('_')[1])
            
            # Weight by scale (prefer medium scales for boulders)
            scale_weight = np.exp(-((scale - 8)**2) / (2 * 4**2))  # Gaussian around scale 8
            
            combined_evidence += features['strength_map'] * scale_weight
            
        # Multiply by shadow-illumination evidence
        final_evidence = combined_evidence * shadow_illumination_map
        
        # Find connected components
        threshold = np.percentile(final_evidence, 95)  # Top 5% of evidence
        binary_evidence = final_evidence > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(binary_evidence)
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            
            # Calculate region properties
            props = measure.regionprops(mask.astype(int))[0]
            
            # Filter by size and shape
            if (props.area > 10 and props.area < 2000 and 
                props.eccentricity < 0.8):  # Roughly circular
                
                candidates.append({
                    'centroid': props.centroid,
                    'area': props.area,
                    'bbox': props.bbox,
                    'mask': mask,
                    'evidence_strength': np.mean(final_evidence[mask])
                })
                
        return candidates
        
    def _validate_boulder_candidates(self, image: np.ndarray, candidates: list,
                                   settings: dict, lunar_coords: dict) -> list:
        """Validate boulder candidates using geometric and spectral analysis"""
        validated = []
        
        for candidate in candidates:
            # Extract region of interest
            bbox = candidate['bbox']
            roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask = candidate['mask'][bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            # Geometric validation
            circularity = self._calculate_circularity(mask)
            if circularity < 0.3:  # Too irregular
                continue
                
            # Size validation
            diameter_pixels = np.sqrt(candidate['area'] / np.pi) * 2
            diameter_meters = diameter_pixels * self.pixel_scale
            
            if diameter_meters < self.min_diameter_m or diameter_meters > self.max_diameter_m:
                continue
                
            # Spectral signature analysis
            spectral_score = self._analyze_spectral_signature(roi, mask)
            if spectral_score < 0.4:  # Poor spectral match
                continue
                
            # Calculate confidence
            confidence = self._calculate_boulder_confidence(
                circularity, spectral_score, candidate['evidence_strength'], diameter_meters
            )
            
            # Convert to real-world coordinates
            cy, cx = candidate['centroid']
            lat, lon = self._pixel_to_lunar_coords(cx, cy, image.shape, lunar_coords)
            
            boulder = {
                "diameter": round(diameter_meters, 2),
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],  # [x1,y1,x2,y2]
                "circularity": round(circularity, 3),
                "spectral_score": round(spectral_score, 3),
                "detection_method": "novel_wavelet_multiscale"
            }
            
            validated.append(boulder)
            
        return validated
        
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity of a binary mask"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
            
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
            
        return 4 * np.pi * area / (perimeter ** 2)
        
    def _analyze_spectral_signature(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Analyze spectral signature for boulder identification"""
        # Boulder regions typically have:
        # 1. Higher brightness on illuminated side
        # 2. Sharp contrast with shadows
        # 3. Intermediate reflectance values
        
        roi_values = roi[mask]
        if len(roi_values) == 0:
            return 0.0
            
        # Calculate spectral properties
        mean_reflectance = np.mean(roi_values)
        std_reflectance = np.std(roi_values)
        
        # Boulder spectral score based on expected properties
        # Prefer moderate reflectance with some variation
        reflectance_score = 1.0 - abs(mean_reflectance - 0.5) * 2  # Peak at 0.5
        variation_score = min(std_reflectance * 4, 1.0)  # Some texture is good
        
        return (reflectance_score + variation_score) / 2
        
    def _calculate_boulder_confidence(self, circularity: float, spectral_score: float,
                                    evidence_strength: float, diameter: float) -> int:
        """Calculate overall boulder detection confidence"""
        # Weight different factors
        circ_weight = 0.3
        spectral_weight = 0.3
        evidence_weight = 0.2
        size_weight = 0.2
        
        # Size score (prefer boulders in 2-20m range)
        optimal_size = 10.0  # meters
        size_score = 1.0 - abs(diameter - optimal_size) / optimal_size
        size_score = max(0.0, min(1.0, size_score))
        
        # Combine scores
        total_score = (circularity * circ_weight + 
                      spectral_score * spectral_weight +
                      evidence_strength * evidence_weight +
                      size_score * size_weight)
        
        confidence = int(total_score * 100)
        return max(70, min(99, confidence))
        
    def _cluster_and_refine(self, boulders: list) -> list:
        """Use clustering to remove duplicate detections and refine results"""
        if len(boulders) < 2:
            return boulders
            
        # Extract positions for clustering
        positions = np.array([[b['lat'], b['lon']] for b in boulders])
        
        # DBSCAN clustering to group nearby detections
        # Use eps based on typical boulder spacing
        eps = 0.0001  # Roughly 10m at lunar scale
        clustering = DBSCAN(eps=eps, min_samples=1).fit(positions)
        
        refined_boulders = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            cluster_indices = np.where(clustering.labels_ == label)[0]
            cluster_boulders = [boulders[i] for i in cluster_indices]
            
            if len(cluster_boulders) == 1:
                refined_boulders.append(cluster_boulders[0])
            else:
                # Merge cluster - keep highest confidence detection
                best_boulder = max(cluster_boulders, key=lambda x: x['confidence'])
                
                # Average position and diameter from cluster
                avg_lat = np.mean([b['lat'] for b in cluster_boulders])
                avg_lon = np.mean([b['lon'] for b in cluster_boulders])
                avg_diameter = np.mean([b['diameter'] for b in cluster_boulders])
                
                best_boulder.update({
                    'lat': round(avg_lat, 6),
                    'lon': round(avg_lon, 6),
                    'diameter': round(avg_diameter, 2),
                    'cluster_size': len(cluster_boulders)
                })
                
                refined_boulders.append(best_boulder)
                
        # Sort by confidence and return top detections
        refined_boulders.sort(key=lambda x: x['confidence'], reverse=True)
        return refined_boulders[:25]  # Top 25 detections
        
    def _pixel_to_lunar_coords(self, x: float, y: float, image_shape: tuple, 
                              lunar_coords: dict = None) -> tuple:
        """Convert pixel coordinates to lunar latitude/longitude"""
        if lunar_coords is None:
            # Use TMC metadata coordinates if available
            # Upper left: -30.2119955876, 5.7322930152
            # Upper right: -30.2249825798, 6.5687430269
            # Lower left: -84.6497387669, 353.6073418869
            # Lower right: -84.7568600411, 0.6446091430
            
            height, width = image_shape[:2]
            
            # Bilinear interpolation between corner coordinates
            u = x / width
            v = y / height
            
            # Corner coordinates from TMC metadata
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
            top_lat = ul_lat * (1 - u) + ur_lat * u
            top_lon = ul_lon * (1 - u) + ur_lon * u
            bottom_lat = ll_lat * (1 - u) + lr_lat * u  
            bottom_lon = ll_lon * (1 - u) + lr_lon * u
            
            lat = top_lat * (1 - v) + bottom_lat * v
            lon = top_lon * (1 - v) + bottom_lon * v
            
            return round(lat, 6), round(lon, 6)
        else:
            # Use provided coordinates
            return lunar_coords.get('lat', 0.0), lunar_coords.get('lon', 0.0)