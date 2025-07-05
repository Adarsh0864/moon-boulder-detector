import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, filters, restoration
from sklearn.cluster import DBSCAN
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class TemporalChangeDetector:
    """
    Novel temporal change detection for landslide activity mapping using:
    1. Multi-temporal image registration and alignment
    2. Change vector analysis for surface displacement
    3. Activity hotspot identification using clustering
    4. Landslide evolution tracking and prediction
    
    Novelty: This approach tracks landslide activity over time using
    multi-temporal Chandrayaan images to identify active regions,
    predict future landslide susceptibility, and map the evolution
    of lunar surface instability.
    """
    
    def __init__(self):
        self.pixel_scale = 5.0  # 5m/pixel from TMC
        self.change_threshold = 0.15  # Normalized change threshold
        self.activity_memory_days = 365  # Track activity for 1 year
        
    def detect_temporal_changes(self, current_image: np.ndarray, 
                              reference_images: List[np.ndarray],
                              timestamps: List[str],
                              current_timestamp: str) -> Dict:
        """
        Detect temporal changes indicating landslide activity
        
        Args:
            current_image: Current TMC image
            reference_images: List of historical TMC images
            timestamps: Timestamps for reference images
            current_timestamp: Timestamp of current image
            
        Returns:
            Dictionary containing change analysis results
        """
        logger.info("Starting temporal change detection analysis")
        
        if not reference_images:
            logger.warning("No reference images provided for temporal analysis")
            return self._create_empty_result()
            
        # Step 1: Image registration and alignment
        aligned_images = self._register_images(current_image, reference_images)
        
        # Step 2: Change vector analysis
        change_maps = self._calculate_change_vectors(current_image, aligned_images)
        
        # Step 3: Activity hotspot identification
        activity_hotspots = self._identify_activity_hotspots(change_maps, timestamps)
        
        # Step 4: Landslide evolution tracking
        evolution_analysis = self._track_landslide_evolution(
            change_maps, activity_hotspots, timestamps, current_timestamp
        )
        
        # Step 5: Susceptibility prediction
        susceptibility_map = self._predict_susceptibility(
            current_image, change_maps, activity_hotspots
        )
        
        # Step 6: Activity statistics
        activity_stats = self._calculate_activity_statistics(
            change_maps, activity_hotspots, timestamps
        )
        
        results = {
            'activity_hotspots': activity_hotspots,
            'evolution_analysis': evolution_analysis,
            'susceptibility_map': susceptibility_map.tolist(),
            'activity_statistics': activity_stats,
            'change_detection_method': 'novel_temporal_analysis',
            'analysis_timestamp': current_timestamp,
            'reference_count': len(reference_images)
        }
        
        logger.info(f"Detected {len(activity_hotspots)} activity hotspots")
        return results
        
    def _register_images(self, current_image: np.ndarray, 
                        reference_images: List[np.ndarray]) -> List[np.ndarray]:
        """Register and align reference images to current image"""
        aligned_images = []
        
        # Convert current image to grayscale for registration
        if len(current_image.shape) == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        else:
            current_gray = current_image.copy()
            
        # Enhance contrast for better feature detection
        current_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
            current_gray.astype(np.uint8)
        )
        
        for ref_image in reference_images:
            try:
                # Convert reference image to grayscale
                if len(ref_image.shape) == 3:
                    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
                else:
                    ref_gray = ref_image.copy()
                    
                # Enhance contrast
                ref_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
                    ref_gray.astype(np.uint8)
                )
                
                # Feature-based registration using ORB
                aligned_image = self._register_orb(current_enhanced, ref_enhanced, ref_gray)
                
                if aligned_image is not None:
                    aligned_images.append(aligned_image)
                else:
                    # Fallback to simple alignment if feature matching fails
                    aligned_images.append(ref_gray)
                    
            except Exception as e:
                logger.warning(f"Registration failed for reference image: {e}")
                # Use unaligned image as fallback
                aligned_images.append(ref_gray if len(ref_image.shape) == 2 else 
                                    cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY))
                
        return aligned_images
        
    def _register_orb(self, current: np.ndarray, reference: np.ndarray, 
                     ref_original: np.ndarray) -> np.ndarray:
        """Register images using ORB feature matching"""
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(current, None)
        kp2, des2 = orb.detectAndCompute(reference, None)
        
        if des1 is None or des2 is None:
            return None
            
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        if len(matches) < 10:  # Need minimum matches for homography
            return None
            
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, 5.0)
        
        if homography is None:
            return None
            
        # Apply transformation
        height, width = current.shape
        aligned = cv2.warpPerspective(ref_original, homography, (width, height))
        
        return aligned
        
    def _calculate_change_vectors(self, current_image: np.ndarray, 
                                aligned_images: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate change vectors between current and reference images"""
        change_maps = []
        
        # Normalize current image
        current_norm = current_image.astype(np.float32) / 255.0
        
        for aligned_image in aligned_images:
            # Normalize aligned image
            aligned_norm = aligned_image.astype(np.float32) / 255.0
            
            # Calculate absolute difference
            diff_abs = np.abs(current_norm - aligned_norm)
            
            # Calculate gradient-based change
            current_grad = np.gradient(current_norm)
            aligned_grad = np.gradient(aligned_norm)
            
            grad_change = np.sqrt(
                (current_grad[0] - aligned_grad[0])**2 + 
                (current_grad[1] - aligned_grad[1])**2
            )
            
            # Combine intensity and gradient changes
            combined_change = 0.6 * diff_abs + 0.4 * grad_change
            
            # Apply morphological filtering to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            filtered_change = cv2.morphologyEx(combined_change, cv2.MORPH_OPEN, kernel)
            
            change_maps.append(filtered_change)
            
        return change_maps
        
    def _identify_activity_hotspots(self, change_maps: List[np.ndarray], 
                                  timestamps: List[str]) -> List[Dict]:
        """Identify activity hotspots using clustering analysis"""
        if not change_maps:
            return []
            
        # Combine all change maps
        combined_change = np.mean(change_maps, axis=0)
        
        # Threshold for significant changes
        change_binary = combined_change > self.change_threshold
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        change_binary = cv2.morphologyEx(change_binary.astype(np.uint8), 
                                       cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        labeled, num_features = ndimage.label(change_binary)
        
        hotspots = []
        
        for i in range(1, num_features + 1):
            hotspot_mask = labeled == i
            props = measure.regionprops(hotspot_mask.astype(int))[0]
            
            # Filter by size
            area_m2 = props.area * (self.pixel_scale ** 2)
            if area_m2 < 1000:  # Minimum 1000 m² for hotspot
                continue
                
            # Calculate activity metrics
            activity_intensity = np.mean(combined_change[hotspot_mask])
            activity_frequency = self._calculate_activity_frequency(
                hotspot_mask, change_maps, timestamps
            )
            
            # Calculate center coordinates
            cy, cx = props.centroid
            lat, lon = self._pixel_to_lunar_coords(cx, cy, combined_change.shape)
            
            # Activity confidence based on intensity and frequency
            confidence = min(100, int((activity_intensity * 50 + activity_frequency * 50)))
            
            hotspot = {
                'id': f"hotspot_{i}",
                'center': [lat, lon],
                'area_m2': round(area_m2, 0),
                'activity_intensity': round(activity_intensity, 3),
                'activity_frequency': round(activity_frequency, 2),
                'confidence': confidence,
                'bbox': [props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]],
                'first_detected': min(timestamps) if timestamps else 'unknown',
                'last_activity': max(timestamps) if timestamps else 'unknown'
            }
            
            hotspots.append(hotspot)
            
        # Sort by activity intensity
        hotspots.sort(key=lambda x: x['activity_intensity'], reverse=True)
        
        return hotspots[:20]  # Top 20 hotspots
        
    def _calculate_activity_frequency(self, hotspot_mask: np.ndarray,
                                    change_maps: List[np.ndarray],
                                    timestamps: List[str]) -> float:
        """Calculate activity frequency for a hotspot"""
        if not change_maps:
            return 0.0
            
        # Count how many change maps show activity in this hotspot
        active_count = 0
        
        for change_map in change_maps:
            hotspot_activity = np.mean(change_map[hotspot_mask])
            if hotspot_activity > self.change_threshold:
                active_count += 1
                
        frequency = active_count / len(change_maps)
        return frequency
        
    def _track_landslide_evolution(self, change_maps: List[np.ndarray],
                                 activity_hotspots: List[Dict],
                                 timestamps: List[str],
                                 current_timestamp: str) -> Dict:
        """Track evolution of landslide activity over time"""
        evolution = {
            'total_active_area_m2': 0,
            'area_change_rate_m2_per_day': 0,
            'activity_trend': 'stable',
            'most_active_region': None,
            'evolution_confidence': 0.5
        }
        
        if not change_maps or not activity_hotspots:
            return evolution
            
        # Calculate total active area
        total_active_pixels = 0
        for hotspot in activity_hotspots:
            total_active_pixels += hotspot['area_m2'] / (self.pixel_scale ** 2)
            
        evolution['total_active_area_m2'] = total_active_pixels * (self.pixel_scale ** 2)
        
        # Find most active region
        if activity_hotspots:
            most_active = max(activity_hotspots, key=lambda x: x['activity_intensity'])
            evolution['most_active_region'] = {
                'location': most_active['center'],
                'intensity': most_active['activity_intensity'],
                'area_m2': most_active['area_m2']
            }
            
        # Estimate activity trend (simplified)
        if len(change_maps) >= 2:
            recent_activity = np.mean(change_maps[-1])
            older_activity = np.mean(change_maps[0])
            
            if recent_activity > older_activity * 1.2:
                evolution['activity_trend'] = 'increasing'
            elif recent_activity < older_activity * 0.8:
                evolution['activity_trend'] = 'decreasing'
            else:
                evolution['activity_trend'] = 'stable'
                
            evolution['evolution_confidence'] = 0.8
            
        return evolution
        
    def _predict_susceptibility(self, current_image: np.ndarray,
                              change_maps: List[np.ndarray],
                              activity_hotspots: List[Dict]) -> np.ndarray:
        """Predict landslide susceptibility based on historical activity"""
        height, width = current_image.shape[:2]
        susceptibility = np.zeros((height, width))
        
        if not change_maps:
            return susceptibility
            
        # Base susceptibility from historical changes
        historical_activity = np.mean(change_maps, axis=0)
        susceptibility += historical_activity * 0.4
        
        # Terrain-based susceptibility
        terrain_susceptibility = self._calculate_terrain_susceptibility(current_image)
        susceptibility += terrain_susceptibility * 0.3
        
        # Proximity to active hotspots
        hotspot_influence = self._calculate_hotspot_influence(
            activity_hotspots, (height, width)
        )
        susceptibility += hotspot_influence * 0.3
        
        # Normalize to [0, 1] range
        susceptibility = susceptibility / np.max(susceptibility) if np.max(susceptibility) > 0 else susceptibility
        
        # Apply smoothing
        susceptibility = filters.gaussian(susceptibility, sigma=2)
        
        return susceptibility
        
    def _calculate_terrain_susceptibility(self, image: np.ndarray) -> np.ndarray:
        """Calculate terrain-based landslide susceptibility"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Calculate slope (gradient magnitude)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize slope
        slope_norm = slope / np.max(slope) if np.max(slope) > 0 else slope
        
        # Higher slopes are more susceptible
        terrain_susceptibility = slope_norm
        
        return terrain_susceptibility
        
    def _calculate_hotspot_influence(self, hotspots: List[Dict], 
                                   image_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate influence of activity hotspots on surrounding areas"""
        height, width = image_shape
        influence = np.zeros((height, width))
        
        for hotspot in hotspots:
            # Convert lat/lon back to pixel coordinates
            cx, cy = self._lunar_coords_to_pixel(
                hotspot['center'][1], hotspot['center'][0], image_shape
            )
            
            # Influence radius based on hotspot size and intensity
            influence_radius = min(
                100,  # Maximum 100 pixels (500m)
                int(np.sqrt(hotspot['area_m2'] / np.pi) / self.pixel_scale * 2)
            )
            
            # Create influence kernel
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Gaussian influence decay
            hotspot_influence = hotspot['activity_intensity'] * np.exp(
                -(distance**2) / (2 * (influence_radius/3)**2)
            )
            
            influence += hotspot_influence
            
        return influence
        
    def _calculate_activity_statistics(self, change_maps: List[np.ndarray],
                                     activity_hotspots: List[Dict],
                                     timestamps: List[str]) -> Dict:
        """Calculate comprehensive activity statistics"""
        stats = {
            'total_hotspots': len(activity_hotspots),
            'total_active_area_km2': 0,
            'average_activity_intensity': 0,
            'max_activity_intensity': 0,
            'activity_distribution': {
                'low': 0,    # 0-0.3
                'medium': 0, # 0.3-0.6
                'high': 0    # 0.6-1.0
            },
            'temporal_coverage_days': 0
        }
        
        if not activity_hotspots:
            return stats
            
        # Calculate total active area
        total_area_m2 = sum(hotspot['area_m2'] for hotspot in activity_hotspots)
        stats['total_active_area_km2'] = round(total_area_m2 / 1e6, 4)
        
        # Calculate intensity statistics
        intensities = [hotspot['activity_intensity'] for hotspot in activity_hotspots]
        stats['average_activity_intensity'] = round(np.mean(intensities), 3)
        stats['max_activity_intensity'] = round(np.max(intensities), 3)
        
        # Activity distribution
        for intensity in intensities:
            if intensity < 0.3:
                stats['activity_distribution']['low'] += 1
            elif intensity < 0.6:
                stats['activity_distribution']['medium'] += 1
            else:
                stats['activity_distribution']['high'] += 1
                
        # Temporal coverage
        if len(timestamps) >= 2:
            try:
                # Simple day calculation (would need proper date parsing in real implementation)
                stats['temporal_coverage_days'] = len(timestamps) * 30  # Assume monthly images
            except:
                stats['temporal_coverage_days'] = 0
                
        return stats
        
    def _create_empty_result(self) -> Dict:
        """Create empty result when no reference images are available"""
        return {
            'activity_hotspots': [],
            'evolution_analysis': {
                'total_active_area_m2': 0,
                'area_change_rate_m2_per_day': 0,
                'activity_trend': 'unknown',
                'most_active_region': None,
                'evolution_confidence': 0.0
            },
            'susceptibility_map': [],
            'activity_statistics': {
                'total_hotspots': 0,
                'total_active_area_km2': 0,
                'average_activity_intensity': 0,
                'max_activity_intensity': 0,
                'activity_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'temporal_coverage_days': 0
            },
            'change_detection_method': 'novel_temporal_analysis',
            'reference_count': 0
        }
        
    def _pixel_to_lunar_coords(self, x: float, y: float, image_shape: tuple) -> tuple:
        """Convert pixel coordinates to lunar coordinates"""
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
        
    def _lunar_coords_to_pixel(self, lon: float, lat: float, image_shape: tuple) -> tuple:
        """Convert lunar coordinates to pixel coordinates"""
        height, width = image_shape[:2]
        
        # TMC metadata coordinates (same as above)
        ul_lat, ul_lon = -30.2119955876, 5.7322930152
        ur_lat, ur_lon = -30.2249825798, 6.5687430269
        ll_lat, ll_lon = -84.6497387669, 353.6073418869
        lr_lat, lr_lon = -84.7568600411, 0.6446091430
        
        # Handle longitude wrap-around
        if ll_lon > 180:
            ll_lon -= 360
        if lr_lon > 180:
            lr_lon -= 360
            
        # Approximate inverse transformation (simplified)
        # In real implementation, would use proper inverse bilinear interpolation
        
        # Estimate u, v from lat, lon
        lat_range = ul_lat - ll_lat
        lon_range = ur_lon - ul_lon
        
        v = (ul_lat - lat) / lat_range if lat_range != 0 else 0.5
        u = (lon - ul_lon) / lon_range if lon_range != 0 else 0.5
        
        # Clamp values
        u = max(0, min(1, u))
        v = max(0, min(1, v))
        
        x = u * width
        y = v * height
        
        return x, y