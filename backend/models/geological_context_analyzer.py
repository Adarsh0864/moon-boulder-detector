import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, filters, morphology
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class GeologicalContextAnalyzer:
    """
    Novel geological context analysis for landslide and boulder source identification using:
    1. Crater proximity analysis for impact-related sources
    2. Topographic signature analysis for structural sources  
    3. Regolith texture analysis for weathering-related sources
    4. Slope stability modeling for gravity-driven sources
    
    Novelty: This approach combines multiple geological indicators to identify
    the most likely source mechanisms for detected landslides and boulders,
    providing insights into lunar geological processes and hazard prediction.
    """
    
    def __init__(self):
        self.pixel_scale = 5.0  # 5m/pixel from TMC
        self.crater_detection_threshold = 0.3
        self.analysis_radius_m = 1000  # 1km analysis radius
        
    def analyze_geological_context(self, image: np.ndarray, 
                                 landslides: List[Dict], 
                                 boulders: List[Dict]) -> Dict:
        """
        Analyze geological context to identify landslide and boulder sources
        
        Args:
            image: Input TMC image
            landslides: List of detected landslides
            boulders: List of detected boulders
            
        Returns:
            Dictionary containing geological context analysis
        """
        logger.info("Starting geological context analysis")
        
        # Step 1: Crater detection and analysis
        craters = self._detect_craters(image)
        
        # Step 2: Topographic signature analysis
        topographic_features = self._analyze_topographic_signatures(image)
        
        # Step 3: Regolith texture analysis
        regolith_analysis = self._analyze_regolith_texture(image)
        
        # Step 4: Slope stability analysis
        slope_stability = self._analyze_slope_stability(image)
        
        # Step 5: Source identification for landslides
        landslide_sources = self._identify_landslide_sources(
            landslides, craters, topographic_features, slope_stability, image
        )
        
        # Step 6: Source identification for boulders
        boulder_sources = self._identify_boulder_sources(
            boulders, craters, topographic_features, regolith_analysis, image
        )
        
        # Step 7: Regional geological assessment
        regional_assessment = self._assess_regional_geology(
            image, craters, topographic_features, regolith_analysis
        )
        
        results = {
            'craters': craters,
            'topographic_features': topographic_features,
            'regolith_analysis': regolith_analysis,
            'slope_stability': slope_stability,
            'landslide_sources': landslide_sources,
            'boulder_sources': boulder_sources,
            'regional_assessment': regional_assessment,
            'analysis_method': 'novel_geological_context'
        }
        
        logger.info(f"Analyzed {len(craters)} craters, {len(landslide_sources)} landslide sources, {len(boulder_sources)} boulder sources")
        return results
        
    def _detect_craters(self, image: np.ndarray) -> List[Dict]:
        """Detect craters using circular Hough transform and morphological analysis"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Enhance contrast for crater detection
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
            gray.astype(np.uint8)
        )
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
        
        # Crater detection using Hough circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,  # Minimum distance between crater centers
            param1=50,   # Upper threshold for edge detection
            param2=30,   # Accumulator threshold for center detection
            minRadius=5, # Minimum radius (25m at 5m/pixel)
            maxRadius=200 # Maximum radius (1km at 5m/pixel)
        )
        
        craters = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validate crater using morphological analysis
                if self._validate_crater(gray, x, y, r):
                    # Convert to real-world coordinates
                    lat, lon = self._pixel_to_lunar_coords(x, y, image.shape)
                    diameter_m = r * 2 * self.pixel_scale
                    
                    # Analyze crater characteristics
                    crater_props = self._analyze_crater_properties(gray, x, y, r)
                    
                    crater = {
                        'center': [lat, lon],
                        'diameter_m': round(diameter_m, 1),
                        'confidence': crater_props['confidence'],
                        'depth_estimate_m': crater_props['depth_estimate'],
                        'freshness': crater_props['freshness'],
                        'impact_energy_estimate': crater_props['impact_energy'],
                        'ejecta_visible': crater_props['ejecta_visible'],
                        'degradation_state': crater_props['degradation_state']
                    }
                    
                    craters.append(crater)
                    
        # Sort by confidence and size
        craters.sort(key=lambda x: (x['confidence'], x['diameter_m']), reverse=True)
        
        return craters[:50]  # Top 50 craters
        
    def _validate_crater(self, image: np.ndarray, x: int, y: int, r: int) -> bool:
        """Validate crater detection using morphological analysis"""
        # Extract circular region
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Get crater interior and rim
        crater_interior = image[mask > 0]
        
        # Get rim region (annulus)
        rim_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(rim_mask, (x, y), int(r * 1.2), 255, -1)
        cv2.circle(rim_mask, (x, y), r, 0, -1)
        
        if np.sum(rim_mask) == 0:
            return False
            
        rim_values = image[rim_mask > 0]
        
        if len(crater_interior) == 0 or len(rim_values) == 0:
            return False
            
        # Craters should have darker interior than rim
        interior_mean = np.mean(crater_interior)
        rim_mean = np.mean(rim_values)
        
        brightness_contrast = (rim_mean - interior_mean) / (rim_mean + interior_mean + 1e-6)
        
        # Valid crater should have positive brightness contrast
        return brightness_contrast > 0.1
        
    def _analyze_crater_properties(self, image: np.ndarray, x: int, y: int, r: int) -> Dict:
        """Analyze detailed crater properties"""
        # Extract crater region
        y_min = max(0, y - r)
        y_max = min(image.shape[0], y + r)
        x_min = max(0, x - r)
        x_max = min(image.shape[1], x + r)
        
        crater_region = image[y_min:y_max, x_min:x_max]
        
        if crater_region.size == 0:
            return {'confidence': 0.0, 'depth_estimate': 0.0, 'freshness': 0.0, 
                   'impact_energy': 0.0, 'ejecta_visible': False, 'degradation_state': 'unknown'}
        
        # Calculate properties
        diameter_m = r * 2 * self.pixel_scale
        
        # Depth estimation using empirical relationships
        # Depth/Diameter ratio for fresh lunar craters is ~0.2
        depth_estimate = diameter_m * 0.15  # Conservative estimate
        
        # Freshness based on rim sharpness
        gradient_magnitude = np.sqrt(
            cv2.Sobel(crater_region, cv2.CV_64F, 1, 0, ksize=3)**2 +
            cv2.Sobel(crater_region, cv2.CV_64F, 0, 1, ksize=3)**2
        )
        rim_sharpness = np.std(gradient_magnitude)
        freshness = min(1.0, rim_sharpness / 50.0)
        
        # Impact energy estimation (simplified)
        # E ∝ D^3 (approximately)
        impact_energy = (diameter_m / 100) ** 3 * 1e12  # Joules (very rough estimate)
        
        # Ejecta detection (look for bright rays)
        ejecta_visible = self._detect_ejecta(image, x, y, r)
        
        # Degradation state
        if freshness > 0.7:
            degradation_state = 'fresh'
        elif freshness > 0.4:
            degradation_state = 'moderately_degraded'
        else:
            degradation_state = 'heavily_degraded'
            
        # Overall confidence
        confidence = min(1.0, (freshness + 0.5) * 0.8)
        
        return {
            'confidence': round(confidence, 2),
            'depth_estimate': round(depth_estimate, 1),
            'freshness': round(freshness, 2),
            'impact_energy': round(impact_energy, 0),
            'ejecta_visible': ejecta_visible,
            'degradation_state': degradation_state
        }
        
    def _detect_ejecta(self, image: np.ndarray, x: int, y: int, r: int) -> bool:
        """Detect ejecta rays around crater"""
        # Look for bright streaks radiating from crater
        ejecta_radius = r * 3  # Ejecta can extend 3x crater radius
        
        # Extract larger region around crater
        y_min = max(0, y - ejecta_radius)
        y_max = min(image.shape[0], y + ejecta_radius)
        x_min = max(0, x - ejecta_radius)
        x_max = min(image.shape[1], x + ejecta_radius)
        
        ejecta_region = image[y_min:y_max, x_min:x_max]
        
        if ejecta_region.size == 0:
            return False
            
        # Look for radial brightness patterns
        center_y, center_x = ejecta_radius, ejecta_radius
        
        # Create radial sampling
        angles = np.linspace(0, 2*np.pi, 16)
        radial_profiles = []
        
        for angle in angles:
            # Sample along radial direction
            distances = np.arange(r, min(ejecta_radius, ejecta_region.shape[0]//2))
            profile = []
            
            for distance in distances:
                sample_y = int(center_y + distance * np.sin(angle))
                sample_x = int(center_x + distance * np.cos(angle))
                
                if (0 <= sample_y < ejecta_region.shape[0] and 
                    0 <= sample_x < ejecta_region.shape[1]):
                    profile.append(ejecta_region[sample_y, sample_x])
                    
            if len(profile) > 3:
                radial_profiles.append(np.array(profile))
                
        if not radial_profiles:
            return False
            
        # Check for brightness enhancement in radial direction
        mean_profile = np.mean([np.mean(profile) for profile in radial_profiles])
        background_brightness = np.mean(ejecta_region)
        
        brightness_enhancement = (mean_profile - background_brightness) / (background_brightness + 1e-6)
        
        return brightness_enhancement > 0.1
        
    def _analyze_topographic_signatures(self, image: np.ndarray) -> Dict:
        """Analyze topographic signatures for structural features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Calculate gradients
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Identify ridges and valleys
        ridges = self._detect_ridges(gray, gradient_magnitude)
        valleys = self._detect_valleys(gray, gradient_magnitude)
        
        # Identify scarps (steep slopes)
        scarps = self._detect_scarps(gradient_magnitude, gradient_direction)
        
        # Identify plateaus
        plateaus = self._detect_plateaus(gray, gradient_magnitude)
        
        topographic_features = {
            'ridges': ridges,
            'valleys': valleys,
            'scarps': scarps,
            'plateaus': plateaus,
            'mean_slope': round(np.mean(gradient_magnitude), 3),
            'max_slope': round(np.max(gradient_magnitude), 3),
            'slope_variation': round(np.std(gradient_magnitude), 3)
        }
        
        return topographic_features
        
    def _detect_ridges(self, image: np.ndarray, gradient_magnitude: np.ndarray) -> List[Dict]:
        """Detect ridge features"""
        # Use morphological opening to identify ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to get ridge mask
        ridge_threshold = np.percentile(tophat, 95)
        ridge_mask = tophat > ridge_threshold
        
        # Find ridge lines
        ridge_skeleton = morphology.skeletonize(ridge_mask)
        
        # Find connected components
        labeled, num_ridges = ndimage.label(ridge_skeleton)
        
        ridges = []
        for i in range(1, num_ridges + 1):
            ridge_points = np.where(labeled == i)
            if len(ridge_points[0]) < 10:  # Minimum ridge length
                continue
                
            # Calculate ridge properties
            ridge_coords = list(zip(ridge_points[1], ridge_points[0]))  # x, y
            ridge_length_pixels = len(ridge_coords)
            ridge_length_m = ridge_length_pixels * self.pixel_scale
            
            # Calculate average position
            avg_x = np.mean(ridge_points[1])
            avg_y = np.mean(ridge_points[0])
            lat, lon = self._pixel_to_lunar_coords(avg_x, avg_y, image.shape)
            
            ridges.append({
                'center': [lat, lon],
                'length_m': round(ridge_length_m, 1),
                'prominence': round(np.mean(tophat[ridge_points]), 2)
            })
            
        return ridges[:20]  # Top 20 ridges
        
    def _detect_valleys(self, image: np.ndarray, gradient_magnitude: np.ndarray) -> List[Dict]:
        """Detect valley features"""
        # Use morphological closing to identify valleys
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to get valley mask
        valley_threshold = np.percentile(blackhat, 95)
        valley_mask = blackhat > valley_threshold
        
        # Find valley lines
        valley_skeleton = morphology.skeletonize(valley_mask)
        
        # Find connected components
        labeled, num_valleys = ndimage.label(valley_skeleton)
        
        valleys = []
        for i in range(1, num_valleys + 1):
            valley_points = np.where(labeled == i)
            if len(valley_points[0]) < 10:  # Minimum valley length
                continue
                
            # Calculate valley properties
            valley_coords = list(zip(valley_points[1], valley_points[0]))  # x, y
            valley_length_pixels = len(valley_coords)
            valley_length_m = valley_length_pixels * self.pixel_scale
            
            # Calculate average position
            avg_x = np.mean(valley_points[1])
            avg_y = np.mean(valley_points[0])
            lat, lon = self._pixel_to_lunar_coords(avg_x, avg_y, image.shape)
            
            valleys.append({
                'center': [lat, lon],
                'length_m': round(valley_length_m, 1),
                'depth': round(np.mean(blackhat[valley_points]), 2)
            })
            
        return valleys[:20]  # Top 20 valleys
        
    def _detect_scarps(self, gradient_magnitude: np.ndarray, gradient_direction: np.ndarray) -> List[Dict]:
        """Detect scarp features (steep slopes)"""
        # Identify steep slopes
        scarp_threshold = np.percentile(gradient_magnitude, 90)
        steep_slopes = gradient_magnitude > scarp_threshold
        
        # Find connected scarp regions
        labeled, num_scarps = ndimage.label(steep_slopes)
        
        scarps = []
        for i in range(1, num_scarps + 1):
            scarp_mask = labeled == i
            scarp_props = measure.regionprops(scarp_mask.astype(int))[0]
            
            # Filter by size
            if scarp_props.area < 50:  # Minimum scarp area
                continue
                
            # Calculate scarp properties
            cy, cx = scarp_props.centroid
            lat, lon = self._pixel_to_lunar_coords(cx, cy, gradient_magnitude.shape)
            
            # Calculate average slope angle
            scarp_gradients = gradient_magnitude[scarp_mask]
            avg_slope = np.mean(scarp_gradients)
            
            # Calculate scarp orientation
            scarp_directions = gradient_direction[scarp_mask]
            avg_direction = np.degrees(np.arctan2(
                np.mean(np.sin(scarp_directions)),
                np.mean(np.cos(scarp_directions))
            ))
            
            scarps.append({
                'center': [lat, lon],
                'area_m2': round(scarp_props.area * (self.pixel_scale ** 2), 0),
                'avg_slope': round(avg_slope, 2),
                'orientation_deg': round(avg_direction, 1),
                'length_m': round(scarp_props.major_axis_length * self.pixel_scale, 1)
            })
            
        return scarps[:15]  # Top 15 scarps
        
    def _detect_plateaus(self, image: np.ndarray, gradient_magnitude: np.ndarray) -> List[Dict]:
        """Detect plateau features (flat elevated areas)"""
        # Identify flat areas (low gradient)
        flat_threshold = np.percentile(gradient_magnitude, 20)
        flat_areas = gradient_magnitude < flat_threshold
        
        # Identify elevated areas (high brightness)
        elevation_threshold = np.percentile(image, 80)
        elevated_areas = image > elevation_threshold
        
        # Plateaus are flat AND elevated
        plateau_mask = flat_areas & elevated_areas
        
        # Find connected plateau regions
        labeled, num_plateaus = ndimage.label(plateau_mask)
        
        plateaus = []
        for i in range(1, num_plateaus + 1):
            plateau_mask_i = labeled == i
            plateau_props = measure.regionprops(plateau_mask_i.astype(int))[0]
            
            # Filter by size
            if plateau_props.area < 100:  # Minimum plateau area
                continue
                
            # Calculate plateau properties
            cy, cx = plateau_props.centroid
            lat, lon = self._pixel_to_lunar_coords(cx, cy, image.shape)
            
            # Calculate average elevation (brightness)
            plateau_elevation = np.mean(image[plateau_mask_i])
            
            plateaus.append({
                'center': [lat, lon],
                'area_m2': round(plateau_props.area * (self.pixel_scale ** 2), 0),
                'elevation_index': round(plateau_elevation / 255.0, 2),
                'flatness': round(1.0 - np.std(gradient_magnitude[plateau_mask_i]), 2)
            })
            
        return plateaus[:10]  # Top 10 plateaus
        
    def _analyze_regolith_texture(self, image: np.ndarray) -> Dict:
        """Analyze regolith texture characteristics"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Calculate texture measures
        texture_measures = {
            'mean_brightness': round(np.mean(gray) / 255.0, 3),
            'brightness_std': round(np.std(gray) / 255.0, 3),
            'texture_energy': 0.0,
            'texture_homogeneity': 0.0,
            'surface_roughness': 0.0,
            'regolith_classification': 'unknown'
        }
        
        # Local texture analysis using GLCM approximation
        texture_energy, texture_homogeneity = self._calculate_texture_measures(gray)
        texture_measures['texture_energy'] = round(texture_energy, 3)
        texture_measures['texture_homogeneity'] = round(texture_homogeneity, 3)
        
        # Surface roughness from gradient variation
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        surface_roughness = np.std(gradient_magnitude) / np.mean(gradient_magnitude + 1e-6)
        texture_measures['surface_roughness'] = round(surface_roughness, 3)
        
        # Regolith classification based on texture
        if surface_roughness > 0.8:
            regolith_class = 'rough_blocky'
        elif surface_roughness > 0.4:
            regolith_class = 'medium_textured'
        else:
            regolith_class = 'smooth_fine'
            
        texture_measures['regolith_classification'] = regolith_class
        
        return texture_measures
        
    def _calculate_texture_measures(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate texture energy and homogeneity (simplified GLCM)"""
        # Simplified texture calculation using local patches
        patch_size = 16
        energy_sum = 0.0
        homogeneity_sum = 0.0
        patch_count = 0
        
        for y in range(0, image.shape[0] - patch_size, patch_size // 2):
            for x in range(0, image.shape[1] - patch_size, patch_size // 2):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Calculate histogram
                hist, _ = np.histogram(patch, bins=16, range=(0, 255))
                hist = hist.astype(float)
                hist = hist / (np.sum(hist) + 1e-6)  # Normalize
                
                # Energy (uniformity)
                energy = np.sum(hist ** 2)
                energy_sum += energy
                
                # Homogeneity (inversely related to contrast)
                homogeneity = np.sum(hist)  # Simplified measure
                homogeneity_sum += homogeneity
                
                patch_count += 1
                
        if patch_count > 0:
            avg_energy = energy_sum / patch_count
            avg_homogeneity = homogeneity_sum / patch_count
        else:
            avg_energy = 0.0
            avg_homogeneity = 0.0
            
        return avg_energy, avg_homogeneity
        
    def _analyze_slope_stability(self, image: np.ndarray) -> Dict:
        """Analyze slope stability for gravity-driven processes"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Calculate slope
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Identify unstable slopes
        stability_threshold = np.percentile(slope_magnitude, 85)
        unstable_slopes = slope_magnitude > stability_threshold
        
        # Calculate stability statistics
        stability_analysis = {
            'mean_slope': round(np.mean(slope_magnitude), 3),
            'max_slope': round(np.max(slope_magnitude), 3),
            'unstable_area_percent': round(np.sum(unstable_slopes) / unstable_slopes.size * 100, 1),
            'stability_index': 0.0,
            'critical_slopes': []
        }
        
        # Calculate stability index (0 = very unstable, 1 = very stable)
        stability_index = 1.0 - (np.mean(slope_magnitude) / np.max(slope_magnitude + 1e-6))
        stability_analysis['stability_index'] = round(stability_index, 2)
        
        # Find critical slope regions
        labeled, num_regions = ndimage.label(unstable_slopes)
        
        for i in range(1, min(num_regions + 1, 11)):  # Top 10 critical slopes
            slope_mask = labeled == i
            slope_props = measure.regionprops(slope_mask.astype(int))[0]
            
            if slope_props.area < 25:  # Minimum area
                continue
                
            cy, cx = slope_props.centroid
            lat, lon = self._pixel_to_lunar_coords(cx, cy, image.shape)
            
            avg_slope = np.mean(slope_magnitude[slope_mask])
            
            stability_analysis['critical_slopes'].append({
                'center': [lat, lon],
                'area_m2': round(slope_props.area * (self.pixel_scale ** 2), 0),
                'avg_slope': round(avg_slope, 2),
                'instability_factor': round(avg_slope / stability_threshold, 2)
            })
            
        return stability_analysis
        
    def _identify_landslide_sources(self, landslides: List[Dict], craters: List[Dict],
                                  topographic_features: Dict, slope_stability: Dict,
                                  image: np.ndarray) -> List[Dict]:
        """Identify probable sources for detected landslides"""
        landslide_sources = []
        
        for landslide in landslides:
            landslide_center = landslide['center']
            
            # Analyze potential sources
            source_analysis = {
                'landslide_id': landslide.get('id', 'unknown'),
                'landslide_center': landslide_center,
                'probable_sources': [],
                'primary_source_type': 'unknown',
                'confidence': 0.0
            }
            
            # Check crater proximity
            crater_sources = self._check_crater_proximity(landslide_center, craters)
            source_analysis['probable_sources'].extend(crater_sources)
            
            # Check scarp proximity
            scarp_sources = self._check_scarp_proximity(landslide_center, topographic_features['scarps'])
            source_analysis['probable_sources'].extend(scarp_sources)
            
            # Check slope instability
            slope_sources = self._check_slope_instability(landslide_center, slope_stability['critical_slopes'])
            source_analysis['probable_sources'].extend(slope_sources)
            
            # Determine primary source type
            if source_analysis['probable_sources']:
                primary_source = max(source_analysis['probable_sources'], key=lambda x: x['confidence'])
                source_analysis['primary_source_type'] = primary_source['type']
                source_analysis['confidence'] = primary_source['confidence']
            
            landslide_sources.append(source_analysis)
            
        return landslide_sources
        
    def _identify_boulder_sources(self, boulders: List[Dict], craters: List[Dict],
                                topographic_features: Dict, regolith_analysis: Dict,
                                image: np.ndarray) -> List[Dict]:
        """Identify probable sources for detected boulders"""
        boulder_sources = []
        
        for boulder in boulders:
            boulder_center = [boulder['lat'], boulder['lon']]
            
            # Analyze potential sources
            source_analysis = {
                'boulder_id': boulder.get('id', 'unknown'),
                'boulder_center': boulder_center,
                'boulder_diameter': boulder['diameter'],
                'probable_sources': [],
                'primary_source_type': 'unknown',
                'confidence': 0.0
            }
            
            # Check crater proximity (impact ejecta)
            crater_sources = self._check_crater_proximity(boulder_center, craters)
            source_analysis['probable_sources'].extend(crater_sources)
            
            # Check ridge proximity (weathering/erosion)
            ridge_sources = self._check_ridge_proximity(boulder_center, topographic_features['ridges'])
            source_analysis['probable_sources'].extend(ridge_sources)
            
            # Check plateau proximity (mass wasting)
            plateau_sources = self._check_plateau_proximity(boulder_center, topographic_features['plateaus'])
            source_analysis['probable_sources'].extend(plateau_sources)
            
            # Determine primary source type
            if source_analysis['probable_sources']:
                primary_source = max(source_analysis['probable_sources'], key=lambda x: x['confidence'])
                source_analysis['primary_source_type'] = primary_source['type']
                source_analysis['confidence'] = primary_source['confidence']
            else:
                # Default to in-situ weathering
                source_analysis['primary_source_type'] = 'in_situ_weathering'
                source_analysis['confidence'] = 0.5
            
            boulder_sources.append(source_analysis)
            
        return boulder_sources
        
    def _check_crater_proximity(self, target_center: List[float], craters: List[Dict]) -> List[Dict]:
        """Check proximity to craters for impact-related sources"""
        crater_sources = []
        
        for crater in craters:
            # Calculate distance
            distance_deg = np.sqrt(
                (target_center[0] - crater['center'][0])**2 +
                (target_center[1] - crater['center'][1])**2
            )
            
            # Convert to approximate distance in meters
            distance_m = distance_deg * 111000  # Rough conversion
            
            # Check if within ejecta range
            ejecta_range_m = crater['diameter_m'] * 5  # Ejecta typically 5x crater diameter
            
            if distance_m <= ejecta_range_m:
                # Calculate confidence based on distance and crater properties
                distance_factor = 1.0 - (distance_m / ejecta_range_m)
                crater_factor = crater['confidence'] * crater['freshness']
                
                confidence = distance_factor * crater_factor
                
                crater_sources.append({
                    'type': 'impact_ejecta',
                    'source_location': crater['center'],
                    'distance_m': round(distance_m, 0),
                    'crater_diameter_m': crater['diameter_m'],
                    'confidence': round(confidence, 2)
                })
                
        return crater_sources
        
    def _check_scarp_proximity(self, target_center: List[float], scarps: List[Dict]) -> List[Dict]:
        """Check proximity to scarps for structural failure sources"""
        scarp_sources = []
        
        for scarp in scarps:
            # Calculate distance
            distance_deg = np.sqrt(
                (target_center[0] - scarp['center'][0])**2 +
                (target_center[1] - scarp['center'][1])**2
            )
            
            distance_m = distance_deg * 111000  # Rough conversion
            
            # Check if within reasonable range for scarp failure
            max_range_m = 500  # 500m maximum range for scarp-related landslides
            
            if distance_m <= max_range_m:
                # Calculate confidence based on distance and scarp properties
                distance_factor = 1.0 - (distance_m / max_range_m)
                scarp_factor = min(1.0, scarp['avg_slope'] / 100.0)  # Steeper scarps more likely
                
                confidence = distance_factor * scarp_factor
                
                scarp_sources.append({
                    'type': 'scarp_failure',
                    'source_location': scarp['center'],
                    'distance_m': round(distance_m, 0),
                    'scarp_slope': scarp['avg_slope'],
                    'confidence': round(confidence, 2)
                })
                
        return scarp_sources
        
    def _check_slope_instability(self, target_center: List[float], critical_slopes: List[Dict]) -> List[Dict]:
        """Check proximity to unstable slopes for gravity-driven sources"""
        slope_sources = []
        
        for slope in critical_slopes:
            # Calculate distance
            distance_deg = np.sqrt(
                (target_center[0] - slope['center'][0])**2 +
                (target_center[1] - slope['center'][1])**2
            )
            
            distance_m = distance_deg * 111000  # Rough conversion
            
            # Check if within reasonable range for slope failure
            max_range_m = 200  # 200m maximum range for slope instability
            
            if distance_m <= max_range_m:
                # Calculate confidence based on distance and slope properties
                distance_factor = 1.0 - (distance_m / max_range_m)
                instability_factor = slope['instability_factor']
                
                confidence = distance_factor * min(1.0, instability_factor)
                
                slope_sources.append({
                    'type': 'slope_instability',
                    'source_location': slope['center'],
                    'distance_m': round(distance_m, 0),
                    'instability_factor': instability_factor,
                    'confidence': round(confidence, 2)
                })
                
        return slope_sources
        
    def _check_ridge_proximity(self, target_center: List[float], ridges: List[Dict]) -> List[Dict]:
        """Check proximity to ridges for weathering/erosion sources"""
        ridge_sources = []
        
        for ridge in ridges:
            # Calculate distance
            distance_deg = np.sqrt(
                (target_center[0] - ridge['center'][0])**2 +
                (target_center[1] - ridge['center'][1])**2
            )
            
            distance_m = distance_deg * 111000  # Rough conversion
            
            # Check if within reasonable range for ridge weathering
            max_range_m = 100  # 100m maximum range for ridge weathering
            
            if distance_m <= max_range_m:
                # Calculate confidence based on distance and ridge properties
                distance_factor = 1.0 - (distance_m / max_range_m)
                prominence_factor = min(1.0, ridge['prominence'] / 50.0)
                
                confidence = distance_factor * prominence_factor
                
                ridge_sources.append({
                    'type': 'ridge_weathering',
                    'source_location': ridge['center'],
                    'distance_m': round(distance_m, 0),
                    'ridge_prominence': ridge['prominence'],
                    'confidence': round(confidence, 2)
                })
                
        return ridge_sources
        
    def _check_plateau_proximity(self, target_center: List[float], plateaus: List[Dict]) -> List[Dict]:
        """Check proximity to plateaus for mass wasting sources"""
        plateau_sources = []
        
        for plateau in plateaus:
            # Calculate distance
            distance_deg = np.sqrt(
                (target_center[0] - plateau['center'][0])**2 +
                (target_center[1] - plateau['center'][1])**2
            )
            
            distance_m = distance_deg * 111000  # Rough conversion
            
            # Check if within reasonable range for plateau edge processes
            max_range_m = 300  # 300m maximum range for plateau mass wasting
            
            if distance_m <= max_range_m:
                # Calculate confidence based on distance and plateau properties
                distance_factor = 1.0 - (distance_m / max_range_m)
                elevation_factor = plateau['elevation_index']
                
                confidence = distance_factor * elevation_factor
                
                plateau_sources.append({
                    'type': 'plateau_mass_wasting',
                    'source_location': plateau['center'],
                    'distance_m': round(distance_m, 0),
                    'plateau_elevation': plateau['elevation_index'],
                    'confidence': round(confidence, 2)
                })
                
        return plateau_sources
        
    def _assess_regional_geology(self, image: np.ndarray, craters: List[Dict],
                               topographic_features: Dict, regolith_analysis: Dict) -> Dict:
        """Assess overall regional geological characteristics"""
        regional_assessment = {
            'crater_density': len(craters) / (image.shape[0] * image.shape[1] * (self.pixel_scale ** 2) / 1e6),  # craters per km²
            'topographic_complexity': 0.0,
            'dominant_processes': [],
            'geological_age_estimate': 'unknown',
            'hazard_assessment': 'unknown'
        }
        
        # Calculate topographic complexity
        num_features = (len(topographic_features['ridges']) + 
                       len(topographic_features['valleys']) + 
                       len(topographic_features['scarps']) + 
                       len(topographic_features['plateaus']))
        
        complexity = num_features / (image.shape[0] * image.shape[1]) * 1e6  # features per km²
        regional_assessment['topographic_complexity'] = round(complexity, 2)
        
        # Identify dominant processes
        if regional_assessment['crater_density'] > 0.1:
            regional_assessment['dominant_processes'].append('impact_cratering')
            
        if topographic_features['slope_variation'] > 50:
            regional_assessment['dominant_processes'].append('tectonic_activity')
            
        if regolith_analysis['surface_roughness'] > 0.6:
            regional_assessment['dominant_processes'].append('mass_wasting')
            
        # Geological age estimate (simplified)
        if regional_assessment['crater_density'] > 0.5:
            age_estimate = 'ancient'
        elif regional_assessment['crater_density'] > 0.1:
            age_estimate = 'mature'
        else:
            age_estimate = 'young'
            
        regional_assessment['geological_age_estimate'] = age_estimate
        
        # Hazard assessment
        if (topographic_features['slope_variation'] > 60 and 
            regolith_analysis['surface_roughness'] > 0.7):
            hazard_level = 'high'
        elif (topographic_features['slope_variation'] > 30 or 
              regolith_analysis['surface_roughness'] > 0.5):
            hazard_level = 'moderate'
        else:
            hazard_level = 'low'
            
        regional_assessment['hazard_assessment'] = hazard_level
        
        return regional_assessment
        
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