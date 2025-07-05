import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import logging
from typing import Dict, List, Tuple
import os

logger = logging.getLogger(__name__)

class AnnotatedMapGenerator:
    """
    Novel annotated map generation system creating comprehensive visualization maps:
    1. Multi-layer geological hazard maps with detection overlays
    2. Statistical heatmaps with confidence indicators
    3. Risk assessment maps with zone classifications
    4. Temporal activity maps showing evolution patterns
    
    Novelty: This system generates publication-quality annotated maps
    specifically designed for lunar geological analysis, with specialized
    symbology for space-based hazard assessment and exploration planning.
    """
    
    def __init__(self):
        self.pixel_scale = 5.0  # 5m/pixel from TMC
        self.output_dpi = 300  # High resolution for publication
        
        # Define color schemes for different map types
        self.color_schemes = {
            'risk_map': ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#8B0000'],  # Green to Red
            'confidence_map': ['#000080', '#4169E1', '#87CEEB', '#F0F8FF'],  # Blue gradient
            'activity_map': ['#4B0082', '#8A2BE2', '#DA70D6', '#FFB6C1'],  # Purple gradient
            'geological_map': ['#8B4513', '#DEB887', '#F4A460', '#FFA500']  # Brown/Orange
        }
        
    def generate_comprehensive_maps(self, image: np.ndarray,
                                  landslides: List[Dict],
                                  boulders: List[Dict],
                                  geological_context: Dict,
                                  statistical_analysis: Dict,
                                  temporal_analysis: Dict = None,
                                  output_dir: str = "output_maps") -> Dict:
        """
        Generate comprehensive annotated maps
        
        Args:
            image: Original TMC image
            landslides: Detected landslides
            boulders: Detected boulders
            geological_context: Geological analysis results
            statistical_analysis: Statistical analysis results
            temporal_analysis: Temporal change analysis results
            output_dir: Directory to save output maps
            
        Returns:
            Dictionary containing map generation results and file paths
        """
        logger.info("Starting comprehensive map generation")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate different types of maps
        map_results = {}
        
        # 1. Main Detection Map
        detection_map_path = self._generate_detection_map(
            image, landslides, boulders, geological_context, 
            os.path.join(output_dir, "lunar_detection_map.png")
        )
        map_results['detection_map'] = detection_map_path
        
        # 2. Risk Assessment Map
        risk_map_path = self._generate_risk_map(
            image, statistical_analysis.get('risk_assessment', {}),
            landslides, boulders,
            os.path.join(output_dir, "risk_assessment_map.png")
        )
        map_results['risk_map'] = risk_map_path
        
        # 3. Statistical Analysis Map
        statistics_map_path = self._generate_statistical_map(
            image, statistical_analysis,
            os.path.join(output_dir, "statistical_analysis_map.png")
        )
        map_results['statistics_map'] = statistics_map_path
        
        # 4. Geological Context Map
        geological_map_path = self._generate_geological_map(
            image, geological_context,
            os.path.join(output_dir, "geological_context_map.png")
        )
        map_results['geological_map'] = geological_map_path
        
        # 5. Activity Hotspot Map
        hotspot_map_path = self._generate_hotspot_map(
            image, statistical_analysis.get('activity_hotspots', []),
            os.path.join(output_dir, "activity_hotspot_map.png")
        )
        map_results['hotspot_map'] = hotspot_map_path
        
        # 6. Temporal Change Map (if available)
        if temporal_analysis:
            temporal_map_path = self._generate_temporal_map(
                image, temporal_analysis,
                os.path.join(output_dir, "temporal_change_map.png")
            )
            map_results['temporal_map'] = temporal_map_path
            
        # 7. Summary Dashboard
        dashboard_path = self._generate_summary_dashboard(
            landslides, boulders, geological_context, statistical_analysis,
            os.path.join(output_dir, "summary_dashboard.png")
        )
        map_results['dashboard'] = dashboard_path
        
        # 8. Export Data Files
        data_files = self._export_data_files(
            landslides, boulders, geological_context, statistical_analysis,
            output_dir
        )
        map_results['data_files'] = data_files
        
        logger.info(f"Generated {len(map_results)} map products in {output_dir}")
        return map_results
        
    def _generate_detection_map(self, image: np.ndarray, landslides: List[Dict],
                              boulders: List[Dict], geological_context: Dict,
                              output_path: str) -> str:
        """Generate main detection map with all features"""
        plt.figure(figsize=(16, 12), dpi=self.output_dpi)
        
        # Display base image
        plt.imshow(image, cmap='gray', alpha=0.8)
        
        # Add landslide detections
        for i, landslide in enumerate(landslides):
            if landslide.get('polygon'):
                # Convert polygon to display coordinates
                polygon_pixels = []
                for lat, lon in landslide['polygon']:
                    px, py = self._lunar_coords_to_pixel(lon, lat, image.shape)
                    polygon_pixels.append([px, py])
                    
                polygon_array = np.array(polygon_pixels)
                
                # Color by confidence
                confidence = landslide.get('confidence', 50)
                color = plt.cm.Reds(confidence / 100.0)
                
                polygon = patches.Polygon(polygon_array, closed=True, 
                                        facecolor=color, edgecolor='red',
                                        alpha=0.6, linewidth=2)
                plt.gca().add_patch(polygon)
                
                # Add label
                center_x, center_y = np.mean(polygon_array, axis=0)
                plt.text(center_x, center_y, f'LS{i+1}', 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
        # Add boulder detections
        for i, boulder in enumerate(boulders):
            px, py = self._lunar_coords_to_pixel(boulder['lon'], boulder['lat'], image.shape)
            
            # Size circle based on diameter
            radius_pixels = (boulder['diameter'] / self.pixel_scale) / 2
            
            # Color by confidence
            confidence = boulder.get('confidence', 50)
            color = plt.cm.Blues(confidence / 100.0)
            
            circle = patches.Circle((px, py), radius_pixels, 
                                  facecolor=color, edgecolor='blue',
                                  alpha=0.7, linewidth=1)
            plt.gca().add_patch(circle)
            
            # Add label for larger boulders
            if boulder['diameter'] > 5.0:
                plt.text(px, py, f'B{i+1}', 
                        fontsize=6, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))
                
        # Add craters from geological context
        craters = geological_context.get('craters', [])
        for crater in craters[:10]:  # Show top 10 craters
            px, py = self._lunar_coords_to_pixel(crater['center'][1], crater['center'][0], image.shape)
            radius_pixels = (crater['diameter_m'] / self.pixel_scale) / 2
            
            circle = patches.Circle((px, py), radius_pixels, 
                                  facecolor='none', edgecolor='yellow',
                                  alpha=0.8, linewidth=2, linestyle='--')
            plt.gca().add_patch(circle)
            
        # Add title and labels
        plt.title('Lunar Landslide and Boulder Detection Map\nChandrayaan TMC Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [
            patches.Patch(color='red', alpha=0.6, label='Landslides'),
            patches.Patch(color='blue', alpha=0.7, label='Boulders'),
            patches.Patch(facecolor='none', edgecolor='yellow', label='Craters'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add scale bar
        self._add_scale_bar(plt.gca(), image.shape)
        
        # Add coordinate grid
        self._add_coordinate_grid(plt.gca(), image.shape)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_risk_map(self, image: np.ndarray, risk_assessment: Dict,
                         landslides: List[Dict], boulders: List[Dict],
                         output_path: str) -> str:
        """Generate risk assessment map"""
        plt.figure(figsize=(14, 10), dpi=self.output_dpi)
        
        # Create risk colormap
        risk_cmap = LinearSegmentedColormap.from_list(
            'risk', self.color_schemes['risk_map'], N=256
        )
        
        # Display risk map if available
        risk_map_data = risk_assessment.get('risk_map', [])
        if risk_map_data:
            risk_array = np.array(risk_map_data)
            plt.imshow(risk_array, cmap=risk_cmap, alpha=0.8, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(shrink=0.6, aspect=20)
            cbar.set_label('Risk Level', fontsize=12, fontweight='bold')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        else:
            # Show base image if no risk map
            plt.imshow(image, cmap='gray', alpha=0.5)
            
        # Add risk zones
        risk_zones = risk_assessment.get('risk_zones', [])
        for zone in risk_zones:
            px, py = self._lunar_coords_to_pixel(zone['center'][1], zone['center'][0], image.shape)
            
            # Zone marker
            if zone['risk_level'] == 'high':
                color = 'red'
                marker = 'X'
                size = 100
            else:
                color = 'orange'
                marker = 'o'
                size = 80
                
            plt.scatter(px, py, c=color, marker=marker, s=size, 
                       edgecolors='black', linewidth=1, alpha=0.9)
            
        # Add detected features for context
        for landslide in landslides[:5]:  # Show top 5
            px, py = self._lunar_coords_to_pixel(landslide['center'][1], landslide['center'][0], image.shape)
            plt.scatter(px, py, c='darkred', marker='s', s=30, alpha=0.7)
            
        for boulder in boulders[:20]:  # Show top 20
            px, py = self._lunar_coords_to_pixel(boulder['lon'], boulder['lat'], image.shape)
            plt.scatter(px, py, c='darkblue', marker='.', s=10, alpha=0.5)
            
        plt.title('Lunar Surface Risk Assessment Map\nHazard Zones and Safety Assessment', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create legend for risk zones
        legend_elements = [
            plt.scatter([], [], c='red', marker='X', s=100, label='High Risk Zone'),
            plt.scatter([], [], c='orange', marker='o', s=80, label='Moderate Risk Zone'),
            plt.scatter([], [], c='darkred', marker='s', s=30, label='Major Landslides'),
            plt.scatter([], [], c='darkblue', marker='.', s=10, label='Boulder Fields'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add risk statistics text box
        risk_stats = self._format_risk_statistics(risk_assessment)
        plt.text(0.02, 0.98, risk_stats, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        self._add_scale_bar(plt.gca(), image.shape)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_statistical_map(self, image: np.ndarray, statistical_analysis: Dict,
                                output_path: str) -> str:
        """Generate statistical analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.output_dpi)
        
        # 1. Size-frequency distribution for landslides
        landslide_dist = statistical_analysis.get('size_frequency_analysis', {}).get('landslide_distribution', {})
        ls_stats = landslide_dist.get('basic_statistics', {})
        
        if ls_stats:
            # Create histogram data (simulated from statistics)
            sizes = np.random.lognormal(np.log(ls_stats.get('mean', 0.1)), 0.5, ls_stats.get('count', 10))
            ax1.hist(sizes, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax1.set_xlabel('Landslide Area (km²)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title('Landslide Size Distribution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
        # 2. Size-frequency distribution for boulders
        boulder_dist = statistical_analysis.get('size_frequency_analysis', {}).get('boulder_distribution', {})
        b_stats = boulder_dist.get('basic_statistics', {})
        
        if b_stats:
            # Create histogram data (simulated from statistics)
            sizes = np.random.lognormal(np.log(b_stats.get('mean', 5)), 0.3, b_stats.get('count', 50))
            ax2.hist(sizes, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_xlabel('Boulder Diameter (m)', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title('Boulder Size Distribution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
        # 3. Spatial clustering analysis
        spatial_analysis = statistical_analysis.get('spatial_analysis', {})
        clustering_data = {
            'Landslides': spatial_analysis.get('landslide_clustering', {}),
            'Boulders': spatial_analysis.get('boulder_clustering', {})
        }
        
        cluster_coeffs = []
        labels = []
        for feature_type, data in clustering_data.items():
            coeff = data.get('clustering_coefficient', 0)
            if coeff > 0:
                cluster_coeffs.append(coeff)
                labels.append(feature_type)
                
        if cluster_coeffs:
            bars = ax3.bar(labels, cluster_coeffs, color=['red', 'blue'], alpha=0.7)
            ax3.set_ylabel('Clustering Coefficient', fontsize=10)
            ax3.set_title('Spatial Clustering Analysis', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                        
        # 4. Activity hotspots visualization
        hotspots = statistical_analysis.get('activity_hotspots', [])
        if hotspots:
            # Create hotspot intensity map
            intensities = [h['activity_intensity'] for h in hotspots[:10]]
            areas = [h['area_m2'] / 1000 for h in hotspots[:10]]  # Convert to hectares
            
            scatter = ax4.scatter(areas, intensities, s=100, alpha=0.7, 
                                c=intensities, cmap='viridis', edgecolors='black')
            ax4.set_xlabel('Hotspot Area (hectares)', fontsize=10)
            ax4.set_ylabel('Activity Intensity', fontsize=10)
            ax4.set_title('Activity Hotspot Analysis', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4, shrink=0.6)
            cbar.set_label('Intensity', fontsize=10)
            
        plt.suptitle('Statistical Analysis of Lunar Surface Features', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_geological_map(self, image: np.ndarray, geological_context: Dict,
                               output_path: str) -> str:
        """Generate geological context map"""
        plt.figure(figsize=(14, 10), dpi=self.output_dpi)
        
        # Display base image
        plt.imshow(image, cmap='gray', alpha=0.6)
        
        # Add craters
        craters = geological_context.get('craters', [])
        for crater in craters:
            px, py = self._lunar_coords_to_pixel(crater['center'][1], crater['center'][0], image.shape)
            radius_pixels = (crater['diameter_m'] / self.pixel_scale) / 2
            
            # Color by freshness
            freshness = crater.get('freshness', 0.5)
            color = plt.cm.YlOrRd(freshness)
            
            circle = patches.Circle((px, py), radius_pixels, 
                                  facecolor=color, edgecolor='darkred',
                                  alpha=0.7, linewidth=1)
            plt.gca().add_patch(circle)
            
        # Add topographic features
        topo_features = geological_context.get('topographic_features', {})
        
        # Ridges
        ridges = topo_features.get('ridges', [])
        for ridge in ridges:
            px, py = self._lunar_coords_to_pixel(ridge['center'][1], ridge['center'][0], image.shape)
            plt.scatter(px, py, c='brown', marker='^', s=50, alpha=0.8, 
                       edgecolors='black', linewidth=0.5)
            
        # Valleys
        valleys = topo_features.get('valleys', [])
        for valley in valleys:
            px, py = self._lunar_coords_to_pixel(valley['center'][1], valley['center'][0], image.shape)
            plt.scatter(px, py, c='darkblue', marker='v', s=50, alpha=0.8,
                       edgecolors='black', linewidth=0.5)
            
        # Scarps
        scarps = topo_features.get('scarps', [])
        for scarp in scarps:
            px, py = self._lunar_coords_to_pixel(scarp['center'][1], scarp['center'][0], image.shape)
            plt.scatter(px, py, c='purple', marker='s', s=40, alpha=0.8,
                       edgecolors='black', linewidth=0.5)
            
        plt.title('Geological Context Map\nStructural Features and Crater Distribution', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [
            patches.Patch(color='darkred', alpha=0.7, label='Impact Craters'),
            plt.scatter([], [], c='brown', marker='^', s=50, label='Ridges'),
            plt.scatter([], [], c='darkblue', marker='v', s=50, label='Valleys'),
            plt.scatter([], [], c='purple', marker='s', s=40, label='Scarps'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add geological statistics
        regional_assessment = geological_context.get('regional_assessment', {})
        geo_stats = self._format_geological_statistics(geological_context, regional_assessment)
        plt.text(0.02, 0.98, geo_stats, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        self._add_scale_bar(plt.gca(), image.shape)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_hotspot_map(self, image: np.ndarray, hotspots: List[Dict],
                            output_path: str) -> str:
        """Generate activity hotspot map"""
        plt.figure(figsize=(12, 10), dpi=self.output_dpi)
        
        # Display base image
        plt.imshow(image, cmap='gray', alpha=0.7)
        
        # Create activity heatmap if hotspots exist
        if hotspots:
            # Create density surface
            height, width = image.shape[:2]
            activity_surface = np.zeros((height, width))
            
            for hotspot in hotspots:
                px, py = self._lunar_coords_to_pixel(hotspot['center'][1], hotspot['center'][0], image.shape)
                
                # Create Gaussian kernel around hotspot
                radius = np.sqrt(hotspot['area_m2'] / np.pi) / self.pixel_scale
                y, x = np.ogrid[:height, :width]
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                
                kernel = np.exp(-(distance**2) / (2 * (radius/2)**2))
                activity_surface += kernel * hotspot['activity_intensity']
                
            # Overlay activity surface
            activity_surface = activity_surface / np.max(activity_surface) if np.max(activity_surface) > 0 else activity_surface
            
            # Create custom colormap for activity
            activity_cmap = LinearSegmentedColormap.from_list(
                'activity', self.color_schemes['activity_map'], N=256
            )
            
            plt.imshow(activity_surface, cmap=activity_cmap, alpha=0.6, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(shrink=0.6, aspect=20)
            cbar.set_label('Activity Level', fontsize=12, fontweight='bold')
            
            # Mark hotspot centers
            for i, hotspot in enumerate(hotspots):
                px, py = self._lunar_coords_to_pixel(hotspot['center'][1], hotspot['center'][0], image.shape)
                
                # Size marker by area
                marker_size = min(200, max(50, hotspot['area_m2'] / 1000))
                
                # Color by risk level
                if hotspot.get('risk_level') == 'high':
                    color = 'red'
                elif hotspot.get('risk_level') == 'moderate':
                    color = 'orange'
                else:
                    color = 'yellow'
                    
                plt.scatter(px, py, c=color, s=marker_size, alpha=0.8,
                           edgecolors='black', linewidth=2)
                
                # Add hotspot ID
                plt.text(px, py, f'H{i+1}', fontsize=8, ha='center', va='center',
                        fontweight='bold', color='white')
                        
        plt.title('Activity Hotspot Map\nRegions of High Geological Activity', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [
            plt.scatter([], [], c='red', s=100, label='High Risk Hotspot'),
            plt.scatter([], [], c='orange', s=100, label='Moderate Risk Hotspot'),
            plt.scatter([], [], c='yellow', s=100, label='Low Risk Hotspot'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add hotspot statistics
        if hotspots:
            hotspot_stats = self._format_hotspot_statistics(hotspots)
            plt.text(0.98, 0.98, hotspot_stats, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.9))
                    
        self._add_scale_bar(plt.gca(), image.shape)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_temporal_map(self, image: np.ndarray, temporal_analysis: Dict,
                             output_path: str) -> str:
        """Generate temporal change analysis map"""
        plt.figure(figsize=(14, 10), dpi=self.output_dpi)
        
        # Display base image
        plt.imshow(image, cmap='gray', alpha=0.7)
        
        # Add temporal activity hotspots
        activity_hotspots = temporal_analysis.get('activity_hotspots', [])
        for hotspot in activity_hotspots:
            px, py = self._lunar_coords_to_pixel(hotspot['center'][1], hotspot['center'][0], image.shape)
            
            # Color by activity frequency
            frequency = hotspot.get('activity_frequency', 0.5)
            color = plt.cm.plasma(frequency)
            
            # Size by area
            marker_size = min(300, max(50, hotspot['area_m2'] / 500))
            
            plt.scatter(px, py, c=[color], s=marker_size, alpha=0.8,
                       edgecolors='white', linewidth=2)
                       
        # Add evolution analysis
        evolution = temporal_analysis.get('evolution_analysis', {})
        most_active = evolution.get('most_active_region')
        
        if most_active:
            px, py = self._lunar_coords_to_pixel(most_active['location'][1], most_active['location'][0], image.shape)
            
            # Highlight most active region
            circle = patches.Circle((px, py), 50, facecolor='none', 
                                  edgecolor='red', linewidth=4, linestyle='-')
            plt.gca().add_patch(circle)
            
            plt.text(px, py-60, 'MOST ACTIVE', fontsize=10, ha='center', va='top',
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                    
        plt.title('Temporal Change Analysis Map\nActivity Evolution and Trends', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Create colorbar for activity frequency
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.6, aspect=20)
        cbar.set_label('Activity Frequency', fontsize=12, fontweight='bold')
        
        # Add temporal statistics
        temporal_stats = self._format_temporal_statistics(temporal_analysis)
        plt.text(0.02, 0.98, temporal_stats, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lavender', alpha=0.9))
        
        self._add_scale_bar(plt.gca(), image.shape)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _generate_summary_dashboard(self, landslides: List[Dict], boulders: List[Dict],
                                  geological_context: Dict, statistical_analysis: Dict,
                                  output_path: str) -> str:
        """Generate comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 14), dpi=self.output_dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Summary statistics (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        summary_stats = statistical_analysis.get('summary_statistics', {})
        self._create_summary_table(ax1, summary_stats)
        
        # Detection overview pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        detection_counts = [len(landslides), len(boulders)]
        labels = ['Landslides', 'Boulders']
        colors = ['red', 'blue']
        if sum(detection_counts) > 0:
            ax2.pie(detection_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, alpha=0.8)
        ax2.set_title('Detection Overview', fontsize=12, fontweight='bold')
        
        # Risk level distribution
        ax3 = fig.add_subplot(gs[0, 3])
        risk_assessment = statistical_analysis.get('risk_assessment', {})
        self._create_risk_chart(ax3, risk_assessment)
        
        # Size distribution comparison (middle row)
        ax4 = fig.add_subplot(gs[1, :2])
        self._create_size_comparison_chart(ax4, landslides, boulders)
        
        # Confidence distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._create_confidence_chart(ax5, landslides, boulders)
        
        # Geological features chart
        ax6 = fig.add_subplot(gs[1, 3])
        self._create_geological_chart(ax6, geological_context)
        
        # Spatial analysis (bottom row)
        ax7 = fig.add_subplot(gs[2, :2])
        spatial_analysis = statistical_analysis.get('spatial_analysis', {})
        self._create_spatial_analysis_chart(ax7, spatial_analysis)
        
        # Activity hotspots
        ax8 = fig.add_subplot(gs[2, 2])
        hotspots = statistical_analysis.get('activity_hotspots', [])
        self._create_hotspot_chart(ax8, hotspots)
        
        # Quality metrics
        ax9 = fig.add_subplot(gs[2, 3])
        self._create_quality_metrics_chart(ax9, summary_stats)
        
        # Main title
        fig.suptitle('Lunar Surface Analysis Dashboard\nChandrayaan TMC Comprehensive Report', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add timestamp and metadata
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        fig.text(0.02, 0.02, f'Generated: {timestamp}\nLunar GeoDetect v2.0 - Novel Algorithm Suite', 
                fontsize=10, alpha=0.7)
        
        plt.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
        
    def _export_data_files(self, landslides: List[Dict], boulders: List[Dict],
                         geological_context: Dict, statistical_analysis: Dict,
                         output_dir: str) -> Dict:
        """Export data in various formats"""
        data_files = {}
        
        # 1. GeoJSON export
        geojson_path = os.path.join(output_dir, "lunar_features.geojson")
        geojson_data = self._create_geojson(landslides, boulders)
        
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        data_files['geojson'] = geojson_path
        
        # 2. CSV export
        csv_path = os.path.join(output_dir, "detection_results.csv")
        self._create_csv_export(landslides, boulders, csv_path)
        data_files['csv'] = csv_path
        
        # 3. Full analysis JSON
        json_path = os.path.join(output_dir, "full_analysis.json")
        full_analysis = {
            'landslides': landslides,
            'boulders': boulders,
            'geological_context': geological_context,
            'statistical_analysis': statistical_analysis,
            'metadata': {
                'pixel_scale_m': self.pixel_scale,
                'coordinate_system': 'Lunar Geographic',
                'analysis_method': 'Novel Multi-Algorithm Suite'
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        data_files['json'] = json_path
        
        return data_files
        
    # Helper methods for chart creation
    def _create_summary_table(self, ax, summary_stats):
        """Create summary statistics table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        
        # Study area info
        study_area = summary_stats.get('study_area', {})
        table_data.append(['Study Area (km²)', f"{study_area.get('total_area_km2', 0):.1f}"])
        table_data.append(['Resolution (m/pixel)', f"{study_area.get('pixel_resolution_m', 5):.1f}"])
        
        # Landslide stats
        ls_stats = summary_stats.get('landslide_statistics', {})
        table_data.append(['Total Landslides', str(ls_stats.get('total_count', 0))])
        table_data.append(['Landslide Density (per km²)', f"{ls_stats.get('density_per_km2', 0):.2f}"])
        
        # Boulder stats
        b_stats = summary_stats.get('boulder_statistics', {})
        table_data.append(['Total Boulders', str(b_stats.get('total_count', 0))])
        table_data.append(['Boulder Density (per km²)', f"{b_stats.get('density_per_km2', 0):.2f}"])
        
        # Quality stats
        quality = summary_stats.get('quality_assessment', {})
        table_data.append(['Mean Confidence', f"{quality.get('overall_mean_confidence', 0):.1f}%"])
        table_data.append(['Detection Reliability', quality.get('detection_reliability', 'Unknown')])
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
    def _create_risk_chart(self, ax, risk_assessment):
        """Create risk level distribution chart"""
        high_risk = risk_assessment.get('total_high_risk_area_km2', 0)
        moderate_risk = risk_assessment.get('total_moderate_risk_area_km2', 0)
        
        if high_risk + moderate_risk > 0:
            risks = [high_risk, moderate_risk]
            labels = ['High Risk', 'Moderate Risk']
            colors = ['red', 'orange']
            
            ax.pie(risks, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90, alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No Risk Data\nAvailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
                   
        ax.set_title('Risk Distribution', fontsize=12, fontweight='bold')
        
    def _create_size_comparison_chart(self, ax, landslides, boulders):
        """Create size distribution comparison"""
        if landslides:
            ls_areas = [ls.get('area_km2', 0) for ls in landslides if ls.get('area_km2', 0) > 0]
            if ls_areas:
                ax.hist(ls_areas, bins=15, alpha=0.7, color='red', label='Landslides (km²)', density=True)
                
        if boulders:
            b_diameters = [b['diameter'] for b in boulders]
            # Normalize to similar scale for comparison
            normalized_diameters = [d/1000 for d in b_diameters]  # Convert to km
            if normalized_diameters:
                ax.hist(normalized_diameters, bins=15, alpha=0.7, color='blue', 
                       label='Boulders (km)', density=True)
                       
        ax.set_xlabel('Size (km / km²)', fontsize=10)
        ax.set_ylabel('Normalized Frequency', fontsize=10)
        ax.set_title('Size Distribution Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _create_confidence_chart(self, ax, landslides, boulders):
        """Create confidence distribution chart"""
        all_confidences = []
        labels = []
        
        if landslides:
            ls_conf = [ls.get('confidence', 50) for ls in landslides]
            all_confidences.extend(ls_conf)
            labels.extend(['Landslide'] * len(ls_conf))
            
        if boulders:
            b_conf = [b.get('confidence', 50) for b in boulders]
            all_confidences.extend(b_conf)
            labels.extend(['Boulder'] * len(b_conf))
            
        if all_confidences:
            unique_labels = list(set(labels))
            for i, label in enumerate(unique_labels):
                conf_subset = [conf for conf, lbl in zip(all_confidences, labels) if lbl == label]
                color = ['red', 'blue'][i % 2]
                ax.hist(conf_subset, bins=15, alpha=0.7, color=color, label=label, density=True)
                
        ax.set_xlabel('Confidence (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Detection Confidence', fontsize=12, fontweight='bold')
        if all_confidences:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _create_geological_chart(self, ax, geological_context):
        """Create geological features chart"""
        feature_counts = []
        feature_names = []
        
        craters = geological_context.get('craters', [])
        if craters:
            feature_counts.append(len(craters))
            feature_names.append('Craters')
            
        topo_features = geological_context.get('topographic_features', {})
        for feature_type in ['ridges', 'valleys', 'scarps', 'plateaus']:
            features = topo_features.get(feature_type, [])
            if features:
                feature_counts.append(len(features))
                feature_names.append(feature_type.capitalize())
                
        if feature_counts:
            colors = plt.cm.Set3(np.linspace(0, 1, len(feature_counts)))
            bars = ax.bar(feature_names, feature_counts, color=colors, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
                       
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Geological Features', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def _create_spatial_analysis_chart(self, ax, spatial_analysis):
        """Create spatial clustering analysis chart"""
        clustering_data = {}
        
        ls_clustering = spatial_analysis.get('landslide_clustering', {})
        if ls_clustering:
            clustering_data['Landslides'] = ls_clustering.get('clustering_coefficient', 0)
            
        b_clustering = spatial_analysis.get('boulder_clustering', {})
        if b_clustering:
            clustering_data['Boulders'] = b_clustering.get('clustering_coefficient', 0)
            
        if clustering_data:
            names = list(clustering_data.keys())
            values = list(clustering_data.values())
            colors = ['red', 'blue'][:len(names)]
            
            bars = ax.bar(names, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                       
        ax.set_ylabel('Clustering Coefficient', fontsize=10)
        ax.set_title('Spatial Clustering Analysis', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
    def _create_hotspot_chart(self, ax, hotspots):
        """Create activity hotspots chart"""
        if not hotspots:
            ax.text(0.5, 0.5, 'No Hotspots\nDetected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Activity Hotspots', fontsize=12, fontweight='bold')
            return
            
        # Show top 5 hotspots
        top_hotspots = hotspots[:5]
        intensities = [h['activity_intensity'] for h in top_hotspots]
        labels = [f"H{i+1}" for i in range(len(top_hotspots))]
        
        bars = ax.bar(labels, intensities, color='purple', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                   
        ax.set_ylabel('Activity Intensity', fontsize=10)
        ax.set_title('Top Activity Hotspots', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    def _create_quality_metrics_chart(self, ax, summary_stats):
        """Create quality metrics chart"""
        quality = summary_stats.get('quality_assessment', {})
        
        if quality:
            mean_conf = quality.get('overall_mean_confidence', 0)
            high_quality_pct = quality.get('high_quality_detections_percent', 0)
            
            metrics = ['Mean Confidence', 'High Quality %']
            values = [mean_conf, high_quality_pct]
            colors = ['green', 'blue']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
                       
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title('Quality Metrics', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Helper methods for coordinate conversion and formatting
    def _lunar_coords_to_pixel(self, lon: float, lat: float, image_shape: tuple) -> tuple:
        """Convert lunar coordinates to pixel coordinates"""
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
            
        # Approximate inverse transformation
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
        
    def _add_scale_bar(self, ax, image_shape):
        """Add scale bar to map"""
        height, width = image_shape[:2]
        
        # Scale bar length in pixels (1 km)
        scale_length_m = 1000  # 1 km
        scale_length_pixels = scale_length_m / self.pixel_scale
        
        # Position scale bar
        x_start = width * 0.02
        y_pos = height * 0.95
        
        # Draw scale bar
        ax.plot([x_start, x_start + scale_length_pixels], [y_pos, y_pos], 
               'k-', linewidth=3)
        ax.text(x_start + scale_length_pixels/2, y_pos - 20, '1 km', 
               ha='center', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
               
    def _add_coordinate_grid(self, ax, image_shape):
        """Add coordinate grid to map"""
        # Simplified coordinate grid (would need actual projection in real implementation)
        height, width = image_shape[:2]
        
        # Add corner coordinates
        corners = [
            (0, 0, "-30.21°N, 5.73°E"),
            (width, 0, "-30.22°N, 6.57°E"),
            (0, height, "-84.65°S, 353.61°E"),
            (width, height, "-84.76°S, 0.64°E")
        ]
        
        for x, y, coord_text in corners:
            ax.text(x, y, coord_text, fontsize=8, ha='left' if x == 0 else 'right',
                   va='top' if y == 0 else 'bottom', alpha=0.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
                   
    def _format_risk_statistics(self, risk_assessment):
        """Format risk statistics for display"""
        high_risk = risk_assessment.get('total_high_risk_area_km2', 0)
        moderate_risk = risk_assessment.get('total_moderate_risk_area_km2', 0)
        confidence = risk_assessment.get('risk_assessment_confidence', 0)
        
        return f"Risk Assessment\n" \
               f"High Risk: {high_risk:.2f} km²\n" \
               f"Moderate Risk: {moderate_risk:.2f} km²\n" \
               f"Confidence: {confidence:.1%}"
               
    def _format_geological_statistics(self, geological_context, regional_assessment):
        """Format geological statistics for display"""
        crater_count = len(geological_context.get('craters', []))
        crater_density = regional_assessment.get('crater_density', 0)
        geological_age = regional_assessment.get('geological_age_estimate', 'unknown')
        hazard_level = regional_assessment.get('hazard_assessment', 'unknown')
        
        return f"Geological Context\n" \
               f"Craters: {crater_count}\n" \
               f"Density: {crater_density:.2f}/km²\n" \
               f"Age: {geological_age.title()}\n" \
               f"Hazard: {hazard_level.title()}"
               
    def _format_hotspot_statistics(self, hotspots):
        """Format hotspot statistics for display"""
        total_hotspots = len(hotspots)
        if hotspots:
            total_area = sum(h['area_m2'] for h in hotspots) / 1e6  # Convert to km²
            max_intensity = max(h['activity_intensity'] for h in hotspots)
            high_risk_count = sum(1 for h in hotspots if h.get('risk_level') == 'high')
        else:
            total_area = 0
            max_intensity = 0
            high_risk_count = 0
            
        return f"Activity Hotspots\n" \
               f"Total: {total_hotspots}\n" \
               f"Total Area: {total_area:.2f} km²\n" \
               f"Max Intensity: {max_intensity:.2f}\n" \
               f"High Risk: {high_risk_count}"
               
    def _format_temporal_statistics(self, temporal_analysis):
        """Format temporal statistics for display"""
        evolution = temporal_analysis.get('evolution_analysis', {})
        total_active_area = evolution.get('total_active_area_m2', 0) / 1e6  # Convert to km²
        activity_trend = evolution.get('activity_trend', 'unknown')
        confidence = evolution.get('evolution_confidence', 0)
        
        return f"Temporal Analysis\n" \
               f"Active Area: {total_active_area:.2f} km²\n" \
               f"Trend: {activity_trend.title()}\n" \
               f"Confidence: {confidence:.1%}"
               
    def _create_geojson(self, landslides, boulders):
        """Create GeoJSON export"""
        features = []
        
        # Add landslides
        for i, landslide in enumerate(landslides):
            if landslide.get('polygon'):
                # Convert polygon coordinates
                coordinates = [[[lon, lat] for lat, lon in landslide['polygon']]]
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": f"landslide_{i+1}",
                        "type": "landslide",
                        "area_km2": landslide.get('area_km2', 0),
                        "confidence": landslide.get('confidence', 50),
                        "detection_method": landslide.get('detection_method', 'unknown')
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    }
                }
                features.append(feature)
                
        # Add boulders
        for i, boulder in enumerate(boulders):
            feature = {
                "type": "Feature",
                "properties": {
                    "id": f"boulder_{i+1}",
                    "type": "boulder",
                    "diameter_m": boulder['diameter'],
                    "confidence": boulder.get('confidence', 50),
                    "detection_method": boulder.get('detection_method', 'unknown')
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [boulder['lon'], boulder['lat']]
                }
            }
            features.append(feature)
            
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::4326"
                }
            },
            "features": features
        }
        
    def _create_csv_export(self, landslides, boulders, output_path):
        """Create CSV export"""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Type', 'ID', 'Latitude', 'Longitude', 'Size', 'Unit', 
                           'Confidence', 'Detection_Method'])
            
            # Landslides
            for i, landslide in enumerate(landslides):
                writer.writerow(['Landslide', f'LS{i+1}', landslide['center'][0], 
                               landslide['center'][1], landslide.get('area_km2', 0), 
                               'km²', landslide.get('confidence', 50),
                               landslide.get('detection_method', 'unknown')])
                               
            # Boulders
            for i, boulder in enumerate(boulders):
                writer.writerow(['Boulder', f'B{i+1}', boulder['lat'], boulder['lon'],
                               boulder['diameter'], 'm', boulder.get('confidence', 50),
                               boulder.get('detection_method', 'unknown')])