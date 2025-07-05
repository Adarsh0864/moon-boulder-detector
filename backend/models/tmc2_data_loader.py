"""
TMC-2 Data Loader for Chandrayaan-2 Terrain Mapping Camera
Handles real TMC-2 ortho images and DTM data integration
"""

import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os
import json

logger = logging.getLogger(__name__)

class TMC2DataLoader:
    """
    Loads and processes Chandrayaan-2 TMC data including:
    - Ortho images (OTH) - Radiometrically and geometrically corrected
    - Digital Terrain Models (DTM) - Height data
    - Metadata extraction from XML files
    """
    
    def __init__(self, data_path: str = "/Users/adarshmishra/Developer/isro/lunar-geodetect/data"):
        self.data_path = data_path
        self.datasets = self._discover_datasets()
        
    def _discover_datasets(self) -> Dict[str, Dict[str, str]]:
        """Discover available TMC-2 datasets"""
        datasets = {}
        
        # Check for extracted data
        data_dir = os.path.join(self.data_path, "data", "derived")
        if os.path.exists(data_dir):
            for date_folder in os.listdir(data_dir):
                date_path = os.path.join(data_dir, date_folder)
                if os.path.isdir(date_path):
                    dataset_info = {
                        'date': date_folder,
                        'path': date_path,
                        'files': {}
                    }
                    
                    # Find OTH and DTM files
                    for file in os.listdir(date_path):
                        if file.endswith('.tif'):
                            if '_d_oth_' in file:
                                dataset_info['files']['ortho'] = os.path.join(date_path, file)
                                dataset_info['files']['ortho_xml'] = os.path.join(date_path, file.replace('.tif', '.xml'))
                            elif '_d_dtm_' in file:
                                dataset_info['files']['dtm'] = os.path.join(date_path, file)
                                dataset_info['files']['dtm_xml'] = os.path.join(date_path, file.replace('.tif', '.xml'))
                    
                    if dataset_info['files']:
                        datasets[date_folder] = dataset_info
                        
        logger.info(f"Discovered {len(datasets)} TMC-2 datasets: {list(datasets.keys())}")
        return datasets
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get formatted dataset information for frontend"""
        formatted_datasets = {}
        
        for date, dataset in self.datasets.items():
            # Extract metadata if available
            metadata = {}
            if 'ortho_xml' in dataset['files']:
                metadata = self.extract_metadata(dataset['files']['ortho_xml'])
            
            formatted_datasets[date] = {
                'id': f'tmc2_{date}',
                'name': f'TMC-2 {date}',
                'description': f'Chandrayaan-2 TMC data from {date}',
                'mission': 'Chandrayaan-2 TMC',
                'date': date,
                'resolution': '5m/pixel',
                'has_ortho': 'ortho' in dataset['files'],
                'has_dtm': 'dtm' in dataset['files'],
                'metadata': metadata,
                'features': ['Real Data', 'TMC-2', 'DTM Available'] if 'dtm' in dataset['files'] else ['Real Data', 'TMC-2']
            }
            
        return formatted_datasets
    
    def extract_metadata(self, xml_path: str) -> Dict[str, Any]:
        """Extract metadata from TMC-2 XML file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Define namespace
            ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
            
            metadata = {}
            
            # Time coordinates
            time_coords = root.find('.//pds:Time_Coordinates', ns)
            if time_coords is not None:
                start_time = time_coords.find('pds:start_date_time', ns)
                stop_time = time_coords.find('pds:stop_date_time', ns)
                if start_time is not None:
                    metadata['start_time'] = start_time.text
                if stop_time is not None:
                    metadata['stop_time'] = stop_time.text
            
            # Array 2D Image details
            array_2d = root.find('.//pds:Array_2D_Image', ns)
            if array_2d is not None:
                axes = array_2d.findall('.//pds:Axis_Array', ns)
                for axis in axes:
                    axis_name = axis.find('pds:axis_name', ns)
                    elements = axis.find('pds:elements', ns)
                    if axis_name is not None and elements is not None:
                        if axis_name.text.lower() == 'line':
                            metadata['lines'] = int(elements.text)
                        elif axis_name.text.lower() == 'sample':
                            metadata['samples'] = int(elements.text)
            
            # Target identification (Moon)
            target = root.find('.//pds:Target_Identification', ns)
            if target is not None:
                target_name = target.find('pds:name', ns)
                if target_name is not None:
                    metadata['target'] = target_name.text
            
            # Instrument information
            instrument = root.find('.//pds:Observing_System_Component[pds:type="Instrument"]', ns)
            if instrument is not None:
                inst_name = instrument.find('pds:name', ns)
                if inst_name is not None:
                    metadata['instrument'] = inst_name.text
            
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {xml_path}: {e}")
            return {}
    
    def load_ortho_image(self, dataset_date: str, max_size: Optional[Tuple[int, int]] = (800, 800)) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load TMC-2 ortho image with optional downsampling for processing
        
        Args:
            dataset_date: Date identifier for dataset
            max_size: Maximum dimensions for downsampling (width, height)
            
        Returns:
            Tuple of (image_array, metadata)
        """
        if dataset_date not in self.datasets:
            raise ValueError(f"Dataset {dataset_date} not found")
        
        dataset = self.datasets[dataset_date]
        if 'ortho' not in dataset['files']:
            raise ValueError(f"No ortho image found for {dataset_date}")
        
        ortho_path = dataset['files']['ortho']
        logger.info(f"Loading ortho image: {ortho_path}")
        
        try:
            # Use rasterio for geospatial TIF files
            with rasterio.open(ortho_path) as src:
                # Get image metadata
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs) if src.crs else None,
                    'transform': list(src.transform),
                    'bounds': src.bounds
                }
                
                # Calculate downsampling if needed
                if max_size and (src.width > max_size[0] or src.height > max_size[1]):
                    scale_x = max_size[0] / src.width
                    scale_y = max_size[1] / src.height
                    scale = min(scale_x, scale_y)
                    
                    new_width = int(src.width * scale)
                    new_height = int(src.height * scale)
                    
                    # Ensure even dimensions for processing
                    if new_width % 2 != 0:
                        new_width += 1
                    if new_height % 2 != 0:
                        new_height += 1
                    
                    # Read with resampling
                    image = src.read(
                        out_shape=(src.count, new_height, new_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    
                    metadata['downsampled'] = True
                    metadata['original_size'] = (src.width, src.height)
                    metadata['current_size'] = (new_width, new_height)
                    metadata['scale_factor'] = scale
                    
                else:
                    # Read at full resolution
                    image = src.read()
                    metadata['downsampled'] = False
                    metadata['scale_factor'] = 1.0
                
                # Convert to numpy array and handle single band
                if image.shape[0] == 1:
                    image = image[0]  # Remove band dimension for single band
                
                # Normalize to 0-255 range for processing
                image_normalized = self._normalize_image(image)
                
                logger.info(f"Loaded image with shape: {image_normalized.shape}")
                return image_normalized, metadata
                
        except Exception as e:
            logger.error(f"Error loading ortho image: {e}")
            raise
    
    def load_dtm_data(self, dataset_date: str, max_size: Optional[Tuple[int, int]] = (800, 800)) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load TMC-2 DTM (Digital Terrain Model) data"""
        if dataset_date not in self.datasets:
            raise ValueError(f"Dataset {dataset_date} not found")
        
        dataset = self.datasets[dataset_date]
        if 'dtm' not in dataset['files']:
            raise ValueError(f"No DTM data found for {dataset_date}")
        
        dtm_path = dataset['files']['dtm']
        logger.info(f"Loading DTM data: {dtm_path}")
        
        try:
            with rasterio.open(dtm_path) as src:
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs) if src.crs else None,
                    'transform': list(src.transform),
                    'bounds': src.bounds,
                    'nodata': src.nodata
                }
                
                # Handle downsampling similar to ortho
                if max_size and (src.width > max_size[0] or src.height > max_size[1]):
                    scale_x = max_size[0] / src.width
                    scale_y = max_size[1] / src.height
                    scale = min(scale_x, scale_y)
                    
                    new_width = int(src.width * scale)
                    new_height = int(src.height * scale)
                    
                    # Ensure even dimensions for processing
                    if new_width % 2 != 0:
                        new_width += 1
                    if new_height % 2 != 0:
                        new_height += 1
                    
                    dtm_data = src.read(
                        1,  # Read first band
                        out_shape=(new_height, new_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    
                    metadata['downsampled'] = True
                    metadata['original_size'] = (src.width, src.height)
                    metadata['current_size'] = (new_width, new_height)
                    metadata['scale_factor'] = scale
                    
                else:
                    dtm_data = src.read(1)
                    metadata['downsampled'] = False
                    metadata['scale_factor'] = 1.0
                
                # Calculate elevation statistics
                valid_data = dtm_data[dtm_data != src.nodata] if src.nodata else dtm_data
                if len(valid_data) > 0:
                    metadata['elevation_stats'] = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data))
                    }
                
                logger.info(f"Loaded DTM with shape: {dtm_data.shape}")
                return dtm_data, metadata
                
        except Exception as e:
            logger.error(f"Error loading DTM data: {e}")
            raise
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        if image.dtype == np.uint8:
            return image
        
        # Handle different data types
        if image.dtype in [np.uint16, np.int16, np.int32, np.float32, np.float64]:
            # Normalize to 0-1 range first
            img_min, img_max = np.min(image), np.max(image)
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(image, dtype=np.float32)
            
            # Convert to 0-255 range
            return (normalized * 255).astype(np.uint8)
        
        return image.astype(np.uint8)
    
    def get_dataset_info(self, dataset_date: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific dataset"""
        if dataset_date not in self.datasets:
            raise ValueError(f"Dataset {dataset_date} not found")
        
        dataset = self.datasets[dataset_date]
        info = {
            'date': dataset_date,
            'available_files': list(dataset['files'].keys()),
            'file_paths': dataset['files']
        }
        
        # Add metadata if XML files exist
        if 'ortho_xml' in dataset['files']:
            info['ortho_metadata'] = self.extract_metadata(dataset['files']['ortho_xml'])
        
        if 'dtm_xml' in dataset['files']:
            info['dtm_metadata'] = self.extract_metadata(dataset['files']['dtm_xml'])
        
        return info
    
    def prepare_for_analysis(self, dataset_date: str) -> Dict[str, Any]:
        """
        Prepare TMC-2 dataset for analysis by novel algorithms
        Returns data in format expected by existing detection algorithms
        """
        try:
            # Load ortho image
            ortho_image, ortho_metadata = self.load_ortho_image(dataset_date)
            
            # Load DTM if available
            dtm_data = None
            dtm_metadata = None
            if 'dtm' in self.datasets[dataset_date]['files']:
                dtm_data, dtm_metadata = self.load_dtm_data(dataset_date)
            
            # Extract XML metadata
            xml_metadata = {}
            if 'ortho_xml' in self.datasets[dataset_date]['files']:
                xml_metadata = self.extract_metadata(self.datasets[dataset_date]['files']['ortho_xml'])
            
            # Prepare analysis package
            analysis_data = {
                'image': ortho_image,
                'dtm': dtm_data,
                'metadata': {
                    'mission': 'Chandrayaan-2',
                    'instrument': 'TMC-2',
                    'date': dataset_date,
                    'dataset_type': 'real_data',
                    'ortho_metadata': ortho_metadata,
                    'dtm_metadata': dtm_metadata,
                    'xml_metadata': xml_metadata,
                    'coordinates': self._extract_coordinates(xml_metadata),
                    'solar_geometry': self._extract_solar_geometry(xml_metadata)
                }
            }
            
            logger.info(f"Prepared TMC-2 dataset {dataset_date} for analysis")
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error preparing dataset {dataset_date}: {e}")
            raise
    
    def _extract_coordinates(self, xml_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract coordinate information from metadata"""
        # For now, return mock coordinates - would need to parse from XML geospatial info
        return {
            'center_lat': 0.0,
            'center_lon': 0.0,
            'bounds': {
                'north': 0.0,
                'south': 0.0, 
                'east': 0.0,
                'west': 0.0
            }
        }
    
    def _extract_solar_geometry(self, xml_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract solar geometry for shadow analysis"""
        # Would need to calculate from time and coordinates
        return {
            'elevation': 45.0,  # degrees
            'azimuth': 135.0,   # degrees
            'phase_angle': 30.0  # degrees
        }

if __name__ == "__main__":
    # Test the loader
    loader = TMC2DataLoader()
    datasets = loader.get_available_datasets()
    print(f"Available datasets: {list(datasets.keys())}")
    
    if datasets:
        first_dataset = list(datasets.keys())[0]
        info = loader.get_dataset_info(first_dataset)
        print(f"Dataset info: {info}")