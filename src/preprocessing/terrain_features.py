"""Terrain feature extraction module.

This module provides functionality for extracting various terrain features
from Digital Elevation Models including slope, curvature, flow accumulation,
and Height Above Nearest Drainage (HAND).
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional, Union
import warnings
from pathlib import Path


class TerrainFeatureExtractor:
    """Extracts terrain features for flood risk analysis.
    
    This class provides methods for computing various topographic indices
    and morphometric features from elevation data.
    """
    
    def __init__(self, cell_size: float = 30.0):
        """Initialize terrain feature extractor.
        
        Args:
            cell_size: DEM cell size in meters
        """
        self.cell_size = cell_size
    
    def compute_slope(self, elevation: np.ndarray, 
                     units: str = 'degrees') -> np.ndarray:
        """Compute slope from elevation data.
        
        Args:
            elevation: Input elevation array
            units: Output units ('degrees' or 'percent')
            
        Returns:
            Slope array in specified units
            
        Raises:
            ValueError: If invalid units specified
        """
        if units not in ['degrees', 'percent']:
            raise ValueError("Units must be 'degrees' or 'percent'")
        
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
        else:
            elev_data = elevation
        
        # Compute gradients using central differences
        dy, dx = np.gradient(elev_data, self.cell_size)
        
        # Compute slope magnitude
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
        
        if units == 'degrees':
            return np.degrees(slope_radians)
        else:  # percent
            return np.tan(slope_radians) * 100
    
    def compute_aspect(self, elevation: np.ndarray) -> np.ndarray:
        """Compute aspect (slope direction) from elevation data.
        
        Args:
            elevation: Input elevation array
            
        Returns:
            Aspect array in degrees (0-360, with 0 = North)
        """
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
        else:
            elev_data = elevation
        
        # Compute gradients
        dy, dx = np.gradient(elev_data, self.cell_size)
        
        # Compute aspect in radians
        aspect_radians = np.arctan2(-dx, dy)
        
        # Convert to degrees (0-360)
        aspect_degrees = np.degrees(aspect_radians)
        aspect_degrees = np.where(aspect_degrees < 0, 
                                aspect_degrees + 360, 
                                aspect_degrees)
        
        return aspect_degrees
    
    def compute_curvature(self, elevation: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various curvature measures from elevation data.
        
        Args:
            elevation: Input elevation array
            
        Returns:
            Dictionary containing:
            - profile_curvature: Curvature in direction of steepest slope
            - planform_curvature: Curvature perpendicular to steepest slope
            - mean_curvature: Mean curvature
            - gaussian_curvature: Gaussian curvature
        """
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
        else:
            elev_data = elevation
        
        # Smooth elevation to reduce noise
        smoothed = gaussian_filter(elev_data, sigma=0.5)
        
        # Compute first derivatives
        fy, fx = np.gradient(smoothed, self.cell_size)
        
        # Compute second derivatives
        fyy, fyx = np.gradient(fy, self.cell_size)
        fxy, fxx = np.gradient(fx, self.cell_size)
        
        # Ensure symmetry
        fxy = (fxy + fyx) / 2
        
        # Compute curvature components
        p = fx
        q = fy
        p2 = p**2
        q2 = q**2
        pq = p * q
        
        # Avoid division by zero
        denominator = p2 + q2
        small_slope_mask = denominator < 1e-10
        
        # Profile curvature (curvature in direction of steepest slope)
        profile_curvature = np.where(
            small_slope_mask,
            0,
            (fxx * p2 + 2 * fxy * pq + fyy * q2) / (denominator * np.sqrt(denominator + 1))
        )
        
        # Planform curvature (curvature perpendicular to steepest slope)
        planform_curvature = np.where(
            small_slope_mask,
            0,
            (fxx * q2 - 2 * fxy * pq + fyy * p2) / (denominator * np.sqrt(1 + denominator))
        )
        
        # Mean curvature
        mean_curvature = np.where(
            small_slope_mask,
            (fxx + fyy) / 2,
            ((1 + q2) * fxx - 2 * pq * fxy + (1 + p2) * fyy) / 
            (2 * (1 + p2 + q2)**(3/2))
        )
        
        # Gaussian curvature
        gaussian_curvature = np.where(
            small_slope_mask,
            fxx * fyy - fxy**2,
            (fxx * fyy - fxy**2) / (1 + p2 + q2)**2
        )
        
        return {
            'profile_curvature': profile_curvature,
            'planform_curvature': planform_curvature,
            'mean_curvature': mean_curvature,
            'gaussian_curvature': gaussian_curvature
        }
    
    def compute_topographic_wetness_index(self, flow_accumulation: np.ndarray,
                                        slope: np.ndarray) -> np.ndarray:
        """Compute Topographic Wetness Index (TWI).
        
        Args:
            flow_accumulation: Flow accumulation array
            slope: Slope array in radians
            
        Returns:
            Topographic Wetness Index array
        """
        # Convert slope to radians if in degrees
        if np.max(slope) > np.pi:  # Likely in degrees
            slope_rad = np.radians(slope)
        else:
            slope_rad = slope
        
        # Specific catchment area (flow accumulation * cell area / contour length)
        catchment_area = flow_accumulation * self.cell_size
        
        # Avoid division by zero and log of zero
        slope_rad = np.maximum(slope_rad, 1e-6)
        catchment_area = np.maximum(catchment_area, self.cell_size)
        
        # Compute TWI
        twi = np.log(catchment_area / np.tan(slope_rad))
        
        return twi
    
    def compute_stream_power_index(self, flow_accumulation: np.ndarray,
                                 slope: np.ndarray) -> np.ndarray:
        """Compute Stream Power Index (SPI).
        
        Args:
            flow_accumulation: Flow accumulation array
            slope: Slope array in radians
            
        Returns:
            Stream Power Index array
        """
        # Convert slope to radians if in degrees
        if np.max(slope) > np.pi:  # Likely in degrees
            slope_rad = np.radians(slope)
        else:
            slope_rad = slope
        
        # Specific catchment area
        catchment_area = flow_accumulation * self.cell_size
        
        # Compute SPI
        spi = catchment_area * np.tan(slope_rad)
        
        return spi
    
    def compute_hand(self, elevation: np.ndarray, 
                    streams: np.ndarray,
                    max_search_distance: float = 1000.0) -> np.ndarray:
        """Compute Height Above Nearest Drainage (HAND).
        
        Args:
            elevation: Input elevation array
            streams: Binary stream network array
            max_search_distance: Maximum search distance in meters
            
        Returns:
            HAND array (height above nearest drainage)
        """
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
            mask = elevation.mask
        else:
            elev_data = elevation
            mask = np.isnan(elev_data)
        
        # Find stream cells
        stream_cells = streams > 0
        
        if not np.any(stream_cells):
            warnings.warn("No stream cells found, HAND will be elevation")
            return elev_data.copy()
        
        # Initialize HAND array
        hand = np.full_like(elev_data, np.inf)
        
        # Convert max search distance to pixels
        max_pixels = int(max_search_distance / self.cell_size)
        
        # Get stream coordinates
        stream_coords = np.where(stream_cells)
        stream_elevations = elev_data[stream_coords]
        
        # For each cell, find nearest drainage
        rows, cols = elev_data.shape
        
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    hand[i, j] = np.nan
                    continue
                
                cell_elevation = elev_data[i, j]
                min_hand = np.inf
                
                # Search within max distance
                for si, sj, se in zip(stream_coords[0], stream_coords[1], stream_elevations):
                    # Calculate distance
                    distance = np.sqrt((i - si)**2 + (j - sj)**2)
                    
                    if distance <= max_pixels:
                        # Calculate height above drainage
                        height_diff = cell_elevation - se
                        
                        if height_diff >= 0 and height_diff < min_hand:
                            min_hand = height_diff
                
                hand[i, j] = min_hand if min_hand != np.inf else np.nan
        
        return hand
    
    def compute_roughness(self, elevation: np.ndarray, 
                         window_size: int = 3) -> np.ndarray:
        """Compute terrain roughness using standard deviation.
        
        Args:
            elevation: Input elevation array
            window_size: Size of moving window (must be odd)
            
        Returns:
            Terrain roughness array
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
        else:
            elev_data = elevation
        
        # Compute local standard deviation
        from scipy.ndimage import generic_filter
        
        def local_std(values):
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                return np.nan
            return np.std(valid_values)
        
        roughness = generic_filter(elev_data, local_std, size=window_size)
        
        return roughness
    
    def compute_terrain_ruggedness_index(self, elevation: np.ndarray) -> np.ndarray:
        """Compute Terrain Ruggedness Index (TRI).
        
        Args:
            elevation: Input elevation array
            
        Returns:
            Terrain Ruggedness Index array
        """
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
        else:
            elev_data = elevation
        
        # Define 3x3 kernel for neighbors
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        # Apply convolution to get TRI
        tri = np.abs(ndimage.convolve(elev_data, kernel, mode='constant', cval=0))
        
        return tri
    
    def extract_all_features(self, elevation: np.ndarray,
                           flow_accumulation: Optional[np.ndarray] = None,
                           streams: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Extract all terrain features from elevation data.
        
        Args:
            elevation: Input elevation array
            flow_accumulation: Optional flow accumulation array
            streams: Optional binary stream network
            
        Returns:
            Dictionary containing all computed terrain features
        """
        features = {}
        
        # Basic slope and aspect
        features['slope_degrees'] = self.compute_slope(elevation, 'degrees')
        features['slope_percent'] = self.compute_slope(elevation, 'percent')
        features['aspect'] = self.compute_aspect(elevation)
        
        # Curvature measures
        curvatures = self.compute_curvature(elevation)
        features.update(curvatures)
        
        # Roughness measures
        features['roughness'] = self.compute_roughness(elevation)
        features['tri'] = self.compute_terrain_ruggedness_index(elevation)
        
        # Flow-based indices (if flow accumulation provided)
        if flow_accumulation is not None:
            slope_rad = np.radians(features['slope_degrees'])
            features['twi'] = self.compute_topographic_wetness_index(
                flow_accumulation, slope_rad)
            features['spi'] = self.compute_stream_power_index(
                flow_accumulation, slope_rad)
        
        # HAND (if streams provided)
        if streams is not None:
            features['hand'] = self.compute_hand(elevation, streams)
        
        return features
    
    def save_features(self, features: Dict[str, np.ndarray],
                     profile: Dict,
                     output_dir: Union[str, Path]) -> None:
        """Save terrain features to raster files.
        
        Args:
            features: Dictionary of feature arrays
            profile: Rasterio profile for output
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, array in features.items():
            # Update profile
            out_profile = profile.copy()
            out_profile.update(dtype='float32', nodata=-9999)
            
            # Save feature
            output_file = output_dir / f"{name}.tif"
            
            with rasterio.open(output_file, 'w', **out_profile) as dst:
                # Handle masked arrays and NaN values
                if hasattr(array, 'mask'):
                    data = array.filled(out_profile['nodata'])
                else:
                    data = np.where(np.isnan(array), out_profile['nodata'], array)
                
                dst.write(data.astype(np.float32), 1)
    
    def validate_features(self, features: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate computed terrain features.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Dictionary of validation results
        """
        validation = {}
        
        # Check slope ranges
        if 'slope_degrees' in features:
            slope = features['slope_degrees']
            validation['slope_range_valid'] = (
                np.nanmin(slope) >= 0 and np.nanmax(slope) <= 90
            )
        
        # Check aspect ranges
        if 'aspect' in features:
            aspect = features['aspect']
            validation['aspect_range_valid'] = (
                np.nanmin(aspect) >= 0 and np.nanmax(aspect) <= 360
            )
        
        # Check for reasonable TWI values
        if 'twi' in features:
            twi = features['twi']
            validation['twi_reasonable'] = (
                np.nanmin(twi) > -20 and np.nanmax(twi) < 30
            )
        
        # Check HAND values
        if 'hand' in features:
            hand = features['hand']
            validation['hand_positive'] = np.all(hand[~np.isnan(hand)] >= 0)
        
        return validation