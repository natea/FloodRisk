"""Digital Elevation Model (DEM) preprocessing module.

This module provides functionality for hydrological conditioning of DEMs,
including sink filling, flow direction computation, and stream network delineation.
"""

import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label, binary_erosion
from typing import Tuple, Optional, Union, Dict, Any
import warnings
from pathlib import Path


class DEMProcessor:
    """Digital Elevation Model processor for hydrological conditioning.
    
    This class provides methods for preprocessing DEMs for flood risk analysis,
    including sink filling, flow direction calculation, and hydrological corrections.
    """
    
    def __init__(self, fill_sinks: bool = True, min_drainage_area: float = 1000.0):
        """Initialize DEM processor.
        
        Args:
            fill_sinks: Whether to fill sinks in DEM
            min_drainage_area: Minimum drainage area in square meters for stream delineation
        """
        self.fill_sinks = fill_sinks
        self.min_drainage_area = min_drainage_area
        
        # D8 flow direction kernel (8-connected neighbors)
        self.d8_directions = np.array([
            [-1, -1], [-1,  0], [-1,  1],
            [ 0, -1],           [ 0,  1],
            [ 1, -1], [ 1,  0], [ 1,  1]
        ])
        
        # D8 flow direction codes
        self.d8_codes = np.array([32, 64, 128, 16, 1, 8, 4, 2])
    
    def load_dem(self, dem_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DEM from file.
        
        Args:
            dem_path: Path to DEM file
            
        Returns:
            Tuple of (elevation array, rasterio profile)
            
        Raises:
            FileNotFoundError: If DEM file doesn't exist
            ValueError: If DEM data is invalid
        """
        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        try:
            with rasterio.open(dem_path) as src:
                elevation = src.read(1, masked=True)
                profile = src.profile.copy()
                
                # Validate elevation data
                if elevation.size == 0:
                    raise ValueError("DEM contains no valid data")
                    
                if np.all(elevation.mask):
                    raise ValueError("DEM is completely masked")
                
                return elevation, profile
                
        except Exception as e:
            raise ValueError(f"Failed to load DEM: {str(e)}")
    
    def fill_depressions(self, elevation: np.ndarray, 
                        max_iterations: int = 1000) -> np.ndarray:
        """Fill depressions (sinks) in DEM using priority flood algorithm.
        
        Args:
            elevation: Input elevation array
            max_iterations: Maximum iterations for filling algorithm
            
        Returns:
            Depression-filled elevation array
        """
        if not self.fill_sinks:
            return elevation.copy()
        
        # Convert masked array to regular array with no-data handling
        if hasattr(elevation, 'mask'):
            filled_elev = elevation.filled(np.nan)
            mask = elevation.mask
        else:
            filled_elev = elevation.copy()
            mask = np.isnan(filled_elev)
        
        # Create output array
        result = filled_elev.copy()
        
        # Simple depression filling using morphological operations
        # This is a simplified approach - more sophisticated algorithms exist
        kernel = np.ones((3, 3))
        
        for _ in range(max_iterations):
            # Dilate the image (expand high values)
            dilated = ndimage.grey_dilation(result, footprint=kernel)
            
            # Only allow increases (filling depressions)
            new_result = np.maximum(result, 
                                  np.minimum(dilated, filled_elev + 1e-6))
            
            # Check for convergence
            if np.allclose(result[~mask], new_result[~mask], rtol=1e-6):
                break
                
            result = new_result
        
        # Restore original values where no depression filling was needed
        result = np.where(mask, np.nan, result)
        
        return result
    
    def compute_flow_direction(self, elevation: np.ndarray) -> np.ndarray:
        """Compute D8 flow direction from DEM.
        
        Args:
            elevation: Input elevation array
            
        Returns:
            Flow direction array using D8 encoding
        """
        rows, cols = elevation.shape
        flow_dir = np.zeros((rows, cols), dtype=np.uint8)
        
        # Handle masked arrays
        if hasattr(elevation, 'mask'):
            elev_data = elevation.filled(np.nan)
            mask = elevation.mask
        else:
            elev_data = elevation
            mask = np.isnan(elev_data)
        
        # Compute flow direction for each cell
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if mask[i, j]:
                    continue
                    
                center_elev = elev_data[i, j]
                max_slope = -np.inf
                flow_direction = 0
                
                # Check all 8 neighbors
                for k, (di, dj) in enumerate(self.d8_directions):
                    ni, nj = i + di, j + dj
                    
                    if mask[ni, nj]:
                        continue
                    
                    neighbor_elev = elev_data[ni, nj]
                    
                    # Calculate slope (considering diagonal distance)
                    distance = np.sqrt(di**2 + dj**2)
                    slope = (center_elev - neighbor_elev) / distance
                    
                    if slope > max_slope:
                        max_slope = slope
                        flow_direction = self.d8_codes[k]
                
                flow_dir[i, j] = flow_direction
        
        return flow_dir
    
    def compute_flow_accumulation(self, flow_direction: np.ndarray) -> np.ndarray:
        """Compute flow accumulation from flow direction.
        
        Args:
            flow_direction: D8 flow direction array
            
        Returns:
            Flow accumulation array (number of upstream cells)
        """
        rows, cols = flow_direction.shape
        flow_acc = np.ones((rows, cols), dtype=np.float32)
        
        # Create a processing order (downstream to upstream)
        # This is a simplified approach - topological sorting would be better
        for iteration in range(rows * cols):
            updated = False
            
            for i in range(rows):
                for j in range(cols):
                    if flow_direction[i, j] == 0:
                        continue
                    
                    # Find downstream cell
                    flow_code = flow_direction[i, j]
                    direction_idx = np.where(self.d8_codes == flow_code)[0]
                    
                    if len(direction_idx) == 0:
                        continue
                    
                    di, dj = self.d8_directions[direction_idx[0]]
                    downstream_i, downstream_j = i + di, j + dj
                    
                    # Check bounds
                    if (0 <= downstream_i < rows and 0 <= downstream_j < cols):
                        flow_acc[downstream_i, downstream_j] += flow_acc[i, j]
                        updated = True
            
            if not updated:
                break
        
        return flow_acc
    
    def extract_stream_network(self, flow_accumulation: np.ndarray,
                             cell_size: float) -> np.ndarray:
        """Extract stream network based on flow accumulation threshold.
        
        Args:
            flow_accumulation: Flow accumulation array
            cell_size: Cell size in meters
            
        Returns:
            Binary stream network array
        """
        # Convert drainage area threshold to number of cells
        cell_area = cell_size ** 2
        threshold_cells = self.min_drainage_area / cell_area
        
        # Create binary stream network
        streams = flow_accumulation >= threshold_cells
        
        return streams.astype(np.uint8)
    
    def condition_dem(self, elevation: np.ndarray, 
                     streams: Optional[np.ndarray] = None,
                     stream_depth: float = 1.0) -> np.ndarray:
        """Apply hydrological conditioning to DEM.
        
        Args:
            elevation: Input elevation array
            streams: Optional binary stream network
            stream_depth: Depth to lower stream cells (meters)
            
        Returns:
            Hydrologically conditioned DEM
        """
        conditioned = elevation.copy()
        
        if streams is not None:
            # Lower stream cells to enforce drainage
            conditioned = np.where(streams, 
                                 conditioned - stream_depth, 
                                 conditioned)
        
        return conditioned
    
    def process_dem(self, dem_path: Union[str, Path], 
                   output_path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
        """Complete DEM processing pipeline.
        
        Args:
            dem_path: Input DEM file path
            output_path: Optional output directory for processed files
            
        Returns:
            Dictionary containing processed arrays:
            - elevation: Original elevation
            - filled_elevation: Depression-filled elevation
            - flow_direction: D8 flow direction
            - flow_accumulation: Flow accumulation
            - streams: Stream network
            - conditioned_elevation: Hydrologically conditioned DEM
        """
        # Load DEM
        elevation, profile = self.load_dem(dem_path)
        
        # Get cell size from transform
        cell_size = abs(profile['transform'][0])
        
        # Fill depressions
        filled_elevation = self.fill_depressions(elevation)
        
        # Compute flow direction
        flow_direction = self.compute_flow_direction(filled_elevation)
        
        # Compute flow accumulation
        flow_accumulation = self.compute_flow_accumulation(flow_direction)
        
        # Extract stream network
        streams = self.extract_stream_network(flow_accumulation, cell_size)
        
        # Apply hydrological conditioning
        conditioned_elevation = self.condition_dem(filled_elevation, streams)
        
        results = {
            'elevation': elevation,
            'filled_elevation': filled_elevation,
            'flow_direction': flow_direction,
            'flow_accumulation': flow_accumulation,
            'streams': streams,
            'conditioned_elevation': conditioned_elevation
        }
        
        # Save results if output path provided
        if output_path is not None:
            self._save_results(results, profile, output_path)
        
        return results
    
    def _save_results(self, results: Dict[str, np.ndarray], 
                     profile: Dict[str, Any],
                     output_path: Union[str, Path]) -> None:
        """Save processing results to files.
        
        Args:
            results: Dictionary of result arrays
            profile: Rasterio profile
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, array in results.items():
            # Update profile for different data types
            out_profile = profile.copy()
            
            if name == 'flow_direction':
                out_profile.update(dtype='uint8', nodata=0)
            elif name == 'streams':
                out_profile.update(dtype='uint8', nodata=0)
            else:
                out_profile.update(dtype='float32', nodata=-9999)
            
            # Save to file
            output_file = output_path / f"{name}.tif"
            
            with rasterio.open(output_file, 'w', **out_profile) as dst:
                # Handle masked arrays
                if hasattr(array, 'mask'):
                    data = array.filled(out_profile['nodata'])
                else:
                    data = np.where(np.isnan(array), out_profile['nodata'], array)
                
                dst.write(data, 1)
    
    def validate_processing(self, results: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate processing results.
        
        Args:
            results: Dictionary of processing results
            
        Returns:
            Dictionary of validation results
        """
        validation = {}
        
        # Check if depression filling increased elevation values
        elev_diff = results['filled_elevation'] - results['elevation']
        validation['depression_filling_valid'] = np.all(elev_diff >= -1e-6)
        
        # Check if flow accumulation is reasonable
        flow_acc = results['flow_accumulation']
        validation['flow_accumulation_valid'] = (
            np.all(flow_acc >= 1) and 
            np.max(flow_acc) <= flow_acc.size
        )
        
        # Check stream network connectivity
        streams = results['streams']
        validation['streams_exist'] = np.any(streams > 0)
        
        # Check conditioned elevation
        cond_diff = results['conditioned_elevation'] - results['filled_elevation']
        validation['conditioning_valid'] = np.all(cond_diff <= 1e-6)
        
        return validation