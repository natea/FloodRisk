"""Multi-scale patch extraction module.

This module provides functionality for extracting patches from raster data
at multiple scales (256m, 512m, 1024m) for machine learning applications.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from typing import List, Tuple, Dict, Optional, Union, Generator, Any
from pathlib import Path
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


@dataclass
class PatchInfo:
    """Information about an extracted patch."""
    patch_id: str
    center_x: float
    center_y: float
    patch_size_m: int
    patch_size_pixels: int
    window: Window
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    transform: Any
    crs: str


class PatchExtractor:
    """Extracts multi-scale patches from raster data for ML training.
    
    This class provides functionality to extract patches at different scales
    from input raster data, with support for overlapping patches, data validation,
    and batch processing.
    """
    
    def __init__(self, patch_sizes_m: List[int] = [256, 512, 1024],
                 overlap_ratio: float = 0.0,
                 min_valid_pixels: float = 0.8):
        """Initialize patch extractor.
        
        Args:
            patch_sizes_m: List of patch sizes in meters
            overlap_ratio: Overlap ratio between adjacent patches (0.0 = no overlap)
            min_valid_pixels: Minimum fraction of valid (non-nodata) pixels required
        """
        self.patch_sizes_m = patch_sizes_m
        self.overlap_ratio = overlap_ratio
        self.min_valid_pixels = min_valid_pixels
        
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("Overlap ratio must be in range [0, 1)")
        
        if not 0.0 <= min_valid_pixels <= 1.0:
            raise ValueError("Min valid pixels must be in range [0, 1]")
    
    def _get_patch_size_pixels(self, patch_size_m: int, cell_size: float) -> int:
        """Convert patch size from meters to pixels.
        
        Args:
            patch_size_m: Patch size in meters
            cell_size: Raster cell size in meters
            
        Returns:
            Patch size in pixels
        """
        return int(np.round(patch_size_m / cell_size))
    
    def _get_step_size(self, patch_size_pixels: int) -> int:
        """Calculate step size between patches based on overlap.
        
        Args:
            patch_size_pixels: Patch size in pixels
            
        Returns:
            Step size in pixels
        """
        return int(patch_size_pixels * (1 - self.overlap_ratio))
    
    def _is_patch_valid(self, data: np.ndarray, nodata_value: Optional[float] = None) -> bool:
        """Check if patch has sufficient valid data.
        
        Args:
            data: Patch data array
            nodata_value: No-data value to check for
            
        Returns:
            True if patch has sufficient valid data
        """
        if nodata_value is not None:
            valid_mask = data != nodata_value
        else:
            valid_mask = ~np.isnan(data)
        
        valid_fraction = np.sum(valid_mask) / data.size
        return valid_fraction >= self.min_valid_pixels
    
    def _generate_patch_grid(self, raster_height: int, raster_width: int,
                           patch_size_pixels: int) -> List[Tuple[int, int]]:
        """Generate grid of patch starting positions.
        
        Args:
            raster_height: Height of input raster in pixels
            raster_width: Width of input raster in pixels
            patch_size_pixels: Patch size in pixels
            
        Returns:
            List of (row, col) starting positions
        """
        step_size = self._get_step_size(patch_size_pixels)
        
        positions = []
        
        # Generate grid positions
        for row in range(0, raster_height - patch_size_pixels + 1, step_size):
            for col in range(0, raster_width - patch_size_pixels + 1, step_size):
                positions.append((row, col))
        
        return positions
    
    def _create_patch_info(self, row: int, col: int, patch_size_pixels: int,
                          patch_size_m: int, transform: Any, crs: str) -> PatchInfo:
        """Create patch information object.
        
        Args:
            row: Starting row position
            col: Starting column position
            patch_size_pixels: Patch size in pixels
            patch_size_m: Patch size in meters
            transform: Raster transform
            crs: Coordinate reference system
            
        Returns:
            PatchInfo object
        """
        # Create window
        window = Window(col, row, patch_size_pixels, patch_size_pixels)
        
        # Calculate bounds
        minx, maxy = transform * (col, row)
        maxx, miny = transform * (col + patch_size_pixels, row + patch_size_pixels)
        bounds = (minx, miny, maxx, maxy)
        
        # Calculate center coordinates
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        
        # Create patch transform
        patch_transform = from_bounds(minx, miny, maxx, maxy, 
                                     patch_size_pixels, patch_size_pixels)
        
        # Generate patch ID
        patch_id = f"patch_{patch_size_m}m_{row}_{col}"
        
        return PatchInfo(
            patch_id=patch_id,
            center_x=center_x,
            center_y=center_y,
            patch_size_m=patch_size_m,
            patch_size_pixels=patch_size_pixels,
            window=window,
            bounds=bounds,
            transform=patch_transform,
            crs=crs
        )
    
    def extract_patches_from_file(self, raster_path: Union[str, Path],
                                 patch_size_m: int,
                                 output_dir: Optional[Union[str, Path]] = None,
                                 save_patches: bool = True) -> List[Dict[str, Any]]:
        """Extract patches from a single raster file.
        
        Args:
            raster_path: Path to input raster file
            patch_size_m: Patch size in meters
            output_dir: Directory to save patches (if save_patches=True)
            save_patches: Whether to save patches to files
            
        Returns:
            List of patch dictionaries containing data and metadata
        """
        raster_path = Path(raster_path)
        
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        patches = []
        
        with rasterio.open(raster_path) as src:
            # Get raster properties
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            cell_size = abs(transform[0])
            
            # Convert patch size to pixels
            patch_size_pixels = self._get_patch_size_pixels(patch_size_m, cell_size)
            
            if patch_size_pixels >= min(height, width):
                warnings.warn(f"Patch size ({patch_size_pixels}px) is larger than raster dimensions ({height}x{width}px)")
                return patches
            
            # Generate patch grid
            positions = self._generate_patch_grid(height, width, patch_size_pixels)
            
            # Setup output directory
            if save_patches and output_dir is not None:
                output_dir = Path(output_dir)
                patch_dir = output_dir / f"patches_{patch_size_m}m"
                patch_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract patches
            for i, (row, col) in enumerate(positions):
                # Create patch info
                patch_info = self._create_patch_info(
                    row, col, patch_size_pixels, patch_size_m, transform, str(crs)
                )
                
                # Read patch data
                try:
                    patch_data = src.read(window=patch_info.window)
                    
                    # Validate patch
                    if not self._is_patch_valid(patch_data, nodata):
                        continue
                    
                    # Create patch dictionary
                    patch_dict = {
                        'data': patch_data,
                        'info': patch_info,
                        'source_file': str(raster_path),
                        'band_count': src.count,
                        'dtype': str(patch_data.dtype),
                        'nodata': nodata
                    }
                    
                    patches.append(patch_dict)
                    
                    # Save patch if requested
                    if save_patches and output_dir is not None:
                        self._save_patch(patch_dict, patch_dir)
                        
                except Exception as e:
                    warnings.warn(f"Failed to extract patch at ({row}, {col}): {str(e)}")
                    continue
        
        return patches
    
    def extract_multiscale_patches(self, raster_path: Union[str, Path],
                                  output_dir: Optional[Union[str, Path]] = None,
                                  save_patches: bool = True) -> Dict[int, List[Dict[str, Any]]]:
        """Extract patches at all configured scales from a raster file.
        
        Args:
            raster_path: Path to input raster file
            output_dir: Directory to save patches (if save_patches=True)
            save_patches: Whether to save patches to files
            
        Returns:
            Dictionary mapping patch sizes to lists of patch dictionaries
        """
        multiscale_patches = {}
        
        for patch_size_m in self.patch_sizes_m:
            print(f"Extracting {patch_size_m}m patches...")
            
            patches = self.extract_patches_from_file(
                raster_path=raster_path,
                patch_size_m=patch_size_m,
                output_dir=output_dir,
                save_patches=save_patches
            )
            
            multiscale_patches[patch_size_m] = patches
            print(f"Extracted {len(patches)} patches at {patch_size_m}m scale")
        
        return multiscale_patches
    
    def extract_patches_batch(self, raster_paths: List[Union[str, Path]],
                            output_dir: Union[str, Path],
                            max_workers: int = 4) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """Extract patches from multiple raster files in parallel.
        
        Args:
            raster_paths: List of paths to input raster files
            output_dir: Directory to save patches
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping file names to multiscale patch dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        def process_file(raster_path):
            file_name = Path(raster_path).stem
            file_output_dir = output_dir / file_name
            
            try:
                patches = self.extract_multiscale_patches(
                    raster_path=raster_path,
                    output_dir=file_output_dir,
                    save_patches=True
                )
                return file_name, patches
            except Exception as e:
                warnings.warn(f"Failed to process {raster_path}: {str(e)}")
                return file_name, {}
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, path): path for path in raster_paths}
            
            for future in as_completed(futures):
                file_name, patches = future.result()
                results[file_name] = patches
        
        return results
    
    def _save_patch(self, patch_dict: Dict[str, Any], output_dir: Path) -> None:
        """Save a single patch to file.
        
        Args:
            patch_dict: Patch dictionary
            output_dir: Output directory
        """
        patch_info = patch_dict['info']
        patch_data = patch_dict['data']
        
        # Create output profile
        profile = {
            'driver': 'GTiff',
            'height': patch_info.patch_size_pixels,
            'width': patch_info.patch_size_pixels,
            'count': patch_data.shape[0] if patch_data.ndim == 3 else 1,
            'dtype': patch_data.dtype,
            'crs': patch_info.crs,
            'transform': patch_info.transform,
            'nodata': patch_dict.get('nodata'),
            'compress': 'lzw'
        }
        
        # Save raster
        output_file = output_dir / f"{patch_info.patch_id}.tif"
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            if patch_data.ndim == 3:
                for i in range(patch_data.shape[0]):
                    dst.write(patch_data[i], i + 1)
            else:
                dst.write(patch_data, 1)
        
        # Save metadata
        metadata = {
            'patch_id': patch_info.patch_id,
            'center_coordinates': [patch_info.center_x, patch_info.center_y],
            'patch_size_m': patch_info.patch_size_m,
            'patch_size_pixels': patch_info.patch_size_pixels,
            'bounds': list(patch_info.bounds),
            'crs': patch_info.crs,
            'source_file': patch_dict['source_file'],
            'band_count': patch_dict['band_count'],
            'dtype': patch_dict['dtype'],
            'nodata': patch_dict['nodata']
        }
        
        metadata_file = output_dir / f"{patch_info.patch_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_patches_from_directory(self, patch_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load patches from a directory.
        
        Args:
            patch_dir: Directory containing patch files
            
        Returns:
            List of patch dictionaries
        """
        patch_dir = Path(patch_dir)
        
        if not patch_dir.exists():
            raise FileNotFoundError(f"Patch directory not found: {patch_dir}")
        
        patches = []
        
        # Find all .tif files
        tif_files = list(patch_dir.glob("*.tif"))
        
        for tif_file in tif_files:
            # Skip if it's not a patch file
            if not tif_file.stem.startswith('patch_'):
                continue
            
            try:
                # Load raster data
                with rasterio.open(tif_file) as src:
                    data = src.read()
                    if data.shape[0] == 1:
                        data = data[0]  # Remove single band dimension
                
                # Load metadata
                metadata_file = patch_dir / f"{tif_file.stem}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Reconstruct patch info
                    patch_info = PatchInfo(
                        patch_id=metadata['patch_id'],
                        center_x=metadata['center_coordinates'][0],
                        center_y=metadata['center_coordinates'][1],
                        patch_size_m=metadata['patch_size_m'],
                        patch_size_pixels=metadata['patch_size_pixels'],
                        window=None,  # Not needed for loaded patches
                        bounds=tuple(metadata['bounds']),
                        transform=None,  # Will be loaded from raster if needed
                        crs=metadata['crs']
                    )
                    
                    patch_dict = {
                        'data': data,
                        'info': patch_info,
                        'source_file': metadata.get('source_file'),
                        'band_count': metadata.get('band_count'),
                        'dtype': metadata.get('dtype'),
                        'nodata': metadata.get('nodata')
                    }
                    
                    patches.append(patch_dict)
                    
                else:
                    warnings.warn(f"No metadata found for {tif_file}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load patch from {tif_file}: {str(e)}")
                continue
        
        return patches
    
    def get_patch_statistics(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for a list of patches.
        
        Args:
            patches: List of patch dictionaries
            
        Returns:
            Dictionary of patch statistics
        """
        if not patches:
            return {'num_patches': 0}
        
        # Basic statistics
        num_patches = len(patches)
        patch_sizes = [p['info'].patch_size_m for p in patches]
        patch_sizes_pixels = [p['info'].patch_size_pixels for p in patches]
        
        # Data statistics
        all_data = []
        for patch in patches:
            data = patch['data']
            if data.ndim > 2:
                # Multi-band data
                all_data.extend(data.flatten())
            else:
                all_data.extend(data.flatten())
        
        all_data = np.array(all_data)
        
        # Remove nodata values
        nodata_vals = [p['nodata'] for p in patches if p['nodata'] is not None]
        if nodata_vals:
            for nodata in set(nodata_vals):
                all_data = all_data[all_data != nodata]
        
        # Remove NaN values
        all_data = all_data[~np.isnan(all_data)]
        
        statistics = {
            'num_patches': num_patches,
            'patch_sizes_m': {
                'unique': list(set(patch_sizes)),
                'min': min(patch_sizes),
                'max': max(patch_sizes),
                'mean': np.mean(patch_sizes)
            },
            'patch_sizes_pixels': {
                'min': min(patch_sizes_pixels),
                'max': max(patch_sizes_pixels),
                'mean': np.mean(patch_sizes_pixels)
            },
            'data_statistics': {
                'min': float(np.min(all_data)) if len(all_data) > 0 else None,
                'max': float(np.max(all_data)) if len(all_data) > 0 else None,
                'mean': float(np.mean(all_data)) if len(all_data) > 0 else None,
                'std': float(np.std(all_data)) if len(all_data) > 0 else None,
                'num_values': len(all_data)
            },
            'spatial_extent': self._calculate_spatial_extent(patches)
        }
        
        return statistics
    
    def _calculate_spatial_extent(self, patches: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate spatial extent of patches.
        
        Args:
            patches: List of patch dictionaries
            
        Returns:
            Dictionary with spatial extent information
        """
        if not patches:
            return {}
        
        all_bounds = [p['info'].bounds for p in patches]
        
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[2] for bounds in all_bounds)
        max_y = max(bounds[3] for bounds in all_bounds)
        
        return {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'area': (max_x - min_x) * (max_y - min_y)
        }
    
    def filter_patches(self, patches: List[Dict[str, Any]],
                      center_bounds: Optional[Tuple[float, float, float, float]] = None,
                      min_data_value: Optional[float] = None,
                      max_data_value: Optional[float] = None) -> List[Dict[str, Any]]:
        """Filter patches based on various criteria.
        
        Args:
            patches: List of patch dictionaries
            center_bounds: Bounding box to filter patch centers (minx, miny, maxx, maxy)
            min_data_value: Minimum data value threshold
            max_data_value: Maximum data value threshold
            
        Returns:
            Filtered list of patches
        """
        filtered_patches = []
        
        for patch in patches:
            # Check spatial filter
            if center_bounds is not None:
                center_x = patch['info'].center_x
                center_y = patch['info'].center_y
                
                if not (center_bounds[0] <= center_x <= center_bounds[2] and
                        center_bounds[1] <= center_y <= center_bounds[3]):
                    continue
            
            # Check data value filters
            if min_data_value is not None or max_data_value is not None:
                data = patch['data']
                nodata = patch.get('nodata')
                
                # Get valid data
                if nodata is not None:
                    valid_data = data[data != nodata]
                else:
                    valid_data = data[~np.isnan(data)]
                
                if len(valid_data) == 0:
                    continue
                
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                
                if min_data_value is not None and data_max < min_data_value:
                    continue
                
                if max_data_value is not None and data_min > max_data_value:
                    continue
            
            filtered_patches.append(patch)
        
        return filtered_patches
    
    def create_patch_index(self, patches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create spatial index of patches for efficient lookup.
        
        Args:
            patches: List of patch dictionaries
            
        Returns:
            Dictionary mapping patch IDs to patch information
        """
        index = {}
        
        for patch in patches:
            patch_id = patch['info'].patch_id
            
            index[patch_id] = {
                'center_x': patch['info'].center_x,
                'center_y': patch['info'].center_y,
                'bounds': patch['info'].bounds,
                'patch_size_m': patch['info'].patch_size_m,
                'source_file': patch.get('source_file'),
                'data_shape': patch['data'].shape,
                'dtype': patch.get('dtype')
            }
        
        return index