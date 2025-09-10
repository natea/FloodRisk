#!/usr/bin/env python3
"""
Process precipitation grid data for flood risk modeling.
Handles ASCII grids, NetCDF, and other formats.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import xarray as xr
from scipy import interpolate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecipitationGridProcessor:
    """Process precipitation grid data for Nashville flood modeling."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize processor."""
        self.data_dir = data_dir or Path("data/v2_additional/precipitation_grids")
        self.output_dir = Path("data/processed/precipitation_grids")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nashville bounding box
        self.nashville_bbox = {
            'north': 36.3,
            'south': 35.9,
            'east': -86.5,
            'west': -87.1
        }
        
        # Target resolution (approximately 1km)
        self.target_resolution = 0.01  # degrees
        
    def read_ascii_grid(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Read ESRI ASCII grid file.
        
        Returns:
            Tuple of (data array, metadata dict)
        """
        metadata = {}
        
        with open(file_path, 'r') as f:
            # Read header
            for i in range(6):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    key = line[0].lower()
                    value = line[1]
                    
                    if key in ['ncols', 'nrows']:
                        metadata[key] = int(value)
                    elif key in ['xllcorner', 'yllcorner', 'cellsize']:
                        metadata[key] = float(value)
                    elif key == 'nodata_value':
                        metadata['nodata'] = float(value)
            
            # Read data
            data = []
            for line in f:
                row = [float(x) for x in line.strip().split()]
                data.append(row)
        
        data = np.array(data)
        
        # Replace nodata values with NaN
        if 'nodata' in metadata:
            data[data == metadata['nodata']] = np.nan
        
        logger.info(f"Read ASCII grid: {data.shape}")
        logger.info(f"Metadata: {metadata}")
        
        return data, metadata
    
    def read_netcdf(self, file_path: Path) -> xr.Dataset:
        """Read NetCDF file."""
        try:
            ds = xr.open_dataset(file_path)
            logger.info(f"Read NetCDF: {file_path}")
            logger.info(f"Variables: {list(ds.data_vars)}")
            logger.info(f"Dimensions: {dict(ds.dims)}")
            return ds
        except Exception as e:
            logger.error(f"Failed to read NetCDF: {e}")
            return None
    
    def clip_to_bbox(self, data: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Clip grid data to Nashville bounding box.
        """
        # Calculate grid coordinates
        xll = metadata['xllcorner']
        yll = metadata['yllcorner']
        cellsize = metadata['cellsize']
        ncols = metadata['ncols']
        nrows = metadata['nrows']
        
        # Calculate column indices for x (longitude)
        col_start = int((self.nashville_bbox['west'] - xll) / cellsize)
        col_end = int((self.nashville_bbox['east'] - xll) / cellsize) + 1
        
        # Calculate row indices for y (latitude)
        # In ASCII grids, row 0 is at the TOP (north), so we need to flip
        row_start_from_bottom = int((self.nashville_bbox['south'] - yll) / cellsize)
        row_end_from_bottom = int((self.nashville_bbox['north'] - yll) / cellsize) + 1
        
        # Convert to top-down indexing (row 0 = north)
        row_start = nrows - row_end_from_bottom
        row_end = nrows - row_start_from_bottom
        
        # Ensure indices are within bounds
        col_start = max(0, col_start)
        col_end = min(ncols, col_end)
        row_start = max(0, row_start)
        row_end = min(nrows, row_end)
        
        logger.info(f"Clipping indices: cols [{col_start}:{col_end}], rows [{row_start}:{row_end}]")
        
        # Clip data
        clipped_data = data[row_start:row_end, col_start:col_end]
        
        # Update metadata
        clipped_metadata = metadata.copy()
        clipped_metadata['xllcorner'] = xll + col_start * cellsize
        clipped_metadata['yllcorner'] = yll + row_start_from_bottom * cellsize
        clipped_metadata['ncols'] = clipped_data.shape[1]
        clipped_metadata['nrows'] = clipped_data.shape[0]
        
        logger.info(f"Clipped to bbox: {clipped_data.shape}")
        logger.info(f"New extent: ({clipped_metadata['xllcorner']:.4f}, {clipped_metadata['yllcorner']:.4f})")
        
        # Check if we have valid data
        valid_data = clipped_data[clipped_data != metadata.get('nodata', -9999)]
        if len(valid_data) > 0:
            logger.info(f"Valid data points: {len(valid_data)}, range: {valid_data.min():.2f} to {valid_data.max():.2f}")
        else:
            logger.warning("No valid data points after clipping!")
        
        return clipped_data, clipped_metadata
    
    def resample_grid(self, data: np.ndarray, metadata: Dict, 
                     target_resolution: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Resample grid to target resolution using interpolation.
        """
        if target_resolution is None:
            target_resolution = self.target_resolution
        
        # Current grid coordinates
        xll = metadata['xllcorner']
        yll = metadata['yllcorner']
        cellsize = metadata['cellsize']
        ncols = metadata['ncols']
        nrows = metadata['nrows']
        
        x_old = np.linspace(xll, xll + (ncols - 1) * cellsize, ncols)
        y_old = np.linspace(yll, yll + (nrows - 1) * cellsize, nrows)
        
        # Target grid coordinates
        x_new = np.arange(xll, xll + ncols * cellsize, target_resolution)
        y_new = np.arange(yll, yll + nrows * cellsize, target_resolution)
        
        # Use RegularGridInterpolator for scipy >= 1.14
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        f = RegularGridInterpolator((y_old, x_old), data, 
                                   method='linear', 
                                   bounds_error=False, 
                                   fill_value=np.nan)
        
        # Create meshgrid for new points
        X_new, Y_new = np.meshgrid(x_new, y_new)
        points = np.array([Y_new.ravel(), X_new.ravel()]).T
        
        # Resample
        resampled_data = f(points).reshape(len(y_new), len(x_new))
        
        # Update metadata
        new_metadata = metadata.copy()
        new_metadata['cellsize'] = target_resolution
        new_metadata['ncols'] = len(x_new)
        new_metadata['nrows'] = len(y_new)
        
        logger.info(f"Resampled from {data.shape} to {resampled_data.shape}")
        
        return resampled_data, new_metadata
    
    def convert_units(self, data: np.ndarray, from_unit: str = 'inches', 
                     to_unit: str = 'mm') -> np.ndarray:
        """Convert precipitation units."""
        conversions = {
            ('inches', 'mm'): 25.4,
            ('mm', 'inches'): 1 / 25.4,
            ('inches', 'm'): 0.0254,
            ('mm', 'm'): 0.001,
        }
        
        key = (from_unit, to_unit)
        if key in conversions:
            return data * conversions[key]
        else:
            logger.warning(f"Unknown unit conversion: {from_unit} to {to_unit}")
            return data
    
    def create_ensemble_grids(self, base_grid: np.ndarray, 
                            num_members: int = 20,
                            cv: float = 0.2) -> List[np.ndarray]:
        """
        Create ensemble of precipitation grids with uncertainty.
        
        Args:
            base_grid: Base precipitation grid
            num_members: Number of ensemble members
            cv: Coefficient of variation for uncertainty
        """
        ensemble = []
        
        for i in range(num_members):
            # Add spatially correlated noise
            noise = np.random.normal(0, cv, base_grid.shape)
            
            # Apply Gaussian smoothing for spatial correlation
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=2)
            
            # Create perturbed grid
            perturbed = base_grid * (1 + noise)
            perturbed = np.maximum(perturbed, 0)  # No negative precipitation
            
            ensemble.append(perturbed)
        
        logger.info(f"Created {num_members} ensemble members")
        
        return ensemble
    
    def process_sample_grid(self):
        """Process the sample grid we created earlier."""
        sample_file = self.data_dir / "sample_100yr_24hr.asc"
        
        if not sample_file.exists():
            logger.error(f"Sample file not found: {sample_file}")
            return None
        
        # Read sample grid
        data, metadata = self.read_ascii_grid(sample_file)
        
        # Convert from inches to mm
        data_mm = self.convert_units(data, 'inches', 'mm')
        
        # Clip to Nashville bbox
        clipped_data, clipped_metadata = self.clip_to_bbox(data_mm, metadata)
        
        # Resample to target resolution
        resampled_data, resampled_metadata = self.resample_grid(
            clipped_data, clipped_metadata
        )
        
        # Create ensemble
        ensemble = self.create_ensemble_grids(resampled_data, num_members=10)
        
        # Save processed data
        output_file = self.output_dir / "processed_100yr_24hr.npz"
        np.savez_compressed(
            output_file,
            mean=resampled_data,
            ensemble=np.array(ensemble),
            metadata=resampled_metadata,
            bbox=self.nashville_bbox
        )
        
        logger.info(f"Saved processed grid: {output_file}")
        
        # Create summary statistics
        stats = {
            'min_precipitation_mm': float(np.nanmin(resampled_data)),
            'max_precipitation_mm': float(np.nanmax(resampled_data)),
            'mean_precipitation_mm': float(np.nanmean(resampled_data)),
            'std_precipitation_mm': float(np.nanstd(resampled_data)),
            'grid_shape': resampled_data.shape,
            'resolution_degrees': resampled_metadata['cellsize'],
            'ensemble_members': len(ensemble),
        }
        
        stats_file = self.output_dir / "grid_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics: {stats}")
        
        return resampled_data, resampled_metadata
    
    def process_cpc_data(self):
        """Process CPC unified precipitation data if downloaded."""
        cpc_file = self.data_dir / "alternative_precip.V1.0.2020.nc"
        
        if not cpc_file.exists():
            logger.info("CPC data not found")
            return None
        
        # Read NetCDF
        ds = self.read_netcdf(cpc_file)
        
        if ds is not None:
            # Extract precipitation variable
            precip_vars = ['precip', 'precipitation', 'ppt', 'prcp']
            precip_data = None
            
            for var in precip_vars:
                if var in ds.data_vars:
                    precip_data = ds[var]
                    break
            
            if precip_data is not None:
                logger.info(f"Found precipitation variable: {precip_data.shape}")
                
                # Clip to Nashville region if coordinates available
                if 'lat' in ds.coords and 'lon' in ds.coords:
                    lat_mask = (ds.lat >= self.nashville_bbox['south']) & \
                              (ds.lat <= self.nashville_bbox['north'])
                    lon_mask = (ds.lon >= self.nashville_bbox['west']) & \
                              (ds.lon <= self.nashville_bbox['east'])
                    
                    clipped = precip_data.sel(lat=lat_mask, lon=lon_mask)
                    logger.info(f"Clipped CPC data: {clipped.shape}")
                    
                    # Save as numpy array
                    output_file = self.output_dir / "cpc_precipitation.npz"
                    np.savez_compressed(
                        output_file,
                        data=clipped.values,
                        lat=clipped.lat.values,
                        lon=clipped.lon.values,
                        time=clipped.time.values if 'time' in clipped.dims else None
                    )
                    
                    logger.info(f"Saved CPC data: {output_file}")
    
    def create_visualization(self, data: np.ndarray, metadata: Dict, 
                           output_file: str = "precipitation_grid.png"):
        """Create visualization of precipitation grid."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create precipitation colormap (white to blue to purple)
            colors = ['white', 'lightblue', 'blue', 'purple', 'darkred']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('precipitation', colors, N=n_bins)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create extent for plotting
            xll = metadata['xllcorner']
            yll = metadata['yllcorner']
            cellsize = metadata['cellsize']
            ncols = metadata['ncols']
            nrows = metadata['nrows']
            
            extent = [xll, xll + ncols * cellsize, 
                     yll, yll + nrows * cellsize]
            
            # Plot data
            im = ax.imshow(data, extent=extent, cmap=cmap, 
                          origin='lower', interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Precipitation (mm)')
            
            # Add labels
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('100-Year 24-Hour Precipitation Grid\nNashville Region')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save figure
            output_path = self.output_dir / output_file
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {output_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
    
    def run_processing(self):
        """Run the complete processing pipeline."""
        logger.info("=" * 60)
        logger.info("Precipitation Grid Processing Pipeline")
        logger.info("=" * 60)
        
        # Process sample grid
        logger.info("\nProcessing sample grid...")
        result = self.process_sample_grid()
        
        if result:
            data, metadata = result
            # Create visualization
            self.create_visualization(data, metadata)
        
        # Process CPC data if available
        logger.info("\nProcessing CPC data...")
        self.process_cpc_data()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Files created:")
        
        for file in self.output_dir.glob("*"):
            size = file.stat().st_size / 1024  # KB
            logger.info(f"  - {file.name} ({size:.1f} KB)")
        
        logger.info("\nNext steps:")
        logger.info("1. Manual download NOAA grids when available")
        logger.info("2. Process real NOAA data with this pipeline")
        logger.info("3. Integrate with ML model training")
        
        return True


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import xarray
        import scipy
    except ImportError:
        import subprocess
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray", "scipy", "netcdf4"])
    
    processor = PrecipitationGridProcessor()
    processor.run_processing()