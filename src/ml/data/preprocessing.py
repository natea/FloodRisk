"""
Data preprocessing pipeline for flood prediction ML model.
Based on APPROACH.md specifications for DEM/rainfall processing.
"""

import logging
import numpy as np
import rasterio
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from rasterio import features, warp, transform
from rasterio.crs import CRS
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class DEMProcessor:
    """Process Digital Elevation Model data."""
    
    def __init__(self, target_crs: str = "EPSG:3857", resolution: float = 10.0):
        """
        Initialize DEM processor.
        
        Args:
            target_crs: Target coordinate reference system (metric CRS recommended)
            resolution: Target resolution in meters (~10m per APPROACH.md)
        """
        self.target_crs = CRS.from_string(target_crs)
        self.resolution = resolution
        
    def load_and_reproject(self, dem_path: Path) -> xr.DataArray:
        """
        Load DEM and reproject to target CRS with target resolution.
        
        Args:
            dem_path: Path to DEM raster file
            
        Returns:
            Reprojected DEM as xarray DataArray
        """
        logger.info(f"Loading DEM from {dem_path}")
        
        with rasterio.open(dem_path) as src:
            # Calculate new transform and dimensions
            dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                src.crs, self.target_crs, src.width, src.height, *src.bounds,
                resolution=self.resolution
            )
            
            # Create destination array
            dst_array = np.empty((dst_height, dst_width), dtype=src.dtypes[0])
            
            # Reproject
            warp.reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=self.target_crs,
                resampling=warp.Resampling.bilinear
            )
            
            # Convert to xarray
            coords_x = np.linspace(
                dst_transform[2], 
                dst_transform[2] + dst_transform[0] * dst_width,
                dst_width
            )
            coords_y = np.linspace(
                dst_transform[5], 
                dst_transform[5] + dst_transform[4] * dst_height, 
                dst_height
            )
            
            dem_da = xr.DataArray(
                dst_array,
                coords={'y': coords_y, 'x': coords_x},
                dims=['y', 'x'],
                attrs={
                    'crs': str(self.target_crs),
                    'transform': dst_transform,
                    'resolution': self.resolution
                }
            )
            
        logger.info(f"DEM reprojected to {self.target_crs} at {self.resolution}m resolution")
        return dem_da
    
    def compute_derived_features(self, dem: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        Compute derived topographic features as recommended in APPROACH.md.
        
        Args:
            dem: Digital elevation model
            
        Returns:
            Dictionary of derived features: slope, curvature, flow_accumulation, HAND
        """
        logger.info("Computing derived topographic features")
        
        features = {}
        dem_array = dem.values
        
        # Compute slope
        gy, gx = np.gradient(dem_array, self.resolution)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        features['slope'] = dem.copy(data=slope)
        
        # Compute curvature (plan and profile)
        gyy, gyx = np.gradient(gy, self.resolution)
        gxy, gxx = np.gradient(gx, self.resolution)
        
        # Plan curvature
        plan_curvature = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / (gx**2 + gy**2 + 1e-8)**1.5
        features['plan_curvature'] = dem.copy(data=plan_curvature)
        
        # Profile curvature  
        profile_curvature = (gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / (gx**2 + gy**2 + 1e-8)**1.5
        features['profile_curvature'] = dem.copy(data=profile_curvature)
        
        # Flow accumulation (simplified - using gaussian filter as proxy)
        # In practice, you'd use a proper flow routing algorithm
        flow_accum = ndimage.gaussian_filter(-dem_array, sigma=3)
        features['flow_accumulation'] = dem.copy(data=np.log10(np.abs(flow_accum) + 1))
        
        # HAND (Height Above Nearest Drainage) - simplified approximation
        # In practice, this requires proper stream network delineation
        streams = flow_accum < np.percentile(flow_accum, 5)  # Approximate streams
        distance_to_stream = ndimage.distance_transform_edt(~streams) * self.resolution
        hand = dem_array - ndimage.minimum_filter(dem_array, size=int(distance_to_stream.max() / self.resolution))
        features['hand'] = dem.copy(data=np.maximum(hand, 0))
        
        logger.info(f"Computed {len(features)} derived features")
        return features

class RainfallProcessor:
    """Process rainfall data from NOAA Atlas 14."""
    
    def __init__(self):
        self.return_periods = {
            '100yr': 100,
            '500yr': 500,
            '10yr': 10,  # For negative examples
            '25yr': 25   # For negative examples
        }
        
    def create_uniform_rainfall_raster(
        self, 
        rainfall_depth_mm: float, 
        template_raster: xr.DataArray,
        spatial_variability: float = 0.0
    ) -> xr.DataArray:
        """
        Create uniform rainfall raster matching template geometry.
        
        Args:
            rainfall_depth_mm: 24-hour total rainfall depth in mm
            template_raster: Template raster for geometry (e.g., DEM)
            spatial_variability: Optional mild spatial variability (0-0.15)
            
        Returns:
            Rainfall raster as xarray DataArray
        """
        logger.info(f"Creating uniform rainfall raster: {rainfall_depth_mm}mm")
        
        # Create base uniform field
        rainfall = np.full_like(template_raster.values, rainfall_depth_mm, dtype=np.float32)
        
        # Add optional spatial variability
        if spatial_variability > 0:
            noise = np.random.normal(1.0, spatial_variability, rainfall.shape)
            noise = ndimage.gaussian_filter(noise, sigma=2)  # Smooth spatial gradients
            rainfall = rainfall * noise
            
        rainfall_da = template_raster.copy(data=rainfall)
        rainfall_da.attrs.update({
            'units': 'mm',
            'description': '24-hour total rainfall depth',
            'spatial_variability': spatial_variability
        })
        
        return rainfall_da

class TileGenerator:
    """Generate training tiles with overlap as specified in APPROACH.md."""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        """
        Initialize tile generator.
        
        Args:
            tile_size: Tile size in pixels (512x512 per APPROACH.md)
            overlap: Overlap in pixels (~64 per APPROACH.md)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
    def generate_tiles(
        self, 
        data_arrays: List[xr.DataArray], 
        flood_labels: Optional[xr.DataArray] = None
    ) -> List[Dict]:
        """
        Generate overlapping tiles from input data arrays.
        
        Args:
            data_arrays: List of input data arrays (DEM, rainfall, derived features)
            flood_labels: Optional flood extent labels
            
        Returns:
            List of tile dictionaries with data and metadata
        """
        logger.info(f"Generating {self.tile_size}x{self.tile_size} tiles with {self.overlap}px overlap")
        
        # Ensure all arrays have same shape
        shapes = [arr.shape for arr in data_arrays]
        if flood_labels is not None:
            shapes.append(flood_labels.shape)
        assert len(set(shapes)) == 1, "All data arrays must have same shape"
        
        height, width = shapes[0]
        tiles = []
        
        for i in range(0, height - self.tile_size + 1, self.stride):
            for j in range(0, width - self.tile_size + 1, self.stride):
                # Extract tile from each data array
                tile_data = {}
                for k, arr in enumerate(data_arrays):
                    key = arr.name if hasattr(arr, 'name') and arr.name else f'data_{k}'
                    tile_data[key] = arr.isel(
                        y=slice(i, i + self.tile_size),
                        x=slice(j, j + self.tile_size)
                    ).values
                
                # Extract labels if provided
                if flood_labels is not None:
                    tile_data['labels'] = flood_labels.isel(
                        y=slice(i, i + self.tile_size),
                        x=slice(j, j + self.tile_size)
                    ).values
                
                # Add metadata
                tile_data.update({
                    'tile_id': f'{i}_{j}',
                    'row': i,
                    'col': j,
                    'bounds': {
                        'row_start': i, 'row_end': i + self.tile_size,
                        'col_start': j, 'col_end': j + self.tile_size
                    }
                })
                
                tiles.append(tile_data)
        
        logger.info(f"Generated {len(tiles)} tiles")
        return tiles
    
    def balanced_sampling(
        self, 
        tiles: List[Dict], 
        flood_threshold: float = 0.02,
        flooded_ratio: float = 0.7
    ) -> List[Dict]:
        """
        Perform balanced sampling as specified in APPROACH.md.
        ~70% tiles contain ≥2-5% flooded pixels; ~30% random tiles.
        
        Args:
            tiles: List of tile dictionaries
            flood_threshold: Minimum fraction of flooded pixels (2-5%)
            flooded_ratio: Ratio of flooded tiles to keep (0.7)
            
        Returns:
            Balanced subset of tiles
        """
        if 'labels' not in tiles[0]:
            logger.warning("No labels found for balanced sampling, returning all tiles")
            return tiles
            
        logger.info(f"Performing balanced sampling with {flood_threshold*100:.1f}% flood threshold")
        
        flooded_tiles = []
        dry_tiles = []
        
        for tile in tiles:
            flood_fraction = np.mean(tile['labels'] > 0)
            if flood_fraction >= flood_threshold:
                flooded_tiles.append(tile)
            else:
                dry_tiles.append(tile)
        
        logger.info(f"Found {len(flooded_tiles)} flooded tiles, {len(dry_tiles)} dry tiles")
        
        # Calculate target counts
        n_flooded = len(flooded_tiles)
        n_dry_target = int(n_flooded * (1 - flooded_ratio) / flooded_ratio)
        
        # Sample dry tiles
        if len(dry_tiles) > n_dry_target:
            dry_sampled = np.random.choice(
                len(dry_tiles), size=n_dry_target, replace=False
            )
            dry_tiles = [dry_tiles[i] for i in dry_sampled]
        
        balanced_tiles = flooded_tiles + dry_tiles
        logger.info(f"Balanced sampling: {len(flooded_tiles)} flooded + {len(dry_tiles)} dry = {len(balanced_tiles)} total")
        
        return balanced_tiles

def normalize_features(
    features: Dict[str, np.ndarray], 
    method: str = 'per_tile'
) -> Dict[str, np.ndarray]:
    """
    Normalize features to improve generalization across cities.
    
    Args:
        features: Dictionary of feature arrays
        method: Normalization method ('per_tile' or 'global')
        
    Returns:
        Normalized features
    """
    normalized = {}
    
    for name, data in features.items():
        if method == 'per_tile':
            # Per-tile normalization to prevent location leakage
            mean = np.mean(data)
            std = np.std(data)
            normalized[name] = (data - mean) / (std + 1e-8)
        elif method == 'global':
            # Global normalization (requires pre-computed statistics)
            scaler = StandardScaler()
            normalized[name] = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
        else:
            normalized[name] = data
            
    return normalized