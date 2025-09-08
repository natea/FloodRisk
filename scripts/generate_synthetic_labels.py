#!/usr/bin/env python3
"""
Generate synthetic flood labels for testing when LISFLOOD-FP is not available.
This creates realistic-looking flood extent maps based on DEM topography.
"""

import numpy as np
import rasterio
from pathlib import Path
import argparse
from scipy import ndimage
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_flood_extent(
    dem_array: np.ndarray,
    rainfall_mm: float,
    flood_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic flood extent based on DEM topography.
    
    Args:
        dem_array: Digital elevation model array
        rainfall_mm: Rainfall amount in mm
        flood_threshold: Minimum depth for flood classification (m)
        
    Returns:
        Tuple of (flood_depth, flood_extent_binary)
    """
    # Normalize DEM
    dem_norm = (dem_array - np.nanmin(dem_array)) / (np.nanmax(dem_array) - np.nanmin(dem_array))
    
    # Find low-lying areas (potential flood zones)
    # Use inverse DEM as proxy for water accumulation
    accumulation = 1.0 - dem_norm
    
    # Apply gaussian filter to simulate water spreading
    sigma = 2.0 + (rainfall_mm / 100.0)  # Spread increases with rainfall
    water_spread = ndimage.gaussian_filter(accumulation, sigma=sigma)
    
    # Calculate synthetic flood depth based on rainfall and topography
    # Higher rainfall = deeper floods in low areas
    base_depth = (rainfall_mm / 1000.0) * 5  # Convert mm to m with multiplier
    flood_depth = water_spread * base_depth * (1 + np.random.normal(0, 0.1, water_spread.shape))
    
    # Add some realistic spatial patterns
    # Create flow channels
    gradient_y, gradient_x = np.gradient(dem_array)
    slope = np.sqrt(gradient_y**2 + gradient_x**2)
    
    # Water accumulates more in flat areas
    flat_areas = slope < np.percentile(slope, 30)
    flood_depth[flat_areas] *= 1.5
    
    # Apply threshold for binary extent
    flood_extent = (flood_depth >= flood_threshold).astype(np.uint8)
    
    # Clean up small isolated patches
    flood_extent = ndimage.binary_opening(flood_extent, iterations=1)
    flood_extent = ndimage.binary_closing(flood_extent, iterations=1)
    
    # Remove very small components
    labeled, num_features = ndimage.label(flood_extent)
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled == i)
        if component_size < 20:  # Remove patches smaller than 20 pixels
            flood_extent[labeled == i] = 0
    
    return flood_depth.astype(np.float32), flood_extent.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic flood labels for ML training"
    )
    parser.add_argument(
        "--dem-file",
        type=Path,
        required=True,
        help="Path to DEM file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for synthetic labels"
    )
    parser.add_argument(
        "--return-periods",
        type=str,
        default="100,500",
        help="Return periods to generate (comma-separated)"
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=0.05,
        help="Minimum flood depth threshold in meters"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define rainfall amounts for different return periods (mm)
    rainfall_amounts = {
        "10": 75.0,
        "25": 100.0,
        "100": 150.0,
        "500": 200.0
    }
    
    # Read DEM
    logger.info(f"Reading DEM from {args.dem_file}")
    with rasterio.open(args.dem_file) as src:
        dem_array = src.read(1)
        profile = src.profile.copy()
    
    # Handle nodata values
    if 'nodata' in profile and profile['nodata'] is not None:
        dem_array[dem_array == profile['nodata']] = np.nan
    
    # Generate labels for each return period
    return_periods = args.return_periods.split(',')
    
    for period in return_periods:
        if period not in rainfall_amounts:
            logger.warning(f"Unknown return period: {period}, skipping")
            continue
            
        rainfall_mm = rainfall_amounts[period]
        logger.info(f"Generating {period}-year flood extent (rainfall: {rainfall_mm}mm)")
        
        # Generate synthetic flood
        flood_depth, flood_extent = generate_synthetic_flood_extent(
            dem_array, 
            rainfall_mm,
            args.flood_threshold
        )
        
        # Save flood depth
        depth_file = args.output_dir / f"flood_depth_{period}yr.tif"
        depth_profile = profile.copy()
        depth_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
        with rasterio.open(depth_file, 'w', **depth_profile) as dst:
            dst.write(flood_depth, 1)
        logger.info(f"Saved flood depth: {depth_file}")
        
        # Save flood extent
        extent_file = args.output_dir / f"flood_extent_{period}yr.tif"
        extent_profile = profile.copy()
        extent_profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        
        with rasterio.open(extent_file, 'w', **extent_profile) as dst:
            dst.write(flood_extent, 1)
        logger.info(f"Saved flood extent: {extent_file}")
        
        # Print statistics
        flood_pixels = np.sum(flood_extent)
        total_pixels = flood_extent.size
        flood_percentage = (flood_pixels / total_pixels) * 100
        
        logger.info(f"  Flood coverage: {flood_percentage:.2f}% ({flood_pixels:,} pixels)")
        logger.info(f"  Max depth: {np.nanmax(flood_depth):.2f}m")
        logger.info(f"  Mean depth (flooded areas): {np.mean(flood_depth[flood_extent > 0]):.2f}m")
    
    logger.info("Synthetic label generation complete!")
    
    # Create a simple metadata file
    metadata_file = args.output_dir / "synthetic_labels_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Synthetic Flood Labels\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated from: {args.dem_file}\n")
        f.write(f"Return periods: {args.return_periods}\n")
        f.write(f"Flood threshold: {args.flood_threshold}m\n")
        f.write(f"Method: Topography-based synthetic generation\n")
        f.write("\nNote: These are synthetic labels for testing.\n")
        f.write("For production, use physics-based simulation (LISFLOOD-FP).\n")
    
    logger.info(f"Metadata saved: {metadata_file}")

if __name__ == "__main__":
    main()