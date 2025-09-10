#!/usr/bin/env python3
"""
Create training tiles from DEM and precipitation data for ML model training.
Splits large rasters into manageable tiles with optional overlap.
"""

import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tiles(
    input_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    overlap: int = 64,
    min_valid_pixels: float = 0.5
) -> List[dict]:
    """
    Create tiles from a raster file.
    
    Args:
        input_path: Path to input raster file
        output_dir: Output directory for tiles
        tile_size: Size of each tile in pixels
        overlap: Overlap between tiles in pixels
        min_valid_pixels: Minimum fraction of valid (non-nodata) pixels required
        
    Returns:
        List of tile metadata dictionaries
    """
    tiles_metadata = []
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        # Calculate stride (how much to move the window)
        stride = tile_size - overlap
        
        # Get raster dimensions
        height, width = src.shape
        transform = src.transform
        
        # Calculate number of tiles
        n_tiles_x = (width - overlap) // stride + 1
        n_tiles_y = (height - overlap) // stride + 1
        
        logger.info(f"Creating {n_tiles_x * n_tiles_y} tiles from {input_path.name}")
        
        tile_idx = 0
        for row in tqdm(range(n_tiles_y), desc="Rows"):
            for col in range(n_tiles_x):
                # Calculate window bounds
                x_start = col * stride
                y_start = row * stride
                
                # Ensure we don't exceed raster bounds
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)
                
                # Adjust start if we're at the edge
                if x_end - x_start < tile_size:
                    x_start = max(0, x_end - tile_size)
                if y_end - y_start < tile_size:
                    y_start = max(0, y_end - tile_size)
                
                # Create window
                window = Window(x_start, y_start, 
                               x_end - x_start, y_end - y_start)
                
                # Read tile data
                tile_data = src.read(window=window)
                
                # Check valid pixel ratio
                if src.nodata is not None:
                    valid_mask = tile_data != src.nodata
                else:
                    valid_mask = ~np.isnan(tile_data)
                    
                valid_ratio = np.sum(valid_mask) / valid_mask.size
                
                if valid_ratio < min_valid_pixels:
                    continue
                
                # Save tile
                tile_name = f"tile_{tile_idx:05d}.tif"
                tile_path = output_dir / tile_name
                
                # Get transform for this tile
                tile_transform = rasterio.windows.transform(window, transform)
                
                # Write tile
                profile = src.profile.copy()
                profile.update({
                    'width': window.width,
                    'height': window.height,
                    'transform': tile_transform
                })
                
                with rasterio.open(tile_path, 'w', **profile) as dst:
                    dst.write(tile_data)
                
                # Store metadata
                metadata = {
                    'tile_id': tile_idx,
                    'tile_name': tile_name,
                    'source_file': input_path.name,
                    'row': row,
                    'col': col,
                    'x_start': x_start,
                    'y_start': y_start,
                    'width': window.width,
                    'height': window.height,
                    'valid_pixel_ratio': float(valid_ratio),
                    'bounds': list(rasterio.windows.bounds(window, transform))
                }
                tiles_metadata.append(metadata)
                
                tile_idx += 1
        
        logger.info(f"Created {tile_idx} valid tiles")
        
    return tiles_metadata


def create_training_tiles(
    dem_path: Path,
    precipitation_dir: Optional[Path],
    output_dir: Path,
    tile_size: int = 512,
    overlap: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> None:
    """
    Create training, validation, and test tiles from DEM and precipitation data.
    
    Args:
        dem_path: Path to DEM file
        precipitation_dir: Directory containing precipitation files
        output_dir: Output directory for tiles
        tile_size: Size of each tile
        overlap: Overlap between tiles
        train_ratio: Fraction of tiles for training
        val_ratio: Fraction of tiles for validation
        test_ratio: Fraction of tiles for testing
    """
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / split / 'dem').mkdir(parents=True, exist_ok=True)
        if precipitation_dir:
            (output_dir / split / 'precipitation').mkdir(parents=True, exist_ok=True)
    
    # Process DEM
    logger.info("Processing DEM data...")
    dem_tiles = create_tiles(
        dem_path,
        output_dir / 'temp_dem',
        tile_size,
        overlap
    )
    
    # Process precipitation if provided
    precip_tiles_dict = {}
    if precipitation_dir and precipitation_dir.exists():
        logger.info("Processing precipitation data...")
        precip_files = list(precipitation_dir.glob("*.tif")) + list(precipitation_dir.glob("*.nc"))
        
        for precip_file in precip_files:
            logger.info(f"Processing {precip_file.name}")
            precip_tiles = create_tiles(
                precip_file,
                output_dir / 'temp_precip' / precip_file.stem,
                tile_size,
                overlap
            )
            precip_tiles_dict[precip_file.stem] = precip_tiles
    
    # Split tiles into train/val/test
    n_tiles = len(dem_tiles)
    indices = np.random.permutation(n_tiles)
    
    n_train = int(n_tiles * train_ratio)
    n_val = int(n_tiles * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    split_mapping = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Move tiles to appropriate splits
    logger.info("Organizing tiles into train/val/test splits...")
    
    for split_name, split_indices in split_mapping.items():
        split_metadata = []
        
        for idx in split_indices:
            tile_info = dem_tiles[idx]
            tile_id = tile_info['tile_id']
            
            # Move DEM tile
            src_dem = output_dir / 'temp_dem' / f"tile_{tile_id:05d}.tif"
            dst_dem = output_dir / split_name / 'dem' / f"tile_{tile_id:05d}.tif"
            if src_dem.exists():
                src_dem.rename(dst_dem)
            
            # Move precipitation tiles
            if precip_tiles_dict:
                for precip_name, precip_tiles in precip_tiles_dict.items():
                    if idx < len(precip_tiles):
                        src_precip = output_dir / 'temp_precip' / precip_name / f"tile_{tile_id:05d}.tif"
                        dst_precip = output_dir / split_name / 'precipitation' / f"{precip_name}_tile_{tile_id:05d}.tif"
                        if src_precip.exists():
                            src_precip.rename(dst_precip)
            
            split_metadata.append(tile_info)
        
        # Save metadata
        metadata_path = output_dir / split_name / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        logger.info(f"{split_name}: {len(split_indices)} tiles")
    
    # Cleanup temporary directories
    if (output_dir / 'temp_dem').exists():
        import shutil
        shutil.rmtree(output_dir / 'temp_dem')
    if (output_dir / 'temp_precip').exists():
        import shutil
        shutil.rmtree(output_dir / 'temp_precip')
    
    # Save overall configuration
    config = {
        'tile_size': tile_size,
        'overlap': overlap,
        'n_tiles_total': n_tiles,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
        'dem_source': str(dem_path),
        'precipitation_sources': [str(f) for f in precipitation_dir.glob("*")] if precipitation_dir else []
    }
    
    with open(output_dir / 'tiling_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Tiling complete! Created {n_tiles} tiles total")
    logger.info(f"   Train: {len(train_indices)} tiles")
    logger.info(f"   Val: {len(val_indices)} tiles")
    logger.info(f"   Test: {len(test_indices)} tiles")


def main():
    parser = argparse.ArgumentParser(description="Create training tiles from raster data")
    parser.add_argument('--dem-path', type=Path, required=True,
                       help='Path to DEM file (GeoTIFF or NetCDF)')
    parser.add_argument('--precipitation-dir', type=Path,
                       help='Directory containing precipitation files')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for tiles')
    parser.add_argument('--tile-size', type=int, default=512,
                       help='Tile size in pixels (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles in pixels (default: 64)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Fraction of tiles for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Fraction of tiles for validation (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Fraction of tiles for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        parser.error(f"Train + val + test ratios must sum to 1.0 (got {total_ratio})")
    
    # Check input paths
    if not args.dem_path.exists():
        parser.error(f"DEM file not found: {args.dem_path}")
    
    if args.precipitation_dir and not args.precipitation_dir.exists():
        logger.warning(f"Precipitation directory not found: {args.precipitation_dir}")
        args.precipitation_dir = None
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tiles
    create_training_tiles(
        args.dem_path,
        args.precipitation_dir,
        args.output_dir,
        args.tile_size,
        args.overlap,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )


if __name__ == "__main__":
    main()