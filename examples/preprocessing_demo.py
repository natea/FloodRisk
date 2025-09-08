#!/usr/bin/env python3
"""Demonstration of enhanced preprocessing pipeline for FloodRisk ML.

This script demonstrates how to use the enhanced preprocessing pipeline to:
1. Process real-world USGS 3DEP DEM data
2. Integrate NOAA Atlas 14 rainfall data
3. Extract comprehensive terrain features
4. Generate training tiles
5. Run quality assurance checks

Usage:
    python examples/preprocessing_demo.py --dem-path /path/to/dem.tif --config config/preprocessing/nashville_config.json
"""

import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from ml.data.real_data_preprocessing import RealDataPreprocessor, create_preprocessor
    from ml.data.preprocessing_qa import run_qa_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_rainfall_data() -> Dict[str, Any]:
    """Create sample NOAA Atlas 14 rainfall data for Nashville area.
    
    In practice, this would come from the data acquisition module.
    """
    # Sample precipitation depths in mm for Nashville area
    # Based on NOAA Atlas 14 for Middle Tennessee
    return {
        "10": {  # 10-year return period
            "6": 89,    # 6-hour: 89mm (3.5 inches)
            "12": 102,  # 12-hour: 102mm (4.0 inches)
            "24": 127   # 24-hour: 127mm (5.0 inches)
        },
        "25": {  # 25-year return period
            "6": 107,   # 6-hour: 107mm (4.2 inches)
            "12": 122,  # 12-hour: 122mm (4.8 inches)
            "24": 152   # 24-hour: 152mm (6.0 inches)
        },
        "50": {  # 50-year return period
            "6": 119,   # 6-hour: 119mm (4.7 inches)
            "12": 137,  # 12-hour: 137mm (5.4 inches)
            "24": 168   # 24-hour: 168mm (6.6 inches)
        },
        "100": {  # 100-year return period
            "6": 130,   # 6-hour: 130mm (5.1 inches)
            "12": 150,  # 12-hour: 150mm (5.9 inches)
            "24": 183   # 24-hour: 183mm (7.2 inches)
        },
        "500": {  # 500-year return period
            "6": 155,   # 6-hour: 155mm (6.1 inches)
            "12": 178,  # 12-hour: 178mm (7.0 inches)
            "24": 218   # 24-hour: 218mm (8.6 inches)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="FloodRisk Preprocessing Pipeline Demo")
    
    parser.add_argument("--dem-path", 
                       help="Path to USGS 3DEP DEM file",
                       default="data/nashville_dem.tif")
    
    parser.add_argument("--config",
                       help="Path to preprocessing configuration file", 
                       default="config/preprocessing/nashville_config.json")
    
    parser.add_argument("--output-dir",
                       help="Output directory for processed data",
                       default="outputs/preprocessing_demo")
    
    parser.add_argument("--region",
                       help="Region preset (nashville, default)",
                       default="nashville")
    
    parser.add_argument("--enable-caching",
                       action="store_true",
                       default=True,
                       help="Enable data caching")
    
    parser.add_argument("--run-qa",
                       action="store_true", 
                       default=True,
                       help="Run quality assurance checks")
    
    parser.add_argument("--create-tiles",
                       action="store_true",
                       default=True,
                       help="Create training tiles")
    
    parser.add_argument("--tile-size",
                       type=int,
                       default=512,
                       help="Training tile size in pixels")
    
    parser.add_argument("--bounds",
                       nargs=4,
                       type=float,
                       metavar=('minx', 'miny', 'maxx', 'maxy'),
                       help="Bounding box for processing (minx miny maxx maxy)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FloodRisk Enhanced Preprocessing Pipeline Demo")
    logger.info("=" * 60)
    
    # Check if DEM file exists
    dem_path = Path(args.dem_path)
    if not dem_path.exists():
        logger.error(f"DEM file not found: {dem_path}")
        logger.info("Please provide a valid USGS 3DEP DEM file using --dem-path")
        logger.info("You can download DEMs from: https://apps.nationalmap.gov/downloader/")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"DEM file: {dem_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Region: {args.region}")
    
    try:
        # Initialize preprocessor
        logger.info("\n1. Initializing Preprocessor")
        logger.info("-" * 30)
        
        if args.region == "nashville":
            preprocessor = create_preprocessor(
                region="nashville",
                cache_dir=output_dir / "cache",
                enable_caching=args.enable_caching
            )
        else:
            config_path = args.config if Path(args.config).exists() else None
            preprocessor = RealDataPreprocessor(
                config_path=config_path,
                cache_dir=output_dir / "cache",
                enable_caching=args.enable_caching
            )
        
        logger.info("✓ Preprocessor initialized successfully")
        
        # Create sample rainfall data
        logger.info("\n2. Preparing Rainfall Data")
        logger.info("-" * 30)
        
        rainfall_data = create_sample_rainfall_data()
        logger.info("✓ Sample NOAA Atlas 14 data created for Nashville")
        logger.info(f"  - Return periods: {list(rainfall_data.keys())}")
        logger.info(f"  - Durations: {list(rainfall_data['10'].keys())} hours")
        
        # Process region
        logger.info("\n3. Processing Region Data")
        logger.info("-" * 30)
        
        bounds = tuple(args.bounds) if args.bounds else None
        if bounds:
            logger.info(f"Processing bounds: {bounds}")
        
        processed_data = preprocessor.process_region(
            dem_path=dem_path,
            rainfall_data=rainfall_data,
            region_bounds=bounds,
            output_dir=output_dir / "processed_data"
        )
        
        logger.info("✓ Region processing completed")
        logger.info(f"  - Processed datasets: {len(processed_data)}")
        
        # List processed datasets
        for name, data in processed_data.items():
            if hasattr(data, 'shape'):
                logger.info(f"    • {name}: {data.shape}")
            elif isinstance(data, dict):
                logger.info(f"    • {name}: {len(data)} scenarios")
            else:
                logger.info(f"    • {name}: {type(data).__name__}")
        
        # Create training tiles
        if args.create_tiles:
            logger.info("\n4. Creating Training Tiles")
            logger.info("-" * 30)
            
            tiles = preprocessor.create_training_tiles(
                processed_data=processed_data,
                tile_size=args.tile_size,
                overlap=64
            )
            
            logger.info(f"✓ Created {len(tiles)} training tiles")
            logger.info(f"  - Tile size: {args.tile_size}x{args.tile_size} pixels")
            logger.info(f"  - Overlap: 64 pixels")
            
            # Save sample tile info
            if tiles:
                sample_tile = tiles[0]
                tile_info = {
                    "total_tiles": len(tiles),
                    "tile_size": args.tile_size,
                    "features": [k for k in sample_tile.keys() if k not in ['tile_id', 'row', 'col', 'bounds']],
                    "sample_tile_bounds": sample_tile.get('bounds', {})
                }
                
                with open(output_dir / "tiles_info.json", 'w') as f:
                    json.dump(tile_info, f, indent=2)
                
                logger.info(f"  - Features per tile: {len(tile_info['features'])}")
                logger.info(f"  - Tile info saved to: tiles_info.json")
        
        # Run quality assurance
        if args.run_qa:
            logger.info("\n5. Running Quality Assurance")
            logger.info("-" * 30)
            
            # Filter out non-array data for QA
            qa_data = {k: v for k, v in processed_data.items() 
                      if hasattr(v, 'values') and hasattr(v, 'shape')}
            
            qa_results = run_qa_pipeline(
                processed_data=qa_data,
                output_dir=output_dir / "qa_results"
            )
            
            overall_status = qa_results.get('overall_status', 'unknown')
            logger.info(f"✓ QA completed with status: {overall_status.upper()}")
            
            # Report key findings
            data_val = qa_results.get('data_validation', {})
            spatial_val = qa_results.get('spatial_validation', {})
            
            logger.info(f"  - Spatial consistency: {spatial_val.get('spatial_consistency', 'Unknown')}")
            logger.info(f"  - Total memory usage: {qa_results.get('performance_metrics', {}).get('total_memory_mb', 0):.1f} MB")
            
            # Count datasets with issues
            issues = 0
            for name, metrics in data_val.items():
                if isinstance(metrics, dict):
                    if not metrics.get('nodata_acceptable', True) or not metrics.get('range_valid', True):
                        issues += 1
            
            if issues > 0:
                logger.warning(f"  - {issues} dataset(s) have quality issues")
            else:
                logger.info("  - All datasets passed quality checks")
            
            logger.info(f"  - Detailed QA report: {output_dir}/qa_results/")
        
        # Summary
        logger.info("\n6. Processing Summary")
        logger.info("-" * 30)
        
        total_memory = 0
        total_pixels = 0
        
        for name, data in processed_data.items():
            if hasattr(data, 'nbytes'):
                total_memory += data.nbytes
            if hasattr(data, 'size'):
                total_pixels += data.size
        
        logger.info(f"✓ Processing completed successfully!")
        logger.info(f"  - Total data processed: {total_memory / 1024 / 1024:.1f} MB")
        logger.info(f"  - Total pixels: {total_pixels:,}")
        logger.info(f"  - Output directory: {output_dir}")
        
        if args.create_tiles and 'tiles' in locals():
            logger.info(f"  - Training tiles: {len(tiles)}")
        
        # Store processing metadata
        metadata = {
            "timestamp": processed_data.get('validation', {}).get('timestamp', "unknown"),
            "dem_file": str(dem_path),
            "output_directory": str(output_dir),
            "region": args.region,
            "bounds": bounds,
            "total_datasets": len(processed_data),
            "total_memory_mb": total_memory / 1024 / 1024,
            "total_pixels": total_pixels,
            "qa_status": qa_results.get('overall_status', 'unknown') if args.run_qa else 'not_run',
            "tiles_created": len(tiles) if args.create_tiles and 'tiles' in locals() else 0
        }
        
        with open(output_dir / "processing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  - Metadata saved: processing_metadata.json")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Check the logs above for details")
        return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())