#!/usr/bin/env python3
"""
Nashville case study data acquisition script.

This script downloads all required data for the Nashville flood risk case study:
- USGS 3DEP 10m DEM data
- NOAA Atlas 14 precipitation frequency data for multiple return periods
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.manager import DataManager
from src.data.config import DataConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nashville_data_download.log')
        ]
    )


def main():
    """Main data acquisition script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Nashville flood risk case study data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--dem-resolution", "-r",
        type=int,
        default=10,
        choices=[1, 10, 30],
        help="DEM resolution in meters (default: 10)"
    )
    parser.add_argument(
        "--rainfall-spacing", "-s",
        type=float,
        default=0.005,
        help="Grid spacing for rainfall data in degrees (default: 0.005 ≈ 500m)"
    )
    parser.add_argument(
        "--cache-dir", "-c",
        type=Path,
        help="Cache directory for downloaded data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel downloads"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Nashville case study data acquisition")
    
    try:
        # Initialize configuration
        config_kwargs = {}
        if args.output_dir:
            config_kwargs['data_dir'] = args.output_dir
        if args.cache_dir:
            config_kwargs['cache_dir'] = args.cache_dir
            
        config = DataConfig.from_env(**config_kwargs)
        
        # Initialize data manager
        data_manager = DataManager(config)
        
        logger.info(f"Configuration:")
        logger.info(f"  Output directory: {config.data_dir}")
        logger.info(f"  Cache directory: {config.cache_dir}")
        logger.info(f"  DEM resolution: {args.dem_resolution}m")
        logger.info(f"  Rainfall grid spacing: {args.rainfall_spacing}°")
        
        # Download Nashville data
        results = data_manager.download_all_region_data(
            region_name="nashville",
            dem_resolution=args.dem_resolution,
            rainfall_return_periods=[10, 25, 100, 500],  # Key return periods
            rainfall_durations_hours=[1, 3, 6, 12, 24],  # Key durations
            rainfall_grid_spacing=args.rainfall_spacing,
            output_dir=args.output_dir,
            parallel=not args.no_parallel
        )
        
        # Validate downloaded data
        logger.info("Validating downloaded data...")
        
        dem_validation = data_manager.validate_data_integrity(
            results.get('dem', []), 'DEM'
        )
        rainfall_validation = data_manager.validate_data_integrity(
            results.get('rainfall', []), 'rainfall'
        )
        
        # Report results
        logger.info("\nDownload Summary:")
        logger.info(f"DEM files: {len(results.get('dem', []))} downloaded, "
                   f"{len(dem_validation['valid'])} valid")
        logger.info(f"Rainfall files: {len(results.get('rainfall', []))} downloaded, "
                   f"{len(rainfall_validation['valid'])} valid")
        
        total_files = len(results.get('dem', [])) + len(results.get('rainfall', []))
        total_valid = len(dem_validation['valid']) + len(rainfall_validation['valid'])
        
        logger.info(f"Total: {total_files} files downloaded, {total_valid} valid")
        
        if total_valid == total_files and total_files > 0:
            logger.info("✓ All downloads completed successfully!")
            return 0
        elif total_valid > 0:
            logger.warning(f"⚠ Partial success: {total_valid}/{total_files} files valid")
            return 1
        else:
            logger.error("✗ No valid files downloaded")
            return 2
            
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}", exc_info=True)
        return 1
        
    finally:
        # Cleanup
        if 'data_manager' in locals():
            data_manager.cleanup()


if __name__ == "__main__":
    exit(main())