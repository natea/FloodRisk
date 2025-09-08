#!/usr/bin/env python3
"""
FloodRisk Data Acquisition Example

This example demonstrates how to use the FloodRisk data acquisition system
to download DEM and rainfall data for flood risk modeling.

Usage:
    python examples/data_acquisition_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manager import DataManager
from src.data.config import DataConfig, BoundingBox
from src.data.sources.usgs_3dep import USGS3DEPDownloader
from src.data.sources.noaa_atlas14 import NOAAAtlas14Fetcher


def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_1_basic_setup():
    """Example 1: Basic setup and configuration."""
    print("\n=== Example 1: Basic Setup and Configuration ===")
    
    # Create default configuration
    config = DataConfig()
    print(f"Cache directory: {config.cache_dir}")
    print(f"Data directory: {config.data_dir}")
    print(f"Target CRS: {config.target_crs}")
    print(f"Caching enabled: {config.enable_caching}")
    
    # List available regions
    print(f"Available regions: {list(config.regions.keys())}")
    
    # Get Nashville bounding box
    nashville_bbox = config.get_region_bbox('nashville')
    print(f"Nashville bounds: {nashville_bbox.bounds_tuple}")
    
    return config


def example_2_usgs_dem_download():
    """Example 2: Download USGS DEM data."""
    print("\n=== Example 2: USGS DEM Data Download ===")
    
    # Initialize downloader
    downloader = USGS3DEPDownloader()
    
    # Query available DEM products for Nashville
    print("Querying available DEM products...")
    nashville_bbox = BoundingBox(west=-86.9, south=36.0, east=-86.6, north=36.3)
    
    try:
        products = downloader.get_available_data(nashville_bbox, resolution=10)
        print(f"Found {len(products)} DEM products")
        
        if products:
            # Show details of first product
            product = products[0]
            print(f"Example product:")
            print(f"  ID: {product['id']}")
            print(f"  Title: {product['title']}")
            print(f"  Format: {product['format']}")
            print(f"  Size: {product['size_bytes'] / 1024 / 1024:.1f} MB")
    
    except Exception as e:
        print(f"Error querying DEM products: {e}")
    
    # Download data for the Nashville region (uncomment to actually download)
    # print("Downloading DEM data...")
    # try:
    #     dem_files = downloader.download_region(
    #         region_name="nashville",
    #         resolution=10,
    #         output_dir=Path("examples/data/dem")
    #     )
    #     print(f"Downloaded {len(dem_files)} DEM files")
    #     for file_path in dem_files[:3]:  # Show first 3 files
    #         print(f"  {file_path}")
    # except Exception as e:
    #     print(f"Error downloading DEM data: {e}")


def example_3_noaa_rainfall_data():
    """Example 3: Download NOAA rainfall data."""
    print("\n=== Example 3: NOAA Rainfall Data Download ===")
    
    # Initialize fetcher
    fetcher = NOAAAtlas14Fetcher()
    
    # Nashville coordinates
    nashville_coords = (-86.7816, 36.1627)
    
    # Query available data for point
    print("Querying available rainfall data...")
    try:
        available_data = fetcher.get_available_data(nashville_coords)
        print(f"Data source: {available_data['source']}")
        print(f"Return periods: {available_data['return_periods']}")
        print(f"Available durations: {len(available_data['durations_minutes'])}")
    
    except Exception as e:
        print(f"Error querying rainfall data: {e}")
    
    # Get rainfall depths for key return periods (uncomment to actually download)
    # print("Getting rainfall depths for 24-hour duration...")
    # try:
    #     rainfall_depths = fetcher.get_rainfall_depths(
    #         location=nashville_coords,
    #         return_periods=[10, 25, 100, 500],
    #         duration_hours=24
    #     )
    #     print("24-hour rainfall depths (inches):")
    #     for return_period, depth in rainfall_depths.items():
    #         print(f"  {return_period}-year: {depth:.2f} inches")
    # except Exception as e:
    #     print(f"Error getting rainfall depths: {e}")


def example_4_data_manager():
    """Example 4: Using DataManager for coordinated downloads."""
    print("\n=== Example 4: Data Manager Coordinated Downloads ===")
    
    # Initialize data manager
    config = DataConfig(
        data_dir=Path("examples/data"),
        enable_caching=True
    )
    manager = DataManager(config)
    
    print("Data Manager initialized")
    print(f"Output directory: {config.data_dir}")
    
    # Show what would be downloaded for Nashville case study
    print("\nNashville case study would download:")
    print("- DEM: 10m resolution USGS 3DEP data")
    print("- Rainfall: NOAA Atlas 14 data for return periods [10, 25, 100, 500] years")
    print("- Rainfall: Durations [1, 3, 6, 12, 24] hours")
    print("- Rainfall: Grid spacing ~500m (0.005 degrees)")
    
    # Uncomment to actually download Nashville case study data
    # print("Starting Nashville case study download...")
    # try:
    #     results = manager.download_nashville_case_study(
    #         output_dir=Path("examples/data/nashville"),
    #         dem_resolution=10,
    #         rainfall_grid_spacing=0.01  # Coarser grid for example
    #     )
    #     
    #     print(f"Download completed:")
    #     print(f"  DEM files: {len(results.get('dem', []))}")
    #     print(f"  Rainfall files: {len(results.get('rainfall', []))}")
    #     
    #     # Validate downloaded data
    #     print("Validating downloaded data...")
    #     all_files = results.get('dem', []) + results.get('rainfall', [])
    #     validation = manager.validate_data_integrity(all_files)
    #     print(f"  Valid files: {len(validation['valid'])}")
    #     print(f"  Invalid files: {len(validation['invalid'])}")
    #     
    # except Exception as e:
    #     print(f"Error in case study download: {e}")
    # finally:
    #     manager.cleanup()


def example_5_custom_region():
    """Example 5: Define and download data for custom region."""
    print("\n=== Example 5: Custom Region Definition ===")
    
    # Define custom region (smaller area for example)
    custom_region = BoundingBox(
        west=-86.85,
        south=36.10,
        east=-86.75,
        north=36.20,
        crs=4326
    )
    
    print(f"Custom region bounds: {custom_region.bounds_tuple}")
    print(f"Region size: {custom_region.east - custom_region.west:.2f}° × "
          f"{custom_region.north - custom_region.south:.2f}°")
    
    # Add to configuration
    config = DataConfig()
    config.add_region("downtown_nashville", custom_region)
    
    print(f"Added region 'downtown_nashville' to configuration")
    print(f"Available regions: {list(config.regions.keys())}")
    
    # Initialize manager with updated config
    manager = DataManager(config)
    
    # Show what would be downloaded
    print("\nDowntown Nashville region would include:")
    print("- Smaller geographic area for faster downloads")
    print("- Same data types (DEM + rainfall)")
    print("- Suitable for testing and development")
    
    manager.cleanup()


def example_6_validation_and_qa():
    """Example 6: Data validation and quality assurance."""
    print("\n=== Example 6: Data Validation and Quality Assurance ===")
    
    # Show validation configuration
    config = DataConfig()
    print("Validation settings:")
    print(f"  Minimum file size: {config.min_file_size_bytes} bytes")
    print(f"  Validation enabled: {config.validate_downloads}")
    print(f"  Cache expiry: {config.cache_expiry_days} days")
    print(f"  Max retries: {config.max_retries}")
    
    # Example validation workflow
    print("\nValidation workflow:")
    print("1. File existence and size checks")
    print("2. Format validation (GeoTIFF, CSV)")
    print("3. Coordinate reference system validation")
    print("4. Data range validation (elevation, precipitation)")
    print("5. Missing data detection")
    print("6. Spatial consistency checks")
    
    # Show how to use validation script
    print("\nTo validate downloaded data:")
    print("python scripts/data_acquisition/validate_data.py /path/to/data --verbose")


def main():
    """Run all examples."""
    setup_logging()
    
    print("FloodRisk Data Acquisition Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_1_basic_setup()
        example_2_usgs_dem_download()
        example_3_noaa_rainfall_data()
        example_4_data_manager()
        example_5_custom_region()
        example_6_validation_and_qa()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nTo actually download data:")
        print("1. Uncomment the download code in the examples")
        print("2. Ensure you have sufficient disk space")
        print("3. Check internet connectivity")
        print("4. Monitor progress with --verbose flag")
        
        print("\nNext steps:")
        print("- Run the Nashville download script:")
        print("  python scripts/data_acquisition/download_nashville_data.py --verbose")
        print("- Validate downloaded data:")
        print("  python scripts/data_acquisition/validate_data.py data/")
        print("- Integrate with FloodRisk ML pipeline")
        
    except Exception as e:
        logging.error(f"Example execution failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())