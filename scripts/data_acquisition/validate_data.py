#!/usr/bin/env python3
"""
Data validation script for FloodRisk datasets.

Validates DEM and rainfall data files for completeness, format compliance,
and spatial consistency.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import rasterio
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    from pyproj import CRS
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False

from src.data.config import DataConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class DataValidator:
    """Validates data files for FloodRisk pipeline."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize validator.
        
        Args:
            config: Data configuration object
        """
        self.config = config or DataConfig()
        self.logger = logging.getLogger(__name__)
        
        if not GEOSPATIAL_AVAILABLE:
            self.logger.warning(
                "Geospatial libraries not available. "
                "Only basic file validation will be performed."
            )
    
    def validate_dem_files(self, dem_files: List[Path]) -> Dict[str, List[Path]]:
        """Validate DEM files.
        
        Args:
            dem_files: List of DEM file paths
            
        Returns:
            Dictionary with 'valid', 'invalid', and 'warnings' lists
        """
        self.logger.info(f"Validating {len(dem_files)} DEM files")
        
        results = {'valid': [], 'invalid': [], 'warnings': []}
        
        for dem_file in dem_files:
            try:
                # Basic file checks
                if not dem_file.exists():
                    self.logger.error(f"DEM file not found: {dem_file}")
                    results['invalid'].append(dem_file)
                    continue
                
                if dem_file.stat().st_size < self.config.min_file_size_bytes:
                    self.logger.error(f"DEM file too small: {dem_file}")
                    results['invalid'].append(dem_file)
                    continue
                
                # Geospatial validation if libraries available
                if GEOSPATIAL_AVAILABLE:
                    validation_result = self._validate_dem_geospatial(dem_file)
                    
                    if validation_result['valid']:
                        results['valid'].append(dem_file)
                        if validation_result['warnings']:
                            results['warnings'].append(dem_file)
                    else:
                        results['invalid'].append(dem_file)
                else:
                    # Basic validation only
                    results['valid'].append(dem_file)
                
            except Exception as e:
                self.logger.error(f"Error validating DEM file {dem_file}: {e}")
                results['invalid'].append(dem_file)
        
        self.logger.info(
            f"DEM validation complete: {len(results['valid'])} valid, "
            f"{len(results['invalid'])} invalid, {len(results['warnings'])} warnings"
        )
        
        return results
    
    def _validate_dem_geospatial(self, dem_file: Path) -> Dict[str, any]:
        """Perform geospatial validation of DEM file.
        
        Args:
            dem_file: Path to DEM file
            
        Returns:
            Dictionary with validation results
        """
        result = {'valid': True, 'warnings': []}
        
        try:
            with rasterio.open(dem_file) as src:
                # Check CRS
                if src.crs is None:
                    result['warnings'].append("No CRS defined")
                
                # Check data type
                if src.dtypes[0] not in ['float32', 'float64', 'int16', 'int32']:
                    result['warnings'].append(f"Unusual data type: {src.dtypes[0]}")
                
                # Check for NoData values
                if src.nodata is None:
                    result['warnings'].append("No NoData value defined")
                
                # Check raster dimensions
                if src.width < 10 or src.height < 10:
                    result['valid'] = False
                    self.logger.error(f"DEM raster too small: {src.width}x{src.height}")
                
                # Sample data to check for valid elevation values
                sample_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                valid_data = sample_data[sample_data != src.nodata] if src.nodata else sample_data
                
                if len(valid_data) == 0:
                    result['valid'] = False
                    self.logger.error("No valid elevation data found")
                else:
                    # Check elevation range (reasonable for Earth)
                    min_elev, max_elev = valid_data.min(), valid_data.max()
                    if min_elev < -500 or max_elev > 10000:
                        result['warnings'].append(
                            f"Unusual elevation range: {min_elev:.1f} to {max_elev:.1f}m"
                        )
                
        except Exception as e:
            result['valid'] = False
            self.logger.error(f"Geospatial validation failed for {dem_file}: {e}")
        
        return result
    
    def validate_rainfall_files(self, rainfall_files: List[Path]) -> Dict[str, List[Path]]:
        """Validate rainfall data files.
        
        Args:
            rainfall_files: List of rainfall file paths (CSV format)
            
        Returns:
            Dictionary with 'valid', 'invalid', and 'warnings' lists
        """
        self.logger.info(f"Validating {len(rainfall_files)} rainfall files")
        
        results = {'valid': [], 'invalid': [], 'warnings': []}
        
        for rainfall_file in rainfall_files:
            try:
                # Basic file checks
                if not rainfall_file.exists():
                    self.logger.error(f"Rainfall file not found: {rainfall_file}")
                    results['invalid'].append(rainfall_file)
                    continue
                
                if rainfall_file.stat().st_size < 100:  # Very small for CSV
                    self.logger.error(f"Rainfall file too small: {rainfall_file}")
                    results['invalid'].append(rainfall_file)
                    continue
                
                # CSV content validation
                validation_result = self._validate_rainfall_csv(rainfall_file)
                
                if validation_result['valid']:
                    results['valid'].append(rainfall_file)
                    if validation_result['warnings']:
                        results['warnings'].append(rainfall_file)
                else:
                    results['invalid'].append(rainfall_file)
                
            except Exception as e:
                self.logger.error(f"Error validating rainfall file {rainfall_file}: {e}")
                results['invalid'].append(rainfall_file)
        
        self.logger.info(
            f"Rainfall validation complete: {len(results['valid'])} valid, "
            f"{len(results['invalid'])} invalid, {len(results['warnings'])} warnings"
        )
        
        return results
    
    def _validate_rainfall_csv(self, csv_file: Path) -> Dict[str, any]:
        """Validate rainfall CSV file content.
        
        Args:
            csv_file: Path to rainfall CSV file
            
        Returns:
            Dictionary with validation results
        """
        result = {'valid': True, 'warnings': []}
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check required columns
            required_cols = ['return_period_years', 'duration_minutes', 'precipitation_inches']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                result['valid'] = False
                self.logger.error(f"Missing required columns in {csv_file}: {missing_cols}")
                return result
            
            # Check data types and ranges
            if not pd.api.types.is_numeric_dtype(df['return_period_years']):
                result['warnings'].append("Return periods not numeric")
            else:
                return_periods = df['return_period_years'].unique()
                if any(rp < 1 or rp > 1000 for rp in return_periods):
                    result['warnings'].append("Unusual return periods found")
            
            if not pd.api.types.is_numeric_dtype(df['precipitation_inches']):
                result['warnings'].append("Precipitation values not numeric")
            else:
                precip_values = df['precipitation_inches']
                if precip_values.min() < 0:
                    result['valid'] = False
                    self.logger.error("Negative precipitation values found")
                elif precip_values.max() > 50:  # Very high rainfall
                    result['warnings'].append("Very high precipitation values found")
            
            # Check for missing data
            if df.isnull().any().any():
                result['warnings'].append("Missing values found in data")
            
            # Check row count
            if len(df) < 5:
                result['warnings'].append("Very few data rows")
            
        except Exception as e:
            result['valid'] = False
            self.logger.error(f"Failed to validate CSV content: {e}")
        
        return result
    
    def generate_validation_report(
        self,
        dem_results: Dict[str, List[Path]],
        rainfall_results: Dict[str, List[Path]],
        output_file: Path
    ) -> None:
        """Generate validation report.
        
        Args:
            dem_results: DEM validation results
            rainfall_results: Rainfall validation results
            output_file: Path for report output
        """
        import json
        from datetime import datetime
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {
                'dem_files': {
                    'total': len(dem_results['valid']) + len(dem_results['invalid']),
                    'valid': len(dem_results['valid']),
                    'invalid': len(dem_results['invalid']),
                    'warnings': len(dem_results['warnings'])
                },
                'rainfall_files': {
                    'total': len(rainfall_results['valid']) + len(rainfall_results['invalid']),
                    'valid': len(rainfall_results['valid']),
                    'invalid': len(rainfall_results['invalid']),
                    'warnings': len(rainfall_results['warnings'])
                }
            },
            'details': {
                'dem': {
                    'valid_files': [str(p) for p in dem_results['valid']],
                    'invalid_files': [str(p) for p in dem_results['invalid']],
                    'warning_files': [str(p) for p in dem_results['warnings']]
                },
                'rainfall': {
                    'valid_files': [str(p) for p in rainfall_results['valid']],
                    'invalid_files': [str(p) for p in rainfall_results['invalid']],
                    'warning_files': [str(p) for p in rainfall_results['warnings']]
                }
            },
            'configuration': {
                'min_file_size_bytes': self.config.min_file_size_bytes,
                'geospatial_validation': GEOSPATIAL_AVAILABLE
            }
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Validation report saved: {output_file}")


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate FloodRisk data files"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing data files to validate"
    )
    parser.add_argument(
        "--output-report", "-o",
        type=Path,
        default=Path("validation_report.json"),
        help="Output file for validation report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    logger.info(f"Validating data in: {args.data_dir}")
    
    try:
        # Initialize validator
        validator = DataValidator()
        
        # Find DEM files
        dem_files = list(args.data_dir.rglob("*.tif")) + list(args.data_dir.rglob("*.img"))
        logger.info(f"Found {len(dem_files)} DEM files")
        
        # Find rainfall files
        rainfall_files = list(args.data_dir.rglob("*atlas14*.csv"))
        logger.info(f"Found {len(rainfall_files)} rainfall files")
        
        # Validate files
        dem_results = validator.validate_dem_files(dem_files)
        rainfall_results = validator.validate_rainfall_files(rainfall_files)
        
        # Generate report
        validator.generate_validation_report(dem_results, rainfall_results, args.output_report)
        
        # Summary
        total_files = len(dem_files) + len(rainfall_files)
        total_valid = len(dem_results['valid']) + len(rainfall_results['valid'])
        total_invalid = len(dem_results['invalid']) + len(rainfall_results['invalid'])
        
        logger.info(f"\nValidation Summary:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Valid files: {total_valid}")
        logger.info(f"Invalid files: {total_invalid}")
        
        if total_invalid == 0:
            logger.info("✓ All files passed validation!")
            return 0
        else:
            logger.warning(f"⚠ {total_invalid} files failed validation")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())