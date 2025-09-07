"""Data acquisition manager for coordinating multiple data sources."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BoundingBox, DataConfig
from .sources.usgs_3dep import USGS3DEPDownloader
from .sources.noaa_atlas14 import NOAAAtlas14Fetcher
from .sources.base import DataSourceError


logger = logging.getLogger(__name__)


class DataManager:
    """Manages acquisition from multiple data sources.
    
    Provides a unified interface for downloading DEM and precipitation data
    from various sources with coordination, error handling, and progress tracking.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize data manager.
        
        Args:
            config: Data configuration object
        """
        self.config = config or DataConfig()
        
        # Initialize data sources
        self.usgs_downloader = USGS3DEPDownloader(self.config)
        self.noaa_fetcher = NOAAAtlas14Fetcher(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def download_all_region_data(
        self,
        region_name: str,
        dem_resolution: int = 10,
        rainfall_return_periods: Optional[List[int]] = None,
        rainfall_durations_hours: Optional[List[float]] = None,
        rainfall_grid_spacing: float = 0.01,
        output_dir: Optional[Path] = None,
        parallel: bool = True
    ) -> Dict[str, List[Path]]:
        """Download all data types for a region.
        
        Args:
            region_name: Name of predefined region
            dem_resolution: DEM resolution in meters
            rainfall_return_periods: Return periods for rainfall data
            rainfall_durations_hours: Durations for rainfall data
            rainfall_grid_spacing: Grid spacing for rainfall data
            output_dir: Base output directory
            parallel: Whether to download data sources in parallel
            
        Returns:
            Dictionary with 'dem' and 'rainfall' keys containing file paths
            
        Raises:
            DataSourceError: If downloads fail
        """
        bbox = self.config.get_region_bbox(region_name)
        if bbox is None:
            available_regions = list(self.config.regions.keys())
            raise DataSourceError(
                f"Region '{region_name}' not found. "
                f"Available regions: {available_regions}"
            )
        
        if output_dir is None:
            output_dir = self.config.data_dir / "regions" / region_name
        
        self.logger.info(f"Starting complete data download for region: {region_name}")
        
        results = {'dem': [], 'rainfall': []}
        
        if parallel:
            # Download data sources in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit download tasks
                dem_future = executor.submit(
                    self._download_dem_safe,
                    region_name, dem_resolution, output_dir / "dem"
                )
                
                rainfall_future = executor.submit(
                    self._download_rainfall_safe,
                    region_name, rainfall_return_periods, rainfall_durations_hours,
                    rainfall_grid_spacing, output_dir / "rainfall"
                )
                
                # Collect results
                for future in as_completed([dem_future, rainfall_future]):
                    try:
                        source_type, files = future.result()
                        results[source_type] = files
                        self.logger.info(f"Completed {source_type} download: {len(files)} files")
                    except Exception as e:
                        self.logger.error(f"Data source download failed: {e}")
        
        else:
            # Download sequentially
            try:
                results['dem'] = self.usgs_downloader.download_region(
                    region_name=region_name,
                    resolution=dem_resolution,
                    output_dir=output_dir / "dem"
                )
                self.logger.info(f"DEM download completed: {len(results['dem'])} files")
            except Exception as e:
                self.logger.error(f"DEM download failed: {e}")
            
            try:
                results['rainfall'] = self.noaa_fetcher.download_region(
                    region_name=region_name,
                    return_periods=rainfall_return_periods,
                    durations_hours=rainfall_durations_hours,
                    grid_spacing=rainfall_grid_spacing,
                    output_dir=output_dir / "rainfall"
                )
                self.logger.info(f"Rainfall download completed: {len(results['rainfall'])} files")
            except Exception as e:
                self.logger.error(f"Rainfall download failed: {e}")
        
        total_files = len(results['dem']) + len(results['rainfall'])
        self.logger.info(f"Data download completed for {region_name}: {total_files} total files")
        
        return results
    
    def _download_dem_safe(
        self,
        region_name: str,
        resolution: int,
        output_dir: Path
    ) -> tuple[str, List[Path]]:
        """Safely download DEM data with error handling."""
        try:
            files = self.usgs_downloader.download_region(
                region_name=region_name,
                resolution=resolution,
                output_dir=output_dir
            )
            return ('dem', files)
        except Exception as e:
            self.logger.error(f"DEM download failed: {e}")
            return ('dem', [])
    
    def _download_rainfall_safe(
        self,
        region_name: str,
        return_periods: Optional[List[int]],
        durations_hours: Optional[List[float]],
        grid_spacing: float,
        output_dir: Path
    ) -> tuple[str, List[Path]]:
        """Safely download rainfall data with error handling."""
        try:
            files = self.noaa_fetcher.download_region(
                region_name=region_name,
                return_periods=return_periods,
                durations_hours=durations_hours,
                grid_spacing=grid_spacing,
                output_dir=output_dir
            )
            return ('rainfall', files)
        except Exception as e:
            self.logger.error(f"Rainfall download failed: {e}")
            return ('rainfall', [])
    
    def download_nashville_case_study(
        self,
        output_dir: Optional[Path] = None,
        dem_resolution: int = 10,
        rainfall_grid_spacing: float = 0.005  # ~500m spacing for Nashville
    ) -> Dict[str, List[Path]]:
        """Download complete Nashville case study dataset.
        
        Args:
            output_dir: Output directory for Nashville data
            dem_resolution: DEM resolution in meters
            rainfall_grid_spacing: Grid spacing for rainfall data
            
        Returns:
            Dictionary with downloaded file paths
        """
        if output_dir is None:
            output_dir = self.config.data_dir / "case_studies" / "nashville"
        
        self.logger.info("Starting Nashville case study data download")
        
        # Nashville-specific parameters
        return_periods = [10, 25, 100, 500]  # Focus on key return periods
        durations = [1, 3, 6, 12, 24]  # Key durations for flood modeling
        
        results = self.download_all_region_data(
            region_name="nashville",
            dem_resolution=dem_resolution,
            rainfall_return_periods=return_periods,
            rainfall_durations_hours=durations,
            rainfall_grid_spacing=rainfall_grid_spacing,
            output_dir=output_dir,
            parallel=True
        )
        
        # Generate summary report
        self._generate_download_report(results, output_dir / "download_report.json")
        
        return results
    
    def _generate_download_report(
        self,
        results: Dict[str, List[Path]],
        report_path: Path
    ) -> None:
        """Generate download summary report.
        
        Args:
            results: Download results
            report_path: Path to save report
        """
        import json
        from datetime import datetime
        
        report = {
            'download_timestamp': datetime.now().isoformat(),
            'data_sources': {
                'dem': {
                    'source': 'USGS 3DEP',
                    'files_downloaded': len(results.get('dem', [])),
                    'file_paths': [str(p) for p in results.get('dem', [])]
                },
                'rainfall': {
                    'source': 'NOAA Atlas 14',
                    'files_downloaded': len(results.get('rainfall', [])),
                    'file_paths': [str(p) for p in results.get('rainfall', [])]
                }
            },
            'total_files': sum(len(files) for files in results.values()),
            'configuration': {
                'cache_dir': str(self.config.cache_dir),
                'data_dir': str(self.config.data_dir),
                'target_crs': self.config.target_crs,
                'enable_caching': self.config.enable_caching
            }
        }
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Download report saved: {report_path}")
    
    def validate_data_integrity(
        self,
        file_paths: List[Path],
        data_type: str = "unknown"
    ) -> Dict[str, List[Path]]:
        """Validate integrity of downloaded data files.
        
        Args:
            file_paths: List of file paths to validate
            data_type: Type of data (for logging)
            
        Returns:
            Dictionary with 'valid' and 'invalid' file lists
        """
        self.logger.info(f"Validating {len(file_paths)} {data_type} files")
        
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            try:
                if not file_path.exists():
                    self.logger.error(f"File not found: {file_path}")
                    invalid_files.append(file_path)
                    continue
                
                # Check minimum file size
                if file_path.stat().st_size < self.config.min_file_size_bytes:
                    self.logger.error(f"File too small: {file_path}")
                    invalid_files.append(file_path)
                    continue
                
                # Additional format-specific validation could go here
                # (e.g., GDAL validation for GeoTIFF files)
                
                valid_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Validation error for {file_path}: {e}")
                invalid_files.append(file_path)
        
        self.logger.info(
            f"Validation complete: {len(valid_files)} valid, "
            f"{len(invalid_files)} invalid files"
        )
        
        return {'valid': valid_files, 'invalid': invalid_files}
    
    def cleanup(self) -> None:
        """Clean up resources from all data sources."""
        self.usgs_downloader.cleanup()
        self.noaa_fetcher.cleanup()
        self.logger.info("Data manager cleanup completed")