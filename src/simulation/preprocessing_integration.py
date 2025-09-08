"""Integration with existing preprocessing pipeline.

This module provides seamless integration between the preprocessing pipeline
and the LISFLOOD-FP simulation system, ensuring consistent data flow and
format compatibility.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not available - GeoTIFF processing disabled")

from .parameter_generator import ParameterFileGenerator, ReturnPeriodConfig, HyetographConfig
from .batch_orchestrator import SimulationBatch, BatchConfig
from .lisflood_simulator import LisfloodSimulator

logger = logging.getLogger(__name__)


class PreprocessingIntegration:
    """Integrates simulation pipeline with preprocessing system."""
    
    def __init__(self,
                 preprocessing_output_dir: str,
                 simulation_output_dir: str,
                 temp_dir: Optional[str] = None):
        """Initialize preprocessing integration.
        
        Args:
            preprocessing_output_dir: Directory with preprocessed data
            simulation_output_dir: Directory for simulation outputs
            temp_dir: Temporary directory for format conversions
        """
        self.preprocess_dir = Path(preprocessing_output_dir)
        self.simulation_dir = Path(simulation_output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.simulation_dir / "temp"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("PreprocessingIntegration initialized")
        logger.info(f"Preprocessing dir: {self.preprocess_dir}")
        logger.info(f"Simulation dir: {self.simulation_dir}")
    
    def prepare_dem_for_simulation(self,
                                 preprocessed_dem_path: str,
                                 output_path: Optional[str] = None,
                                 target_format: str = "asc") -> str:
        """Convert preprocessed DEM to LISFLOOD-FP format.
        
        Args:
            preprocessed_dem_path: Path to preprocessed DEM file
            output_path: Output file path (auto-generated if None)
            target_format: Target format ('asc' for ASCII grid)
            
        Returns:
            Path to converted DEM file
        """
        dem_path = Path(preprocessed_dem_path)
        
        if not dem_path.exists():
            raise FileNotFoundError(f"Preprocessed DEM not found: {dem_path}")
        
        if output_path is None:
            output_path = self.temp_dir / f"dem_{dem_path.stem}.{target_format}"
        
        output_path = Path(output_path)
        
        logger.info(f"Converting DEM: {dem_path} -> {output_path}")
        
        # If already in ASC format and properly conditioned, just copy
        if dem_path.suffix.lower() == '.asc' and target_format == 'asc':
            import shutil
            shutil.copy2(dem_path, output_path)
            logger.info("DEM already in ASCII format - copied")
            return str(output_path)
        
        # Convert using rasterio if available
        if HAS_RASTERIO:
            return self._convert_dem_with_rasterio(dem_path, output_path, target_format)
        else:
            logger.warning("rasterio not available - limited format conversion")
            # Fallback: assume compatible format
            import shutil
            shutil.copy2(dem_path, output_path)
            return str(output_path)
    
    def _convert_dem_with_rasterio(self, 
                                  input_path: Path, 
                                  output_path: Path,
                                  target_format: str) -> str:
        """Convert DEM using rasterio."""
        
        with rasterio.open(input_path) as src:
            # Read DEM data
            dem_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                # Replace nodata with a reasonable elevation (e.g., minimum valid value)
                valid_mask = dem_data != nodata
                if np.any(valid_mask):
                    min_valid = np.min(dem_data[valid_mask])
                    dem_data[dem_data == nodata] = min_valid
            
            # Convert to ASC format
            if target_format.lower() == 'asc':
                self._write_ascii_grid(dem_data, transform, output_path)
            else:
                # Use rasterio for other formats
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=dem_data.shape[0],
                    width=dem_data.shape[1],
                    count=1,
                    dtype=dem_data.dtype,
                    crs=crs,
                    transform=transform,
                    compress='lzw'
                ) as dst:
                    dst.write(dem_data, 1)
        
        logger.info(f"DEM converted successfully: {output_path}")
        return str(output_path)
    
    def _write_ascii_grid(self, data: np.ndarray, transform, output_path: Path):
        """Write data as ESRI ASCII grid."""
        height, width = data.shape
        
        # Extract geospatial info from transform
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        xllcorner = transform[2]
        yllcorner = transform[5] - height * pixel_height
        
        # Write ASCII grid header and data
        with open(output_path, 'w') as f:
            f.write(f"ncols {width}\n")
            f.write(f"nrows {height}\n")
            f.write(f"xllcorner {xllcorner:.6f}\n")
            f.write(f"yllcorner {yllcorner:.6f}\n")
            f.write(f"cellsize {pixel_width:.6f}\n")
            f.write(f"NODATA_value -9999\n")
            
            # Write data row by row
            for row in data:
                row_str = ' '.join(f'{val:.6f}' for val in row)
                f.write(row_str + '\n')
    
    def create_auxiliary_input_files(self,
                                   dem_bounds: Tuple[float, float, float, float],
                                   grid_shape: Tuple[int, int],
                                   output_dir: str) -> Dict[str, str]:
        """Create auxiliary input files (Manning's n, infiltration) from DEM.
        
        Args:
            dem_bounds: DEM spatial bounds (minx, miny, maxx, maxy)
            grid_shape: Grid dimensions (height, width)
            output_dir: Output directory for auxiliary files
            
        Returns:
            Dictionary mapping file types to paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        height, width = grid_shape
        
        # Create Manning's n file (simplified - constant value for urban areas)
        manning_file = output_path / "manning.asc"
        self._create_constant_ascii_grid(
            value=0.05,  # Typical urban Manning's n
            shape=(height, width),
            bounds=dem_bounds,
            output_file=manning_file,
            description="Manning's roughness coefficient"
        )
        
        # Create infiltration file (simplified - constant Green-Ampt parameters)
        infiltration_file = output_path / "infiltration.asc"
        self._create_constant_ascii_grid(
            value=5.0,  # mm/hr infiltration rate
            shape=(height, width),
            bounds=dem_bounds,
            output_file=infiltration_file,
            description="Infiltration rate"
        )
        
        auxiliary_files = {
            'manning_file': str(manning_file),
            'infiltration_file': str(infiltration_file)
        }
        
        logger.info(f"Created auxiliary input files: {list(auxiliary_files.keys())}")
        return auxiliary_files
    
    def _create_constant_ascii_grid(self,
                                   value: float,
                                   shape: Tuple[int, int],
                                   bounds: Tuple[float, float, float, float],
                                   output_file: Path,
                                   description: str):
        """Create ASCII grid with constant value."""
        height, width = shape
        minx, miny, maxx, maxy = bounds
        
        cellsize = (maxx - minx) / width
        
        with open(output_file, 'w') as f:
            f.write(f"# {description}\n")
            f.write(f"ncols {width}\n")
            f.write(f"nrows {height}\n")
            f.write(f"xllcorner {minx:.6f}\n")
            f.write(f"yllcorner {miny:.6f}\n")
            f.write(f"cellsize {cellsize:.6f}\n")
            f.write(f"NODATA_value -9999\n")
            
            # Write constant value grid
            row_str = ' '.join([str(value)] * width)
            for _ in range(height):
                f.write(row_str + '\n')
    
    def extract_rainfall_from_preprocessing(self,
                                          preprocessing_metadata: Dict,
                                          return_periods: List[int],
                                          location_name: str = "site") -> Dict[int, float]:
        """Extract rainfall depths from preprocessing metadata.
        
        Args:
            preprocessing_metadata: Metadata from preprocessing pipeline
            return_periods: Required return periods (years)
            location_name: Location identifier
            
        Returns:
            Dictionary mapping return periods to 24-hour rainfall depths (mm)
        """
        rainfall_data = {}
        
        # Try to extract from preprocessing metadata
        if 'rainfall_statistics' in preprocessing_metadata:
            stats = preprocessing_metadata['rainfall_statistics']
            
            for rp in return_periods:
                # Look for various naming conventions
                possible_keys = [
                    f'{rp}yr_24h_mm',
                    f'return_period_{rp}',
                    f'rp_{rp}_depth',
                    f'{rp}year'
                ]
                
                for key in possible_keys:
                    if key in stats:
                        rainfall_data[rp] = float(stats[key])
                        break
                else:
                    logger.warning(f"No rainfall data found for {rp}-year return period")
        
        # Use default values if not found in metadata
        defaults = self._get_default_rainfall_depths(location_name)
        for rp in return_periods:
            if rp not in rainfall_data and rp in defaults:
                rainfall_data[rp] = defaults[rp]
                logger.info(f"Using default rainfall for {rp}-year: {defaults[rp]:.1f} mm")
        
        return rainfall_data
    
    def _get_default_rainfall_depths(self, location_name: str) -> Dict[int, float]:
        """Get default rainfall depths for location."""
        # Default values for Nashville (NOAA Atlas 14)
        defaults = {
            'nashville': {
                10: 111.76,   # 4.4 inches
                25: 142.24,   # 5.6 inches 
                100: 177.8,   # 7.01 inches
                500: 222.25   # 8.75 inches
            },
            'default': {
                10: 100.0,
                25: 130.0,
                100: 170.0,
                500: 220.0
            }
        }
        
        location_key = location_name.lower()
        return defaults.get(location_key, defaults['default'])
    
    def setup_simulation_from_preprocessing(self,
                                          preprocessing_config_file: str,
                                          return_periods: Optional[List[int]] = None,
                                          hyetograph_patterns: Optional[List[str]] = None) -> Dict:
        """Set up complete simulation from preprocessing configuration.
        
        Args:
            preprocessing_config_file: Path to preprocessing configuration
            return_periods: Return periods to simulate (uses defaults if None)
            hyetograph_patterns: Hyetograph patterns (uses defaults if None)
            
        Returns:
            Dictionary with simulation setup information
        """
        logger.info("Setting up simulation from preprocessing configuration")
        
        # Load preprocessing configuration
        config_path = Path(preprocessing_config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Preprocessing config not found: {config_path}")
        
        with open(config_path) as f:
            preprocess_config = json.load(f)
        
        logger.info(f"Loaded preprocessing config: {config_path}")
        
        # Extract essential information
        dem_info = preprocess_config.get('dem_info', {})
        bounds = dem_info.get('bounds')  # [minx, miny, maxx, maxy]
        grid_shape = dem_info.get('shape')  # [height, width]
        location_name = preprocess_config.get('location', 'default')
        
        if not bounds or not grid_shape:
            raise ValueError("Incomplete DEM information in preprocessing config")
        
        # Prepare DEM file
        preprocessed_dem = preprocess_config.get('outputs', {}).get('conditioned_dem')
        if not preprocessed_dem:
            raise ValueError("No conditioned DEM found in preprocessing outputs")
        
        simulation_dem = self.prepare_dem_for_simulation(
            preprocessed_dem,
            output_path=self.temp_dir / "simulation_dem.asc"
        )
        
        # Create auxiliary files
        auxiliary_files = self.create_auxiliary_input_files(
            dem_bounds=bounds,
            grid_shape=grid_shape,
            output_dir=str(self.temp_dir)
        )
        
        # Set up return periods
        if return_periods is None:
            return_periods = [100, 500]  # Default design events
        
        rainfall_depths = self.extract_rainfall_from_preprocessing(
            preprocess_config.get('metadata', {}),
            return_periods,
            location_name
        )
        
        # Create return period configurations
        rp_configs = []
        for rp in return_periods:
            if rp in rainfall_depths:
                is_sub_design = rp < 50  # Heuristic
                rp_config = ReturnPeriodConfig(
                    return_period_years=rp,
                    rainfall_depth_24h_mm=rainfall_depths[rp],
                    description=f"{rp}-year return period for {location_name}",
                    is_sub_design=is_sub_design
                )
                rp_configs.append(rp_config)
        
        # Set up hyetograph patterns
        if hyetograph_patterns is None:
            hyetograph_patterns = ['uniform', 'center_loaded', 'front_loaded']
        
        hyet_configs = []
        for pattern in hyetograph_patterns:
            if pattern == 'uniform':
                hyet_configs.append(HyetographConfig('uniform', 0))
            elif pattern == 'center_loaded':
                hyet_configs.append(HyetographConfig('center_loaded', 0, peak_hour=12.0))
            elif pattern == 'front_loaded':
                hyet_configs.append(HyetographConfig('front_loaded', 0, front_factor=2.5))
            elif pattern == 'back_loaded':
                hyet_configs.append(HyetographConfig('back_loaded', 0, front_factor=2.5))
        
        # Compile simulation setup
        setup_info = {
            'location': location_name,
            'dem_file': simulation_dem,
            'auxiliary_files': auxiliary_files,
            'dem_bounds': bounds,
            'grid_shape': grid_shape,
            'return_period_configs': rp_configs,
            'hyetograph_configs': hyet_configs,
            'total_scenarios': len(rp_configs) * len(hyet_configs),
            'preprocessing_config': preprocess_config
        }
        
        logger.info(f"Simulation setup completed:")
        logger.info(f"  - Location: {location_name}")
        logger.info(f"  - Grid shape: {grid_shape}")
        logger.info(f"  - Return periods: {[rp.return_period_years for rp in rp_configs]}")
        logger.info(f"  - Patterns: {[h.pattern_type for h in hyet_configs]}")
        logger.info(f"  - Total scenarios: {setup_info['total_scenarios']}")
        
        return setup_info
    
    def run_integrated_simulation(self,
                                preprocessing_config_file: str,
                                max_parallel_jobs: int = 4,
                                return_periods: Optional[List[int]] = None,
                                hyetograph_patterns: Optional[List[str]] = None) -> Dict:
        """Run complete integrated simulation from preprocessing to results.
        
        Args:
            preprocessing_config_file: Path to preprocessing configuration
            max_parallel_jobs: Number of parallel simulation jobs
            return_periods: Return periods to simulate
            hyetograph_patterns: Hyetograph patterns to use
            
        Returns:
            Complete simulation results and metadata
        """
        logger.info("Starting integrated simulation pipeline")
        
        # Set up simulation from preprocessing
        setup_info = self.setup_simulation_from_preprocessing(
            preprocessing_config_file,
            return_periods,
            hyetograph_patterns
        )
        
        # Initialize simulation components
        simulator = LisfloodSimulator(
            working_directory=str(self.simulation_dir / "runs")
        )
        
        param_generator = ParameterFileGenerator()
        
        batch_config = BatchConfig(
            max_parallel_jobs=max_parallel_jobs,
            validate_results=True,
            cleanup_failed_runs=True
        )
        
        batch_orchestrator = SimulationBatch(simulator, param_generator, batch_config)
        
        # Create and execute batch
        batch_id = batch_orchestrator.create_batch_from_config(
            dem_file=setup_info['dem_file'],
            return_periods=setup_info['return_period_configs'],
            hyetograph_patterns=setup_info['hyetograph_configs'],
            output_dir=str(self.simulation_dir / "scenarios"),
            base_sim_config=setup_info['auxiliary_files']
        )
        
        # Execute batch
        batch_summary = batch_orchestrator.execute_batch()
        
        # Compile integrated results
        integrated_results = {
            'setup_info': setup_info,
            'batch_id': batch_id,
            'batch_summary': batch_summary,
            'simulation_results': batch_orchestrator.results,
            'preprocessing_integration': {
                'preprocessing_config_file': preprocessing_config_file,
                'simulation_dir': str(self.simulation_dir),
                'temp_files_created': list(self.temp_dir.glob('*'))
            }
        }
        
        # Save integrated results
        results_file = self.simulation_dir / "integrated_results.json"
        with open(results_file, 'w') as f:
            json.dump(integrated_results, f, indent=2, default=str)
        
        logger.info(f"Integrated simulation completed!")
        logger.info(f"Success rate: {batch_summary['success_rate']:.1%}")
        logger.info(f"Results saved to: {results_file}")
        
        return integrated_results
    
    def export_for_ml_training(self,
                             integrated_results: Dict,
                             ml_training_dir: str,
                             include_preprocessing_features: bool = True) -> str:
        """Export simulation results for ML training with preprocessing integration.
        
        Args:
            integrated_results: Results from run_integrated_simulation
            ml_training_dir: Directory for ML training data
            include_preprocessing_features: Whether to include preprocessing features
            
        Returns:
            Path to training data manifest
        """
        ml_dir = Path(ml_training_dir)
        ml_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting training data to: {ml_dir}")
        
        # Extract successful simulation results
        sim_results = integrated_results['simulation_results']
        successful = [r for r in sim_results if r.get('status') == 'success']
        
        if not successful:
            raise ValueError("No successful simulations to export")
        
        setup_info = integrated_results['setup_info']
        
        # Create ML training manifest
        training_manifest = {
            'created_at': json.dumps(None, default=str),  # Current timestamp
            'source': 'integrated_preprocessing_simulation',
            'location': setup_info['location'],
            'dem_bounds': setup_info['dem_bounds'],
            'grid_shape': setup_info['grid_shape'],
            'total_samples': len(successful),
            'preprocessing_config': setup_info.get('preprocessing_config', {}),
            'samples': []
        }
        
        # Process each successful simulation
        for result in successful:
            sample = {
                'simulation_id': result['simulation_id'],
                'scenario': result.get('scenario', {}),
                'flood_extent_file': result['outputs'].get('extent_file'),
                'depth_file': result['outputs'].get('depth_file'),
                'statistics': result['outputs'].get('statistics', {}),
                'simulation_metadata': {
                    'runtime_seconds': result.get('runtime_seconds'),
                    'validation': result.get('validation', {}),
                    'config': result.get('config', {})
                }
            }
            
            # Add preprocessing features if requested
            if include_preprocessing_features:
                sample['preprocessing_features'] = {
                    'dem_file': setup_info['dem_file'],
                    'preprocessing_source': setup_info.get('preprocessing_config', {}).get('source'),
                    'conditioning_applied': True  # Since we use conditioned DEM
                }
            
            training_manifest['samples'].append(sample)
        
        # Save training manifest
        manifest_file = ml_dir / "training_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(training_manifest, f, indent=2, default=str)
        
        logger.info(f"Training data manifest exported: {manifest_file}")
        logger.info(f"Training samples available: {len(successful)}")
        
        return str(manifest_file)