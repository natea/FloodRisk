"""LISFLOOD-FP simulation interface for flood modeling.

This module provides the main interface to run LISFLOOD-FP physics-based
flood simulations for generating training labels.
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for LISFLOOD-FP simulation."""
    
    # Essential files
    dem_file: str
    rainfall_file: str
    manning_file: str
    infiltration_file: str
    
    # Simulation parameters
    sim_time: float = 86400.0  # 24 hours in seconds
    initial_timestep: float = 10.0
    
    # Output configuration
    output_prefix: str = "res"
    output_directory: str = "results"
    
    # Physics options
    acceleration: bool = True
    depth_threshold: float = 0.05  # Minimum flood depth (m) for extent mapping
    
    # Optional files
    boundary_file: Optional[str] = None
    floodplain_friction: Optional[float] = None
    
    # Quality control
    max_runtime_hours: float = 24.0
    expected_flooded_fraction: Tuple[float, float] = (0.001, 0.3)  # min, max expected


class LisfloodSimulator:
    """Main interface for LISFLOOD-FP flood simulations."""
    
    def __init__(self, 
                 lisflood_executable: Optional[str] = None,
                 working_directory: Optional[str] = None):
        """Initialize the LISFLOOD-FP simulator.
        
        Args:
            lisflood_executable: Path to LISFLOOD-FP executable
            working_directory: Base directory for simulations
        """
        self.lisflood_exe = self._find_executable(lisflood_executable)
        self.working_dir = Path(working_directory or "simulation_runs")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LISFLOOD-FP simulator initialized with executable: {self.lisflood_exe}")
        logger.info(f"Working directory: {self.working_dir}")
    
    def _find_executable(self, exe_path: Optional[str]) -> str:
        """Find LISFLOOD-FP executable."""
        if exe_path and os.path.isfile(exe_path):
            return exe_path
        
        # Check common locations
        candidates = [
            "LISFLOOD-FP/build/lisflood",
            "build/lisflood", 
            "./lisflood",
            "lisflood"
        ]
        
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        
        # Try system PATH
        try:
            result = subprocess.run(["which", "lisflood"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        raise FileNotFoundError("LISFLOOD-FP executable not found")
    
    def run_simulation(self, 
                      config: SimulationConfig,
                      simulation_id: str,
                      cleanup_temp: bool = True) -> Dict:
        """Run a single LISFLOOD-FP simulation.
        
        Args:
            config: Simulation configuration
            simulation_id: Unique identifier for this simulation
            cleanup_temp: Whether to cleanup temporary files after completion
            
        Returns:
            Dictionary with simulation results and metadata
        """
        logger.info(f"Starting simulation: {simulation_id}")
        
        # Create simulation directory
        sim_dir = self.working_dir / simulation_id
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        
        try:
            # Generate parameter file
            par_file = self._generate_parameter_file(config, sim_dir)
            
            # Copy input files to simulation directory
            self._prepare_input_files(config, sim_dir)
            
            # Run LISFLOOD-FP
            result = self._execute_lisflood(par_file, sim_dir, config.max_runtime_hours)
            
            # Process results
            output_info = self._process_outputs(config, sim_dir, simulation_id)
            
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            # Validate results
            validation_results = self._validate_simulation_results(
                output_info, config, runtime
            )
            
            # Cleanup if requested
            if cleanup_temp:
                self._cleanup_temporary_files(sim_dir)
            
            return {
                'simulation_id': simulation_id,
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'runtime_seconds': runtime,
                'config': config.__dict__,
                'outputs': output_info,
                'validation': validation_results,
                'working_directory': str(sim_dir)
            }
            
        except Exception as e:
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            logger.error(f"Simulation {simulation_id} failed: {str(e)}")
            
            return {
                'simulation_id': simulation_id,
                'status': 'failed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'runtime_seconds': runtime,
                'error': str(e),
                'working_directory': str(sim_dir)
            }
    
    def _generate_parameter_file(self, config: SimulationConfig, sim_dir: Path) -> str:
        """Generate LISFLOOD-FP parameter file."""
        par_content = []
        par_content.append("# LISFLOOD-FP parameter file - Auto-generated")
        par_content.append(f"# Generated: {datetime.now().isoformat()}")
        par_content.append("")
        
        # Essential parameters - use basename since files are copied to sim_dir
        par_content.append(f"DEMfile\t\t\t{os.path.basename(config.dem_file)}")
        par_content.append(f"resroot\t\t\t{config.output_prefix}")
        par_content.append(f"dirroot\t\t\t{config.output_directory}")
        par_content.append(f"sim_time\t\t{config.sim_time}")
        par_content.append(f"initial_tstep\t\t{config.initial_timestep}")
        
        # Input files - use basename since files are copied to sim_dir
        par_content.append(f"rainfall\t\t{os.path.basename(config.rainfall_file)}")
        par_content.append(f"manningfile\t\t{config.manning_file}")
        par_content.append(f"infiltration\t\t{config.infiltration_file}")
        
        # Optional parameters
        if config.boundary_file:
            par_content.append(f"bcifile\t\t\t{config.boundary_file}")
        
        if config.floodplain_friction:
            par_content.append(f"fpfric\t\t\t{config.floodplain_friction}")
        
        # Physics options
        par_content.append("depthoff")
        par_content.append("elevoff")
        
        if config.acceleration:
            par_content.append("acceleration")
        
        # Write parameter file
        par_file = sim_dir / "simulation.par"
        par_file.write_text("\n".join(par_content))
        
        logger.info(f"Generated parameter file: {par_file}")
        return str(par_file)
    
    def _prepare_input_files(self, config: SimulationConfig, sim_dir: Path):
        """Copy and prepare input files for simulation."""
        files_to_copy = [
            config.dem_file,
            config.rainfall_file,
            config.manning_file,
            config.infiltration_file
        ]
        
        if config.boundary_file:
            files_to_copy.append(config.boundary_file)
        
        for file_path in files_to_copy:
            if not os.path.isabs(file_path):
                # Check if file exists relative to current directory first
                source = Path(file_path)
                if not source.exists():
                    # Try relative to LISFLOOD-FP/Nashville as fallback
                    source = Path("LISFLOOD-FP/Nashville") / file_path
            else:
                source = Path(file_path)
            
            if not source.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            dest = sim_dir / source.name
            shutil.copy2(source, dest)
            logger.debug(f"Copied {source} -> {dest}")
    
    def _execute_lisflood(self, par_file: str, sim_dir: Path, max_hours: float) -> subprocess.CompletedProcess:
        """Execute LISFLOOD-FP simulation."""
        # Ensure executable path is absolute
        lisflood_exe_abs = os.path.abspath(self.lisflood_exe)
        # Use just the filename for the parameter file (not full path)
        par_filename = os.path.basename(par_file)
        cmd = [lisflood_exe_abs, par_filename]
        
        logger.info(f"Executing LISFLOOD-FP: {' '.join(cmd)}")
        logger.info(f"Working directory: {sim_dir}")
        
        # Run with timeout
        timeout_seconds = max_hours * 3600
        
        try:
            result = subprocess.run(
                cmd,
                cwd=sim_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"LISFLOOD-FP failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"LISFLOOD-FP execution failed: {result.stderr}")
            
            logger.info("LISFLOOD-FP execution completed successfully")
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Simulation timed out after {max_hours} hours")
            raise TimeoutError(f"Simulation exceeded maximum runtime of {max_hours} hours")
    
    def _process_outputs(self, config: SimulationConfig, sim_dir: Path, simulation_id: str) -> Dict:
        """Process simulation outputs and extract flood extent."""
        outputs = {}
        
        # Find depth output file (.max file)
        results_dir = sim_dir / config.output_directory
        max_files = list(results_dir.glob(f"{config.output_prefix}*.max"))
        
        if not max_files:
            raise FileNotFoundError("No depth output (.max) files found")
        
        depth_file = max_files[0]  # Take the first .max file
        outputs['depth_file'] = str(depth_file)
        
        # Load depth data and create flood extent
        depth_data = self._load_lisflood_output(depth_file)
        flood_extent = self._create_flood_extent(depth_data, config.depth_threshold)
        
        # Save flood extent
        extent_file = results_dir / f"flood_extent_{simulation_id}.npy"
        np.save(extent_file, flood_extent.astype(np.uint8))
        outputs['extent_file'] = str(extent_file)
        
        # Calculate statistics
        total_cells = depth_data.size
        flooded_cells = np.sum(flood_extent)
        flooded_fraction = flooded_cells / total_cells
        max_depth = np.max(depth_data)
        mean_depth_flooded = np.mean(depth_data[flood_extent])
        
        outputs['statistics'] = {
            'total_cells': total_cells,
            'flooded_cells': int(flooded_cells),
            'flooded_fraction': float(flooded_fraction),
            'max_depth_m': float(max_depth),
            'mean_depth_flooded_m': float(mean_depth_flooded),
            'depth_threshold_m': config.depth_threshold
        }
        
        logger.info(f"Processed simulation outputs: {flooded_cells}/{total_cells} cells flooded "
                   f"({flooded_fraction:.3%}), max depth: {max_depth:.2f}m")
        
        return outputs
    
    def _load_lisflood_output(self, file_path: Path) -> np.ndarray:
        """Load LISFLOOD-FP output file (.max format)."""
        # LISFLOOD-FP .max files are binary format
        # Need to determine grid dimensions from DEM or parameter files
        try:
            # Try to read as binary float32 array
            data = np.fromfile(file_path, dtype=np.float32)
            
            # Try to infer dimensions (this is simplified - real implementation 
            # would need to read DEM header or use LISFLOOD-FP output format)
            size = data.shape[0]
            
            # Common grid sizes - try to find square or reasonable rectangle
            possible_dims = []
            for i in range(1, int(np.sqrt(size)) + 1):
                if size % i == 0:
                    possible_dims.append((i, size // i))
            
            # Take the most square-like dimensions
            dims = min(possible_dims, key=lambda x: abs(x[0] - x[1]))
            
            return data.reshape(dims)
            
        except Exception as e:
            logger.error(f"Failed to load LISFLOOD output {file_path}: {e}")
            raise ValueError(f"Unable to parse LISFLOOD-FP output file: {file_path}")
    
    def _create_flood_extent(self, depth_data: np.ndarray, threshold: float) -> np.ndarray:
        """Create binary flood extent from depth data."""
        flood_extent = depth_data >= threshold
        
        # Apply morphological cleaning to remove noise
        from scipy import ndimage
        
        # Remove small isolated flooded areas (less than 4 connected pixels)
        labeled, num_features = ndimage.label(flood_extent)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < 4:
                flood_extent[labeled == i] = False
        
        # Close small gaps (1-2 pixel gaps)
        struct = ndimage.generate_binary_structure(2, 1)
        flood_extent = ndimage.binary_closing(flood_extent, structure=struct, iterations=2)
        
        return flood_extent
    
    def _validate_simulation_results(self, 
                                   output_info: Dict, 
                                   config: SimulationConfig,
                                   runtime_seconds: float) -> Dict:
        """Validate simulation results for quality control."""
        validation = {'status': 'passed', 'warnings': [], 'errors': []}
        
        stats = output_info['statistics']
        flooded_fraction = stats['flooded_fraction']
        max_depth = stats['max_depth_m']
        
        # Check flooded fraction is reasonable
        min_expected, max_expected = config.expected_flooded_fraction
        if flooded_fraction < min_expected:
            validation['warnings'].append(
                f"Low flooded fraction: {flooded_fraction:.3%} < {min_expected:.3%}"
            )
        elif flooded_fraction > max_expected:
            validation['errors'].append(
                f"Excessive flooded fraction: {flooded_fraction:.3%} > {max_expected:.3%}"
            )
        
        # Check maximum depth is reasonable
        if max_depth > 50.0:  # 50m seems excessive for pluvial flooding
            validation['errors'].append(f"Unrealistic maximum depth: {max_depth:.1f}m")
        elif max_depth < 0.1:  # Very shallow flooding
            validation['warnings'].append(f"Very shallow maximum depth: {max_depth:.3f}m")
        
        # Check simulation runtime
        if runtime_seconds > config.max_runtime_hours * 3600 * 0.9:  # 90% of max time
            validation['warnings'].append(
                f"Long simulation runtime: {runtime_seconds/3600:.1f}h"
            )
        
        # Overall status
        if validation['errors']:
            validation['status'] = 'failed'
        elif validation['warnings']:
            validation['status'] = 'passed_with_warnings'
        
        return validation
    
    def _cleanup_temporary_files(self, sim_dir: Path):
        """Clean up temporary files, keeping only essential outputs."""
        # Keep these files
        keep_patterns = [
            "*.max",           # Depth outputs
            "flood_extent_*.npy",  # Processed extent files
            "simulation.par",   # Parameter file
            "*.log"            # Log files
        ]
        
        # Get all files to keep
        keep_files = set()
        for pattern in keep_patterns:
            keep_files.update(sim_dir.rglob(pattern))
        
        # Remove all other files
        for file_path in sim_dir.rglob("*"):
            if file_path.is_file() and file_path not in keep_files:
                try:
                    file_path.unlink()
                except OSError:
                    pass  # Ignore errors during cleanup
        
        logger.info(f"Cleaned up temporary files in {sim_dir}")