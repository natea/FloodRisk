"""Batch orchestration for multiple LISFLOOD-FP simulations.

This module orchestrates the execution of multiple flood simulations in parallel,
supporting different return periods and hyetograph patterns for comprehensive
ML training dataset generation.
"""

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .lisflood_simulator import LisfloodSimulator, SimulationConfig
from .parameter_generator import ParameterFileGenerator, ReturnPeriodConfig, HyetographConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch simulation execution."""
    
    max_parallel_jobs: int = 4
    max_retries: int = 2
    retry_delay_seconds: float = 60.0
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_disk_space_gb: Optional[float] = None
    
    # Quality control
    validate_results: bool = True
    cleanup_failed_runs: bool = True
    keep_intermediate_files: bool = False
    
    # Monitoring
    progress_callback: Optional[Callable[[Dict], None]] = None
    result_callback: Optional[Callable[[Dict], None]] = None


class SimulationBatch:
    """Orchestrates batch execution of multiple LISFLOOD-FP simulations."""
    
    def __init__(self,
                 simulator: LisfloodSimulator,
                 parameter_generator: ParameterFileGenerator,
                 batch_config: Optional[BatchConfig] = None):
        """Initialize batch orchestrator.
        
        Args:
            simulator: LISFLOOD-FP simulator instance
            parameter_generator: Parameter file generator
            batch_config: Configuration for batch execution
        """
        self.simulator = simulator
        self.param_gen = parameter_generator
        self.config = batch_config or BatchConfig()
        
        # Execution state
        self.batch_id = None
        self.scenarios = []
        self.results = []
        self.failed_jobs = []
        self.start_time = None
        self.end_time = None
        
        logger.info("SimulationBatch initialized")
    
    def create_batch_from_config(self,
                                dem_file: str,
                                return_periods: List[ReturnPeriodConfig],
                                hyetograph_patterns: List[HyetographConfig],
                                output_dir: str,
                                base_sim_config: Optional[Dict] = None) -> str:
        """Create batch from configuration parameters.
        
        Args:
            dem_file: Path to DEM file
            return_periods: List of return period configurations  
            hyetograph_patterns: List of rainfall patterns
            output_dir: Output directory for batch
            base_sim_config: Base simulation configuration
            
        Returns:
            Batch ID for tracking
        """
        self.batch_id = f"batch_{int(time.time())}"
        batch_dir = Path(output_dir) / self.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate scenarios
        self.scenarios = self.param_gen.generate_scenario_parameters(
            dem_file=dem_file,
            return_periods=return_periods,
            hyetograph_patterns=hyetograph_patterns,
            output_dir=str(batch_dir),
            base_simulation_config=base_sim_config
        )
        
        # Create batch metadata
        batch_metadata = {
            'batch_id': self.batch_id,
            'created_at': datetime.now().isoformat(),
            'dem_file': dem_file,
            'output_dir': str(batch_dir),
            'total_scenarios': len(self.scenarios),
            'return_periods': [rp.__dict__ for rp in return_periods],
            'hyetograph_patterns': [hp.__dict__ for hp in hyetograph_patterns],
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not callable(v)}
        }
        
        # Save batch metadata
        with open(batch_dir / "batch_metadata.json", 'w') as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        
        logger.info(f"Created batch {self.batch_id} with {len(self.scenarios)} scenarios")
        return self.batch_id
    
    def create_batch_from_scenarios(self, scenarios: List[Dict], output_dir: str) -> str:
        """Create batch from pre-defined scenarios.
        
        Args:
            scenarios: List of scenario configurations
            output_dir: Output directory for batch
            
        Returns:
            Batch ID for tracking
        """
        self.batch_id = f"batch_{int(time.time())}"
        self.scenarios = scenarios
        
        batch_dir = Path(output_dir) / self.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch metadata
        batch_metadata = {
            'batch_id': self.batch_id,
            'created_at': datetime.now().isoformat(),
            'output_dir': str(batch_dir),
            'total_scenarios': len(self.scenarios),
            'scenarios': self.scenarios,
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not callable(v)}
        }
        
        # Save batch metadata
        with open(batch_dir / "batch_metadata.json", 'w') as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        
        logger.info(f"Created batch {self.batch_id} with {len(self.scenarios)} scenarios")
        return self.batch_id
    
    def execute_batch(self) -> Dict:
        """Execute all simulations in the batch.
        
        Returns:
            Batch execution summary
        """
        if not self.scenarios:
            raise ValueError("No scenarios defined. Call create_batch_from_config first.")
        
        self.start_time = datetime.now()
        logger.info(f"Starting batch execution: {self.batch_id}")
        logger.info(f"Total scenarios: {len(self.scenarios)}")
        logger.info(f"Max parallel jobs: {self.config.max_parallel_jobs}")
        
        # Execute simulations
        if self.config.max_parallel_jobs > 1:
            self.results = self._execute_parallel()
        else:
            self.results = self._execute_sequential()
        
        self.end_time = datetime.now()
        
        # Generate summary
        summary = self._generate_batch_summary()
        
        # Save results
        self._save_batch_results(summary)
        
        logger.info(f"Batch execution completed: {summary['success_count']}/{summary['total_scenarios']} successful")
        
        return summary
    
    def _execute_parallel(self) -> List[Dict]:
        """Execute simulations in parallel using ThreadPoolExecutor."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_parallel_jobs
        ) as executor:
            
            # Submit all jobs
            future_to_scenario = {
                executor.submit(
                    self._execute_scenario_with_retry,
                    scenario
                ): scenario
                for scenario in self.scenarios
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress callback
                    if self.config.progress_callback:
                        completed_count += 1
                        progress_info = {
                            'completed': completed_count,
                            'total': len(self.scenarios),
                            'current_result': result,
                            'batch_id': self.batch_id
                        }
                        self.config.progress_callback(progress_info)
                    
                    # Result callback
                    if self.config.result_callback:
                        self.config.result_callback(result)
                        
                except Exception as e:
                    logger.error(f"Scenario {scenario['scenario_id']} failed with exception: {e}")
                    
                    error_result = {
                        'scenario_id': scenario['scenario_id'],
                        'status': 'failed',
                        'error': str(e),
                        'scenario': scenario
                    }
                    results.append(error_result)
                    self.failed_jobs.append(error_result)
        
        return results
    
    def _execute_sequential(self) -> List[Dict]:
        """Execute simulations sequentially."""
        results = []
        
        for i, scenario in enumerate(self.scenarios):
            logger.info(f"Executing scenario {i+1}/{len(self.scenarios)}: {scenario['scenario_id']}")
            
            try:
                result = self._execute_scenario_with_retry(scenario)
                results.append(result)
                
                # Progress callback
                if self.config.progress_callback:
                    progress_info = {
                        'completed': i + 1,
                        'total': len(self.scenarios),
                        'current_result': result,
                        'batch_id': self.batch_id
                    }
                    self.config.progress_callback(progress_info)
                
                # Result callback
                if self.config.result_callback:
                    self.config.result_callback(result)
                    
            except Exception as e:
                logger.error(f"Scenario {scenario['scenario_id']} failed with exception: {e}")
                
                error_result = {
                    'scenario_id': scenario['scenario_id'],
                    'status': 'failed',
                    'error': str(e),
                    'scenario': scenario
                }
                results.append(error_result)
                self.failed_jobs.append(error_result)
        
        return results
    
    def _execute_scenario_with_retry(self, scenario: Dict) -> Dict:
        """Execute a single scenario with retry logic."""
        scenario_id = scenario['scenario_id']
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retrying scenario {scenario_id}, attempt {attempt + 1}")
                time.sleep(self.config.retry_delay_seconds)
            
            try:
                # Create simulation config from scenario
                sim_config = self._scenario_to_simulation_config(scenario)
                
                # Execute simulation
                result = self.simulator.run_simulation(
                    config=sim_config,
                    simulation_id=scenario_id,
                    cleanup_temp=not self.config.keep_intermediate_files
                )
                
                # Add scenario metadata to result
                result['scenario'] = scenario
                
                # Validate if requested
                if self.config.validate_results and result['status'] == 'success':
                    validation = self._validate_scenario_result(result)
                    result['batch_validation'] = validation
                    
                    if not validation['passed']:
                        logger.warning(f"Scenario {scenario_id} failed validation: {validation['issues']}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Scenario {scenario_id} attempt {attempt + 1} failed: {e}")
                
                # Cleanup failed run if requested
                if self.config.cleanup_failed_runs:
                    self._cleanup_failed_scenario(scenario_id)
        
        # All retries failed
        logger.error(f"Scenario {scenario_id} failed after {self.config.max_retries + 1} attempts")
        raise last_error
    
    def _scenario_to_simulation_config(self, scenario: Dict) -> SimulationConfig:
        """Convert scenario dictionary to SimulationConfig."""
        config_dict = scenario['config']
        
        # Use original DEM file path for copying, not the basename
        dem_file = scenario.get('original_dem_file', config_dict['dem_file'])
        
        return SimulationConfig(
            dem_file=dem_file,
            rainfall_file=scenario['rainfall_file'],
            manning_file=config_dict.get('manning_file'), 
            infiltration_file=config_dict.get('infiltration_file'),
            sim_time=config_dict['sim_time'],
            initial_timestep=config_dict.get('initial_timestep', 0.1),
            output_prefix=config_dict['output_prefix'],
            output_directory=config_dict['output_directory'],
            acceleration=config_dict.get('acceleration', True),
            boundary_file=config_dict.get('boundary_file'),
            floodplain_friction=config_dict.get('floodplain_friction')
        )
    
    def _validate_scenario_result(self, result: Dict) -> Dict:
        """Additional validation for batch execution."""
        validation = {'passed': True, 'issues': []}
        
        if result['status'] != 'success':
            validation['passed'] = False
            validation['issues'].append(f"Simulation failed: {result.get('error', 'Unknown error')}")
            return validation
        
        # Check if outputs exist
        outputs = result.get('outputs', {})
        required_outputs = ['depth_file', 'extent_file']
        
        for output in required_outputs:
            if output not in outputs:
                validation['passed'] = False
                validation['issues'].append(f"Missing required output: {output}")
            elif not Path(outputs[output]).exists():
                validation['passed'] = False
                validation['issues'].append(f"Output file not found: {outputs[output]}")
        
        # Check simulation validation
        sim_validation = result.get('validation', {})
        if sim_validation.get('status') == 'failed':
            validation['passed'] = False
            validation['issues'].extend(sim_validation.get('errors', []))
        
        return validation
    
    def _cleanup_failed_scenario(self, scenario_id: str):
        """Clean up files from failed scenario."""
        try:
            sim_dir = self.simulator.working_dir / scenario_id
            if sim_dir.exists():
                import shutil
                shutil.rmtree(sim_dir)
                logger.debug(f"Cleaned up failed scenario directory: {sim_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup scenario {scenario_id}: {e}")
    
    def _generate_batch_summary(self) -> Dict:
        """Generate comprehensive batch execution summary."""
        total_scenarios = len(self.scenarios)
        successful_results = [r for r in self.results if r.get('status') == 'success']
        failed_results = [r for r in self.results if r.get('status') != 'success']
        
        # Calculate execution time
        runtime_seconds = 0
        if self.start_time and self.end_time:
            runtime_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Aggregate statistics from successful simulations
        stats_summary = self._aggregate_simulation_statistics(successful_results)
        
        summary = {
            'batch_id': self.batch_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'runtime_seconds': runtime_seconds,
            'total_scenarios': total_scenarios,
            'success_count': len(successful_results),
            'failure_count': len(failed_results),
            'success_rate': len(successful_results) / total_scenarios if total_scenarios > 0 else 0,
            'failed_scenarios': [r.get('scenario_id', 'unknown') for r in failed_results],
            'statistics_summary': stats_summary,
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not callable(v)},
            'successful_results': successful_results,
            'failed_results': failed_results
        }
        
        return summary
    
    def _aggregate_simulation_statistics(self, successful_results: List[Dict]) -> Dict:
        """Aggregate statistics from successful simulations."""
        if not successful_results:
            return {}
        
        # Extract statistics from all successful runs
        all_stats = []
        for result in successful_results:
            if 'outputs' in result and 'statistics' in result['outputs']:
                all_stats.append(result['outputs']['statistics'])
        
        if not all_stats:
            return {}
        
        # Calculate aggregated metrics
        import numpy as np
        
        flooded_fractions = [s.get('flooded_fraction', 0) for s in all_stats]
        max_depths = [s.get('max_depth_m', 0) for s in all_stats]
        mean_depths = [s.get('mean_depth_flooded_m', 0) for s in all_stats]
        
        summary = {
            'total_simulations': len(all_stats),
            'flooded_fraction': {
                'mean': float(np.mean(flooded_fractions)),
                'std': float(np.std(flooded_fractions)),
                'min': float(np.min(flooded_fractions)),
                'max': float(np.max(flooded_fractions))
            },
            'max_depth_m': {
                'mean': float(np.mean(max_depths)),
                'std': float(np.std(max_depths)),
                'min': float(np.min(max_depths)),
                'max': float(np.max(max_depths))
            },
            'mean_depth_flooded_m': {
                'mean': float(np.mean(mean_depths)),
                'std': float(np.std(mean_depths)),
                'min': float(np.min(mean_depths)),
                'max': float(np.max(mean_depths))
            }
        }
        
        return summary
    
    def _save_batch_results(self, summary: Dict):
        """Save batch results to disk."""
        if not self.batch_id:
            return
        
        batch_dir = self.simulator.working_dir.parent / self.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        with open(batch_dir / "batch_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        with open(batch_dir / "batch_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Batch results saved to {batch_dir}")
    
    def get_successful_results(self) -> List[Dict]:
        """Get list of successful simulation results."""
        return [r for r in self.results if r.get('status') == 'success']
    
    def get_failed_results(self) -> List[Dict]:
        """Get list of failed simulation results."""
        return [r for r in self.results if r.get('status') != 'success']
    
    def export_training_data_paths(self, output_file: str) -> Dict:
        """Export paths to flood extent files for ML training.
        
        Args:
            output_file: Path to save training data manifest
            
        Returns:
            Dictionary mapping scenario IDs to extent file paths
        """
        successful_results = self.get_successful_results()
        
        training_data = {}
        for result in successful_results:
            # Handle both scenario_id and simulation_id keys
            scenario_id = result.get('scenario_id', result.get('simulation_id', 'unknown'))
            if 'outputs' in result and 'extent_file' in result['outputs']:
                extent_file = result['outputs']['extent_file']
                if Path(extent_file).exists():
                    training_data[scenario_id] = {
                        'extent_file': extent_file,
                        'depth_file': result['outputs'].get('depth_file'),
                        'scenario': result['scenario'],
                        'statistics': result['outputs'].get('statistics', {})
                    }
        
        # Save manifest
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(training_data)} training data paths to {output_file}")
        
        return training_data