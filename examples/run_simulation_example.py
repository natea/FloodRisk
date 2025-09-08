#!/usr/bin/env python3
"""
Example script demonstrating the LISFLOOD-FP simulation pipeline.

This script shows how to:
1. Set up simulation parameters
2. Generate multiple scenarios (return periods + hyetograph patterns)
3. Run batch simulations
4. Process results for ML training
5. Validate outputs and track metadata

Usage:
    python examples/run_simulation_example.py --dem-file LISFLOOD-FP/Nashville/final_dem.asc
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import (
    LisfloodSimulator,
    ParameterFileGenerator, 
    SimulationBatch,
    ResultProcessor,
    SimulationValidator,
    SimulationMetadata
)
from simulation.lisflood_simulator import SimulationConfig
from simulation.parameter_generator import ReturnPeriodConfig, HyetographConfig
from simulation.batch_orchestrator import BatchConfig
from simulation.result_processor import ProcessingConfig
from simulation.validation import ValidationThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_example_scenarios():
    """Set up example return periods and hyetograph patterns."""
    
    # Define return periods for Nashville (from APPROACH.md)
    return_periods = [
        # Main design events for positive examples
        ReturnPeriodConfig(
            return_period_years=100,
            rainfall_depth_24h_mm=177.8,  # 7.01 inches
            description="100-year design storm - primary training target"
        ),
        ReturnPeriodConfig(
            return_period_years=500,
            rainfall_depth_24h_mm=222.25,  # 8.75 inches
            description="500-year extreme event - secondary training target"
        ),
        
        # Sub-design events for negative examples
        ReturnPeriodConfig(
            return_period_years=10,
            rainfall_depth_24h_mm=111.76,  # 4.4 inches
            description="10-year sub-design event - negative example",
            is_sub_design=True
        ),
        ReturnPeriodConfig(
            return_period_years=25,
            rainfall_depth_24h_mm=142.24,  # 5.6 inches
            description="25-year sub-design event - negative example",
            is_sub_design=True
        )
    ]
    
    # Define hyetograph patterns for diverse training
    hyetograph_patterns = [
        HyetographConfig(
            pattern_type='uniform',
            total_depth_mm=0  # Will be set by return period
        ),
        HyetographConfig(
            pattern_type='front_loaded',
            total_depth_mm=0,
            front_factor=2.5
        ),
        HyetographConfig(
            pattern_type='center_loaded',
            total_depth_mm=0,
            peak_hour=12.0,
            center_factor=3.0
        )
    ]
    
    return return_periods, hyetograph_patterns


def run_single_simulation_example(dem_file: str, output_dir: str):
    """Example of running a single simulation."""
    logger.info("=== Single Simulation Example ===")
    
    # Initialize simulator
    try:
        simulator = LisfloodSimulator(
            lisflood_executable="LISFLOOD-FP/build/lisflood",
            working_directory=str(Path(output_dir) / "single_sim")
        )
    except FileNotFoundError as e:
        logger.error(f"LISFLOOD-FP executable not found: {e}")
        logger.info("Make sure LISFLOOD-FP is compiled in LISFLOOD-FP/build/lisflood")
        return None
    
    # Create simulation configuration
    config = SimulationConfig(
        dem_file=dem_file,
        rainfall_file="LISFLOOD-FP/Nashville/rain_input_100yr.rain",
        manning_file="LISFLOOD-FP/Nashville/manning.asc",
        infiltration_file="LISFLOOD-FP/Nashville/infiltration.asc",
        sim_time=86400.0,  # 24 hours
        initial_timestep=10.0,
        acceleration=True
    )
    
    # Run simulation
    logger.info("Running single simulation...")
    result = simulator.run_simulation(config, "example_100yr_uniform")
    
    if result['status'] == 'success':
        logger.info(f"Simulation completed successfully!")
        logger.info(f"Runtime: {result['runtime_seconds']:.1f} seconds")
        
        stats = result['outputs']['statistics']
        logger.info(f"Flood statistics:")
        logger.info(f"  - Flooded fraction: {stats['flood_fraction']:.3%}")
        logger.info(f"  - Max depth: {stats['max_depth_m']:.2f} m")
        logger.info(f"  - Flooded area: {stats['flooded_area_km2']:.2f} km²")
        
        return result
    else:
        logger.error(f"Simulation failed: {result.get('error', 'Unknown error')}")
        return None


def run_batch_simulation_example(dem_file: str, output_dir: str):
    """Example of running batch simulations."""
    logger.info("=== Batch Simulation Example ===")
    
    # Initialize components
    try:
        simulator = LisfloodSimulator(
            lisflood_executable="LISFLOOD-FP/build/lisflood",
            working_directory=str(Path(output_dir) / "batch_sim")
        )
    except FileNotFoundError as e:
        logger.error(f"LISFLOOD-FP executable not found: {e}")
        return None
    
    param_generator = ParameterFileGenerator(
        base_config_dir="LISFLOOD-FP/Nashville"
    )
    
    # Configure batch execution (reduced for example)
    batch_config = BatchConfig(
        max_parallel_jobs=2,  # Reduced for example
        max_retries=1,
        validate_results=True,
        cleanup_failed_runs=True
    )
    
    batch_orchestrator = SimulationBatch(simulator, param_generator, batch_config)
    
    # Set up scenarios (limited subset for example)
    return_periods, hyetograph_patterns = setup_example_scenarios()
    
    # Use only subset for example
    return_periods_subset = return_periods[:2]  # 100yr and 500yr
    hyetographs_subset = hyetograph_patterns[:2]  # uniform and front_loaded
    
    logger.info(f"Creating batch with {len(return_periods_subset)} return periods × "
               f"{len(hyetographs_subset)} hyetograph patterns = "
               f"{len(return_periods_subset) * len(hyetographs_subset)} simulations")
    
    # Create batch
    batch_id = batch_orchestrator.create_batch_from_config(
        dem_file=dem_file,
        return_periods=return_periods_subset,
        hyetograph_patterns=hyetographs_subset,
        output_dir=str(Path(output_dir) / "batch_scenarios"),
        base_sim_config={
            'manning_file': 'manning.asc',
            'infiltration_file': 'infiltration.asc'
        }
    )
    
    logger.info(f"Created batch: {batch_id}")
    
    # Set up progress callback
    def progress_callback(progress_info):
        completed = progress_info['completed']
        total = progress_info['total']
        current_result = progress_info['current_result']
        
        logger.info(f"Progress: {completed}/{total} - "
                   f"Latest: {current_result['simulation_id']} "
                   f"({current_result['status']})")
    
    batch_orchestrator.config.progress_callback = progress_callback
    
    # Execute batch
    logger.info("Executing batch simulations...")
    batch_summary = batch_orchestrator.execute_batch()
    
    logger.info(f"Batch execution completed!")
    logger.info(f"Success rate: {batch_summary['success_rate']:.1%} "
               f"({batch_summary['success_count']}/{batch_summary['total_scenarios']})")
    
    if batch_summary['failure_count'] > 0:
        logger.warning(f"Failed scenarios: {batch_summary['failed_scenarios']}")
    
    return batch_orchestrator, batch_summary


def process_results_example(batch_orchestrator, output_dir: str):
    """Example of processing simulation results for ML training."""
    logger.info("=== Result Processing Example ===")
    
    successful_results = batch_orchestrator.get_successful_results()
    
    if not successful_results:
        logger.warning("No successful simulation results to process")
        return None
    
    # Initialize result processor
    processing_config = ProcessingConfig(
        flood_depth_threshold_m=0.05,  # 5cm threshold from APPROACH.md
        remove_small_areas=True,
        min_flood_area_pixels=4,
        save_numpy=True,
        save_geotiff=False,  # Disabled if no rasterio
        save_statistics=True
    )
    
    result_processor = ResultProcessor(processing_config)
    
    # Process batch results
    logger.info(f"Processing {len(successful_results)} simulation results...")
    
    processing_output_dir = Path(output_dir) / "processed_results"
    
    batch_processing_summary = result_processor.process_batch_outputs(
        simulation_results=successful_results,
        output_dir=str(processing_output_dir)
    )
    
    logger.info("Result processing completed!")
    logger.info(f"Successfully processed: {batch_processing_summary['successful_processing']} simulations")
    
    # Show aggregate statistics
    if 'batch_statistics' in batch_processing_summary:
        stats = batch_processing_summary['batch_statistics']
        logger.info("Aggregate flood statistics:")
        logger.info(f"  - Flood fraction range: {stats['flood_fraction']['min']:.3%} - {stats['flood_fraction']['max']:.3%}")
        logger.info(f"  - Mean flood fraction: {stats['flood_fraction']['mean']:.3%}")
        logger.info(f"  - Max depth range: {stats['max_depth_m']['min']:.2f} - {stats['max_depth_m']['max']:.2f} m")
        logger.info(f"  - Total flooded area: {stats['flooded_area_km2']['total']:.1f} km²")
    
    return batch_processing_summary


def validate_results_example(batch_orchestrator, output_dir: str):
    """Example of result validation."""
    logger.info("=== Validation Example ===")
    
    successful_results = batch_orchestrator.get_successful_results()
    
    if not successful_results:
        logger.warning("No results to validate")
        return None
    
    # Configure validation thresholds
    thresholds = ValidationThresholds(
        min_flood_fraction=0.001,     # 0.1% minimum
        max_flood_fraction=0.3,       # 30% maximum for pluvial flooding
        max_reasonable_depth_m=20.0,  # 20m max for urban pluvial
        target_success_rate=0.8       # 80% success rate target
    )
    
    validator = SimulationValidator(thresholds)
    
    # Validate batch
    logger.info(f"Validating {len(successful_results)} simulation results...")
    
    batch_validation = validator.validate_batch_results(
        simulation_results=successful_results,
        detailed_validation=True
    )
    
    logger.info("Validation completed!")
    summary = batch_validation['summary']
    logger.info(f"Validation results: {summary['passed_count']} passed, "
               f"{summary['warning_count']} warnings, {summary['failed_count']} failed")
    logger.info(f"Average validation score: {summary['average_score']:.1f}/100")
    
    # Show common issues
    quality = batch_validation['quality_assessment']
    if quality['common_warnings']:
        logger.info("Most common warnings:")
        for warning, count in list(quality['common_warnings'].items())[:3]:
            logger.info(f"  - {warning}: {count} occurrences")
    
    if quality['common_errors']:
        logger.warning("Common errors found:")
        for error, count in list(quality['common_errors'].items())[:3]:
            logger.warning(f"  - {error}: {count} occurrences")
    
    return batch_validation


def track_metadata_example(batch_orchestrator, output_dir: str):
    """Example of metadata tracking."""
    logger.info("=== Metadata Tracking Example ===")
    
    metadata_dir = Path(output_dir) / "metadata"
    metadata_tracker = SimulationMetadata(
        metadata_dir=str(metadata_dir),
        enable_file_checksums=True,
        track_environment=True
    )
    
    # Create batch provenance
    batch_config = batch_orchestrator.config.__dict__
    scenarios = batch_orchestrator.scenarios
    
    batch_provenance = metadata_tracker.create_batch_provenance(
        batch_orchestrator.batch_id,
        batch_config,
        scenarios
    )
    
    # Update with execution results
    results = batch_orchestrator.results
    batch_summary = {
        'total_scenarios': len(scenarios),
        'successful_simulations': len([r for r in results if r.get('status') == 'success'])
    }
    
    metadata_tracker.update_batch_execution(
        batch_provenance,
        results,
        batch_summary
    )
    
    # Save batch provenance
    metadata_tracker.save_batch_provenance(batch_provenance)
    
    logger.info(f"Metadata tracking completed!")
    logger.info(f"Batch provenance saved: {batch_provenance.batch_id}")
    logger.info(f"Metadata directory: {metadata_dir}")
    
    # Create lineage report for one simulation
    successful_results = [r for r in results if r.get('status') == 'success']
    if successful_results:
        example_sim_id = successful_results[0]['simulation_id']
        
        # Create individual simulation provenance (simplified for example)
        sim_provenance = metadata_tracker.create_simulation_provenance(
            example_sim_id,
            {'dem': 'final_dem.asc'},  # Simplified
            {'return_period': 100}
        )
        
        metadata_tracker.save_simulation_provenance(sim_provenance)
        
        # Generate lineage report
        lineage_report = metadata_tracker.create_lineage_report(example_sim_id)
        
        logger.info(f"Example lineage report for {example_sim_id}:")
        logger.info(f"  - Input files: {lineage_report['input_summary']['file_count']}")
        logger.info(f"  - Output files: {lineage_report['output_summary']['file_count']}")
        logger.info(f"  - Quality flags: {lineage_report['quality_assessment']['quality_flags']}")
    
    return metadata_tracker


def export_training_data_example(batch_orchestrator, processing_summary, output_dir: str):
    """Example of exporting training data manifest."""
    logger.info("=== Training Data Export Example ===")
    
    # Export training data paths
    training_manifest_file = Path(output_dir) / "training_data_manifest.json"
    
    training_data = batch_orchestrator.export_training_data_paths(
        str(training_manifest_file)
    )
    
    logger.info(f"Training data manifest exported: {training_manifest_file}")
    logger.info(f"Available training samples: {len(training_data)}")
    
    # Show example training data entry
    if training_data:
        example_id = list(training_data.keys())[0]
        example_data = training_data[example_id]
        
        logger.info(f"Example training sample: {example_id}")
        logger.info(f"  - Flood extent file: {Path(example_data['extent_file']).name}")
        logger.info(f"  - Flood fraction: {example_data['statistics']['flood_fraction']:.3%}")
        logger.info(f"  - Return period: {example_data['scenario']['return_period']['return_period_years']} years")
        logger.info(f"  - Pattern: {example_data['scenario']['hyetograph']['pattern_type']}")
    
    return training_data


def main():
    """Main example execution."""
    parser = argparse.ArgumentParser(description="LISFLOOD-FP Simulation Pipeline Example")
    parser.add_argument("--dem-file", required=True,
                       help="Path to DEM file (e.g., LISFLOOD-FP/Nashville/final_dem.asc)")
    parser.add_argument("--output-dir", default="simulation_example_output",
                       help="Output directory for results")
    parser.add_argument("--mode", choices=['single', 'batch', 'full'], default='full',
                       help="Example mode: single simulation, batch, or full pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate DEM file
    if not Path(args.dem_file).exists():
        logger.error(f"DEM file not found: {args.dem_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting simulation pipeline example")
    logger.info(f"DEM file: {args.dem_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'single':
            # Single simulation example
            result = run_single_simulation_example(args.dem_file, str(output_dir))
            
        elif args.mode == 'batch':
            # Batch simulation example
            batch_orchestrator, batch_summary = run_batch_simulation_example(
                args.dem_file, str(output_dir)
            )
            
        elif args.mode == 'full':
            # Full pipeline example
            logger.info("Running full pipeline example...")
            
            # 1. Run batch simulations
            batch_orchestrator, batch_summary = run_batch_simulation_example(
                args.dem_file, str(output_dir)
            )
            
            if batch_summary['success_count'] == 0:
                logger.error("No successful simulations - cannot proceed with pipeline")
                sys.exit(1)
            
            # 2. Process results
            processing_summary = process_results_example(
                batch_orchestrator, str(output_dir)
            )
            
            # 3. Validate results
            validation_summary = validate_results_example(
                batch_orchestrator, str(output_dir)
            )
            
            # 4. Track metadata
            metadata_tracker = track_metadata_example(
                batch_orchestrator, str(output_dir)
            )
            
            # 5. Export training data
            training_data = export_training_data_example(
                batch_orchestrator, processing_summary, str(output_dir)
            )
            
            logger.info("=== Full Pipeline Example Completed ===")
            logger.info(f"Results available in: {output_dir}")
            logger.info(f"Training samples ready: {len(training_data) if training_data else 0}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()