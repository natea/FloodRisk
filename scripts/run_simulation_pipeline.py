#!/usr/bin/env python3
"""
Command-line interface for LISFLOOD-FP simulation pipeline.

This script provides a comprehensive CLI for running flood simulations,
processing results, and generating training data for ML models.

Usage Examples:
    # Run single simulation
    python scripts/run_simulation_pipeline.py single \\
        --dem-file LISFLOOD-FP/Nashville/final_dem.asc \\
        --rainfall-file LISFLOOD-FP/Nashville/rain_input_100yr.rain \\
        --output-dir results/single_sim

    # Run batch simulations
    python scripts/run_simulation_pipeline.py batch \\
        --dem-file LISFLOOD-FP/Nashville/final_dem.asc \\
        --return-periods 100,500 \\
        --patterns uniform,center_loaded \\
        --output-dir results/batch_sim \\
        --parallel-jobs 4

    # Process existing simulation results
    python scripts/run_simulation_pipeline.py process \\
        --input-dir results/batch_sim \\
        --output-dir results/processed \\
        --flood-threshold 0.05

    # Validate simulation results
    python scripts/run_simulation_pipeline.py validate \\
        --input-dir results/batch_sim \\
        --output-file validation_report.json
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_return_periods(rp_str: str) -> List[ReturnPeriodConfig]:
    """Parse return periods from command line string."""
    
    # Default Nashville return periods with NOAA Atlas 14 values
    defaults = {
        10: ReturnPeriodConfig(10, 111.76, "10-year return period", is_sub_design=True),
        25: ReturnPeriodConfig(25, 142.24, "25-year return period", is_sub_design=True),
        100: ReturnPeriodConfig(100, 177.8, "100-year design storm"),
        500: ReturnPeriodConfig(500, 222.25, "500-year extreme event")
    }
    
    periods = []
    for rp_part in rp_str.split(','):
        rp_part = rp_part.strip()
        
        if ':' in rp_part:
            # Format: "100:177.8" (return_period:depth_mm)
            rp_years, depth_mm = rp_part.split(':')
            rp_years = int(rp_years)
            depth_mm = float(depth_mm)
            
            is_sub_design = rp_years < 50  # Heuristic for sub-design events
            periods.append(ReturnPeriodConfig(
                rp_years, depth_mm, 
                f"{rp_years}-year return period",
                is_sub_design=is_sub_design
            ))
        else:
            # Just return period, use defaults
            rp_years = int(rp_part)
            if rp_years in defaults:
                periods.append(defaults[rp_years])
            else:
                logger.warning(f"No default rainfall for {rp_years}-year return period")
    
    return periods


def parse_hyetograph_patterns(patterns_str: str) -> List[HyetographConfig]:
    """Parse hyetograph patterns from command line string."""
    patterns = []
    
    for pattern in patterns_str.split(','):
        pattern = pattern.strip()
        
        if pattern == 'uniform':
            patterns.append(HyetographConfig('uniform', 0))
        elif pattern == 'front_loaded':
            patterns.append(HyetographConfig('front_loaded', 0, front_factor=2.5))
        elif pattern == 'center_loaded':
            patterns.append(HyetographConfig('center_loaded', 0, peak_hour=12.0, center_factor=3.0))
        elif pattern == 'back_loaded':
            patterns.append(HyetographConfig('back_loaded', 0, front_factor=2.5))
        else:
            logger.warning(f"Unknown hyetograph pattern: {pattern}")
    
    return patterns


def cmd_single(args) -> int:
    """Run single simulation."""
    logger.info("Running single simulation")
    
    # Validate inputs
    if not Path(args.dem_file).exists():
        logger.error(f"DEM file not found: {args.dem_file}")
        return 1
    
    if not Path(args.rainfall_file).exists():
        logger.error(f"Rainfall file not found: {args.rainfall_file}")
        return 1
    
    # Initialize simulator
    try:
        simulator = LisfloodSimulator(
            lisflood_executable=args.lisflood_exe,
            working_directory=args.output_dir
        )
    except FileNotFoundError as e:
        logger.error(f"LISFLOOD-FP executable not found: {e}")
        return 1
    
    # Create configuration
    config = SimulationConfig(
        dem_file=args.dem_file,
        rainfall_file=args.rainfall_file,
        manning_file=args.manning_file or "LISFLOOD-FP/Nashville/manning.asc",
        infiltration_file=args.infiltration_file or "LISFLOOD-FP/Nashville/infiltration.asc",
        sim_time=args.sim_time,
        initial_timestep=args.initial_timestep,
        acceleration=not args.no_acceleration
    )
    
    # Run simulation
    logger.info(f"Starting simulation: {args.simulation_id}")
    result = simulator.run_simulation(config, args.simulation_id)
    
    # Save result
    output_dir = Path(args.output_dir)
    result_file = output_dir / f"{args.simulation_id}_result.json"
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Report results
    if result['status'] == 'success':
        logger.info(f"Simulation completed successfully!")
        logger.info(f"Runtime: {result['runtime_seconds']:.1f} seconds")
        
        stats = result['outputs']['statistics']
        logger.info(f"Flood statistics:")
        logger.info(f"  - Flooded fraction: {stats['flood_fraction']:.3%}")
        logger.info(f"  - Max depth: {stats['max_depth_m']:.2f} m")
        logger.info(f"  - Flooded area: {stats['flooded_area_km2']:.2f} km²")
        logger.info(f"Result saved to: {result_file}")
        
        return 0
    else:
        logger.error(f"Simulation failed: {result.get('error', 'Unknown error')}")
        return 1


def cmd_batch(args) -> int:
    """Run batch simulations."""
    logger.info("Running batch simulations")
    
    # Validate inputs
    if not Path(args.dem_file).exists():
        logger.error(f"DEM file not found: {args.dem_file}")
        return 1
    
    # Parse configurations
    return_periods = parse_return_periods(args.return_periods)
    hyetograph_patterns = parse_hyetograph_patterns(args.patterns)
    
    logger.info(f"Configured {len(return_periods)} return periods × "
               f"{len(hyetograph_patterns)} patterns = "
               f"{len(return_periods) * len(hyetograph_patterns)} simulations")
    
    # Initialize components
    try:
        simulator = LisfloodSimulator(
            lisflood_executable=args.lisflood_exe,
            working_directory=str(Path(args.output_dir) / "simulations")
        )
    except FileNotFoundError as e:
        logger.error(f"LISFLOOD-FP executable not found: {e}")
        return 1
    
    param_generator = ParameterFileGenerator(
        base_config_dir=args.base_config_dir
    )
    
    # Configure batch execution
    batch_config = BatchConfig(
        max_parallel_jobs=args.parallel_jobs,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay,
        validate_results=not args.no_validation,
        cleanup_failed_runs=args.cleanup_failed,
        keep_intermediate_files=args.keep_intermediate
    )
    
    # Set up progress callback
    def progress_callback(progress_info):
        completed = progress_info['completed']
        total = progress_info['total']
        current = progress_info['current_result']
        
        # Handle both scenario_id and simulation_id keys
        sim_id = current.get('scenario_id', current.get('simulation_id', 'unknown'))
        logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) - "
                   f"{sim_id}: {current['status']}")
    
    batch_config.progress_callback = progress_callback
    
    batch_orchestrator = SimulationBatch(simulator, param_generator, batch_config)
    
    # Create batch
    logger.info("Creating simulation batch...")
    batch_id = batch_orchestrator.create_batch_from_config(
        dem_file=args.dem_file,
        return_periods=return_periods,
        hyetograph_patterns=hyetograph_patterns,
        output_dir=args.output_dir,
        base_sim_config={
            'manning_file': args.manning_file or 'manning.asc',
            'infiltration_file': args.infiltration_file or 'infiltration.asc',
            'sim_time': args.sim_time,
            'initial_timestep': args.initial_timestep
        }
    )
    
    logger.info(f"Created batch: {batch_id}")
    
    # Execute batch
    logger.info("Executing batch simulations...")
    batch_summary = batch_orchestrator.execute_batch()
    
    # Save results
    output_dir = Path(args.output_dir)
    summary_file = output_dir / "batch_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2, default=str)
    
    # Export training data manifest
    if batch_summary['success_count'] > 0:
        training_file = output_dir / "training_data_manifest.json"
        training_data = batch_orchestrator.export_training_data_paths(str(training_file))
        
        logger.info(f"Training data manifest saved: {training_file}")
        logger.info(f"Available training samples: {len(training_data)}")
    
    # Report results
    logger.info(f"Batch execution completed!")
    logger.info(f"Success rate: {batch_summary['success_rate']:.1%} "
               f"({batch_summary['success_count']}/{batch_summary['total_scenarios']})")
    logger.info(f"Runtime: {batch_summary['runtime_seconds']/3600:.1f} hours")
    
    if batch_summary['failure_count'] > 0:
        logger.warning(f"Failed scenarios: {batch_summary['failed_scenarios']}")
        return 1 if batch_summary['success_count'] == 0 else 0
    
    return 0


def cmd_process(args) -> int:
    """Process simulation results."""
    logger.info("Processing simulation results")
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Load simulation results
    results_file = input_dir / "batch_results.json"
    if not results_file.exists():
        logger.error(f"Batch results file not found: {results_file}")
        logger.info("Expected file: batch_results.json in input directory")
        return 1
    
    with open(results_file) as f:
        simulation_results = json.load(f)
    
    logger.info(f"Loaded {len(simulation_results)} simulation results")
    
    # Configure processing
    processing_config = ProcessingConfig(
        flood_depth_threshold_m=args.flood_threshold,
        remove_small_areas=not args.no_morphology,
        min_flood_area_pixels=args.min_area_pixels,
        close_gaps=not args.no_morphology,
        save_numpy=True,
        save_geotiff=args.save_geotiff,
        save_statistics=True
    )
    
    if args.dem_bounds:
        # Parse bounds: "minx,miny,maxx,maxy"
        bounds = [float(x) for x in args.dem_bounds.split(',')]
        if len(bounds) != 4:
            logger.error("DEM bounds must be in format: minx,miny,maxx,maxy")
            return 1
    else:
        bounds = None
    
    # Process results
    result_processor = ResultProcessor(processing_config)
    
    processing_summary = result_processor.process_batch_outputs(
        simulation_results=simulation_results,
        output_dir=args.output_dir,
        dem_bounds=bounds
    )
    
    # Save processing summary
    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(processing_summary, f, indent=2, default=str)
    
    # Report results
    logger.info(f"Processing completed!")
    logger.info(f"Successfully processed: {processing_summary['successful_processing']} simulations")
    logger.info(f"Failed processing: {processing_summary['failed_processing']}")
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0


def cmd_validate(args) -> int:
    """Validate simulation results."""
    logger.info("Validating simulation results")
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Load simulation results
    results_file = input_dir / "batch_results.json"
    if not results_file.exists():
        logger.error(f"Batch results file not found: {results_file}")
        return 1
    
    with open(results_file) as f:
        simulation_results = json.load(f)
    
    logger.info(f"Loaded {len(simulation_results)} simulation results")
    
    # Configure validation
    thresholds = ValidationThresholds(
        min_flood_fraction=args.min_flood_fraction,
        max_flood_fraction=args.max_flood_fraction,
        max_reasonable_depth_m=args.max_depth,
        target_success_rate=args.target_success_rate
    )
    
    validator = SimulationValidator(thresholds)
    
    # Run validation
    validation_summary = validator.validate_batch_results(
        simulation_results=simulation_results,
        detailed_validation=not args.quick_validation
    )
    
    # Save validation results
    with open(args.output_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    # Report results
    logger.info(f"Validation completed!")
    summary = validation_summary['summary']
    logger.info(f"Results: {summary['passed_count']} passed, "
               f"{summary['warning_count']} warnings, {summary['failed_count']} failed")
    logger.info(f"Average score: {summary['average_score']:.1f}/100")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Report saved to: {args.output_file}")
    
    # Show common issues
    quality = validation_summary['quality_assessment']
    if quality['common_warnings']:
        logger.info("Most common warnings:")
        for warning, count in list(quality['common_warnings'].items())[:3]:
            logger.info(f"  - {warning}: {count} occurrences")
    
    if quality['common_errors']:
        logger.warning("Common errors:")
        for error, count in list(quality['common_errors'].items())[:3]:
            logger.warning(f"  - {error}: {count} occurrences")
    
    # Return non-zero if validation found serious issues
    if not quality['meets_target_success_rate']:
        logger.warning(f"Batch does not meet target success rate of {thresholds.target_success_rate:.1%}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LISFLOOD-FP Simulation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--lisflood-exe", 
                       default="LISFLOOD-FP/build/lisflood",
                       help="Path to LISFLOOD-FP executable")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single simulation command
    single_parser = subparsers.add_parser('single', help='Run single simulation')
    single_parser.add_argument("--dem-file", required=True,
                              help="Path to DEM file")
    single_parser.add_argument("--rainfall-file", required=True,
                              help="Path to rainfall file")
    single_parser.add_argument("--manning-file",
                              help="Path to Manning's n file")
    single_parser.add_argument("--infiltration-file", 
                              help="Path to infiltration file")
    single_parser.add_argument("--simulation-id", default="single_sim",
                              help="Simulation identifier")
    single_parser.add_argument("--output-dir", required=True,
                              help="Output directory")
    single_parser.add_argument("--sim-time", type=float, default=86400.0,
                              help="Simulation time in seconds (default: 86400)")
    single_parser.add_argument("--initial-timestep", type=float, default=10.0,
                              help="Initial timestep in seconds")
    single_parser.add_argument("--no-acceleration", action="store_true",
                              help="Disable acceleration")
    
    # Batch simulation command
    batch_parser = subparsers.add_parser('batch', help='Run batch simulations')
    batch_parser.add_argument("--dem-file", required=True,
                             help="Path to DEM file")
    batch_parser.add_argument("--return-periods", 
                             default="100,500",
                             help="Return periods (comma-separated, e.g., '100,500' or '100:177.8,500:222.25')")
    batch_parser.add_argument("--patterns",
                             default="uniform,center_loaded",
                             help="Hyetograph patterns (comma-separated)")
    batch_parser.add_argument("--base-config-dir",
                             default="LISFLOOD-FP/Nashville",
                             help="Base configuration directory")
    batch_parser.add_argument("--manning-file",
                             help="Manning's n file (relative to base config)")
    batch_parser.add_argument("--infiltration-file",
                             help="Infiltration file (relative to base config)")
    batch_parser.add_argument("--output-dir", required=True,
                             help="Output directory")
    batch_parser.add_argument("--parallel-jobs", type=int, default=4,
                             help="Number of parallel jobs")
    batch_parser.add_argument("--max-retries", type=int, default=2,
                             help="Maximum retries for failed simulations")
    batch_parser.add_argument("--retry-delay", type=float, default=60.0,
                             help="Delay between retries in seconds")
    batch_parser.add_argument("--sim-time", type=float, default=86400.0,
                             help="Simulation time in seconds")
    batch_parser.add_argument("--initial-timestep", type=float, default=10.0,
                             help="Initial timestep in seconds")
    batch_parser.add_argument("--no-validation", action="store_true",
                             help="Skip result validation")
    batch_parser.add_argument("--cleanup-failed", action="store_true",
                             help="Cleanup failed simulation files")
    batch_parser.add_argument("--keep-intermediate", action="store_true",
                             help="Keep intermediate files")
    
    # Result processing command
    process_parser = subparsers.add_parser('process', help='Process simulation results')
    process_parser.add_argument("--input-dir", required=True,
                               help="Input directory with batch results")
    process_parser.add_argument("--output-dir", required=True,
                               help="Output directory for processed results")
    process_parser.add_argument("--flood-threshold", type=float, default=0.05,
                               help="Flood depth threshold in meters")
    process_parser.add_argument("--dem-bounds",
                               help="DEM bounds for GeoTIFF: minx,miny,maxx,maxy")
    process_parser.add_argument("--save-geotiff", action="store_true",
                               help="Save results as GeoTIFF")
    process_parser.add_argument("--no-morphology", action="store_true",
                               help="Skip morphological processing")
    process_parser.add_argument("--min-area-pixels", type=int, default=4,
                               help="Minimum flood area in pixels")
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate simulation results')
    validate_parser.add_argument("--input-dir", required=True,
                                help="Input directory with batch results")
    validate_parser.add_argument("--output-file", required=True,
                                help="Output validation report file")
    validate_parser.add_argument("--quick-validation", action="store_true",
                                help="Skip detailed validation (faster)")
    validate_parser.add_argument("--min-flood-fraction", type=float, default=0.001,
                                help="Minimum acceptable flood fraction")
    validate_parser.add_argument("--max-flood-fraction", type=float, default=0.3,
                                help="Maximum acceptable flood fraction")
    validate_parser.add_argument("--max-depth", type=float, default=50.0,
                                help="Maximum reasonable depth in meters")
    validate_parser.add_argument("--target-success-rate", type=float, default=0.8,
                                help="Target batch success rate")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'single':
            return cmd_single(args)
        elif args.command == 'batch':
            return cmd_batch(args)
        elif args.command == 'process':
            return cmd_process(args)
        elif args.command == 'validate':
            return cmd_validate(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())