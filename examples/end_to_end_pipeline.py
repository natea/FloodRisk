#!/usr/bin/env python3
"""
Complete end-to-end pipeline demonstration.

This example shows the full integration from preprocessing through simulation
to ML-ready training data generation.

Usage:
    python examples/end_to_end_pipeline.py --location nashville --output-dir pipeline_demo
"""

import sys
import argparse
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import PreprocessingIntegration
from simulation.lisflood_simulator import LisfloodSimulator
from simulation.parameter_generator import ParameterFileGenerator, ReturnPeriodConfig, HyetographConfig
from simulation.batch_orchestrator import SimulationBatch, BatchConfig
from simulation.result_processor import ResultProcessor, ProcessingConfig
from simulation.validation import SimulationValidator, ValidationThresholds
from simulation.metadata_tracker import SimulationMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_preprocessing_config(location: str, output_dir: Path) -> str:
    """Create a mock preprocessing configuration for demonstration."""
    
    if location.lower() == 'nashville':
        # Use actual Nashville data structure
        config = {
            "location": "Nashville, TN",
            "dem_info": {
                "source": "USGS 3DEP",
                "resolution_m": 10.0,
                "bounds": [-86.91, 36.06, -86.66, 36.31],  # Nashville area
                "shape": [2500, 2500],  # Approximate grid size
                "crs": "EPSG:4326"
            },
            "outputs": {
                "conditioned_dem": "LISFLOOD-FP/Nashville/final_dem.asc",
                "flow_directions": "derived/flow_dirs.tif",
                "flow_accumulation": "derived/flow_accum.tif"
            },
            "metadata": {
                "processing_date": "2024-01-15",
                "rainfall_statistics": {
                    "10yr_24h_mm": 111.76,
                    "25yr_24h_mm": 142.24, 
                    "100yr_24h_mm": 177.8,
                    "500yr_24h_mm": 222.25,
                    "source": "NOAA Atlas 14"
                },
                "validation": {
                    "dem_quality_score": 0.92,
                    "drainage_completeness": 0.89,
                    "sink_removal_success": True
                }
            },
            "source": "FloodRisk preprocessing pipeline v1.0"
        }
    else:
        # Generic location template
        config = {
            "location": f"{location}",
            "dem_info": {
                "source": "User provided",
                "resolution_m": 10.0,
                "bounds": [-1.0, -1.0, 1.0, 1.0],  # Placeholder bounds
                "shape": [1000, 1000],
                "crs": "EPSG:4326"
            },
            "outputs": {
                "conditioned_dem": "path/to/conditioned_dem.asc"
            },
            "metadata": {
                "processing_date": "2024-01-15",
                "rainfall_statistics": {
                    "100yr_24h_mm": 150.0,  # Generic value
                    "500yr_24h_mm": 200.0
                }
            },
            "source": "Mock preprocessing config"
        }
    
    # Save configuration
    config_file = output_dir / "preprocessing_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created preprocessing config: {config_file}")
    return str(config_file)


def demonstrate_component_integration(output_dir: Path):
    """Demonstrate individual component integration."""
    logger.info("=== Component Integration Demonstration ===")
    
    # 1. Parameter Generation
    logger.info("1. Parameter file generation...")
    param_generator = ParameterFileGenerator()
    
    return_periods = [
        ReturnPeriodConfig(100, 177.8, "100-year Nashville storm"),
        ReturnPeriodConfig(10, 111.76, "10-year sub-design", is_sub_design=True)
    ]
    
    patterns = [
        HyetographConfig('uniform', 0),
        HyetographConfig('center_loaded', 0, peak_hour=12.0)
    ]
    
    scenarios = param_generator.generate_scenario_parameters(
        dem_file="LISFLOOD-FP/Nashville/final_dem.asc",
        return_periods=return_periods,
        hyetograph_patterns=patterns,
        output_dir=str(output_dir / "scenarios")
    )
    
    logger.info(f"Generated {len(scenarios)} simulation scenarios")
    
    # 2. Validation Configuration
    logger.info("2. Validation framework setup...")
    thresholds = ValidationThresholds(
        min_flood_fraction=0.001,
        max_flood_fraction=0.25,
        max_reasonable_depth_m=20.0,
        target_success_rate=0.8
    )
    
    validator = SimulationValidator(thresholds)
    logger.info("Validation framework configured")
    
    # 3. Result Processing Configuration
    logger.info("3. Result processing setup...")
    processing_config = ProcessingConfig(
        flood_depth_threshold_m=0.05,
        remove_small_areas=True,
        save_numpy=True,
        save_statistics=True
    )
    
    processor = ResultProcessor(processing_config)
    logger.info("Result processor configured")
    
    # 4. Metadata Tracking
    logger.info("4. Metadata tracking setup...")
    metadata_tracker = SimulationMetadata(
        metadata_dir=str(output_dir / "metadata"),
        enable_file_checksums=True,
        track_environment=True
    )
    logger.info("Metadata tracker initialized")
    
    return {
        'scenarios': scenarios,
        'validator': validator,
        'processor': processor,
        'metadata_tracker': metadata_tracker
    }


def demonstrate_batch_simulation(output_dir: Path, components: dict):
    """Demonstrate batch simulation execution."""
    logger.info("=== Batch Simulation Demonstration ===")
    
    # Check for LISFLOOD-FP executable
    lisflood_exe = Path("LISFLOOD-FP/build/lisflood")
    if not lisflood_exe.exists():
        logger.warning(f"LISFLOOD-FP executable not found: {lisflood_exe}")
        logger.info("Simulation will be mocked for demonstration")
        return mock_simulation_results(components['scenarios'])
    
    try:
        # Initialize simulator
        simulator = LisfloodSimulator(
            lisflood_executable=str(lisflood_exe),
            working_directory=str(output_dir / "simulations")
        )
        
        # Set up batch configuration
        batch_config = BatchConfig(
            max_parallel_jobs=2,  # Conservative for demo
            max_retries=1,
            validate_results=True
        )
        
        param_generator = ParameterFileGenerator()
        batch = SimulationBatch(simulator, param_generator, batch_config)
        
        # Create mini-batch for demonstration
        demo_scenarios = components['scenarios'][:2]  # Just 2 scenarios
        
        logger.info(f"Running {len(demo_scenarios)} demonstration simulations...")
        
        # This would normally create and execute batch, but we'll mock for demo
        logger.info("Note: Real simulation execution would happen here")
        return mock_simulation_results(demo_scenarios)
        
    except Exception as e:
        logger.warning(f"Could not run real simulations: {e}")
        logger.info("Continuing with mocked results for demonstration")
        return mock_simulation_results(components['scenarios'][:2])


def mock_simulation_results(scenarios: List[dict]) -> List[dict]:
    """Create mock simulation results for demonstration."""
    import numpy as np
    
    logger.info("Creating mock simulation results for demonstration...")
    
    results = []
    for i, scenario in enumerate(scenarios):
        # Create realistic mock statistics
        flood_fraction = np.random.uniform(0.02, 0.15)
        max_depth = np.random.uniform(0.5, 3.0)
        
        result = {
            'simulation_id': scenario['scenario_id'],
            'status': 'success',
            'start_time': '2024-01-15T10:00:00',
            'end_time': '2024-01-15T10:30:00',
            'runtime_seconds': 1800 + np.random.randint(-300, 300),
            'outputs': {
                'depth_file': f"mock_depth_{i}.max",
                'extent_file': f"mock_extent_{i}.npy",
                'statistics': {
                    'total_pixels': 250000,
                    'flooded_pixels': int(250000 * flood_fraction),
                    'flood_fraction': flood_fraction,
                    'max_depth_m': max_depth,
                    'mean_depth_flooded_m': max_depth * 0.3,
                    'flooded_area_km2': flood_fraction * 25.0  # 5km x 5km domain
                }
            },
            'validation': {
                'status': 'passed',
                'warnings': [],
                'errors': []
            },
            'scenario': scenario
        }
        results.append(result)
    
    logger.info(f"Created {len(results)} mock simulation results")
    return results


def demonstrate_integrated_pipeline(location: str, output_dir: Path):
    """Demonstrate the complete integrated pipeline."""
    logger.info("=== Integrated Pipeline Demonstration ===")
    
    # Create mock preprocessing config
    preprocessing_config = create_mock_preprocessing_config(location, output_dir)
    
    # Initialize integration
    integration = PreprocessingIntegration(
        preprocessing_output_dir="preprocessing_results",  # Mock directory
        simulation_output_dir=str(output_dir / "integrated"),
        temp_dir=str(output_dir / "temp")
    )
    
    # Set up simulation from preprocessing
    logger.info("Setting up simulation from preprocessing configuration...")
    setup_info = integration.setup_simulation_from_preprocessing(
        preprocessing_config_file=preprocessing_config,
        return_periods=[100, 500],  # Reduced set for demo
        hyetograph_patterns=['uniform', 'center_loaded']
    )
    
    logger.info("Integration setup completed:")
    logger.info(f"  - Location: {setup_info['location']}")
    logger.info(f"  - Total scenarios: {setup_info['total_scenarios']}")
    logger.info(f"  - DEM file: {Path(setup_info['dem_file']).name}")
    
    # Mock the full simulation run (would normally call run_integrated_simulation)
    logger.info("Mock integrated simulation execution...")
    
    mock_integrated_results = {
        'setup_info': setup_info,
        'batch_id': 'demo_batch_001',
        'batch_summary': {
            'total_scenarios': setup_info['total_scenarios'],
            'success_count': setup_info['total_scenarios'] - 1,  # One failure for realism
            'success_rate': (setup_info['total_scenarios'] - 1) / setup_info['total_scenarios'],
            'runtime_seconds': 3600
        },
        'simulation_results': mock_simulation_results(
            [{'scenario_id': f'mock_scenario_{i}'} for i in range(setup_info['total_scenarios'])]
        )
    }
    
    # Export for ML training
    logger.info("Exporting training data...")
    training_manifest = integration.export_for_ml_training(
        integrated_results=mock_integrated_results,
        ml_training_dir=str(output_dir / "ml_training"),
        include_preprocessing_features=True
    )
    
    logger.info(f"Training data manifest created: {training_manifest}")
    
    return mock_integrated_results


def demonstrate_quality_assessment(simulation_results: List[dict], output_dir: Path):
    """Demonstrate quality assessment and validation."""
    logger.info("=== Quality Assessment Demonstration ===")
    
    # Validation
    thresholds = ValidationThresholds()
    validator = SimulationValidator(thresholds)
    
    validation_summary = validator.validate_batch_results(
        simulation_results=simulation_results,
        detailed_validation=False  # Quick validation for demo
    )
    
    logger.info("Validation Results:")
    summary = validation_summary['summary']
    logger.info(f"  - Success rate: {summary['success_rate']:.1%}")
    logger.info(f"  - Average score: {summary['average_score']:.1f}/100")
    logger.info(f"  - Passed: {summary['passed_count']}, "
               f"Warnings: {summary['warning_count']}, "
               f"Failed: {summary['failed_count']}")
    
    # Save validation report
    validation_file = output_dir / "validation_report.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    logger.info(f"Validation report saved: {validation_file}")
    
    # Result processing demonstration
    processing_config = ProcessingConfig(
        flood_depth_threshold_m=0.05,
        save_statistics=True
    )
    
    processor = ResultProcessor(processing_config)
    
    # Mock processing summary
    processing_summary = {
        'successful_processing': len([r for r in simulation_results if r['status'] == 'success']),
        'total_simulations': len(simulation_results),
        'batch_statistics': {
            'flood_fraction': {
                'mean': 0.08,
                'std': 0.04,
                'min': 0.02,
                'max': 0.15
            }
        }
    }
    
    logger.info("Processing Summary:")
    logger.info(f"  - Successfully processed: {processing_summary['successful_processing']} simulations")
    logger.info(f"  - Mean flood fraction: {processing_summary['batch_statistics']['flood_fraction']['mean']:.3%}")
    
    return validation_summary, processing_summary


def create_summary_report(output_dir: Path, 
                         components: dict,
                         simulation_results: List[dict],
                         validation_summary: dict,
                         processing_summary: dict):
    """Create comprehensive summary report."""
    logger.info("=== Creating Summary Report ===")
    
    successful_sims = [r for r in simulation_results if r['status'] == 'success']
    
    report = {
        'pipeline_demonstration': {
            'timestamp': '2024-01-15T12:00:00',
            'output_directory': str(output_dir),
            'components_tested': list(components.keys())
        },
        'simulation_summary': {
            'total_scenarios': len(simulation_results),
            'successful_simulations': len(successful_sims),
            'success_rate': len(successful_sims) / len(simulation_results) if simulation_results else 0,
            'simulation_ids': [r['simulation_id'] for r in successful_sims]
        },
        'validation_summary': {
            'overall_score': validation_summary['summary']['average_score'],
            'success_rate': validation_summary['summary']['success_rate'],
            'quality_passed': validation_summary['quality_assessment']['meets_target_success_rate']
        },
        'processing_summary': {
            'processed_simulations': processing_summary['successful_processing'],
            'flood_statistics': processing_summary['batch_statistics']['flood_fraction']
        },
        'key_outputs': {
            'validation_report': 'validation_report.json',
            'training_manifest': 'ml_training/training_manifest.json',
            'simulation_scenarios': 'scenarios/scenarios.json',
            'metadata': 'metadata/'
        },
        'next_steps': [
            'Review validation report for quality assessment',
            'Use training manifest for ML model development',
            'Scale up to full return period and pattern combinations',
            'Integrate with preprocessing pipeline for additional locations',
            'Optimize simulation parameters based on validation feedback'
        ]
    }
    
    # Save summary report
    report_file = output_dir / "pipeline_demonstration_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Summary report saved: {report_file}")
    
    # Print summary
    logger.info("=== Pipeline Demonstration Summary ===")
    logger.info(f"Location: {report['simulation_summary'].get('location', 'Demo')}")
    logger.info(f"Simulations: {report['simulation_summary']['successful_simulations']}/{report['simulation_summary']['total_scenarios']} successful")
    logger.info(f"Validation score: {report['validation_summary']['overall_score']:.1f}/100")
    logger.info(f"Training samples ready: {report['processing_summary']['processed_simulations']}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return report


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="End-to-end pipeline demonstration")
    parser.add_argument("--location", default="nashville",
                       help="Location for demonstration (nashville or custom)")
    parser.add_argument("--output-dir", default="pipeline_demo",
                       help="Output directory for demonstration")
    parser.add_argument("--mode", choices=['components', 'integrated', 'full'], default='full',
                       help="Demonstration mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting FloodRisk simulation pipeline demonstration")
    logger.info(f"Location: {args.location}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        if args.mode in ['components', 'full']:
            # Component demonstration
            components = demonstrate_component_integration(output_dir)
            simulation_results = demonstrate_batch_simulation(output_dir, components)
            validation_summary, processing_summary = demonstrate_quality_assessment(
                simulation_results, output_dir
            )
        
        if args.mode in ['integrated', 'full']:
            # Integrated pipeline demonstration
            integrated_results = demonstrate_integrated_pipeline(args.location, output_dir)
            
            # Use results from integrated pipeline if available
            if args.mode == 'integrated':
                simulation_results = integrated_results['simulation_results']
                components = {'integrated': True}
                validation_summary, processing_summary = demonstrate_quality_assessment(
                    simulation_results, output_dir
                )
        
        # Create comprehensive summary
        summary_report = create_summary_report(
            output_dir, components, simulation_results, 
            validation_summary, processing_summary
        )
        
        logger.info("=== Demonstration Completed Successfully ===")
        logger.info("Key takeaways:")
        logger.info("1. Physics-based simulation integration provides high-quality training labels")
        logger.info("2. Comprehensive validation ensures data quality for ML training")
        logger.info("3. Automated pipeline scales from single simulations to large datasets")
        logger.info("4. Integration with preprocessing enables end-to-end workflows")
        logger.info("5. Metadata tracking ensures reproducibility and provenance")
        
        logger.info(f"\\nFor production use:")
        logger.info("- Compile LISFLOOD-FP for actual simulations")
        logger.info("- Configure location-specific return periods and patterns")
        logger.info("- Scale parallel execution based on available resources")
        logger.info("- Integrate with your preprocessing and ML training pipelines")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()