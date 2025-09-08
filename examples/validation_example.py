#!/usr/bin/env python3
"""
FloodRisk Validation Framework - Complete Example

This example demonstrates the comprehensive data validation and quality assurance
framework for the FloodRisk ML pipeline, including:

- End-to-end pipeline validation
- ML integration validation
- Automated QA reporting and dashboards
- Real-time monitoring and alerting

Author: FloodRisk QA Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import json

# Import validation framework components
from src.validation.pipeline_validator import (
    PipelineValidator, DEMValidator, RainfallValidator, 
    SpatialConsistencyValidator, SimulationValidator, TileQualityValidator
)
from src.validation.ml_integration_validator import MLIntegrationValidator
from src.validation.qa_dashboard import QADashboard, ValidationDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for validation demonstration"""
    
    # Create temporary directory for sample files
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Creating sample data in {temp_dir}")
    
    # 1. Create sample DEM file
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    
    # Generate realistic DEM data
    np.random.seed(42)
    dem_data = np.random.uniform(10, 500, (200, 200)).astype(np.float32)
    # Add some spatial correlation
    from scipy.ndimage import gaussian_filter
    dem_data = gaussian_filter(dem_data, sigma=3)
    
    dem_path = temp_dir / 'sample_dem.tif'
    with rasterio.open(
        dem_path, 'w',
        driver='GTiff', height=200, width=200, count=1,
        dtype=rasterio.float32, crs=CRS.from_epsg(4326),
        transform=from_bounds(-120, 30, -110, 40, 200, 200)
    ) as dst:
        dst.write(dem_data, 1)
    
    # 2. Create sample rainfall data
    rainfall_data = np.random.exponential(2.0, (200, 200)).astype(np.float32)
    rainfall_data = np.clip(rainfall_data, 0, 100)  # Cap at 100mm/hr
    
    # 3. Create sample simulation results
    simulation_results = {
        'depths': np.random.exponential(0.3, (200, 200)).astype(np.float32),
        'velocities': np.random.normal(0, 1, (2, 200, 200)).astype(np.float32),
        'convergence': {
            'final_residual': 1e-7,
            'iterations': 45,
            'residuals': [10**(-i/2) for i in range(1, 15)]
        },
        'inflow': 1000.0,
        'outflow': 950.0
    }
    
    # 4. Create sample tiles
    tiles_data = []
    np.random.seed(42)
    for i in range(200):
        # Create tile with varying flood ratios
        tile = np.random.uniform(0, 1, (64, 64))
        flood_ratio = np.random.uniform(0.1, 0.8)
        threshold = np.percentile(tile, (1 - flood_ratio) * 100)
        tile_flood = (tile > threshold).astype(float) * np.random.uniform(0.1, 3.0, (64, 64))
        tiles_data.append({'data': tile_flood})
    
    tiles_info = {
        'tiles': tiles_data,
        'metadata': {
            'tile_size': (64, 64),
            'overlap': 0.1,
            'projection': 'EPSG:4326',
            'bounds': [-120, 30, -110, 40]
        }
    }
    
    # 5. Create sample ML training data
    class SampleDataset:
        def __init__(self, size=1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Sample input: DEM + rainfall + land use
            input_tensor = np.random.randn(3, 256, 256).astype(np.float32)
            # Sample target: flood depths
            target_tensor = np.random.exponential(0.2, (1, 256, 256)).astype(np.float32)
            return input_tensor, target_tensor
    
    # Sample labels for ML training
    sample_labels = np.random.choice([0, 1], size=(100, 256, 256), p=[0.7, 0.3])
    
    # Sample model training results
    model_results = {
        'training_history': {
            'train_loss': [2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28],
            'val_loss': [2.2, 1.7, 1.2, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38],
            'train_accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.88, 0.89],
            'val_accuracy': [0.48, 0.58, 0.68, 0.73, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86]
        },
        'test_metrics': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'iou': 0.78
        },
        'real_data_metrics': {
            'accuracy': 0.85,
            'iou': 0.78
        },
        'dummy_data_metrics': {
            'accuracy': 0.72,  # Dummy should be worse
            'iou': 0.65
        },
        'inference_metrics': {
            'avg_time_ms': 65.0,
            'max_time_ms': 95.0,
            'memory_mb': 512.0,
            'throughput': 15.4
        },
        'model_info': {
            'parameters': 18_500_000,
            'size_mb': 72.0,
            'architecture': 'U-Net',
            'input_shape': (3, 256, 256),
            'output_shape': (1, 256, 256)
        }
    }
    
    return {
        'temp_dir': temp_dir,
        'dem_path': dem_path,
        'rainfall_data': rainfall_data,
        'simulation_results': simulation_results,
        'tiles_info': tiles_info,
        'ml_dataset': SampleDataset(),
        'ml_labels': sample_labels,
        'model_results': model_results
    }


def demonstrate_individual_validators():
    """Demonstrate individual validator components"""
    logger.info("=== Demonstrating Individual Validators ===")
    
    sample_data = create_sample_data()
    
    # 1. DEM Validation
    logger.info("1. DEM Quality Validation")
    dem_validator = DEMValidator({
        'elevation_bounds': (-100, 2000),
        'void_threshold': 0.05,
        'smoothness_threshold': 100
    })
    
    dem_result = dem_validator.validate(sample_data['dem_path'])
    logger.info(f"DEM Validation: {dem_result.status} (Score: {dem_result.score:.3f})")
    if dem_result.issues:
        logger.warning(f"DEM Issues: {dem_result.issues}")
    
    # 2. Rainfall Validation
    logger.info("2. Rainfall Data Validation")
    rainfall_validator = RainfallValidator({
        'max_intensity': 200,
        'min_coverage': 0.9,
        'missing_data_threshold': 0.1
    })
    
    rainfall_result = rainfall_validator.validate(sample_data['rainfall_data'])
    logger.info(f"Rainfall Validation: {rainfall_result.status} (Score: {rainfall_result.score:.3f})")
    if rainfall_result.issues:
        logger.warning(f"Rainfall Issues: {rainfall_result.issues}")
    
    # 3. Simulation Validation
    logger.info("3. Simulation Results Validation")
    sim_validator = SimulationValidator({
        'max_depth': 20.0,
        'mass_conservation_tolerance': 0.05,
        'convergence_threshold': 1e-6
    })
    
    sim_result = sim_validator.validate(sample_data['simulation_results'])
    logger.info(f"Simulation Validation: {sim_result.status} (Score: {sim_result.score:.3f})")
    if sim_result.issues:
        logger.warning(f"Simulation Issues: {sim_result.issues}")
    
    # 4. Tile Quality Validation
    logger.info("4. Tile Quality Validation")
    tile_validator = TileQualityValidator({
        'target_flood_ratio': (0.1, 0.9),
        'min_tiles': 100,
        'edge_threshold': 3
    })
    
    tile_result = tile_validator.validate(sample_data['tiles_info'])
    logger.info(f"Tile Validation: {tile_result.status} (Score: {tile_result.score:.3f})")
    if tile_result.issues:
        logger.warning(f"Tile Issues: {tile_result.issues}")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_data['temp_dir'], ignore_errors=True)
    
    return [dem_result, rainfall_result, sim_result, tile_result]


def demonstrate_pipeline_validation():
    """Demonstrate complete pipeline validation"""
    logger.info("=== Demonstrating Complete Pipeline Validation ===")
    
    sample_data = create_sample_data()
    
    # Initialize pipeline validator
    pipeline_validator = PipelineValidator({
        'dem': {'elevation_bounds': (-100, 2000)},
        'rainfall': {'max_intensity': 200},
        'spatial': {'spatial_tolerance': 0.1},
        'simulation': {'max_depth': 20.0},
        'tiles': {'target_flood_ratio': (0.1, 0.9)}
    })
    
    # Prepare pipeline data
    pipeline_data = {
        'dem_path': sample_data['dem_path'],
        'rainfall_data': sample_data['rainfall_data'],
        'spatial_datasets': [
            {'path': sample_data['dem_path'], 'name': 'DEM', 'type': 'elevation'}
        ],
        'simulation_results': sample_data['simulation_results'],
        'tiles_info': sample_data['tiles_info']
    }
    
    # Run complete pipeline validation
    results = pipeline_validator.validate_full_pipeline(pipeline_data)
    
    logger.info(f"Pipeline validation completed: {len(results)} components validated")
    
    for component, result in results.items():
        logger.info(f"{component}: {result.status} (Score: {result.score:.3f})")
        if result.issues:
            logger.warning(f"  Issues: {result.issues}")
    
    # Generate validation report
    report = pipeline_validator.generate_validation_report()
    overall_score = report['validation_summary']['overall_score']
    overall_status = pipeline_validator.get_pipeline_status()
    
    logger.info(f"Overall Pipeline Status: {overall_status} (Score: {overall_score:.3f})")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_data['temp_dir'], ignore_errors=True)
    
    return results, report


def demonstrate_ml_integration_validation():
    """Demonstrate ML pipeline integration validation"""
    logger.info("=== Demonstrating ML Integration Validation ===")
    
    sample_data = create_sample_data()
    
    # Initialize ML integration validator
    ml_validator = MLIntegrationValidator({
        'data': {
            'expected_input_shape': (3, 256, 256),
            'expected_output_shape': (1, 256, 256),
            'batch_size': 32
        },
        'performance': {
            'min_accuracy': 0.7,
            'max_overfitting_gap': 0.1
        },
        'labels': {
            'min_class_ratio': 0.01,
            'spatial_coherence_threshold': 0.8
        }
    })
    
    # Prepare ML data
    ml_data = {
        'dataset': sample_data['ml_dataset'],
        'labels_data': {'labels': sample_data['ml_labels']},
        'model_results': sample_data['model_results']
    }
    
    # Run ML pipeline validation
    ml_results = ml_validator.validate_ml_pipeline(ml_data)
    
    logger.info(f"ML validation completed: {len(ml_results)} components validated")
    
    for component, result in ml_results.items():
        logger.info(f"{component}: {result.status} (Score: {result.score:.3f})")
        if result.issues:
            logger.warning(f"  Issues: {result.issues}")
    
    # Generate ML validation report
    ml_report = ml_validator.generate_ml_validation_report()
    ml_score = ml_report['ml_validation_summary']['overall_ml_score']
    ml_status = ml_validator.get_ml_pipeline_status()
    
    logger.info(f"Overall ML Pipeline Status: {ml_status} (Score: {ml_score:.3f})")
    
    return ml_results, ml_report


def demonstrate_qa_dashboard():
    """Demonstrate QA dashboard and automated reporting"""
    logger.info("=== Demonstrating QA Dashboard and Reporting ===")
    
    # Initialize QA dashboard
    qa_dashboard = QADashboard("example_validation.db")
    
    # Run sample validations to generate data
    sample_data = create_sample_data()
    
    # Create some sample validation results
    from src.validation.pipeline_validator import ValidationResult
    
    sample_results = [
        ValidationResult(
            component='DEM_Quality',
            status='PASS',
            score=0.92,
            details={'elevation_stats': {'min': 10, 'max': 500}},
            issues=[],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='Rainfall_Quality',
            status='WARN',
            score=0.78,
            details={'value_stats': {'max': 150}},
            issues=['Some extreme rainfall values detected'],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='Simulation_Quality',
            status='PASS',
            score=0.88,
            details={'convergence_analysis': {'converged': True}},
            issues=[],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='ML_Data_Compatibility',
            status='PASS',
            score=0.95,
            details={'dataset_size': 1000},
            issues=[],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='Model_Performance',
            status='PASS',
            score=0.85,
            details={'performance_analysis': {'accuracy': 0.85}},
            issues=[],
            timestamp=datetime.now()
        )
    ]
    
    # Process results through QA dashboard
    dashboard_result = qa_dashboard.process_validation_results(
        sample_results,
        pipeline_type="FloodRisk ML Pipeline Demo",
        generate_report=True,
        report_path=Path("example_qa_report.html")
    )
    
    logger.info(f"QA Dashboard processing completed:")
    logger.info(f"  Run ID: {dashboard_result['run_id']}")
    logger.info(f"  Overall Score: {dashboard_result['overall_score']:.3f}")
    logger.info(f"  Overall Status: {dashboard_result['overall_status']}")
    logger.info(f"  Alert Summary: {dashboard_result['alert_summary']}")
    
    # Get dashboard data
    dashboard_data = qa_dashboard.get_dashboard_data(days=30)
    logger.info(f"Dashboard data includes {len(dashboard_data['validation_history'])} historical runs")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_data['temp_dir'], ignore_errors=True)
    
    return dashboard_result, dashboard_data


def demonstrate_complete_workflow():
    """Demonstrate complete end-to-end validation workflow"""
    logger.info("=== Demonstrating Complete End-to-End Workflow ===")
    
    # 1. Create comprehensive sample data
    sample_data = create_sample_data()
    
    # 2. Initialize all validation components
    pipeline_validator = PipelineValidator()
    ml_validator = MLIntegrationValidator()
    qa_dashboard = QADashboard("complete_workflow.db")
    
    # 3. Run pipeline validation
    logger.info("Step 1: Running Pipeline Validation...")
    pipeline_data = {
        'dem_path': sample_data['dem_path'],
        'rainfall_data': sample_data['rainfall_data'],
        'simulation_results': sample_data['simulation_results'],
        'tiles_info': sample_data['tiles_info']
    }
    
    pipeline_results = pipeline_validator.validate_full_pipeline(pipeline_data)
    
    # 4. Run ML integration validation
    logger.info("Step 2: Running ML Integration Validation...")
    ml_data = {
        'dataset': sample_data['ml_dataset'],
        'labels_data': {'labels': sample_data['ml_labels']},
        'model_results': sample_data['model_results']
    }
    
    ml_results = ml_validator.validate_ml_pipeline(ml_data)
    
    # 5. Combine all results
    all_results = list(pipeline_results.values()) + list(ml_results.values())
    
    # 6. Process through QA dashboard
    logger.info("Step 3: Processing through QA Dashboard...")
    dashboard_result = qa_dashboard.process_validation_results(
        all_results,
        pipeline_type="Complete FloodRisk ML Pipeline",
        generate_report=True,
        report_path=Path("complete_workflow_report.html")
    )
    
    # 7. Generate comprehensive summary
    logger.info("Step 4: Generating Comprehensive Summary...")
    
    total_components = len(all_results)
    passed_components = len([r for r in all_results if r.status == 'PASS'])
    warning_components = len([r for r in all_results if r.status == 'WARN'])
    failed_components = len([r for r in all_results if r.status == 'FAIL'])
    
    overall_score = dashboard_result['overall_score']
    overall_status = dashboard_result['overall_status']
    
    logger.info("="*60)
    logger.info("COMPLETE WORKFLOW VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Components Validated: {total_components}")
    logger.info(f"✅ Passed: {passed_components}")
    logger.info(f"⚠️  Warning: {warning_components}")
    logger.info(f"❌ Failed: {failed_components}")
    logger.info(f"Overall Score: {overall_score:.3f}")
    logger.info(f"Overall Status: {overall_status}")
    logger.info("="*60)
    
    # 8. Component-by-component summary
    logger.info("COMPONENT DETAILS:")
    for result in all_results:
        status_emoji = "✅" if result.status == 'PASS' else "⚠️" if result.status == 'WARN' else "❌"
        logger.info(f"{status_emoji} {result.component}: {result.score:.3f} ({result.status})")
        if result.issues:
            for issue in result.issues:
                logger.info(f"    - {issue}")
    
    # 9. Recommendations
    logger.info("\nRECOMMENDATIONS:")
    if overall_status == 'PASS':
        logger.info("✅ All validations passed successfully!")
        logger.info("   Pipeline is ready for production deployment.")
    elif overall_status == 'WARN':
        logger.info("⚠️  Some components have warnings.")
        logger.info("   Review and address warnings before production deployment.")
    else:
        logger.info("❌ Critical issues detected.")
        logger.info("   Address failed components before proceeding.")
    
    # 10. Save final report
    final_report = {
        'workflow_summary': {
            'total_components': total_components,
            'passed_components': passed_components,
            'warning_components': warning_components,
            'failed_components': failed_components,
            'overall_score': overall_score,
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat()
        },
        'pipeline_results': [r.__dict__ for r in pipeline_results.values()],
        'ml_results': [r.__dict__ for r in ml_results.values()],
        'dashboard_data': dashboard_result
    }
    
    with open('complete_workflow_summary.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info("Complete workflow summary saved to: complete_workflow_summary.json")
    logger.info("Detailed HTML report saved to: complete_workflow_report.html")
    
    # Clean up
    import shutil
    shutil.rmtree(sample_data['temp_dir'], ignore_errors=True)
    
    return final_report


def main():
    """Main demonstration function"""
    print("🌊 FloodRisk Validation Framework - Comprehensive Demo")
    print("="*60)
    
    try:
        # Demonstrate individual validators
        print("\n1️⃣  Individual Validators Demo")
        individual_results = demonstrate_individual_validators()
        
        # Demonstrate pipeline validation
        print("\n2️⃣  Pipeline Validation Demo") 
        pipeline_results, pipeline_report = demonstrate_pipeline_validation()
        
        # Demonstrate ML integration validation
        print("\n3️⃣  ML Integration Validation Demo")
        ml_results, ml_report = demonstrate_ml_integration_validation()
        
        # Demonstrate QA dashboard
        print("\n4️⃣  QA Dashboard Demo")
        dashboard_result, dashboard_data = demonstrate_qa_dashboard()
        
        # Demonstrate complete workflow
        print("\n5️⃣  Complete End-to-End Workflow Demo")
        final_report = demonstrate_complete_workflow()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nGenerated files:")
        print("  - example_qa_report.html")
        print("  - complete_workflow_report.html") 
        print("  - complete_workflow_summary.json")
        print("  - example_validation.db")
        print("  - complete_workflow.db")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())