#!/usr/bin/env python3
"""
FloodRisk Validation Framework - Quick Start Guide

A minimal example showing how to use the validation framework
for common validation tasks.

Author: FloodRisk QA Team
"""

import numpy as np
import tempfile
from pathlib import Path
import logging

# Import validation framework
from src.validation import (
    PipelineValidator, MLIntegrationValidator, QADashboard
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_pipeline_validation():
    """Quick example of pipeline validation"""
    print("🔍 Quick Pipeline Validation Example")
    print("-" * 40)
    
    # Create simple test data
    np.random.seed(42)
    
    # Simple rainfall data (2D array representing rainfall intensities)
    rainfall_data = np.random.exponential(5.0, (100, 100))  # mm/hr
    rainfall_data = np.clip(rainfall_data, 0, 100)  # Realistic max
    
    # Simple simulation results
    simulation_results = {
        'depths': np.random.exponential(0.4, (100, 100)),  # Flood depths in meters
        'convergence': {
            'final_residual': 1e-7,
            'iterations': 30
        }
    }
    
    # Simple tiles for ML training
    tiles_data = []
    for i in range(50):
        # Create flood/dry tiles with reasonable balance
        tile = np.random.rand(64, 64)
        flood_threshold = np.random.uniform(0.3, 0.7)  # 30-70% flood
        tile_flood = (tile > flood_threshold).astype(float) * np.random.uniform(0.1, 2.0, (64, 64))
        tiles_data.append({'data': tile_flood})
    
    tiles_info = {
        'tiles': tiles_data,
        'metadata': {'tile_size': (64, 64)}
    }
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Prepare validation data (minimal example without actual files)
    pipeline_data = {
        'rainfall_data': rainfall_data,
        'simulation_results': simulation_results,
        'tiles_info': tiles_info
    }
    
    # Run validation
    print("Running pipeline validation...")
    results = validator.validate_full_pipeline(pipeline_data)
    
    # Display results
    print(f"\n✅ Validation completed! {len(results)} components validated:")
    for component, result in results.items():
        status_emoji = "✅" if result.status == 'PASS' else "⚠️" if result.status == 'WARN' else "❌"
        print(f"{status_emoji} {component}: {result.score:.3f} ({result.status})")
        
        if result.issues:
            for issue in result.issues[:2]:  # Show first 2 issues
                print(f"    • {issue}")
    
    # Get overall status
    overall_status = validator.get_pipeline_status()
    print(f"\n🎯 Overall Pipeline Status: {overall_status}")
    
    return results


def quick_ml_validation():
    """Quick example of ML integration validation"""
    print("\n🤖 Quick ML Integration Validation Example")
    print("-" * 45)
    
    # Create simple mock dataset class
    class SimpleDataset:
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Simple 3-channel input (like DEM + rainfall + landuse)
            input_tensor = np.random.randn(3, 64, 64).astype(np.float32)
            # Simple binary flood mask output
            target_tensor = (np.random.rand(1, 64, 64) > 0.3).astype(np.float32)
            return input_tensor, target_tensor
    
    # Create sample training labels
    sample_labels = np.random.choice([0, 1], size=(50, 64, 64), p=[0.7, 0.3])
    
    # Create sample model results
    model_results = {
        'training_history': {
            'train_loss': [1.5, 1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.6, 1.1, 0.9, 0.7, 0.6, 0.5],
            'train_accuracy': [0.6, 0.7, 0.75, 0.8, 0.83, 0.85],
            'val_accuracy': [0.58, 0.68, 0.73, 0.78, 0.80, 0.82]
        },
        'test_metrics': {
            'accuracy': 0.81,
            'precision': 0.78,
            'recall': 0.84,
            'f1_score': 0.81
        }\n    }\n    \n    # Initialize ML validator\n    ml_validator = MLIntegrationValidator()\n    \n    # Prepare ML data\n    ml_data = {\n        'dataset': SimpleDataset(),\n        'labels_data': {'labels': sample_labels},\n        'model_results': model_results\n    }\n    \n    # Run ML validation\n    print(\"Running ML integration validation...\")\n    ml_results = ml_validator.validate_ml_pipeline(ml_data)\n    \n    # Display results\n    print(f\"\\n✅ ML validation completed! {len(ml_results)} components validated:\")\n    for component, result in ml_results.items():\n        status_emoji = \"✅\" if result.status == 'PASS' else \"⚠️\" if result.status == 'WARN' else \"❌\"\n        component_name = component.replace('_', ' ').title()\n        print(f\"{status_emoji} {component_name}: {result.score:.3f} ({result.status})\")\n        \n        if result.issues:\n            for issue in result.issues[:2]:  # Show first 2 issues\n                print(f\"    • {issue}\")\n    \n    # Get overall ML status\n    ml_status = ml_validator.get_ml_pipeline_status()\n    print(f\"\\n🎯 Overall ML Pipeline Status: {ml_status}\")\n    \n    return ml_results\n\n\ndef quick_qa_dashboard():\n    \"\"\"Quick example of QA dashboard usage\"\"\"\n    print(\"\\n📊 Quick QA Dashboard Example\")\n    print(\"-\" * 35)\n    \n    # Use in-memory database for this example\n    dashboard = QADashboard(\":memory:\")\n    \n    # Create some sample validation results\n    from src.validation.pipeline_validator import ValidationResult\n    from datetime import datetime\n    \n    sample_results = [\n        ValidationResult(\n            component='Rainfall_Quality',\n            status='PASS',\n            score=0.88,\n            details={'max_intensity': 45.2, 'coverage': 0.98},\n            issues=[],\n            timestamp=datetime.now()\n        ),\n        ValidationResult(\n            component='Simulation_Quality',\n            status='WARN',\n            score=0.74,\n            details={'convergence': True, 'mass_balance_error': 0.08},\n            issues=['Mass balance slightly exceeds tolerance'],\n            timestamp=datetime.now()\n        ),\n        ValidationResult(\n            component='ML_Data_Compatibility',\n            status='PASS',\n            score=0.92,\n            details={'dataset_size': 1000, 'format_valid': True},\n            issues=[],\n            timestamp=datetime.now()\n        )\n    ]\n    \n    # Process through dashboard\n    print(\"Processing validation results through QA dashboard...\")\n    dashboard_result = dashboard.process_validation_results(\n        sample_results,\n        pipeline_type=\"Quick Start Demo\",\n        generate_report=True,\n        report_path=Path(\"quick_start_qa_report.html\")\n    )\n    \n    # Display dashboard results\n    print(f\"\\n📈 Dashboard processing completed:\")\n    print(f\"   Run ID: {dashboard_result['run_id']}\")\n    print(f\"   Overall Score: {dashboard_result['overall_score']:.3f}\")\n    print(f\"   Overall Status: {dashboard_result['overall_status']}\")\n    print(f\"   Active Alerts: {dashboard_result['alert_summary']['total_alerts']}\")\n    \n    if Path(\"quick_start_qa_report.html\").exists():\n        print(f\"\\n📄 QA Report generated: quick_start_qa_report.html\")\n        print(\"   Open this file in your browser to view the interactive report!\")\n    \n    return dashboard_result\n\n\ndef main():\n    \"\"\"Main quick start demonstration\"\"\"\n    print(\"🌊 FloodRisk Validation Framework - Quick Start\")\n    print(\"=\" * 50)\n    print(\"This example demonstrates basic validation functionality\\n\")\n    \n    try:\n        # Run quick examples\n        pipeline_results = quick_pipeline_validation()\n        ml_results = quick_ml_validation() \n        dashboard_result = quick_qa_dashboard()\n        \n        # Summary\n        print(\"\\n\" + \"=\" * 50)\n        print(\"🎉 Quick Start Completed Successfully!\")\n        print(\"=\" * 50)\n        \n        total_components = len(pipeline_results) + len(ml_results)\n        print(f\"✅ Validated {total_components} pipeline components\")\n        print(f\"📊 Generated QA dashboard report\")\n        print(f\"🔍 Overall system status: Ready for production\")\n        \n        print(\"\\n📚 Next Steps:\")\n        print(\"   1. Check out examples/validation_example.py for comprehensive examples\")\n        print(\"   2. Read docs/validation_framework_guide.md for detailed documentation\")\n        print(\"   3. Run tests with: pytest tests/validation/ -v\")\n        print(\"   4. Integrate validation into your existing pipeline\")\n        \n        return 0\n        \n    except Exception as e:\n        logger.error(f\"Quick start failed: {e}\")\n        print(\"\\n❌ Quick start encountered an error. Please check the logs.\")\n        return 1\n\n\nif __name__ == \"__main__\":\n    exit(main())"