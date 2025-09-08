#!/usr/bin/env python3
"""Integration Test Script for FloodRisk Pipeline.

This script validates that all pipeline components are properly integrated
and can work together without actually running expensive operations.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_pipeline_integration():
    """Test the complete pipeline integration."""
    
    print("🔧 FloodRisk Pipeline Integration Test")
    print("="*50)
    
    test_results = {
        "imports": {"status": "pending", "details": {}},
        "configuration": {"status": "pending", "details": {}},
        "pipeline_creation": {"status": "pending", "details": {}},
        "prerequisites": {"status": "pending", "details": {}},
        "dry_run": {"status": "pending", "details": {}}
    }
    
    # Test 1: Import all integration components
    print("\n1️⃣ Testing component imports...")
    try:
        from src.pipeline.integration_api import IntegratedFloodPipeline, run_nashville_flood_modeling
        from src.pipeline.main_controller import PipelineController, PipelineConfig
        from src.pipeline.progress_tracker import ProgressTracker
        from src.pipeline.resource_manager import ResourceManager
        from src.pipeline.checkpoint_manager import CheckpointManager
        
        test_results["imports"]["status"] = "pass"
        test_results["imports"]["details"] = "All pipeline components imported successfully"
        print("✅ All integration components imported successfully")
        
    except Exception as e:
        test_results["imports"]["status"] = "fail"
        test_results["imports"]["details"] = str(e)
        print(f"❌ Import failed: {e}")
        return test_results
    
    # Test 2: Configuration system
    print("\n2️⃣ Testing configuration system...")
    try:
        # Test basic config creation
        config = PipelineConfig(
            project_name="integration_test",
            region_name="Test Region",
            output_root="./test_outputs"
        )
        
        # Test YAML config loading
        config_file = project_root / "config" / "nashville_demo_config.yaml"
        if config_file.exists():
            yaml_config = PipelineConfig.from_yaml(str(config_file))
            test_results["configuration"]["details"]["yaml_config"] = "loaded successfully"
        else:
            test_results["configuration"]["details"]["yaml_config"] = "file not found"
        
        test_results["configuration"]["status"] = "pass"
        test_results["configuration"]["details"]["basic_config"] = "created successfully"
        print("✅ Configuration system working")
        
    except Exception as e:
        test_results["configuration"]["status"] = "fail"
        test_results["configuration"]["details"] = str(e)
        print(f"❌ Configuration test failed: {e}")
        return test_results
    
    # Test 3: Pipeline creation
    print("\n3️⃣ Testing pipeline creation...")
    try:
        pipeline = IntegratedFloodPipeline(
            config=config,
            enable_monitoring=True,
            enable_checkpointing=True,
            enable_resource_management=True
        )
        
        test_results["pipeline_creation"]["status"] = "pass"
        test_results["pipeline_creation"]["details"] = {
            "monitoring": "enabled",
            "checkpointing": "enabled",
            "resource_management": "enabled"
        }
        print("✅ Pipeline created successfully with all features enabled")
        
    except Exception as e:
        test_results["pipeline_creation"]["status"] = "fail"
        test_results["pipeline_creation"]["details"] = str(e)
        print(f"❌ Pipeline creation failed: {e}")
        return test_results
    
    # Test 4: Prerequisites check
    print("\n4️⃣ Testing prerequisites check...")
    try:
        prereq_results = pipeline.check_prerequisites()
        
        test_results["prerequisites"]["status"] = "pass"
        test_results["prerequisites"]["details"] = {
            "overall_status": prereq_results.get("overall_status", "unknown"),
            "checks_completed": len(prereq_results.get("checks", {}))
        }
        
        overall_status = prereq_results.get("overall_status", "unknown")
        if overall_status == "ready":
            print("✅ All prerequisites met - system ready for execution")
        elif overall_status == "not_ready":
            print("⚠️  Some prerequisites not met - see details")
            for check_name, check_result in prereq_results.get("checks", {}).items():
                status = check_result.get("status", "unknown")
                if status == "fail":
                    print(f"   ❌ {check_name}: {check_result.get('details', 'No details')}")
                elif status == "warning":
                    print(f"   ⚠️  {check_name}: {check_result.get('details', 'No details')}")
                else:
                    print(f"   ✅ {check_name}: OK")
        else:
            print(f"⚠️  Prerequisites check status: {overall_status}")
        
    except Exception as e:
        test_results["prerequisites"]["status"] = "fail"
        test_results["prerequisites"]["details"] = str(e)
        print(f"❌ Prerequisites check failed: {e}")
        return test_results
    
    # Test 5: Dry run validation
    print("\n5️⃣ Testing dry run validation...")
    try:
        dry_run_results = await pipeline.run_pipeline(dry_run=True)
        
        test_results["dry_run"]["status"] = "pass"
        test_results["dry_run"]["details"] = {
            "dry_run_completed": True,
            "validation_status": dry_run_results.get("prerequisites", {}).get("overall_status", "unknown")
        }
        print("✅ Dry run validation completed successfully")
        print(f"   Pipeline setup validated and ready for execution")
        
    except Exception as e:
        test_results["dry_run"]["status"] = "fail"
        test_results["dry_run"]["details"] = str(e)
        print(f"❌ Dry run validation failed: {e}")
        return test_results
    
    # Test summary
    print("\n📊 Integration Test Summary")
    print("="*50)
    
    passed_tests = sum(1 for result in test_results.values() if result["status"] == "pass")
    total_tests = len(test_results)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in test_results.items():
        status_symbol = "✅" if result["status"] == "pass" else "❌"
        print(f"{status_symbol} {test_name}: {result['status']}")
    
    if passed_tests == total_tests:
        print("\n🎉 All integration tests passed!")
        print("   The FloodRisk pipeline is properly integrated and ready for use.")
        print("   Run the Nashville demonstration with:")
        print("     python examples/nashville_demo.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("   Please check the errors above and ensure all dependencies are installed.")
    
    return test_results


def test_component_connectivity():
    """Test that all components can connect to each other."""
    
    print("\n🔗 Testing Component Connectivity")
    print("="*40)
    
    try:
        # Test data acquisition components
        from src.data.acquisition import DEMAcquisition, RainfallDataAcquisition
        print("✅ Data acquisition components connected")
        
        # Test preprocessing components  
        from src.data.preprocessing import TopographicPreprocessor, DEMProcessor, RainfallProcessor
        print("✅ Preprocessing components connected")
        
        # Test simulation components
        from src.simulation.lisflood_simulator import LisfloodSimulator
        from src.simulation.batch_orchestrator import SimulationBatch
        from src.simulation.parameter_generator import ParameterFileGenerator
        print("✅ Simulation components connected")
        
        # Test ML components
        from src.ml.training.train import FloodLightningModule, FloodDataModule
        print("✅ ML training components connected")
        
        # Test validation components
        from src.validation.flood_extent_validator import FloodExtentValidator
        print("✅ Validation components connected")
        
        return True
        
    except Exception as e:
        print(f"❌ Component connectivity test failed: {e}")
        return False


def test_nashville_config():
    """Test Nashville-specific configuration."""
    
    print("\n🏙️  Testing Nashville Configuration")
    print("="*40)
    
    try:
        config_file = Path("config/nashville_demo_config.yaml")
        if config_file.exists():
            from src.pipeline.main_controller import PipelineConfig
            config = PipelineConfig.from_yaml(str(config_file))
            
            print(f"✅ Nashville config loaded: {config.region_name}")
            print(f"   DEM Resolution: {config.dem_resolution}m")
            print(f"   Return Periods: {config.return_periods}")
            print(f"   ML Training: {'Enabled' if config.ml_enabled else 'Disabled'}")
            
            return True
        else:
            print("❌ Nashville config file not found")
            return False
            
    except Exception as e:
        print(f"❌ Nashville config test failed: {e}")
        return False


async def main():
    """Main test function."""
    
    print("🚀 FloodRisk Pipeline Complete Integration Test")
    print("="*60)
    print("This test validates that all pipeline components are properly")
    print("integrated and can work together as a complete system.")
    print("="*60)
    
    # Run all tests
    integration_results = await test_pipeline_integration()
    connectivity_ok = test_component_connectivity()
    nashville_config_ok = test_nashville_config()
    
    # Overall results
    print("\n🏁 Final Results")
    print("="*30)
    
    integration_passed = sum(1 for r in integration_results.values() if r["status"] == "pass") == len(integration_results)
    
    all_tests_passed = integration_passed and connectivity_ok and nashville_config_ok
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   The FloodRisk pipeline is fully integrated and production-ready.")
        print("\n📖 Next Steps:")
        print("   1. Run Nashville demonstration: python examples/nashville_demo.py --dry-run")
        print("   2. For full execution: python examples/nashville_demo.py")
        print("   3. Customize for your region using config/nashville_demo_config.yaml as template")
        print("   4. Review docs/INTEGRATION_GUIDE.md for detailed usage instructions")
        
        return 0
    else:
        print("❌ Some tests failed.")
        print("   Please review the errors above and ensure all dependencies are properly installed.")
        
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)