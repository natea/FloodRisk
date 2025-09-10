#!/usr/bin/env python3
"""
Nashville Flood Risk Modeling Demonstration

This script demonstrates the complete end-to-end flood risk modeling pipeline
for Nashville, Tennessee, showcasing all integrated components working together.

Usage:
    python examples/nashville_demo.py [--dry-run] [--output-dir OUTPUT_DIR]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.pipeline.integration_api import IntegratedFloodPipeline, run_nashville_flood_modeling
from src.pipeline.main_controller import PipelineConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NashvilleDemoRunner:
    """Nashville flood modeling demonstration runner."""
    
    def __init__(self, output_dir: str = "./nashville_demo_output"):
        """Initialize demo runner.
        
        Args:
            output_dir: Output directory for demonstration results
        """
        self.output_dir = Path(output_dir)
        self.demo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.demo_output_dir = self.output_dir / f"nashville_demo_{self.demo_timestamp}"
        
        # Create demo configuration optimized for demonstration
        self.config = PipelineConfig(
            project_name="nashville_flood_demo",
            project_description="Nashville flood risk modeling demonstration",
            region_name="Nashville, TN",
            output_root=str(self.demo_output_dir),
            log_level="INFO",
            
            # Nashville bounding box (slightly smaller for demo speed)
            bbox={
                "west": -87.0,
                "south": 36.0,
                "east": -86.6,
                "north": 36.3,
                "crs": "EPSG:4326"
            },
            
            # DEM configuration  
            dem_source="usgs_3dep",
            dem_resolution=10,  # 10m resolution
            target_crs="EPSG:3857",
            
            # Rainfall scenarios (reduced for demo)
            rainfall_source="noaa_atlas14",
            return_periods=[25, 100],  # Focus on key return periods
            storm_durations=[6, 24],   # 6-hour and 24-hour storms
            rainfall_patterns=["scs_type_ii", "uniform"],
            
            # Simulation settings
            max_sim_time=1800.0,  # 30 minutes simulation time
            output_interval=300.0,  # 5-minute output interval
            parallel_simulations=2,  # Conservative for demo
            
            # ML training (enabled but with reduced parameters for demo)
            ml_enabled=True,
            tile_size=128,  # Smaller tiles for faster processing
            batch_size=4,
            max_epochs=10,  # Reduced epochs for demo
            validation_split=0.2,
            
            # Resource management
            max_memory_gb=8.0,
            cleanup_intermediate=False,  # Keep files for inspection
            enable_checkpointing=True,
            checkpoint_interval_minutes=5.0,  # Frequent checkpoints for demo
            
            # Quality control
            validation_enabled=True,
            strict_validation=False,
            min_flood_extent_km2=0.05,  # Lower threshold for demo
            max_flood_extent_km2=500.0
        )
        
        # Initialize pipeline
        self.pipeline = IntegratedFloodPipeline(
            self.config,
            enable_monitoring=True,
            enable_checkpointing=True,
            enable_resource_management=True
        )
        
        # Demo progress tracking
        self.demo_start_time = None
        self.demo_stages_completed = []
        self.demo_metrics = {}
        
        logger.info(f"Nashville Demo initialized")
        logger.info(f"Output directory: {self.demo_output_dir}")
    
    def setup_demo_callbacks(self):
        """Setup callbacks for demonstration progress tracking."""
        
        def progress_callback(progress_data):
            """Handle progress updates."""
            stage = progress_data.get("current_stage", "unknown")
            completion = progress_data.get("overall_progress", 0) * 100
            
            logger.info(f"🔄 Demo Progress: {completion:.1f}% - {stage}")
            
            # Update demo metrics
            self.demo_metrics["last_progress"] = completion
            self.demo_metrics["current_stage"] = stage
            
        def stage_callback(stage_name, stage_data):
            """Handle stage completion."""
            logger.info(f"✅ Demo Stage Completed: {stage_name}")
            self.demo_stages_completed.append({
                "stage": stage_name,
                "timestamp": datetime.now().isoformat(),
                "data": stage_data
            })
        
        def error_callback(stage, error):
            """Handle pipeline errors."""
            logger.error(f"❌ Demo Error in {stage}: {error}")
            self.demo_metrics["error_stage"] = stage
            self.demo_metrics["error_message"] = str(error)
        
        # Register callbacks
        self.pipeline.add_progress_callback(progress_callback)
        
        # Add stage-specific callbacks
        for stage in ["data_acquisition", "simulation_execution", "ml_training_execution"]:
            self.pipeline.add_stage_callback(stage, lambda data, s=stage: stage_callback(s, data))
        
        self.pipeline.add_error_callback(error_callback)
    
    async def run_demonstration(self, dry_run: bool = False) -> Dict[str, any]:
        """Run the complete Nashville demonstration.
        
        Args:
            dry_run: If True, validate setup without execution
            
        Returns:
            Demonstration results
        """
        self.demo_start_time = datetime.now()
        
        logger.info("🚀 Starting Nashville Flood Risk Modeling Demonstration")
        logger.info("="*60)
        logger.info(f"Demo Configuration:")
        logger.info(f"  Region: {self.config.region_name}")
        logger.info(f"  Bounding Box: {self.config.bbox}")
        logger.info(f"  DEM Resolution: {self.config.dem_resolution}m")
        logger.info(f"  Return Periods: {self.config.return_periods}")
        logger.info(f"  Storm Durations: {self.config.storm_durations}h")
        logger.info(f"  ML Training: {'Enabled' if self.config.ml_enabled else 'Disabled'}")
        logger.info(f"  Output Directory: {self.demo_output_dir}")
        logger.info("="*60)
        
        try:
            # Setup monitoring
            self.setup_demo_callbacks()
            
            # Run pipeline
            results = await self.pipeline.run_pipeline(dry_run=dry_run)
            
            # Calculate demo duration
            demo_duration = datetime.now() - self.demo_start_time
            
            # Create comprehensive demo report
            demo_report = self._create_demo_report(results, demo_duration, dry_run)
            
            # Save demo report
            self._save_demo_report(demo_report)
            
            if dry_run:
                logger.info("✅ Nashville Demo Validation Completed Successfully")
                logger.info(f"   Setup validated, ready for full execution")
            else:
                logger.info("🎉 Nashville Demo Completed Successfully!")
                logger.info(f"   Total Duration: {demo_duration}")
                logger.info(f"   Stages Completed: {len(self.demo_stages_completed)}")
                logger.info(f"   Results saved to: {self.demo_output_dir}")
            
            return demo_report
            
        except Exception as e:
            demo_duration = datetime.now() - self.demo_start_time
            logger.error(f"❌ Nashville Demo Failed after {demo_duration}")
            logger.error(f"   Error: {e}")
            
            # Create failure report
            failure_report = self._create_failure_report(e, demo_duration)
            self._save_demo_report(failure_report)
            
            raise
    
    def _create_demo_report(self, pipeline_results: Dict[str, any], 
                           duration: any, dry_run: bool) -> Dict[str, any]:
        """Create comprehensive demonstration report."""
        
        report = {
            "demonstration_info": {
                "demo_type": "Nashville Flood Risk Modeling",
                "dry_run": dry_run,
                "start_time": self.demo_start_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_minutes": duration.total_seconds() / 60,
                "configuration": self.config.__dict__
            },
            "pipeline_results": pipeline_results,
            "demo_metrics": self.demo_metrics,
            "stages_completed": self.demo_stages_completed,
        }
        
        if not dry_run:
            # Add detailed results analysis
            report["results_analysis"] = self._analyze_pipeline_results(pipeline_results)
            report["key_outputs"] = self._identify_key_outputs(pipeline_results)
            report["performance_summary"] = self._summarize_performance(pipeline_results)
        
        # Add recovery information
        if hasattr(self.pipeline, 'checkpoint_manager') and self.pipeline.checkpoint_manager:
            report["recovery_info"] = {
                "checkpoints_available": len(self.pipeline.list_checkpoints()),
                "recovery_options": self.pipeline.get_recovery_options()
            }
        
        report["generated_at"] = datetime.now().isoformat()
        
        return report
    
    def _create_failure_report(self, error: Exception, duration: any) -> Dict[str, any]:
        """Create failure analysis report."""
        
        return {
            "demonstration_info": {
                "demo_type": "Nashville Flood Risk Modeling",
                "status": "FAILED",
                "start_time": self.demo_start_time.isoformat(),
                "failure_time": datetime.now().isoformat(),
                "duration_before_failure_seconds": duration.total_seconds(),
                "configuration": self.config.__dict__
            },
            "error_info": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_stage": self.demo_metrics.get("error_stage", "unknown")
            },
            "stages_completed": self.demo_stages_completed,
            "demo_metrics": self.demo_metrics,
            "recovery_info": self.pipeline.get_recovery_options() if hasattr(self.pipeline, 'get_recovery_options') else [],
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_pipeline_results(self, results: Dict[str, any]) -> Dict[str, any]:
        """Analyze pipeline execution results."""
        
        analysis = {
            "overall_success": results.get("execution_results", {}).get("status") == "success",
            "stages_analysis": {}
        }
        
        # Analyze each stage
        execution_results = results.get("execution_results", {})
        for stage_name, stage_result in execution_results.items():
            if isinstance(stage_result, dict):
                analysis["stages_analysis"][stage_name] = {
                    "status": stage_result.get("status", "unknown"),
                    "has_outputs": bool(stage_result.get("outputs")),
                    "key_metrics": self._extract_stage_metrics(stage_result)
                }
        
        # Overall statistics
        successful_stages = sum(
            1 for stage_analysis in analysis["stages_analysis"].values()
            if stage_analysis["status"] == "success"
        )
        
        analysis["overall_statistics"] = {
            "total_stages": len(analysis["stages_analysis"]),
            "successful_stages": successful_stages,
            "success_rate": successful_stages / len(analysis["stages_analysis"]) if analysis["stages_analysis"] else 0
        }
        
        return analysis
    
    def _extract_stage_metrics(self, stage_result: Dict[str, any]) -> Dict[str, any]:
        """Extract key metrics from stage results."""
        
        metrics = {}
        
        # Data acquisition metrics
        if "dem_file" in stage_result:
            metrics["dem_acquired"] = bool(stage_result["dem_file"])
        
        if "rainfall_data" in stage_result:
            metrics["rainfall_scenarios"] = len(stage_result.get("rainfall_data", {}))
        
        # Simulation metrics
        if "successful_simulations" in stage_result:
            metrics["successful_simulations"] = stage_result["successful_simulations"]
            metrics["failed_simulations"] = stage_result.get("failed_simulations", 0)
        
        # ML metrics
        if "training_metrics" in stage_result:
            ml_metrics = stage_result["training_metrics"]
            if isinstance(ml_metrics, dict):
                metrics.update({f"ml_{k}": v for k, v in ml_metrics.items()})
        
        return metrics
    
    def _identify_key_outputs(self, results: Dict[str, any]) -> Dict[str, str]:
        """Identify key output files for users."""
        
        key_outputs = {}
        
        # Main report
        if "report_file" in results:
            key_outputs["final_report"] = results["report_file"]
        
        # Output files
        output_files = results.get("output_files", {})
        for key, path in output_files.items():
            if isinstance(path, str) and Path(path).exists():
                key_outputs[key] = path
        
        # Batch results
        execution_results = results.get("execution_results", {})
        if "simulation_execution" in execution_results:
            sim_results = execution_results["simulation_execution"]
            if "batch_results" in sim_results:
                key_outputs["batch_results_directory"] = sim_results["batch_results"]
        
        # ML model
        if "ml_training_execution" in execution_results:
            ml_results = execution_results["ml_training_execution"]
            if "model_checkpoint" in ml_results:
                key_outputs["ml_model"] = ml_results["model_checkpoint"]
        
        return key_outputs
    
    def _summarize_performance(self, results: Dict[str, any]) -> Dict[str, any]:
        """Summarize pipeline performance."""
        
        performance = {
            "total_runtime_minutes": 0,
            "stage_performance": {},
            "resource_efficiency": "N/A"
        }
        
        # Extract timing information
        progress_data = results.get("progress_summary", {})
        if progress_data:
            metrics = progress_data.get("metrics", {})
            performance["total_runtime_minutes"] = metrics.get("total_runtime_seconds", 0) / 60
        
        # Resource usage
        resource_usage = results.get("resource_usage", {})
        if resource_usage:
            stats = resource_usage.get("statistics", {})
            if stats:
                memory_stats = stats.get("memory_gb", {})
                performance["peak_memory_gb"] = memory_stats.get("peak", 0)
                performance["average_memory_gb"] = memory_stats.get("average", 0)
        
        return performance
    
    def _save_demo_report(self, report: Dict[str, any]):
        """Save demonstration report."""
        
        # Create outputs directory
        self.demo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main report
        report_file = self.demo_output_dir / "nashville_demo_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = self.demo_output_dir / "demo_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Nashville Flood Risk Modeling Demonstration Summary\n")
            f.write("="*55 + "\n\n")
            
            demo_info = report["demonstration_info"]
            f.write(f"Demo Type: {demo_info['demo_type']}\n")
            f.write(f"Start Time: {demo_info['start_time']}\n")
            f.write(f"Duration: {demo_info.get('duration_minutes', 0):.1f} minutes\n")
            f.write(f"Dry Run: {demo_info.get('dry_run', False)}\n\n")
            
            if "results_analysis" in report:
                analysis = report["results_analysis"]
                f.write("Results Analysis:\n")
                f.write(f"  Overall Success: {analysis['overall_success']}\n")
                stats = analysis.get("overall_statistics", {})
                f.write(f"  Successful Stages: {stats.get('successful_stages', 0)}/{stats.get('total_stages', 0)}\n")
                f.write(f"  Success Rate: {stats.get('success_rate', 0):.1%}\n\n")
            
            if "key_outputs" in report:
                f.write("Key Output Files:\n")
                for name, path in report["key_outputs"].items():
                    f.write(f"  {name}: {path}\n")
            
            f.write(f"\nGenerated: {report['generated_at']}\n")
        
        logger.info(f"📄 Demo report saved to: {report_file}")
        logger.info(f"📋 Demo summary saved to: {summary_file}")


async def main():
    """Main demonstration function."""
    
    parser = argparse.ArgumentParser(description="Nashville Flood Risk Modeling Demonstration")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Validate setup without executing pipeline")
    parser.add_argument("--output-dir", type=str, default="./nashville_demo_output",
                       help="Output directory for demonstration results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run demonstration
    demo = NashvilleDemoRunner(args.output_dir)
    
    try:
        results = await demo.run_demonstration(dry_run=args.dry_run)
        
        if args.dry_run:
            print("\n✅ Nashville Demo Validation Completed!")
            print("   The pipeline setup has been validated and is ready for execution.")
            print("   Run without --dry-run to execute the complete demonstration.")
        else:
            print("\n🎉 Nashville Demo Completed Successfully!")
            print(f"   Results saved to: {demo.demo_output_dir}")
            print(f"   Duration: {results['demonstration_info']['duration_minutes']:.1f} minutes")
            
            # Print key outputs
            if "key_outputs" in results:
                print("\n📁 Key Output Files:")
                for name, path in results["key_outputs"].items():
                    print(f"   {name}: {path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Demonstration interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))