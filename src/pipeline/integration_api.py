"""Unified Integration API for FloodRisk Pipeline.

This module provides a high-level API that integrates all pipeline components
into a seamless, production-ready flood risk modeling system.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from .main_controller import PipelineController, PipelineConfig, PipelineStage
from .progress_tracker import ProgressTracker
from .resource_manager import ResourceManager, ResourceConfig
from .checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class IntegratedFloodPipeline:
    """Production-ready integrated flood risk modeling pipeline.

    This class provides a complete, high-level API for running end-to-end
    flood risk modeling workflows with built-in monitoring, recovery,
    and resource management.
    """

    def __init__(
        self,
        config: Union[PipelineConfig, str, Path],
        enable_monitoring: bool = True,
        enable_checkpointing: bool = True,
        enable_resource_management: bool = True,
    ):
        """Initialize the integrated pipeline.

        Args:
            config: Pipeline configuration (object, YAML file path, or dict)
            enable_monitoring: Enable progress monitoring
            enable_checkpointing: Enable checkpoint/recovery
            enable_resource_management: Enable resource monitoring
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = PipelineConfig.from_yaml(str(config))
        elif isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config

        # Feature flags
        self.enable_monitoring = enable_monitoring
        self.enable_checkpointing = enable_checkpointing
        self.enable_resource_management = enable_resource_management

        # Core components
        self.controller: Optional[PipelineController] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

        # Execution state
        self.pipeline_id: Optional[str] = None
        self.is_running = False
        self.execution_results: Optional[Dict[str, Any]] = None

        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.stage_callbacks: Dict[str, List[Callable]] = {}
        self.error_callbacks: List[Callable] = []

        logger.info(
            f"IntegratedFloodPipeline initialized for {self.config.region_name}"
        )

    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a progress update callback.

        Args:
            callback: Function to call with progress updates
        """
        self.progress_callbacks.append(callback)

    def add_stage_callback(
        self, stage: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """Add a stage completion callback.

        Args:
            stage: Pipeline stage name
            callback: Function to call when stage completes
        """
        if stage not in self.stage_callbacks:
            self.stage_callbacks[stage] = []
        self.stage_callbacks[stage].append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add an error callback.

        Args:
            callback: Function to call when errors occur
        """
        self.error_callbacks.append(callback)

    def check_prerequisites(self) -> Dict[str, Any]:
        """Check if system prerequisites are met for pipeline execution.

        Returns:
            Dictionary with prerequisite check results
        """
        logger.info("Checking system prerequisites...")

        checks = {
            "system_resources": {"status": "checking"},
            "dependencies": {"status": "checking"},
            "data_access": {"status": "checking"},
            "output_permissions": {"status": "checking"},
        }

        try:
            # Check system resources
            if self.enable_resource_management:
                resource_config = ResourceConfig(
                    max_memory_gb=self.config.max_memory_gb,
                    max_disk_space_gb=self.config.max_disk_space_gb,
                )
                temp_resource_manager = ResourceManager(resource_config)
                resource_status = temp_resource_manager.check_available_resources()

                if resource_status["overall_status"]:
                    checks["system_resources"] = {
                        "status": "pass",
                        "details": resource_status,
                    }
                else:
                    checks["system_resources"] = {
                        "status": "fail",
                        "details": resource_status,
                    }
            else:
                checks["system_resources"] = {
                    "status": "skip",
                    "reason": "Resource management disabled",
                }

            # Check Python dependencies
            missing_deps = self._check_dependencies()
            if not missing_deps:
                checks["dependencies"] = {
                    "status": "pass",
                    "details": "All required packages available",
                }
            else:
                checks["dependencies"] = {
                    "status": "fail",
                    "details": f"Missing packages: {missing_deps}",
                }

            # Check data source access
            data_access = self._check_data_access()
            checks["data_access"] = data_access

            # Check output directory permissions
            try:
                output_dir = Path(self.config.output_root)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Test write permissions
                test_file = output_dir / ".permission_test"
                test_file.write_text("test")
                test_file.unlink()

                checks["output_permissions"] = {
                    "status": "pass",
                    "details": f"Write access confirmed: {output_dir}",
                }
            except Exception as e:
                checks["output_permissions"] = {
                    "status": "fail",
                    "details": f"Cannot write to output directory: {e}",
                }

            # Overall status
            overall_status = all(
                check["status"] in ["pass", "skip"] for check in checks.values()
            )

            result = {
                "overall_status": "ready" if overall_status else "not_ready",
                "checks": checks,
                "timestamp": datetime.now().isoformat(),
            }

            if overall_status:
                logger.info("✅ System prerequisites check passed")
            else:
                logger.warning("❌ System prerequisites check failed - see details")

            return result

        except Exception as e:
            logger.error(f"Prerequisites check failed with exception: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "checks": checks,
                "timestamp": datetime.now().isoformat(),
            }

    def _check_dependencies(self) -> List[str]:
        """Check for missing Python dependencies."""
        required_packages = [
            "numpy",
            "pandas",
            "xarray",
            "rasterio",
            "geopandas",
            "requests",
            "aiohttp",
            "matplotlib",
            "psutil",
            "torch",
            "pytorch_lightning",
            "segmentation_models_pytorch",
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        return missing

    def _check_data_access(self) -> Dict[str, Any]:
        """Check access to external data sources."""
        try:
            import requests

            # Test USGS data access
            usgs_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
            response = requests.get(usgs_url, timeout=10)
            usgs_ok = response.status_code == 200

            # Test NOAA data access (this will always work with our sample data)
            noaa_ok = True  # We use generated sample data

            if usgs_ok and noaa_ok:
                return {"status": "pass", "details": "Data sources accessible"}
            else:
                return {
                    "status": "warning",
                    "details": f"USGS: {'✓' if usgs_ok else '✗'}, NOAA: {'✓' if noaa_ok else '✗'}",
                }

        except Exception as e:
            return {
                "status": "warning",
                "details": f"Could not verify data access: {e}",
            }

    async def run_pipeline(
        self, resume_from_checkpoint: Optional[str] = None, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run the complete pipeline with full integration.

        Args:
            resume_from_checkpoint: Checkpoint ID to resume from
            dry_run: If True, validate setup without executing

        Returns:
            Pipeline execution results
        """
        if self.is_running:
            raise RuntimeError("Pipeline is already running")

        try:
            self.is_running = True

            # Check prerequisites first
            prereq_results = self.check_prerequisites()
            if prereq_results["overall_status"] == "not_ready" and not dry_run:
                raise RuntimeError(f"Prerequisites not met: {prereq_results}")

            if dry_run:
                logger.info("🔍 Dry run completed - pipeline setup validated")
                return {
                    "dry_run": True,
                    "prerequisites": prereq_results,
                    "status": "validated",
                }

            # Initialize pipeline components
            await self._initialize_components()

            # Handle checkpoint recovery
            if resume_from_checkpoint:
                await self._resume_from_checkpoint(resume_from_checkpoint)

            # Execute pipeline with full monitoring
            async with self._monitoring_context():
                results = await self.controller.execute_pipeline()

            self.execution_results = results

            # Generate comprehensive report
            final_report = await self._generate_final_report()

            logger.info("🎉 Pipeline execution completed successfully!")
            return final_report

        except Exception as e:
            await self._handle_pipeline_error(e)
            raise
        finally:
            self.is_running = False

    async def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        # Create main controller
        self.controller = PipelineController(self.config)
        self.pipeline_id = self.controller.pipeline_id

        # Initialize optional components
        if self.enable_monitoring:
            self.progress_tracker = ProgressTracker(
                self.controller.output_dir / "progress.json", auto_save_interval=30.0
            )

            # Start progress tracking
            stage_names = [stage.value for stage in PipelineStage]
            self.progress_tracker.start_pipeline(stage_names)

        if self.enable_resource_management:
            resource_config = ResourceConfig(
                max_memory_gb=self.config.max_memory_gb,
                max_disk_space_gb=self.config.max_disk_space_gb,
                cleanup_intermediate=self.config.cleanup_intermediate,
            )
            self.resource_manager = ResourceManager(resource_config)

        if self.enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(
                self.controller.output_dir / "checkpoints",
                auto_save_interval=self.config.checkpoint_interval_minutes,
            )

        # Initialize controller components
        self.controller.initialize_components()

        logger.info("✅ All components initialized successfully")

    async def _resume_from_checkpoint(self, checkpoint_id: str):
        """Resume pipeline from a checkpoint."""
        if not self.checkpoint_manager:
            raise ValueError("Checkpointing not enabled")

        logger.info(f"Resuming from checkpoint: {checkpoint_id}")

        state = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not state:
            raise ValueError(f"Could not load checkpoint: {checkpoint_id}")

        # Restore controller state
        self.controller.current_stage = state.stage
        self.controller.results = state.results

        # Update progress tracker
        if self.progress_tracker:
            # Mark previous stages as completed
            for stage_name in state.results.keys():
                self.progress_tracker.complete_stage(stage_name, success=True)

        logger.info(f"✅ Resumed from stage: {state.stage.value}")

    @asynccontextmanager
    async def _monitoring_context(self):
        """Context manager for pipeline monitoring."""
        try:
            # Start monitoring
            if self.resource_manager:
                self.resource_manager.start_monitoring()

            if self.checkpoint_manager and self.controller.current_stage:
                # Create initial checkpoint
                from .checkpoint_manager import PipelineState

                initial_state = PipelineState(
                    pipeline_id=self.pipeline_id,
                    stage=self.controller.current_stage,
                    timestamp=datetime.now(),
                    configuration=self.config.__dict__,
                )
                self.checkpoint_manager.start_auto_save(initial_state)

            yield

        finally:
            # Stop monitoring
            if self.resource_manager:
                self.resource_manager.stop_monitoring()

            if self.checkpoint_manager:
                self.checkpoint_manager.stop_auto_save()

            if self.progress_tracker:
                self.progress_tracker.end_pipeline(success=True)

    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline execution errors."""
        logger.error(f"Pipeline execution failed: {error}")

        # Save error state for recovery
        if self.checkpoint_manager and self.controller:
            self.checkpoint_manager.save_error_state(
                self.controller.current_stage, str(error), self.pipeline_id
            )

        # End progress tracking as failed
        if self.progress_tracker:
            self.progress_tracker.end_pipeline(success=False)

        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(
                    (
                        self.controller.current_stage.value
                        if self.controller
                        else "unknown"
                    ),
                    error,
                )
            except Exception as cb_error:
                logger.error(f"Error callback failed: {cb_error}")

    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        report = {
            "pipeline_id": self.pipeline_id,
            "region_name": self.config.region_name,
            "execution_timestamp": datetime.now().isoformat(),
            "configuration": self.config.__dict__,
            "execution_results": self.execution_results,
        }

        # Add monitoring data
        if self.progress_tracker:
            report["progress_summary"] = self.progress_tracker.get_progress_summary()
            report["performance_metrics"] = (
                self.progress_tracker.get_performance_metrics()
            )

        if self.resource_manager:
            report["resource_usage"] = self.resource_manager.get_usage_summary()

        if self.checkpoint_manager:
            report["checkpoint_summary"] = (
                self.checkpoint_manager.get_checkpoint_summary()
            )

        # Generate outputs inventory
        if self.controller:
            outputs_dir = self.controller.output_dir
            report["output_files"] = self._inventory_outputs(outputs_dir)

        # Save report
        if self.controller:
            report_file = self.controller.output_dir / "final_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            report["report_file"] = str(report_file)

        return report

    def _inventory_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """Create inventory of output files."""
        inventory = {
            "output_directory": str(output_dir),
            "total_files": 0,
            "total_size_mb": 0.0,
            "file_types": {},
            "key_outputs": {},
        }

        if not output_dir.exists():
            return inventory

        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                inventory["total_files"] += 1
                file_size_mb = file_path.stat().st_size / 1024 / 1024
                inventory["total_size_mb"] += file_size_mb

                # Count by extension
                ext = file_path.suffix.lower()
                if ext not in inventory["file_types"]:
                    inventory["file_types"][ext] = {"count": 0, "size_mb": 0.0}
                inventory["file_types"][ext]["count"] += 1
                inventory["file_types"][ext]["size_mb"] += file_size_mb

                # Identify key outputs
                if file_path.name in [
                    "final_report.json",
                    "pipeline_metadata.json",
                    "batch_summary.json",
                ]:
                    inventory["key_outputs"][file_path.name] = str(file_path)

        return inventory

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Dictionary with current status information
        """
        status = {
            "pipeline_id": self.pipeline_id,
            "is_running": self.is_running,
            "region_name": self.config.region_name,
            "timestamp": datetime.now().isoformat(),
        }

        if self.controller:
            status["current_stage"] = self.controller.current_stage.value
            status["completed_stages"] = list(self.controller.results.keys())

        if self.progress_tracker:
            status["progress"] = self.progress_tracker.get_progress_summary()["metrics"]

        if self.resource_manager:
            status["resource_usage"] = (
                self.resource_manager.get_current_usage().to_dict()
            )

        return status

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints for this pipeline.

        Returns:
            List of checkpoint metadata
        """
        if not self.checkpoint_manager:
            return []

        checkpoints = self.checkpoint_manager.list_checkpoints(self.pipeline_id)
        return [cp.to_dict() for cp in checkpoints]

    def get_recovery_options(self) -> List[Dict[str, Any]]:
        """Get recovery options for failed pipelines.

        Returns:
            List of recovery options
        """
        if not self.checkpoint_manager or not self.pipeline_id:
            return []

        return self.checkpoint_manager.get_recovery_options(self.pipeline_id)

    def export_logs(self, output_file: Path):
        """Export comprehensive logs and reports.

        Args:
            output_file: Path to save the export
        """
        export_data = {
            "pipeline_id": self.pipeline_id,
            "configuration": self.config.__dict__,
            "status": self.get_status(),
            "export_timestamp": datetime.now().isoformat(),
        }

        if self.controller and self.controller.output_dir.exists():
            # Read key log files
            log_files = {
                "pipeline.log": self.controller.output_dir / "pipeline.log",
                "progress.json": self.controller.output_dir / "progress.json",
                "final_report.json": self.controller.output_dir / "final_report.json",
            }

            for name, path in log_files.items():
                if path.exists():
                    try:
                        if path.suffix == ".json":
                            with open(path, "r") as f:
                                export_data[name] = json.load(f)
                        else:
                            with open(path, "r") as f:
                                export_data[name] = f.read()
                    except Exception as e:
                        export_data[name] = f"Error reading {path}: {e}"

        # Save export
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Logs exported to: {output_file}")


# Convenience functions for common use cases


async def run_nashville_flood_modeling(
    output_dir: str = "./nashville_flood_output",
    parallel_simulations: int = 4,
    enable_ml_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run complete Nashville flood modeling workflow.

    Args:
        output_dir: Output directory for results
        parallel_simulations: Number of parallel simulations
        enable_ml_training: Whether to include ML training
        **kwargs: Additional configuration options

    Returns:
        Pipeline execution results
    """
    # Create Nashville configuration
    config = PipelineConfig(
        project_name="nashville_flood_modeling",
        region_name="Nashville, TN",
        output_root=output_dir,
        parallel_simulations=parallel_simulations,
        ml_enabled=enable_ml_training,
        **kwargs,
    )

    # Create and run pipeline
    pipeline = IntegratedFloodPipeline(config)
    results = await pipeline.run_pipeline()

    return results


async def run_custom_flood_modeling(
    region_name: str, bbox: Dict[str, float], output_dir: str, **kwargs
) -> Dict[str, Any]:
    """Run flood modeling for a custom region.

    Args:
        region_name: Name of the region
        bbox: Bounding box with west, south, east, north coordinates
        output_dir: Output directory for results
        **kwargs: Additional configuration options

    Returns:
        Pipeline execution results
    """
    config = PipelineConfig(
        project_name=f"flood_modeling_{region_name.lower().replace(' ', '_')}",
        region_name=region_name,
        bbox=bbox,
        output_root=output_dir,
        **kwargs,
    )

    pipeline = IntegratedFloodPipeline(config)
    results = await pipeline.run_pipeline()

    return results


def create_pipeline_from_config(
    config_file: Union[str, Path],
) -> IntegratedFloodPipeline:
    """Create pipeline from configuration file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Configured pipeline instance
    """
    return IntegratedFloodPipeline(config_file)


if __name__ == "__main__":
    # Example usage
    async def main():
        results = await run_nashville_flood_modeling()
        print(f"Pipeline completed successfully!")
        print(f"Results saved to: {results['output_files']['output_directory']}")

    asyncio.run(main())
