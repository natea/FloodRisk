"""FloodRisk Pipeline Integration Module.

This module provides the main integration and orchestration layer
for the complete flood risk modeling pipeline, connecting data acquisition,
preprocessing, simulation, validation, and ML training components.
"""

__version__ = "1.0.0"

# Main pipeline components
from .main_controller import PipelineController, PipelineConfig
from .progress_tracker import ProgressTracker, ProgressMetrics
from .resource_manager import ResourceManager, ResourceConfig
from .checkpoint_manager import CheckpointManager, PipelineState
from .integration_api import IntegratedFloodPipeline

__all__ = [
    "PipelineController",
    "PipelineConfig",
    "ProgressTracker",
    "ProgressMetrics",
    "ResourceManager",
    "ResourceConfig",
    "CheckpointManager",
    "PipelineState",
    "IntegratedFloodPipeline",
]
