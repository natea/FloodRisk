"""Physics-based flood simulation integration module.

This module provides integration with LISFLOOD-FP physics-based flood modeling
for generating high-quality training labels for the ML pipeline.

Key Components:
- LisfloodSimulator: Main interface to LISFLOOD-FP
- ParameterFileGenerator: Auto-generation of .par files from inputs
- SimulationBatch: Orchestration of multiple simulation scenarios
- ResultProcessor: Post-processing of simulation outputs to ML labels
- ValidationFramework: Quality control and validation of results
"""

from .lisflood_simulator import LisfloodSimulator
from .parameter_generator import ParameterFileGenerator
from .batch_orchestrator import SimulationBatch
from .result_processor import ResultProcessor
from .validation import SimulationValidator
from .metadata_tracker import SimulationMetadata
from .preprocessing_integration import PreprocessingIntegration

__all__ = [
    "LisfloodSimulator",
    "ParameterFileGenerator",
    "SimulationBatch",
    "ResultProcessor",
    "SimulationValidator",
    "SimulationMetadata",
    "PreprocessingIntegration",
]

__version__ = "1.0.0"
