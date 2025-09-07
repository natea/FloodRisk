"""
Flood Risk Prediction Models

This package contains neural network architectures for flood depth prediction
and related hydrological modeling tasks.
"""

from .flood_cnn import (
    FloodDepthPredictor,
    PhysicsInformedLoss,
    DimensionlessFeatureProcessor,
    MultiScaleInputProcessor,
    RainfallScalingModule,
    AttentionFusion,
    FloodModelTrainer,
    create_flood_model
)

__all__ = [
    'FloodDepthPredictor',
    'PhysicsInformedLoss',
    'DimensionlessFeatureProcessor',
    'MultiScaleInputProcessor',
    'RainfallScalingModule',
    'AttentionFusion',
    'FloodModelTrainer',
    'create_flood_model'
]