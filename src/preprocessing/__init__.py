"""Data preprocessing package for FloodRisk.

This package provides comprehensive data preprocessing capabilities for
flood risk analysis, including:

- DEM hydrological conditioning and flow analysis
- Terrain feature extraction (slope, curvature, HAND, etc.)
- Rainfall scenario generation from NOAA Atlas 14 data
- Multi-scale patch extraction for machine learning
- Data normalization and dimensionless feature computation
"""

from .dem_processor import DEMProcessor
from .terrain_features import TerrainFeatureExtractor
from .rainfall_generator import RainfallGenerator
from .patch_extractor import PatchExtractor, PatchInfo
from .normalizer import DataNormalizer, NormalizationParams

__all__ = [
    'DEMProcessor',
    'TerrainFeatureExtractor', 
    'RainfallGenerator',
    'PatchExtractor',
    'PatchInfo',
    'DataNormalizer',
    'NormalizationParams'
]

__version__ = '0.1.0'