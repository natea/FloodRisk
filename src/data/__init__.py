"""Data acquisition and management package for FloodRisk.

This package provides comprehensive data acquisition capabilities for
flood risk analysis, including:

- USGS 3DEP DEM data acquisition from multiple sources
- NOAA Atlas 14 precipitation frequency data retrieval
- Spatial data validation and quality checks
- Automated data caching and resume capabilities
- CRS handling and reprojection utilities
"""

from .sources.usgs_3dep import USGS3DEPDownloader
from .sources.noaa_atlas14 import NOAAAtlas14Fetcher
from .manager import DataManager
from .config import DataConfig

__all__ = [
    'USGS3DEPDownloader',
    'NOAAAtlas14Fetcher', 
    'DataManager',
    'DataConfig'
]

__version__ = '0.1.0'