"""Data acquisition configuration management."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pyproj import CRS


@dataclass
class BoundingBox:
    """Geographic bounding box definition."""
    west: float
    south: float  
    east: float
    north: float
    crs: Union[str, int] = 4326
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.west >= self.east:
            raise ValueError("West coordinate must be less than east coordinate")
        if self.south >= self.north:
            raise ValueError("South coordinate must be less than north coordinate")
            
    @property
    def bounds_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounds as (west, south, east, north) tuple."""
        return (self.west, self.south, self.east, self.north)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'west': self.west,
            'south': self.south,
            'east': self.east, 
            'north': self.north,
            'crs': self.crs
        }


@dataclass
class DataConfig:
    """Configuration for data acquisition operations."""
    
    # Directory paths
    cache_dir: Path = Path.home() / ".floodrisk" / "cache"
    data_dir: Path = Path("data")
    temp_dir: Path = Path("tmp")
    
    # Target CRS for processing (Web Mercator for ML pipeline)
    target_crs: Union[str, int] = 3857
    
    # USGS 3DEP Configuration
    usgs_api_base_url: str = "https://tnmaccess.nationalmap.gov/api/v1/products"
    usgs_static_service_url: str = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster"
    preferred_dem_resolution: int = 10  # meters
    dem_formats: List[str] = field(default_factory=lambda: ["GeoTIFF", "IMG"])
    
    # NOAA Atlas 14 Configuration  
    noaa_pfds_base_url: str = "https://hdsc.nws.noaa.gov/pfds"
    noaa_api_base_url: str = "https://pfds.weather.gov/pfds_api"
    return_periods_years: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 500])
    durations_hours: List[float] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    
    # Request settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 300
    chunk_size: int = 8192
    
    # Caching settings
    enable_caching: bool = True
    cache_expiry_days: int = 30
    
    # Data validation settings
    validate_downloads: bool = True
    min_file_size_bytes: int = 500  # Lowered to accommodate small CSV files
    
    # Regional configurations
    regions: Dict[str, BoundingBox] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default regions if not provided."""
        if not self.regions:
            self.regions.update({
                'nashville': BoundingBox(
                    west=-87.1284, south=35.9728,
                    east=-86.4637, north=36.4427,
                    crs=4326
                ),
                'tennessee': BoundingBox(
                    west=-90.3103, south=34.9829,
                    east=-81.6469, north=36.6781,
                    crs=4326
                ),
                'middle_tennessee': BoundingBox(
                    west=-88.2034, south=35.3619,
                    east=-85.6060, north=36.6781,
                    crs=4326
                )
            })
            
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True) 
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def target_crs_obj(self) -> CRS:
        """Get target CRS as pyproj CRS object."""
        return CRS.from_epsg(self.target_crs) if isinstance(self.target_crs, int) else CRS.from_string(self.target_crs)
    
    def get_region_bbox(self, region_name: str) -> Optional[BoundingBox]:
        """Get bounding box for named region."""
        return self.regions.get(region_name.lower())
    
    def add_region(self, name: str, bbox: BoundingBox) -> None:
        """Add a new region configuration."""
        self.regions[name.lower()] = bbox
    
    @classmethod
    def from_env(cls, **kwargs) -> 'DataConfig':
        """Create config from environment variables with overrides."""
        config = cls(**kwargs)
        
        # Override with environment variables if present
        if cache_dir := os.getenv('FLOODRISK_CACHE_DIR'):
            config.cache_dir = Path(cache_dir)
        if data_dir := os.getenv('FLOODRISK_DATA_DIR'):
            config.data_dir = Path(data_dir)
        if target_crs := os.getenv('FLOODRISK_TARGET_CRS'):
            config.target_crs = int(target_crs) if target_crs.isdigit() else target_crs
            
        return config


# Default configuration instance
default_config = DataConfig()