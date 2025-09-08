# Data Acquisition Documentation

## Overview

The FloodRisk data acquisition system provides automated download and validation of geospatial datasets required for flood risk modeling. The system supports multiple data sources and provides robust error handling, caching, and quality assurance.

## Supported Data Sources

### USGS 3D Elevation Program (3DEP)

The primary source for Digital Elevation Model (DEM) data in the United States.

**Available Resolutions:**
- 1m: High-resolution data (limited coverage, urban areas)
- 10m: Standard resolution (complete CONUS coverage) - **Recommended**
- 30m: Lower resolution (global coverage available)

**Data Access Methods:**
- **Static Service**: Faster downloads for 10m, 30m, and 60m resolutions
- **TNM API**: Complete product catalog access
- **Cloud-Optimized GeoTIFF (COG)**: Efficient partial data access

**Coverage:**
- Complete coverage of CONUS (48 contiguous states)
- Hawaii and U.S. territories
- Limited 1m coverage in urban areas

### NOAA Atlas 14 Precipitation Frequency

Precipitation frequency estimates for the United States based on observed data.

**Available Data:**
- **Return Periods**: 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000 years
- **Durations**: 5 minutes to 60 days (22 standard durations)
- **Series Types**: Partial Duration Series (PDS), Annual Maximum Series (AMS)
- **Units**: English (inches) or Metric (millimeters)

**Data Access Methods:**
- **PFDS API**: Point-specific precipitation frequency estimates
- **CSV Downloads**: Tabular data for analysis
- **Grid Downloads**: Spatially distributed estimates

## Installation

### Required Dependencies

The data acquisition system requires several geospatial Python libraries:

```bash
# Core dependencies (already in requirements.txt)
pip install requests pandas numpy

# Geospatial dependencies for validation (optional but recommended)
pip install rasterio geopandas pyproj

# Alternative lightweight packages for 3DEP access
pip install py3dep seamless3dep
```

### Additional Tools

For enhanced functionality, consider installing:

```bash
# GDAL tools for advanced raster operations
conda install gdal

# Jupyter for interactive data exploration
pip install jupyter
```

## Quick Start

### Nashville Case Study Download

Download all required data for Nashville flood risk analysis:

```bash
# Using the provided script
python scripts/data_acquisition/download_nashville_data.py

# With custom parameters
python scripts/data_acquisition/download_nashville_data.py \
    --output-dir ./data/nashville \
    --dem-resolution 10 \
    --rainfall-spacing 0.005 \
    --verbose
```

### Programmatic Usage

```python
from src.data.manager import DataManager
from src.data.config import DataConfig

# Initialize data manager
config = DataConfig()
manager = DataManager(config)

# Download Nashville case study data
results = manager.download_nashville_case_study(
    output_dir=Path("data/nashville"),
    dem_resolution=10,
    rainfall_grid_spacing=0.005
)

# Results contain paths to downloaded files
print(f"DEM files: {len(results['dem'])}")
print(f"Rainfall files: {len(results['rainfall'])}")
```

## Configuration

### Environment Variables

Configure data acquisition using environment variables:

```bash
export FLOODRISK_CACHE_DIR="/path/to/cache"
export FLOODRISK_DATA_DIR="/path/to/data"
export FLOODRISK_TARGET_CRS="3857"  # Web Mercator for ML pipeline
```

### Configuration Files

Data sources are configured in `config/data/data_sources.yaml`. Key settings:

```yaml
usgs_3dep:
  api_base_url: "https://tnmaccess.nationalmap.gov/api/v1/products"
  preferred_resolution: 10
  
noaa_atlas14:
  pfds_base_url: "https://hdsc.nws.noaa.gov/pfds"
  return_periods: [10, 25, 100, 500]
  durations_hours: [1, 3, 6, 12, 24]
```

### Regional Configurations

Predefined regions for common study areas:

```yaml
regions:
  nashville:
    bbox:
      west: -87.1284
      south: 35.9728
      east: -86.4637
      north: 36.4427
    recommended_dem_resolution: 10
    recommended_rainfall_spacing: 0.005
```

## Data Validation

### Automated Validation

All downloaded data undergoes automated validation:

**DEM Validation:**
- File format and readability
- Coordinate reference system (CRS) presence
- Reasonable elevation ranges (-500m to 10,000m)
- NoData value handling
- Raster dimensions and data types

**Rainfall Data Validation:**
- CSV format compliance
- Required columns presence
- Data type validation
- Range checks for precipitation values
- Missing data detection

### Manual Validation

Use the validation script for comprehensive checks:

```bash
# Validate all data in a directory
python scripts/data_acquisition/validate_data.py data/nashville \
    --output-report validation_report.json \
    --verbose

# Check specific file types
python scripts/data_acquisition/validate_data.py data/dem --verbose
```

## API Reference

### DataManager Class

Primary interface for data acquisition operations.

```python
class DataManager:
    def __init__(self, config: Optional[DataConfig] = None)
    
    def download_all_region_data(
        self,
        region_name: str,
        dem_resolution: int = 10,
        rainfall_return_periods: Optional[List[int]] = None,
        rainfall_durations_hours: Optional[List[float]] = None,
        rainfall_grid_spacing: float = 0.01,
        output_dir: Optional[Path] = None,
        parallel: bool = True
    ) -> Dict[str, List[Path]]
    
    def download_nashville_case_study(
        self,
        output_dir: Optional[Path] = None,
        dem_resolution: int = 10,
        rainfall_grid_spacing: float = 0.005
    ) -> Dict[str, List[Path]]
    
    def validate_data_integrity(
        self,
        file_paths: List[Path],
        data_type: str = "unknown"
    ) -> Dict[str, List[Path]]
```

### USGS3DEPDownloader Class

Specialized downloader for USGS 3DEP data.

```python
class USGS3DEPDownloader:
    def download_region(
        self,
        region_name: str,
        resolution: int = 10,
        output_dir: Optional[Path] = None
    ) -> List[Path]
    
    def get_seamless_dem(
        self,
        bbox: Union[BoundingBox, Dict[str, float]],
        resolution: int = 10,
        output_path: Optional[Path] = None
    ) -> Path
```

### NOAAAtlas14Fetcher Class

Specialized fetcher for NOAA Atlas 14 data.

```python
class NOAAAtlas14Fetcher:
    def download_point_data(
        self,
        location: Union[Tuple[float, float], Dict[str, float]],
        return_periods: Optional[List[int]] = None,
        durations_hours: Optional[List[float]] = None,
        output_dir: Optional[Path] = None
    ) -> Path
    
    def get_rainfall_depths(
        self,
        location: Union[Tuple[float, float], Dict[str, float]],
        return_periods: Optional[List[int]] = None,
        duration_hours: float = 24
    ) -> Dict[int, float]
```

## Data Formats

### Downloaded DEM Files

- **Format**: GeoTIFF (.tif) or IMG (.img)
- **Coordinate System**: Original (typically Geographic NAD83) 
- **Data Type**: Float32 (elevations in meters)
- **NoData**: Typically -9999 or -3.4e+38
- **Compression**: LZW or DEFLATE

### Downloaded Rainfall Files

- **Format**: CSV (.csv)
- **Encoding**: UTF-8
- **Columns**:
  - `return_period_years`: Return period (integer)
  - `duration_minutes`: Duration in minutes (integer)
  - `duration_label`: Human-readable duration (string)
  - `precipitation_inches`: Precipitation depth in inches (float)
  - `lower_bound`: 90% confidence interval lower bound (float)
  - `upper_bound`: 90% confidence interval upper bound (float)

## Error Handling

### Common Issues and Solutions

**Connection Errors:**
- **Issue**: Network timeouts or service unavailable
- **Solution**: Built-in retry mechanism with exponential backoff
- **Configuration**: Adjust `max_retries` and `timeout_seconds` in config

**Authentication Errors:**
- **Issue**: Access denied to data services
- **Solution**: Most data sources are public domain, no authentication needed
- **Note**: Some services may have rate limiting

**Data Quality Issues:**
- **Issue**: Downloaded files are corrupted or incomplete
- **Solution**: File validation with automatic re-download
- **Prevention**: Enable resume capability for large files

**Storage Issues:**
- **Issue**: Insufficient disk space
- **Solution**: Configure cache cleanup and use efficient storage locations
- **Monitoring**: Check available space before large downloads

### Logging and Debugging

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific loggers
logging.getLogger('src.data').setLevel(logging.DEBUG)
```

## Performance Optimization

### Caching Strategy

- **Persistent Cache**: Downloaded files cached indefinitely unless manually cleared
- **Metadata Cache**: API responses cached for 30 days (configurable)
- **Resume Capability**: Partial downloads automatically resumed
- **Validation Cache**: Valid files not re-validated for cache period

### Parallel Downloads

- **DEM and Rainfall**: Downloaded in parallel by default
- **Multiple Files**: Individual files downloaded sequentially to avoid overwhelming servers
- **Configuration**: Use `parallel=False` for sequential downloads

### Storage Optimization

- **Compression**: GeoTIFF files use LZW compression
- **COG Format**: Cloud-Optimized GeoTIFF when available
- **Efficient Spatial Queries**: Only download data within specified boundaries
- **Progressive Download**: Large areas split into manageable chunks

## License and Attribution

### Data Licenses

**USGS 3DEP Data:**
- **License**: Public Domain
- **Attribution**: U.S. Geological Survey
- **Restrictions**: None

**NOAA Atlas 14 Data:**
- **License**: Public Domain  
- **Attribution**: NOAA National Weather Service
- **Restrictions**: None

### Code License

The FloodRisk data acquisition system is licensed under [LICENSE]. See the project root for details.

## Support and Contributing

### Getting Help

1. **Documentation**: Check this documentation first
2. **Issues**: Report bugs and request features on GitHub
3. **Discussions**: Ask questions in GitHub Discussions

### Contributing

1. **Bug Reports**: Include error messages, logs, and reproduction steps
2. **Feature Requests**: Describe use case and expected behavior
3. **Code Contributions**: Follow the project's contributing guidelines

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd FloodRisk

# Install development dependencies  
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run data acquisition tests
pytest tests/test_data_acquisition.py -v
```

---

*Last updated: September 2024*
*Version: 0.1.0*