# Enhanced Preprocessing Pipeline Integration

## Overview

The enhanced preprocessing pipeline bridges real-world data acquisition with the FloodRisk ML pipeline, providing comprehensive preprocessing for USGS 3DEP DEMs and NOAA Atlas 14 rainfall data.

## Architecture

### Core Components

1. **RealDataPreprocessor**: Main orchestrator class
2. **EnhancedDEMProcessor**: Advanced DEM processing with RichDEM integration
3. **EnhancedRainfallProcessor**: NOAA Atlas 14 rainfall data processing
4. **SpatialProcessor**: Spatial alignment and validation
5. **DataValidator**: Comprehensive data quality validation
6. **PreprocessingQA**: Quality assurance system with visualizations

### Key Features

- **Advanced Flow Routing**: Uses RichDEM for state-of-the-art hydrological algorithms
- **Robust Spatial Handling**: Automatic CRS handling and spatial alignment
- **Intelligent Caching**: Hash-based caching for incremental processing
- **Configurable Pipelines**: JSON-based configuration for different regions
- **Comprehensive QA**: Statistical validation, spatial consistency checks, visualizations
- **Memory Efficiency**: Chunked processing for large datasets

## Integration with Existing Pipeline

### Data Flow

```
Real Data Sources → Enhanced Preprocessing → ML Pipeline
     ↓                        ↓                  ↓
USGS 3DEP DEM         RealDataPreprocessor    Training/Inference
NOAA Atlas 14         Advanced Processing     Flood Prediction
Stream Networks       Quality Assurance       Model Outputs
```

### Existing Module Integration

The enhanced pipeline integrates with existing preprocessing modules:

- `src/preprocessing/dem_processor.py` - Enhanced with RichDEM algorithms
- `src/preprocessing/rainfall_generator.py` - Extended for NOAA Atlas 14 data
- `src/preprocessing/terrain_features.py` - Used for comprehensive feature extraction
- `src/preprocessing/dem/hydrological_conditioning.py` - Advanced flow routing

### Configuration System

Region-specific configurations enable customized processing:

```json
{
  "region_name": "Nashville, TN",
  "bounds": {...},
  "dem_processing": {...},
  "spatial_processing": {...},
  "rainfall_processing": {...},
  "feature_extraction": {...},
  "validation": {...}
}
```

## Usage Examples

### Basic Usage

```python
from ml.data.real_data_preprocessing import create_preprocessor

# Create Nashville-optimized preprocessor
preprocessor = create_preprocessor(region="nashville")

# Process region data
processed_data = preprocessor.process_region(
    dem_path="data/nashville_dem.tif",
    rainfall_data=noaa_atlas14_data
)

# Create training tiles
tiles = preprocessor.create_training_tiles(processed_data)
```

### Advanced Configuration

```python
from ml.data.real_data_preprocessing import RealDataPreprocessor

# Custom configuration
preprocessor = RealDataPreprocessor(
    config_path="config/custom_region.json",
    target_crs="EPSG:3857",
    target_resolution=10.0,
    enable_caching=True
)
```

### Quality Assurance

```python
from ml.data.preprocessing_qa import run_qa_pipeline

# Run comprehensive QA
qa_results = run_qa_pipeline(
    processed_data=data,
    output_dir="qa_results"
)

print(f"QA Status: {qa_results['overall_status']}")
```

## Data Products

### Processed Outputs

1. **dem**: Original elevation data
2. **filled_dem**: Depression-filled elevation
3. **flow_direction**: D8 flow direction
4. **flow_accumulation**: Flow accumulation
5. **streams**: Stream network
6. **slope_degrees**: Slope in degrees
7. **hand**: Height Above Nearest Drainage
8. **curvature**: Profile/planform curvature
9. **twi**: Topographic Wetness Index
10. **rainfall_scenarios**: Multiple return period/duration scenarios

### Training Tiles

Each tile contains:
- Multi-band feature data (DEM, slope, flow accumulation, HAND, rainfall)
- Optional flood labels
- Metadata (location, bounds, tile ID)
- Consistent 512x512 pixel size with 64-pixel overlap

## Quality Assurance

### Validation Checks

1. **Data Quality**:
   - NoData percentage < 10%
   - Value ranges within expected bounds
   - Statistical distribution validation

2. **Spatial Consistency**:
   - Coordinate alignment
   - Resolution matching
   - CRS consistency

3. **Physical Consistency**:
   - DEM-slope correlation
   - Flow accumulation-streams consistency
   - HAND-elevation relationships

### Visualization Outputs

- Data overview plots
- Statistical summaries
- Correlation heatmaps
- Quality metrics dashboards

## Performance Optimizations

### Memory Management

- Chunked processing for large rasters
- Configurable memory limits
- Lazy loading with xarray

### Caching System

- SHA256-based cache keys
- Automatic cache invalidation
- Configurable cache directory

### Parallel Processing

- Multi-threaded tile generation
- Configurable worker count
- Memory-aware scheduling

## Configuration Reference

### Nashville Configuration

Located at `config/preprocessing/nashville_config.json`:

- Optimized for urban flood modeling
- 10m resolution processing
- Smaller drainage area thresholds
- Enhanced urban feature extraction

### Custom Regions

Create new configurations by:

1. Defining region bounds
2. Setting processing parameters
3. Specifying validation thresholds
4. Configuring output options

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk_size in configuration
2. **CRS Misalignment**: Check input data projections
3. **Missing Dependencies**: Install RichDEM for advanced algorithms
4. **Cache Issues**: Clear cache directory and restart

### Performance Tuning

1. **Large DEMs**: Increase chunk_size and max_memory_gb
2. **Slow Processing**: Enable parallel processing
3. **Storage Issues**: Enable compression in output config

## Integration Checklist

- [ ] Install dependencies (rasterio, xarray, richdem)
- [ ] Configure region-specific parameters
- [ ] Test with sample data
- [ ] Validate QA outputs
- [ ] Integrate with ML training pipeline
- [ ] Set up monitoring and logging

## Future Enhancements

### Planned Features

1. **Multi-source DEM fusion**
2. **Temporal rainfall analysis**
3. **Real-time processing capabilities**
4. **Cloud processing integration**
5. **Automated parameter tuning**

### Integration Points

1. **Data Acquisition**: Direct integration with acquisition modules
2. **ML Pipeline**: Seamless training data generation
3. **Validation**: Integration with LISFLOOD-FP and NFIP validators
4. **API**: REST API endpoints for processing requests

## References

- [USGS 3DEP Documentation](https://www.usgs.gov/3d-elevation-program)
- [NOAA Atlas 14](https://www.weather.gov/owp/hdsc_precip)
- [RichDEM Documentation](https://richdem.readthedocs.io/)
- [FloodRisk APPROACH.md](../APPROACH.md)