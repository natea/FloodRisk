# LISFLOOD-FP Simulation Pipeline Documentation

## Overview

The simulation pipeline integrates LISFLOOD-FP physics-based flood modeling with the FloodRisk ML training pipeline. It provides automated generation of high-quality flood extent labels from hydrodynamic simulations for multiple return periods and rainfall patterns.

## Architecture

### Core Components

1. **LisfloodSimulator** - Direct interface to LISFLOOD-FP executable
2. **ParameterFileGenerator** - Automated generation of simulation parameter files
3. **SimulationBatch** - Orchestration of multiple parallel simulations
4. **ResultProcessor** - Post-processing of simulation outputs to ML-ready formats
5. **SimulationValidator** - Comprehensive quality control and validation
6. **SimulationMetadata** - Provenance tracking and metadata management
7. **PreprocessingIntegration** - Integration with existing preprocessing pipeline

### Pipeline Flow

```
Preprocessing Pipeline → DEM/Rainfall Data → Parameter Generation → 
LISFLOOD-FP Simulation → Result Processing → Validation → ML Training Labels
```

## Installation and Setup

### Prerequisites

1. **LISFLOOD-FP**: Compiled executable available at `LISFLOOD-FP/build/lisflood`
2. **Python Dependencies**: numpy, scipy, rasterio (optional), pathlib
3. **System Resources**: Sufficient disk space and memory for flood simulations

### Compilation of LISFLOOD-FP

The LISFLOOD-FP submodule is already compiled. If recompilation is needed:

```bash
cd LISFLOOD-FP
cmake -S . -B build
cmake --build build
```

### Python Environment

Ensure required packages are installed:

```bash
pip install numpy scipy rasterio  # rasterio optional for GeoTIFF support
```

## Usage Examples

### Single Simulation

```python
from src.simulation import LisfloodSimulator, SimulationConfig

# Initialize simulator
simulator = LisfloodSimulator(
    lisflood_executable="LISFLOOD-FP/build/lisflood",
    working_directory="simulation_runs"
)

# Create configuration
config = SimulationConfig(
    dem_file="LISFLOOD-FP/Nashville/final_dem.asc",
    rainfall_file="LISFLOOD-FP/Nashville/rain_input_100yr.rain",
    manning_file="LISFLOOD-FP/Nashville/manning.asc",
    infiltration_file="LISFLOOD-FP/Nashville/infiltration.asc",
    sim_time=86400.0,  # 24 hours
    acceleration=True
)

# Run simulation
result = simulator.run_simulation(config, "test_100yr")

if result['status'] == 'success':
    print(f"Flooded fraction: {result['outputs']['statistics']['flood_fraction']:.3%}")
```

### Batch Simulations

```python
from src.simulation import SimulationBatch, ParameterFileGenerator, BatchConfig
from src.simulation.parameter_generator import ReturnPeriodConfig, HyetographConfig

# Set up components
simulator = LisfloodSimulator()
param_generator = ParameterFileGenerator()

batch_config = BatchConfig(
    max_parallel_jobs=4,
    validate_results=True
)

batch = SimulationBatch(simulator, param_generator, batch_config)

# Define scenarios
return_periods = [
    ReturnPeriodConfig(100, 177.8, "100-year storm"),
    ReturnPeriodConfig(500, 222.25, "500-year storm")
]

patterns = [
    HyetographConfig('uniform', 0),
    HyetographConfig('center_loaded', 0, peak_hour=12.0)
]

# Create and execute batch
batch_id = batch.create_batch_from_config(
    dem_file="final_dem.asc",
    return_periods=return_periods,
    hyetograph_patterns=patterns,
    output_dir="batch_results"
)

summary = batch.execute_batch()
print(f"Success rate: {summary['success_rate']:.1%}")
```

### Command Line Interface

#### Single Simulation

```bash
python scripts/run_simulation_pipeline.py single \\
    --dem-file LISFLOOD-FP/Nashville/final_dem.asc \\
    --rainfall-file LISFLOOD-FP/Nashville/rain_input_100yr.rain \\
    --output-dir results/single_sim \\
    --simulation-id nashville_100yr
```

#### Batch Simulations

```bash
python scripts/run_simulation_pipeline.py batch \\
    --dem-file LISFLOOD-FP/Nashville/final_dem.asc \\
    --return-periods 100,500 \\
    --patterns uniform,center_loaded,front_loaded \\
    --output-dir results/batch_sim \\
    --parallel-jobs 4
```

#### Result Processing

```bash
python scripts/run_simulation_pipeline.py process \\
    --input-dir results/batch_sim \\
    --output-dir results/processed \\
    --flood-threshold 0.05 \\
    --save-geotiff
```

#### Validation

```bash
python scripts/run_simulation_pipeline.py validate \\
    --input-dir results/batch_sim \\
    --output-file validation_report.json \\
    --target-success-rate 0.8
```

## Configuration

### Return Periods

Standard return periods are configured with NOAA Atlas 14 rainfall depths for Nashville:

| Return Period | 24-hour Depth | Usage |
|---------------|----------------|-------|
| 10-year | 111.76 mm (4.4") | Sub-design (negative examples) |
| 25-year | 142.24 mm (5.6") | Sub-design (negative examples) |
| 100-year | 177.8 mm (7.01") | Primary design storm |
| 500-year | 222.25 mm (8.75") | Extreme event |

### Hyetograph Patterns

Multiple temporal rainfall distributions for training diversity:

- **Uniform**: Constant intensity over 24 hours
- **Front-loaded**: Higher intensity in first 6 hours (factor: 2.5x)
- **Center-loaded**: Bell curve peaked at hour 12 (factor: 3.0x)
- **Back-loaded**: Higher intensity in last 6 hours (factor: 2.5x)

### Simulation Parameters

Default LISFLOOD-FP parameters optimized for urban pluvial flooding:

```python
SimulationConfig(
    sim_time=86400.0,        # 24 hours simulation
    initial_timestep=10.0,   # 10 second initial timestep
    acceleration=True,       # Enable adaptive timestepping
    depth_threshold=0.05     # 5cm flood depth threshold
)
```

## Output Formats

### Simulation Outputs

Each simulation produces:

- **Depth raster** (.max file): Water depths at each grid cell
- **Flood extent** (.npy): Binary flood map (≥0.05m depth)
- **Statistics** (.json): Flood metrics and validation results
- **Metadata** (.json): Complete provenance and configuration

### Training Data Structure

```
results/
├── batch_summary.json          # Overall batch statistics
├── training_data_manifest.json # ML training data catalog
├── processed_results/          # Post-processed outputs
│   ├── sim_001_flood_extent.npy
│   ├── sim_001_depth.npy
│   └── sim_001_statistics.json
└── metadata/                   # Provenance tracking
    ├── simulations.json
    └── sim_001.json
```

### Training Manifest Format

```json
{
  "created_at": "2024-01-15T10:30:00",
  "total_samples": 16,
  "flood_threshold_m": 0.05,
  "samples": [
    {
      "simulation_id": "100yr_uniform_178mm",
      "flood_extent_file": "sim_001_flood_extent.npy",
      "depth_file": "sim_001_depth.npy",
      "statistics": {
        "flood_fraction": 0.023,
        "max_depth_m": 1.47,
        "flooded_area_km2": 2.34
      },
      "return_period": {"return_period_years": 100, "rainfall_depth_24h_mm": 177.8},
      "hyetograph": {"pattern_type": "uniform"}
    }
  ]
}
```

## Quality Control

### Validation Framework

Comprehensive validation ensures simulation reliability:

#### Physical Validation
- Depth range checks (0.01m - 50m reasonable for pluvial)
- Mass conservation assessment
- Depth distribution analysis

#### Spatial Validation
- Flood fraction bounds (0.01% - 30% for urban pluvial)
- Flood area validation (100m² - 1000km²)
- Spatial continuity analysis

#### Temporal Validation
- Runtime bounds (10s - 24h reasonable)
- Convergence assessment

#### Data Quality
- Completeness checks for all outputs
- Consistency validation across metrics
- NaN/infinite value detection

### Quality Metrics

- **Overall Score**: 0-100 validation score
- **Status Categories**: passed, warning, failed
- **Issue Tracking**: Detailed error and warning logs
- **Batch Success Rate**: Target ≥80% for production use

## Performance Optimization

### Parallel Execution

- **Configurable Parallelism**: 1-16 concurrent simulations
- **Resource Management**: Memory and CPU monitoring
- **Fault Tolerance**: Automatic retry with exponential backoff
- **Load Balancing**: Dynamic job distribution

### Computational Requirements

| Scenario Count | Est. Runtime | Disk Space | Memory |
|----------------|--------------|------------|---------|
| Single (100yr) | 2-5 minutes | 50 MB | 512 MB |
| Standard Batch (16) | 30-60 minutes | 800 MB | 2 GB |
| Full Dataset (64) | 2-4 hours | 3.2 GB | 4 GB |

### Optimization Strategies

1. **Preprocessing**: Use conditioned DEMs to improve convergence
2. **Timestep Adaptation**: Enable acceleration for automatic timestep adjustment
3. **Grid Resolution**: Balance accuracy vs computational cost (10m recommended)
4. **Parallel Scaling**: Optimal performance at 4-8 concurrent jobs on modern hardware

## Integration with ML Pipeline

### Data Flow Integration

```python
from src.simulation.preprocessing_integration import PreprocessingIntegration

# Set up integration
integration = PreprocessingIntegration(
    preprocessing_output_dir="preprocessing_results",
    simulation_output_dir="simulation_results"
)

# Run integrated pipeline
results = integration.run_integrated_simulation(
    preprocessing_config_file="preprocessing_config.json",
    max_parallel_jobs=4
)

# Export for ML training
training_manifest = integration.export_for_ml_training(
    results, 
    ml_training_dir="ml_training_data"
)
```

### Feature Engineering

The simulation pipeline provides rich features for ML training:

- **Input Features**: DEM, rainfall depth, terrain derivatives
- **Label Features**: Binary flood extent, depth values
- **Metadata Features**: Return period, hyetograph pattern, validation metrics

### Training Data Characteristics

Based on APPROACH.md specifications:

- **Spatial Resolution**: ~10m pixels (consistent with 3DEP DEM)
- **Tile Size**: 512×512 pixels with 64px overlap
- **Label Threshold**: ≥0.05m depth for binary classification
- **Morphological Cleaning**: Remove <4 connected pixels, close 1-2 pixel gaps
- **Sampling Strategy**: 70% tiles with ≥2-5% flooding, 30% random tiles

## Troubleshooting

### Common Issues

#### LISFLOOD-FP Executable Not Found
```bash
# Ensure LISFLOOD-FP is compiled
cd LISFLOOD-FP
cmake --build build

# Or specify custom path
python scripts/run_simulation_pipeline.py --lisflood-exe /path/to/lisflood
```

#### Memory Issues
```python
# Reduce parallel jobs
batch_config = BatchConfig(max_parallel_jobs=2)

# Enable cleanup
batch_config.cleanup_failed_runs = True
batch_config.keep_intermediate_files = False
```

#### Simulation Failures
```python
# Enable detailed validation
batch_config.validate_results = True
batch_config.max_retries = 3

# Check validation reports
validator = SimulationValidator()
validation = validator.validate_batch_results(results)
```

### Validation Failures

| Issue | Cause | Solution |
|-------|--------|----------|
| Low flood fraction | Insufficient rainfall/infiltration too high | Check rainfall depths, reduce infiltration rates |
| Excessive flooding | Rainfall too high/poor drainage | Validate return period data, check DEM conditioning |
| Long runtime | Large domain/small timestep | Enable acceleration, check grid size |
| Convergence issues | Poor DEM quality | Use hydrologically conditioned DEM |

### Performance Issues

- **Slow simulations**: Check DEM conditioning, enable acceleration
- **Memory usage**: Reduce parallel jobs, enable cleanup
- **Disk space**: Monitor output size, cleanup intermediate files
- **Failed validations**: Review threshold settings, check input data quality

## API Reference

### Core Classes

#### LisfloodSimulator
```python
LisfloodSimulator(lisflood_executable, working_directory)
run_simulation(config, simulation_id, cleanup_temp=True)
```

#### ParameterFileGenerator
```python
ParameterFileGenerator(base_config_dir, template_dir)
generate_scenario_parameters(dem_file, return_periods, hyetograph_patterns, output_dir)
create_standard_return_periods()
create_standard_hyetographs()
```

#### SimulationBatch
```python
SimulationBatch(simulator, parameter_generator, batch_config)
create_batch_from_config(dem_file, return_periods, hyetograph_patterns, output_dir)
execute_batch()
export_training_data_paths(output_file)
```

#### ResultProcessor
```python
ResultProcessor(config)
process_simulation_output(depth_file, output_dir, simulation_id)
process_batch_outputs(simulation_results, output_dir)
```

#### SimulationValidator
```python
SimulationValidator(thresholds)
validate_single_simulation(simulation_result, depth_data, flood_extent)
validate_batch_results(simulation_results, detailed_validation)
```

#### SimulationMetadata
```python
SimulationMetadata(metadata_dir, enable_file_checksums, track_environment)
create_simulation_provenance(simulation_id, input_files, parameter_config)
save_simulation_provenance(provenance)
create_lineage_report(simulation_id)
```

### Configuration Classes

#### SimulationConfig
```python
SimulationConfig(
    dem_file, rainfall_file, manning_file, infiltration_file,
    sim_time=86400.0, initial_timestep=10.0, acceleration=True
)
```

#### ReturnPeriodConfig
```python
ReturnPeriodConfig(
    return_period_years, rainfall_depth_24h_mm, 
    description="", is_sub_design=False
)
```

#### HyetographConfig
```python
HyetographConfig(
    pattern_type, total_depth_mm, duration_hours=24.0,
    peak_hour=None, front_factor=2.0, center_factor=3.0
)
```

## Extending the Pipeline

### Adding New Locations

1. **Obtain DEM and Rainfall Data**
2. **Configure Return Periods** with local rainfall statistics
3. **Update Default Values** in `preprocessing_integration.py`
4. **Test and Validate** with local validation thresholds

### Custom Hyetograph Patterns

```python
# Add custom pattern to ParameterFileGenerator
def _create_custom_pattern(self, config, total_depth_mm):
    # Implement custom temporal distribution
    pass
```

### Advanced Validation

```python
# Extend ValidationThresholds for specific requirements
custom_thresholds = ValidationThresholds(
    min_flood_fraction=0.005,  # Location-specific
    max_reasonable_depth_m=30.0,  # Adjusted for terrain
    target_success_rate=0.9  # Higher standard
)
```

## Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8, use type hints
2. **Testing**: Write unit tests for new components
3. **Documentation**: Update docstrings and this documentation
4. **Validation**: Ensure all changes pass existing tests

### Testing

```bash
# Run simulation tests
python -m pytest tests/test_simulation.py -v

# Run integration tests
python -m pytest tests/test_simulation.py::TestFullPipeline -v
```

### Adding Features

1. **Create Feature Branch**: `git checkout -b feature/new-component`
2. **Implement and Test**: Add code with comprehensive tests
3. **Update Documentation**: Include usage examples
4. **Submit Pull Request**: With detailed description

## Support and Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: This guide and inline docstrings
- **LISFLOOD-FP Manual**: Detailed simulation model documentation
- **Example Scripts**: `examples/run_simulation_example.py`

## License and Citation

This simulation pipeline integrates with LISFLOOD-FP under its respective license terms. When using this pipeline in research, please cite both the FloodRisk project and LISFLOOD-FP accordingly.