# LISFLOOD-FP Simulation Module

## Overview

The LISFLOOD-FP simulation module provides complete integration with physics-based flood modeling for generating high-quality training labels for the FloodRisk ML pipeline. It automates the entire workflow from parameter generation through simulation execution to ML-ready data export.

## Architecture

```
                    FloodRisk Simulation Pipeline
    
Preprocessing ──► Parameter ──► LISFLOOD-FP ──► Result ──► Validation ──► ML Training
   Pipeline      Generation     Simulation    Processing               Data
      │              │              │            │           │            │
      ▼              ▼              ▼            ▼           ▼            ▼
DEM/Rainfall ─► .par/.rain ─► Depth/Extent ─► Binary ─► Quality ─► Labeled
   Data         Files          Maps          Maps      Control     Datasets
```

## Key Features

- **Automated Parameter Generation**: Creates LISFLOOD-FP parameter files from DEM and rainfall inputs
- **Batch Simulation Orchestration**: Parallel execution of multiple flood scenarios  
- **Multiple Return Periods**: 10, 25, 100, 500-year events with NOAA Atlas 14 rainfall
- **Diverse Hyetograph Patterns**: Uniform, front-loaded, center-loaded, back-loaded temporal distributions
- **Comprehensive Result Processing**: Converts depth maps to binary flood extents with morphological cleaning
- **Quality Control Framework**: Multi-level validation ensuring reliable training data
- **Metadata Tracking**: Complete provenance and lineage tracking for reproducibility
- **Preprocessing Integration**: Seamless connection with existing data preprocessing pipeline

## Quick Start

### Basic Single Simulation

```python
from src.simulation import LisfloodSimulator, SimulationConfig

# Initialize simulator
simulator = LisfloodSimulator()

# Configure simulation  
config = SimulationConfig(
    dem_file="LISFLOOD-FP/Nashville/final_dem.asc",
    rainfall_file="LISFLOOD-FP/Nashville/rain_input_100yr.rain",
    manning_file="LISFLOOD-FP/Nashville/manning.asc", 
    infiltration_file="LISFLOOD-FP/Nashville/infiltration.asc"
)

# Run simulation
result = simulator.run_simulation(config, "nashville_100yr")
print(f"Flooded area: {result['outputs']['statistics']['flooded_area_km2']:.2f} km²")
```

### Batch Simulation Pipeline

```python
from src.simulation import SimulationBatch, ParameterFileGenerator
from src.simulation.parameter_generator import ReturnPeriodConfig, HyetographConfig

# Set up batch processing
param_gen = ParameterFileGenerator()
batch = SimulationBatch(simulator, param_gen)

# Define scenarios (Nashville standard)
return_periods = ParameterFileGenerator.create_standard_return_periods()
patterns = ParameterFileGenerator.create_standard_hyetographs()

# Execute batch
batch_id = batch.create_batch_from_config(
    dem_file="LISFLOOD-FP/Nashville/final_dem.asc",
    return_periods=return_periods,
    hyetograph_patterns=patterns,
    output_dir="batch_results"
)

summary = batch.execute_batch()
print(f"Success rate: {summary['success_rate']:.1%}")
```

### Command Line Interface

```bash
# Batch simulations
python scripts/run_simulation_pipeline.py batch \\
    --dem-file LISFLOOD-FP/Nashville/final_dem.asc \\
    --return-periods 100,500 \\
    --patterns uniform,center_loaded \\
    --output-dir results/nashville_batch \\
    --parallel-jobs 4

# Process results
python scripts/run_simulation_pipeline.py process \\
    --input-dir results/nashville_batch \\
    --output-dir results/processed \\
    --flood-threshold 0.05

# Validate results
python scripts/run_simulation_pipeline.py validate \\
    --input-dir results/nashville_batch \\
    --output-file validation_report.json
```

### Complete Example

```bash
# Run complete end-to-end demonstration
python examples/end_to_end_pipeline.py \\
    --location nashville \\
    --output-dir pipeline_demo \\
    --mode full
```

## Module Components

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `LisfloodSimulator` | LISFLOOD-FP execution interface | `run_simulation()` |
| `ParameterFileGenerator` | Automated parameter file creation | `generate_scenario_parameters()` |
| `SimulationBatch` | Batch orchestration and parallel execution | `execute_batch()` |
| `ResultProcessor` | Post-processing to ML formats | `process_batch_outputs()` |
| `SimulationValidator` | Quality control and validation | `validate_batch_results()` |
| `SimulationMetadata` | Provenance tracking | `create_simulation_provenance()` |
| `PreprocessingIntegration` | Preprocessing pipeline connection | `run_integrated_simulation()` |

### Configuration Classes

- **SimulationConfig**: LISFLOOD-FP simulation parameters
- **ReturnPeriodConfig**: Return period and rainfall specifications  
- **HyetographConfig**: Temporal rainfall pattern definitions
- **BatchConfig**: Batch execution settings
- **ProcessingConfig**: Result processing parameters
- **ValidationThresholds**: Quality control criteria

## Default Configurations

### Nashville Return Periods (NOAA Atlas 14)

| Return Period | 24-hour Depth | Purpose |
|---------------|---------------|---------|
| 10-year | 111.76 mm (4.4") | Sub-design negative examples |
| 25-year | 142.24 mm (5.6") | Sub-design negative examples |
| 100-year | 177.8 mm (7.01") | Primary design storm |
| 500-year | 222.25 mm (8.75") | Extreme event |

### Hyetograph Patterns

- **Uniform**: Constant 24-hour intensity
- **Front-loaded**: 2.5x intensity in first 6 hours
- **Center-loaded**: 3.0x peak at hour 12
- **Back-loaded**: 2.5x intensity in last 6 hours

### Processing Parameters

- **Flood Threshold**: 0.05m depth (per APPROACH.md)
- **Morphological Cleaning**: Remove <4 pixel areas, close 1-2 pixel gaps
- **Output Formats**: NumPy arrays (.npy), GeoTIFF (optional), JSON statistics

## Output Structure

```
simulation_output/
├── batch_summary.json              # Execution summary
├── training_data_manifest.json     # ML training catalog  
├── scenarios/                      # Generated parameter files
│   ├── scenarios.json
│   ├── simulation_100yr_uniform.par
│   └── rainfall_100yr_uniform.rain
├── runs/                          # Simulation working directories
│   └── simulation_id/
│       ├── results_*/
│       │   └── res_*.max         # LISFLOOD-FP depth output
│       └── flood_extent_*.npy    # Processed binary extent
├── processed_results/             # Post-processed outputs
│   ├── processing_summary.json
│   ├── *_flood_extent.npy        # Binary flood maps
│   ├── *_depth.npy               # Depth arrays
│   └── *_statistics.json         # Flood statistics
└── metadata/                      # Provenance tracking
    ├── simulations.json          # Complete simulation database
    ├── batches.json              # Batch execution records
    └── sim_*.json                # Individual provenance files
```

## Integration Points

### With Preprocessing Pipeline

```python
from src.simulation import PreprocessingIntegration

# Initialize integration
integration = PreprocessingIntegration(
    preprocessing_output_dir="preprocessing_results",
    simulation_output_dir="simulation_results"
)

# Run integrated pipeline
results = integration.run_integrated_simulation(
    preprocessing_config_file="preprocessing_config.json"
)

# Export for ML training
training_manifest = integration.export_for_ml_training(
    results, "ml_training_data"
)
```

### With ML Training Pipeline

The simulation module produces training data in standardized formats:

```python
# Load training manifest
with open("training_data_manifest.json") as f:
    manifest = json.load(f)

# Access flood extent labels
for sample in manifest['samples']:
    extent = np.load(sample['flood_extent_file'])  # Binary flood map
    stats = sample['statistics']                   # Flood statistics
    scenario = sample['scenario']                  # Return period/pattern info
```

## Quality Control

### Validation Levels

1. **Physical Validation**: Depth ranges, mass conservation, distribution analysis
2. **Spatial Validation**: Flood fractions, area bounds, continuity checks  
3. **Temporal Validation**: Runtime bounds, convergence assessment
4. **Data Quality**: Completeness, consistency, invalid value detection

### Quality Metrics

- **Overall Score**: 0-100 comprehensive validation score
- **Status Categories**: passed, warning, failed
- **Success Rate Target**: ≥80% for production datasets
- **Issue Tracking**: Detailed error/warning logs with frequencies

### Common Quality Issues

| Issue | Typical Cause | Solution |
|-------|---------------|----------|
| Low flood fraction | Insufficient rainfall or high infiltration | Check return period data, adjust parameters |
| Excessive flooding | Poor drainage or unrealistic rainfall | Validate DEM conditioning |
| Long runtimes | Large domain or stability issues | Enable acceleration, check DEM quality |
| Convergence failures | Poor DEM or extreme parameters | Use conditioned DEM, adjust timesteps |

## Performance Optimization

### Computational Requirements

| Scale | Scenarios | Runtime | Disk Space | Memory |
|-------|-----------|---------|------------|--------|
| Single | 1 | 2-5 min | 50 MB | 512 MB |
| Standard | 16 (4×4) | 30-60 min | 800 MB | 2 GB |
| Full | 64 (4×4×4) | 2-4 hours | 3.2 GB | 4 GB |
| Production | 256+ | 8+ hours | 12+ GB | 8+ GB |

### Optimization Strategies

1. **Parallel Execution**: 4-8 concurrent jobs optimal for most systems
2. **Resource Management**: Monitor memory usage, cleanup intermediate files
3. **DEM Conditioning**: Use hydrologically conditioned DEMs for faster convergence
4. **Parameter Tuning**: Enable acceleration, optimize timesteps
5. **Storage Management**: Regular cleanup, compressed archives for long-term storage

### Scaling Guidelines

```python
# For development/testing
BatchConfig(max_parallel_jobs=2, cleanup_failed_runs=True)

# For production
BatchConfig(max_parallel_jobs=8, max_retries=3, validate_results=True)

# For high-throughput
BatchConfig(max_parallel_jobs=16, keep_intermediate_files=False)
```

## Error Handling and Troubleshooting

### Common Errors

1. **LISFLOOD-FP Not Found**: Ensure executable is compiled at `LISFLOOD-FP/build/lisflood`
2. **Simulation Failures**: Check DEM conditioning, parameter validity
3. **Memory Issues**: Reduce parallel jobs, enable cleanup
4. **Validation Failures**: Review quality thresholds, input data quality

### Debug Modes

```python
# Enable detailed logging
import logging
logging.getLogger('simulation').setLevel(logging.DEBUG)

# Preserve intermediate files
BatchConfig(keep_intermediate_files=True, cleanup_failed_runs=False)

# Extended validation
SimulationValidator(detailed_validation=True)
```

## Testing

### Unit Tests

```bash
python -m pytest tests/test_simulation.py -v
```

### Integration Tests

```bash
python -m pytest tests/test_simulation.py::TestFullPipeline -v
```

### End-to-End Testing

```bash
python examples/run_simulation_example.py --mode single --verbose
```

## Extending the Module

### Adding New Locations

1. Create location-specific return period configurations
2. Update default rainfall depths in `preprocessing_integration.py`
3. Validate with local data and adjust thresholds as needed

### Custom Hyetograph Patterns

```python
# Implement custom pattern in ParameterFileGenerator
class CustomHyetographConfig(HyetographConfig):
    def create_pattern(self, total_depth_mm):
        # Custom temporal distribution logic
        return hourly_intensities
```

### Advanced Validation

```python
# Custom validation thresholds
custom_thresholds = ValidationThresholds(
    min_flood_fraction=0.005,      # Location-specific
    max_reasonable_depth_m=30.0,   # Terrain-appropriate
    custom_checks=my_custom_checks  # Domain-specific validation
)
```

## API Reference

See [complete API documentation](../docs/simulation_pipeline.md) for detailed method signatures and parameters.

## Citation and License

When using this simulation module in research, please cite:

- FloodRisk project and this simulation integration
- LISFLOOD-FP: Bates, P.D., Horritt, M.S., Fewtrell, T.J. (2010)
- NOAA Atlas 14 for rainfall statistics (where applicable)

## Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See `docs/simulation_pipeline.md`
- **Examples**: Check `examples/` directory
- **Testing**: Use `tests/test_simulation.py` as reference