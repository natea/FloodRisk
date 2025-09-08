# FloodRisk Pipeline Integration Guide

## Overview

The FloodRisk ML pipeline provides a complete end-to-end solution for flood risk modeling, integrating data acquisition, preprocessing, hydrodynamic simulation, validation, and machine learning components into a seamless production-ready system.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Acquisition│────│  Preprocessing  │────│   Simulation    │
│                 │    │                 │    │                 │
│ • DEM Download  │    │ • Topographic   │    │ • LISFLOOD-FP   │
│ • Rainfall Data │    │   Analysis      │    │ • Batch Exec    │
│ • Validation    │    │ • Feature Eng   │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ML Training    │────│   Integration   │────│   Monitoring    │
│                 │    │                 │    │                 │
│ • Data Prep     │    │ • Pipeline      │    │ • Progress      │
│ • Model Train   │    │   Controller    │    │ • Resources     │
│ • Validation    │    │ • Error Handling│    │ • Checkpointing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Basic Nashville Demonstration

```python
from src.pipeline.integration_api import run_nashville_flood_modeling

# Run complete Nashville demonstration
results = await run_nashville_flood_modeling(
    output_dir="./nashville_results",
    parallel_simulations=4,
    enable_ml_training=True
)
```

### 2. Command Line Interface

```bash
# Run Nashville demonstration
python examples/nashville_demo.py

# Validate setup without execution
python examples/nashville_demo.py --dry-run

# Custom output directory
python examples/nashville_demo.py --output-dir ./my_results
```

### 3. Configuration-Based Execution

```python
from src.pipeline.integration_api import IntegratedFloodPipeline

# Load configuration from YAML
pipeline = IntegratedFloodPipeline("config/nashville_demo_config.yaml")

# Run with monitoring
results = await pipeline.run_pipeline()
```

## Core Components

### Pipeline Controller (`src/pipeline/main_controller.py`)

The main orchestrator that manages the complete workflow:

- **Data Acquisition**: DEM and rainfall data download
- **Preprocessing**: Topographic analysis and feature extraction
- **Simulation Setup**: Parameter generation and batch configuration
- **Simulation Execution**: Parallel LISFLOOD-FP runs
- **Validation**: Results quality control
- **ML Training**: Model training and validation

### Integration API (`src/pipeline/integration_api.py`)

High-level production API providing:

```python
class IntegratedFloodPipeline:
    def __init__(config, enable_monitoring=True, enable_checkpointing=True)
    def check_prerequisites() -> Dict[str, Any]
    def run_pipeline(resume_from_checkpoint=None, dry_run=False) -> Dict[str, Any]
    def get_status() -> Dict[str, Any]
    def list_checkpoints() -> List[Dict[str, Any]]
    def get_recovery_options() -> List[Dict[str, Any]]
```

### Progress Tracking (`src/pipeline/progress_tracker.py`)

Real-time monitoring of pipeline execution:

- Stage-by-stage progress tracking
- Performance metrics collection
- Completion time estimation
- Detailed execution timeline

### Resource Management (`src/pipeline/resource_manager.py`)

System resource monitoring and management:

- Memory usage tracking
- Disk space monitoring
- Automatic cleanup triggers
- Resource limit enforcement

### Checkpoint System (`src/pipeline/checkpoint_manager.py`)

Robust checkpoint and recovery system:

- Automatic periodic checkpoints
- Manual checkpoint creation
- Pipeline state restoration
- Recovery option analysis

## Configuration System

### Complete Configuration Structure

```yaml
# Project Configuration
project_name: "my_flood_project"
project_description: "Custom flood risk analysis"
output_root: "./outputs"
log_level: "INFO"

# Region Definition
region_name: "My Study Area"
bbox:
  west: -87.0
  south: 36.0
  east: -86.5
  north: 36.5
  crs: "EPSG:4326"

# Data Configuration
dem_source: "usgs_3dep"
dem_resolution: 10  # meters
rainfall_source: "noaa_atlas14"
return_periods: [25, 50, 100]
storm_durations: [6, 24]  # hours

# Simulation Configuration
max_sim_time: 3600.0  # seconds
parallel_simulations: 4
ml_enabled: true

# Performance Configuration  
max_memory_gb: 16.0
cleanup_intermediate: true
enable_checkpointing: true
```

### Nashville Pre-configured Settings

The Nashville demonstration uses optimized settings:

```yaml
# Nashville-specific bounding box
bbox:
  west: -87.0
  south: 36.0
  east: -86.6
  north: 36.3
  crs: "EPSG:4326"

# Optimized for Nashville topography
dem_resolution: 10  # 10-meter USGS data
return_periods: [10, 25, 50, 100, 500]
storm_durations: [6, 12, 24]
rainfall_patterns: ["scs_type_ii", "uniform", "chicago"]
```

## Execution Modes

### 1. Full Pipeline Execution

Complete end-to-end processing with all components:

```python
pipeline = IntegratedFloodPipeline(config)
results = await pipeline.run_pipeline()
```

Includes:
- Data acquisition and preprocessing
- Hydrodynamic simulation batch execution
- Results validation and quality control
- ML model training and validation
- Comprehensive reporting

### 2. Dry Run Validation

Validate configuration and prerequisites without execution:

```python
results = await pipeline.run_pipeline(dry_run=True)
```

Checks:
- System resources availability
- Data source accessibility
- Configuration validity
- Output directory permissions

### 3. Checkpoint Recovery

Resume execution from previous checkpoint:

```python
# List available checkpoints
checkpoints = pipeline.list_checkpoints()

# Resume from specific checkpoint
results = await pipeline.run_pipeline(
    resume_from_checkpoint="cp_nashville_simulation_execution_a1b2c3d4"
)
```

## Monitoring and Diagnostics

### Progress Monitoring

Real-time progress tracking with callbacks:

```python
def progress_callback(progress_data):
    stage = progress_data.get("current_stage")
    completion = progress_data.get("overall_progress") * 100
    print(f"Stage: {stage}, Progress: {completion:.1f}%")

pipeline.add_progress_callback(progress_callback)
```

### Resource Monitoring

System resource usage tracking:

```python
# Get current status
status = pipeline.get_status()
print(f"Memory: {status['resource_usage']['memory_used_gb']:.1f} GB")
print(f"Disk: {status['resource_usage']['disk_available_gb']:.1f} GB")
```

### Error Handling

Comprehensive error handling with recovery options:

```python
def error_callback(stage, error):
    print(f"Error in {stage}: {error}")
    
    # Get recovery options
    recovery_options = pipeline.get_recovery_options()
    for option in recovery_options:
        print(f"Can resume from: {option['stage']} ({option['timestamp']})")

pipeline.add_error_callback(error_callback)
```

## Output Structure

### Directory Organization

```
pipeline_output/
├── pipeline_metadata.json      # Configuration and metadata
├── pipeline.log               # Complete execution log
├── progress.json              # Progress tracking data
├── final_report.json          # Comprehensive results
├── data/                      # Downloaded and processed data
│   ├── dem/                   # DEM files
│   └── rainfall/              # Rainfall data
├── preprocessed/              # Processed topographic data
│   ├── conditioned_dem.tif
│   ├── flow_accumulation.tif
│   ├── slope.tif
│   └── hand.tif
├── simulations/               # LISFLOOD-FP outputs
│   ├── scenario_0001/
│   ├── scenario_0002/
│   └── ...
├── batch_results/             # Batch processing results
│   ├── batch_summary.json
│   └── batch_results.json
├── training_manifest.json     # ML training data index
├── ml_training/               # ML model and training outputs
│   ├── best_model.ckpt
│   ├── training.log
│   └── metrics/
└── checkpoints/               # Recovery checkpoints
    ├── checkpoint_metadata.json
    └── cp_*.pkl
```

### Key Output Files

1. **final_report.json**: Comprehensive pipeline results and metrics
2. **training_manifest.json**: Index of simulation outputs for ML training
3. **batch_summary.json**: Summary of all flood simulations
4. **best_model.ckpt**: Trained ML model checkpoint
5. **pipeline.log**: Complete execution log

## Performance Optimization

### Resource Configuration

Optimize for your system:

```yaml
# For high-memory systems
max_memory_gb: 32.0
parallel_simulations: 8
batch_size: 16

# For limited systems
max_memory_gb: 8.0
parallel_simulations: 2
batch_size: 4
cleanup_intermediate: true
```

### Processing Efficiency

- **Parallel Processing**: Utilize all available CPU cores
- **Memory Management**: Automatic cleanup of intermediate files  
- **Disk Optimization**: Compressed storage for large datasets
- **GPU Utilization**: Automatic detection for ML training

### Estimated System Requirements

| Component | Minimum | Recommended | High-Performance |
|-----------|---------|-------------|------------------|
| RAM | 8 GB | 16 GB | 32+ GB |
| Disk Space | 30 GB | 50 GB | 100+ GB |
| CPU Cores | 4 | 8 | 16+ |
| GPU Memory | N/A | 8 GB | 16+ GB |

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Errors

**Problem**: Out of memory during processing
**Solutions**:
- Reduce `parallel_simulations`
- Enable `cleanup_intermediate: true`
- Lower `max_memory_gb` setting
- Reduce DEM resolution

#### 2. Disk Space Issues

**Problem**: Insufficient disk space
**Solutions**:
- Enable automatic cleanup
- Use compressed storage
- Monitor with resource manager
- Clean old checkpoints

#### 3. Simulation Failures

**Problem**: LISFLOOD-FP simulations failing
**Solutions**:
- Check DEM quality and conditioning
- Verify rainfall input format
- Reduce simulation time step
- Check boundary conditions

#### 4. Data Acquisition Failures

**Problem**: Cannot download DEM or rainfall data
**Solutions**:
- Check internet connectivity
- Verify data source availability
- Use cached data if available
- Try alternative data sources

### Recovery Procedures

#### From Checkpoints

```python
# List available recovery options
recovery_options = pipeline.get_recovery_options()

# Create recovery plan
plan = checkpoint_manager.create_recovery_plan(checkpoint_id)

# Resume execution
results = await pipeline.run_pipeline(
    resume_from_checkpoint=checkpoint_id
)
```

#### Manual Recovery

1. **Identify last successful stage**
2. **Check checkpoint availability**
3. **Verify data integrity**
4. **Resume from appropriate point**

### Debugging Mode

Enable verbose logging for debugging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Or in configuration
log_level: "DEBUG"
```

## Customization Guide

### Adding New Regions

```python
custom_config = PipelineConfig(
    region_name="My Region",
    bbox={
        "west": -123.0,
        "south": 45.0, 
        "east": -122.0,
        "north": 46.0,
        "crs": "EPSG:4326"
    },
    dem_resolution=10,
    return_periods=[25, 100]
)

pipeline = IntegratedFloodPipeline(custom_config)
```

### Custom Processing Steps

Extend the pipeline with custom processing:

```python
class CustomPipelineController(PipelineController):
    async def _stage_custom_processing(self):
        # Custom processing logic
        return {"status": "success", "custom_result": "data"}
    
    # Add to stage list
    async def execute_pipeline(self):
        # Insert custom stage
        stages.insert(3, (CustomStage.CUSTOM_PROCESSING, self._stage_custom_processing))
        return await super().execute_pipeline()
```

### Integration with External Systems

```python
# Database integration
def save_results_to_database(results):
    # Custom database storage logic
    pass

# External API integration
def notify_external_system(status):
    # Send notifications to external systems
    pass

# Register callbacks
pipeline.add_stage_callback("completion", save_results_to_database)
pipeline.add_progress_callback(notify_external_system)
```

## Production Deployment

### Containerization

Example Docker setup:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY src/ /app/src/
COPY config/ /app/config/

WORKDIR /app
CMD ["python", "-m", "src.pipeline.integration_api"]
```

### Scalable Deployment

For large-scale processing:

1. **Kubernetes Deployment**: Scale simulation pods
2. **Distributed Storage**: Use cloud storage for data
3. **Message Queues**: Coordinate batch processing
4. **Monitoring**: Prometheus/Grafana integration

### CI/CD Integration

```yaml
# GitHub Actions example
name: FloodRisk Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
    - name: Dry run demo
      run: python examples/nashville_demo.py --dry-run
```

## API Reference

### Core Classes

#### `IntegratedFloodPipeline`

Main pipeline interface.

**Methods**:
- `__init__(config, enable_monitoring=True, enable_checkpointing=True, enable_resource_management=True)`
- `check_prerequisites() -> Dict[str, Any]`
- `run_pipeline(resume_from_checkpoint=None, dry_run=False) -> Dict[str, Any]`
- `get_status() -> Dict[str, Any]`
- `list_checkpoints() -> List[Dict[str, Any]]`
- `get_recovery_options() -> List[Dict[str, Any]]`

#### `PipelineController`

Core pipeline orchestration.

**Methods**:
- `__init__(config: PipelineConfig)`
- `initialize_components()`
- `execute_pipeline() -> Dict[str, Any]`

#### `PipelineConfig`

Configuration management.

**Methods**:
- `from_yaml(config_path: str) -> PipelineConfig`
- `to_yaml(output_path: str)`

### Utility Functions

```python
# Convenience functions
run_nashville_flood_modeling(**kwargs)
run_custom_flood_modeling(region_name, bbox, output_dir, **kwargs)
create_pipeline_from_config(config_file)
```

## Support and Development

### Getting Help

1. **Documentation**: Check this integration guide
2. **Examples**: Review `examples/nashville_demo.py`
3. **Configuration**: Use `config/nashville_demo_config.yaml` as template
4. **Logs**: Check pipeline execution logs
5. **Issues**: Report on project repository

### Contributing

1. **Fork** the repository
2. **Create** feature branch
3. **Implement** changes with tests
4. **Test** with Nashville demonstration
5. **Submit** pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd FloodRisk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run demonstration
python examples/nashville_demo.py --dry-run
```

## Conclusion

The FloodRisk pipeline provides a comprehensive, production-ready solution for flood risk modeling that integrates multiple sophisticated components into a seamless workflow. The system is designed for both research and operational use, with extensive monitoring, error handling, and recovery capabilities.

The Nashville demonstration showcases the complete capabilities of the system and serves as both a validation tool and a template for custom deployments. The modular architecture allows for easy customization and extension while maintaining robustness and reliability.

For additional support or to contribute to the project, please refer to the project repository and documentation.