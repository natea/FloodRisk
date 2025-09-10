# FloodRisk ML Implementation Guide

## Overview

This document describes the implementation of the flood prediction ML pipeline based on the specifications in `docs/APPROACH.md`. The implementation follows the phased approach outlined in the research document.

## Architecture

### Core Components

1. **Data Pipeline** (`src/ml/data/`)
   - `preprocessing.py`: DEM and rainfall data processing
   - Tile generation with 512×512 tiles and 64px overlap
   - Balanced sampling (70% flooded tiles, 30% random)
   - Per-tile normalization to prevent location leakage

2. **Model Architecture** (`src/ml/models/`)
   - `flood_unet.py`: UNet-ResNet34 implementation
   - Pretrained ResNet-34 encoder adapted for 2-6 input channels
   - Optional Feature Pyramid Network (FPN) for multi-scale context
   - MC Dropout for uncertainty estimation

3. **Training Pipeline** (`src/ml/training/`)
   - `train.py`: PyTorch Lightning training module
   - `losses.py`: BCE+Dice, Focal, Tversky loss implementations
   - `metrics.py`: IoU, F1, AUCPR, Brier score evaluation
   - `augmentations.py`: Hydrology-safe augmentations

4. **Inference Pipeline** (`src/ml/inference/`)
   - `geotiff_pipeline.py`: Large raster prediction with sliding windows
   - GeoTIFF outputs: probability, binary extent, vector polygons
   - Morphological cleaning and uncertainty estimation

## Implementation Phases

### Phase 1: Baseline (Current)
- **Goal**: IoU ≥ 0.70 on held-out neighborhoods
- **Inputs**: [DEM, Rainfall] (2 channels)
- **Architecture**: UNet-ResNet34 
- **Loss**: BCE + Dice (0.5 + 0.5)
- **Duration**: 3-5 days

```bash
# Run Phase 1 training
python scripts/train_model.py --phase baseline --debug
```

### Phase 2: Multi-scale & Features
- **Additions**: FPN head, derived features (slope, HAND, flow accumulation)
- **Inputs**: [DEM, Rainfall, Slope, HAND, FlowAccum, Curvature] (6 channels)
- **Focus**: Improve transferability across cities

```bash
# Run Phase 2 training  
python scripts/train_model.py --phase multiscale --debug
```

### Phase 3: Generalization
- **Focus**: Cross-city validation, few-shot adaptation
- **Validation**: Leave-one-city-out (LOCO) protocol
- **Adaptation**: Few-shot fine-tuning for new cities

```bash
# Run Phase 3 training
python scripts/train_model.py --phase generalization --data-dir /path/to/multi-city-data
```

## Configuration System

The system uses Hydra for configuration management with YAML files:

```yaml
# configs/model_config.yaml
model:
  type: "unet"
  in_channels: 2  # [DEM, Rain] baseline
  encoder_name: "resnet34"
  use_fpn: false

training:
  optimizer: "adamw"
  encoder_lr: 1e-4    # Lower LR for pretrained encoder
  decoder_lr: 1e-3    # Higher LR for new layers
  freeze_encoder_epochs: 10
  total_epochs: 60
  batch_size: 8
  precision: 16       # Mixed precision training

loss:
  type: "bce_dice"
  bce_weight: 0.5
  dice_weight: 0.5
  flood_weight: 3.0   # 3x weight for flood class
```

## Data Requirements

### Input Data Format
- **DEM**: USGS 3DEP (~10m resolution), reprojected to metric CRS
- **Rainfall**: NOAA Atlas 14, 24-hour totals for return periods
- **Labels**: Physics-based simulation results (≥0.05m flood threshold)

### Expected Directory Structure
```
data/
├── nashville/
│   ├── dem.tif           # Digital elevation model
│   ├── rainfall_100yr.tif # 100-year return period
│   ├── rainfall_500yr.tif # 500-year return period  
│   └── flood_labels/
│       ├── flood_100yr.tif
│       └── flood_500yr.tif
└── city2/
    └── ... (same structure)
```

### Preprocessing Steps
1. Reproject DEM to metric CRS (EPSG:3857) at 10m resolution
2. Compute derived features: slope, curvature, flow accumulation, HAND
3. Create uniform rainfall rasters with optional spatial variability
4. Generate overlapping tiles with balanced sampling
5. Apply per-tile normalization

## Training Features

### Hydrology-Safe Augmentations
- 90° rotations and flips (preserve flow patterns)
- Small translations (≤5% of tile size)
- Rainfall domain randomization (±15% scaling)
- Spatial rainfall gradients (≤10% variation)
- **Avoids**: Elastic warps that distort flow pathways

### Multi-Return Period Training
- Joint training on 100-yr and 500-yr events (80%)
- Sub-design negatives: 10-yr and 25-yr events (20%)
- Balanced sampling to maintain flood/dry ratios

### Advanced Features
- **Freeze→Unfreeze**: Train decoder 10 epochs, then unfreeze encoder
- **Gradient Clipping**: Norm clipping at 5.0
- **Mixed Precision**: FP16 training for memory efficiency  
- **MC Dropout**: Uncertainty estimation during inference

## Evaluation Metrics

### Primary Metrics (APPROACH.md)
- **IoU (Jaccard)**: Primary metric for flood extent overlap
- **F1 Score**: Harmonic mean of precision and recall
- **AUCPR**: Area under precision-recall curve (robust with imbalance)
- **Brier Score**: Probability calibration assessment

### Threshold Optimization
- Youden's J statistic for optimal threshold selection
- Precision-recall curve analysis
- Reliability curves for calibration assessment

## Inference Pipeline

### Large Raster Prediction
```python
from src.ml.inference.geotiff_pipeline import GeoTIFFInferencePipeline

# Initialize pipeline
pipeline = GeoTIFFInferencePipeline(model, device='cuda')

# Run inference
outputs = pipeline.predict_large_raster(
    input_data=dataset,
    output_dir="predictions/",
    threshold=0.5,
    apply_morphology=True,
    mc_samples=10  # For uncertainty
)
```

### Output Formats
- `flood_prob.tif`: Float32 probability map (0-1)
- `flood_extent.tif`: Binary extent (uint8, morphology-cleaned)
- `flood_extent.gpkg`: Vector polygons (dissolved)
- `flood_uncertainty.tif`: MC Dropout uncertainty (optional)
- `prediction_metadata.json`: Prediction metadata

## Installation & Setup

### Prerequisites
```bash
# Create conda environment
conda create -n floodrisk-ml python=3.11
conda activate floodrisk-ml

# Install GDAL (system dependencies)
conda install -c conda-forge gdal

# Install ML dependencies
pip install -r requirements-ml.txt
```

### Key Dependencies
- `torch>=2.0.0`: PyTorch deep learning framework
- `pytorch-lightning>=2.0.0`: Training framework
- `segmentation-models-pytorch>=0.3.3`: UNet-ResNet models
- `rasterio>=1.3.0`: Geospatial raster I/O
- `xarray>=2023.6.0`: N-dimensional data arrays
- `hydra-core>=1.3.0`: Configuration management
- `wandb>=0.15.0`: Experiment tracking (optional)

### Quick Start
```bash
# Phase 1: Baseline training with dummy data
python scripts/train_model.py --phase baseline --debug --wandb

# Monitor training
tensorboard --logdir outputs/tensorboard_logs
```

## Development Workflow

### Adding New Features
1. Update configuration in `configs/model_config.yaml`
2. Implement feature in appropriate module
3. Add tests in `tests/ml/`
4. Update documentation

### Custom Loss Functions
```python
# In src/ml/training/losses.py
class CustomLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Implementation
    
    def forward(self, logits, targets):
        return loss_value

# Register in create_loss_function()
```

### Model Variants
```python
# In src/ml/models/flood_unet.py
class CustomFloodModel(nn.Module):
    def __init__(self, ...):
        # Custom architecture
        
# Register in create_model()
```

## Performance Optimization

### Memory Optimization
- Mixed precision training (FP16)
- Gradient accumulation for large effective batch sizes
- Tile-based inference for large rasters
- Efficient data loading with persistent workers

### Speed Optimization
- Pre-computed derived features
- Cached tile generation
- Multi-GPU training support
- Optimized data augmentation pipeline

## Monitoring & Logging

### Experiment Tracking
- Weights & Biases integration
- MLflow support
- TensorBoard logging
- Custom metric logging

### Model Checkpointing
- Save top-k models by validation IoU
- Automatic resuming from checkpoints
- Model versioning and metadata

## Troubleshooting

### Common Issues

**GDAL Installation Problems**
```bash
# Use conda for GDAL
conda install -c conda-forge gdal
export GDAL_DATA=$(gdal-config --datadir)
```

**CUDA Memory Issues**
```bash
# Reduce batch size or use gradient accumulation
python scripts/train_model.py --config configs/small_batch.yaml
```

**Data Loading Errors**
```bash
# Enable debug mode to use dummy data
python scripts/train_model.py --debug
```

## Next Steps

1. **Phase 1 Implementation**: Complete baseline UNet-ResNet34
2. **Data Integration**: Connect to real DEM/simulation data
3. **Multi-City Expansion**: Add datasets for generalization testing
4. **Production Deployment**: Package for inference at scale
5. **QGIS Integration**: Create styling and visualization tools

## References

- Original approach: `docs/APPROACH.md`
- UNOSAT methodology: Rapid flood mapping with U-Net
- SRKabir surrogate modeling: Physics-informed ML for hydraulics
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch