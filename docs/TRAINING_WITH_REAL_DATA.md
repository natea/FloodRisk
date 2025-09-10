# Training with Real Data - Next Steps

## Current Status ✅
- Simple model trained successfully with dummy data
- Model saved to `outputs/baseline/simple_model.pth` (97MB)
- GDAL and geospatial dependencies installed
- Nashville synthetic precipitation data available

## Next Steps for Real Data Training

### Step 1: Prepare Real Training Data

Since we have synthetic precipitation data for Nashville, we need to:

```bash
# 1. Generate training tiles from your data
python scripts/create_training_tiles.py \
    --dem-path data/nashville/test_dem.npy \
    --precipitation-dir data/v2_additional/precipitation_grids/processed \
    --output-dir data/training_tiles \
    --tile-size 512 \
    --overlap 64
```

### Step 2: Run LISFLOOD-FP Simulations (Optional but Recommended)

Generate physics-based flood labels:

```bash
# Run batch simulations for different scenarios
python scripts/run_simulation_pipeline.py batch \
    --dem-file data/nashville/test_dem.npy \
    --return-periods 100,500 \
    --patterns uniform,center_loaded \
    --output-dir results/simulations \
    --parallel-jobs 4
```

### Step 3: Configure Training for Real Data

Create a configuration file `configs/nashville_real.yaml`:

```yaml
model:
  type: "unet"
  in_channels: 6  # DEM, Rainfall, Slope, HAND, FlowAccum, Curvature
  encoder_name: "resnet34"
  use_fpn: true
  pretrained: true

data:
  train_dir: "data/training_tiles/train"
  val_dir: "data/training_tiles/val"
  tile_size: 512
  overlap: 64
  augmentation: true
  normalize: true

training:
  optimizer: "adamw"
  encoder_lr: 1e-4
  decoder_lr: 1e-3
  freeze_encoder_epochs: 10
  total_epochs: 60
  batch_size: 8
  precision: 16  # Mixed precision

loss:
  type: "combined"
  bce_weight: 0.4
  dice_weight: 0.4
  physics_weight: 0.2
  flood_class_weight: 3.0

validation:
  iou_threshold: 0.70
  save_best_only: true
  patience: 10
```

### Step 4: Train with Real Data

Run the full training pipeline:

```bash
# Using the main training script
python scripts/train_model.py \
    --config configs/nashville_real.yaml \
    --phase baseline \
    --wandb  # Optional: enable Weights & Biases tracking
```

Or use the modular approach:

```python
from src.ml.training.train import FloodModelTrainer
from src.ml.data.dataset import FloodDataset
from src.ml.models.flood_unet import create_model

# Load data
train_dataset = FloodDataset(
    data_dir="data/training_tiles/train",
    tile_size=512,
    transform=True
)

val_dataset = FloodDataset(
    data_dir="data/training_tiles/val",
    tile_size=512,
    transform=False
)

# Create model
model = create_model(
    model_type="unet",
    in_channels=6,
    encoder_name="resnet34",
    use_fpn=True
)

# Initialize trainer
trainer = FloodModelTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config_path="configs/nashville_real.yaml"
)

# Train
trainer.fit()
```

### Step 5: Validate Against Real Events

Compare predictions with historical data:

```bash
# Validate against 2010 Nashville flood
python scripts/validate_historical.py \
    --model-path outputs/nashville_real/best_model.ckpt \
    --event "2010_nashville_flood" \
    --rainfall-data data/historical/2010_nashville_rainfall.csv \
    --claims-data data/nfip/nashville_claims.csv
```

### Step 6: Fine-tune with Physics Constraints

Add physics-informed training:

```bash
python scripts/train_model.py \
    --config configs/nashville_real.yaml \
    --phase physics_informed \
    --pretrained outputs/nashville_real/best_model.ckpt \
    --physics-loss-weight 0.3
```

## Data Requirements Checklist

### ✅ Available:
- [x] Synthetic precipitation grids (18 scenarios)
- [x] Test DEM data
- [x] Model architecture (UNet-ResNet34)
- [x] Training infrastructure

### ⚠️ Still Needed:
- [ ] Real Nashville DEM at 10m resolution
- [ ] LISFLOOD-FP simulations for labels
- [ ] NFIP claims data for validation
- [ ] Historical flood extent maps

## Quick Commands Reference

```bash
# Check data status
python scripts/check_data_status.py

# Create training manifest
python scripts/create_training_manifest.py \
    --simulations-dir results/simulations \
    --output data/training_manifest.json

# Monitor training
tensorboard --logdir outputs/tensorboard_logs

# Export model for production
python scripts/export_model.py \
    --checkpoint outputs/nashville_real/best_model.ckpt \
    --output models/flood_model_v1.onnx
```

## Performance Targets

Based on your documentation, aim for:
- **IoU**: ≥ 0.70 (baseline), ≥ 0.75 (enhanced)
- **F1 Score**: ≥ 0.75
- **RMSE**: < 0.5m for flood depths
- **Inference Speed**: < 10 seconds for 100 km²

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
--accumulate-grad-batches 2

# Enable CPU offloading
--cpu-offload true
```

### Convergence Issues
```bash
# Use different optimizer
--optimizer sgd --lr 0.01

# Adjust learning rate schedule
--scheduler cosine --warmup-epochs 5

# Change loss weights
--bce-weight 0.5 --dice-weight 0.5
```

### Data Issues
```bash
# Validate data integrity
python scripts/validate_data.py --data-dir data/training_tiles

# Check class balance
python scripts/analyze_dataset.py --data-dir data/training_tiles

# Rebalance dataset
python scripts/rebalance_tiles.py --ratio 0.7
```

## Next Immediate Action

Since you have the infrastructure ready, the most important next step is:

**Get real Nashville DEM data:**
```bash
python scripts/data_acquisition/download_nashville_data.py \
    --output-dir data/nashville_real \
    --dem-resolution 10 \
    --force  # Force download even if cached
```

Then process it for training:
```bash
python scripts/preprocess_dem.py \
    --input data/nashville_real/dem.tif \
    --output data/preprocessed \
    --compute-derivatives  # Slope, HAND, flow accumulation
```

Ready to train with real data! 🚀