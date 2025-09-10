# 🚀 FloodRisk Production ML Setup Guide

## Quick Start: From Zero to Flood Predictions

### Prerequisites

1. **System Requirements**
   - Python 3.11+
   - 8GB+ RAM (16GB recommended)
   - 10GB+ free disk space
   - GDAL system libraries

2. **Install System Dependencies**

**macOS:**
```bash
brew install gdal
brew install proj
brew install geos
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
sudo apt-get install libproj-dev libgeos-dev
```

### Step 1: Switch to ML Training Branch

```bash
# Switch to the ml-training branch
git checkout ml-training

# Pull latest changes
git pull origin ml-training
```

### Step 2: Create ML Environment

```bash
# Create a new conda environment for ML
conda create -n floodrisk-ml python=3.11
conda activate floodrisk-ml

# Install GDAL through conda (more reliable than pip)
conda install -c conda-forge gdal rasterio

# Install ML dependencies
pip install -r requirements-ml.txt
```

### Step 3: Download Nashville Demo Data

```bash
# Create data directory
mkdir -p data/nashville

# Run the data acquisition script
python scripts/data_acquisition/download_nashville_data.py \
    --output-dir data/nashville \
    --verbose

# This will download:
# - USGS 3DEP DEM (10m resolution)
# - NOAA Atlas 14 rainfall data
# - Takes ~5-10 minutes depending on connection
```

### Step 4: Run LISFLOOD-FP Simulations (Optional)

If you have LISFLOOD-FP installed:

```bash
# Generate flood simulation labels
python scripts/run_simulation_pipeline.py batch \
    --dem-file data/nashville/dem_10m.tif \
    --output-dir data/nashville/simulations \
    --return-periods 100,500 \
    --patterns uniform \
    --parallel-jobs 4
```

**Note:** If you don't have LISFLOOD-FP, you can use synthetic labels for testing:

```bash
# Generate synthetic flood labels for testing
python scripts/generate_synthetic_labels.py \
    --dem-file data/nashville/dem_10m.tif \
    --output-dir data/nashville/labels
```

### Step 5: Preprocess Data for ML

```bash
# Run preprocessing pipeline
python examples/preprocessing_demo.py \
    --dem-path data/nashville/dem_10m.tif \
    --rainfall-config data/nashville/rainfall_config.json \
    --output-dir data/nashville/processed \
    --generate-tiles \
    --tile-size 512 \
    --overlap 64
```

### Step 6: Train the ML Model

```bash
# Phase 1: Baseline training with real data
python scripts/train_model.py \
    --phase baseline \
    --data-dir data/nashville/processed \
    --output-dir outputs/baseline \
    --wandb  # Optional: enable experiment tracking

# Monitor training progress
tensorboard --logdir outputs/baseline/tensorboard_logs
```

### Step 7: Run End-to-End Nashville Demo

```bash
# Complete integrated pipeline demo
python examples/nashville_demo.py \
    --config config/nashville_demo_config.yaml \
    --output-dir results/nashville_demo \
    --dry-run  # First run with dry-run to validate

# If dry-run succeeds, run the full pipeline
python examples/nashville_demo.py \
    --config config/nashville_demo_config.yaml \
    --output-dir results/nashville_demo
```

## 📁 Expected Directory Structure

After setup, you should have:

```
FloodRisk/
├── data/
│   └── nashville/
│       ├── dem_10m.tif              # USGS 3DEP DEM
│       ├── rainfall_config.json     # NOAA Atlas 14 data
│       ├── processed/               # Preprocessed tiles
│       │   ├── tiles/              # 512x512 training tiles
│       │   └── features/           # Derived features
│       └── labels/                 # Flood extent labels
├── outputs/
│   └── baseline/
│       ├── checkpoints/            # Model checkpoints
│       └── tensorboard_logs/       # Training logs
└── results/
    └── nashville_demo/
        ├── flood_prob.tif          # Probability predictions
        ├── flood_extent.tif        # Binary flood extent
        └── flood_extent.gpkg       # Vector polygons
```

## 🔧 Troubleshooting Common Issues

### Issue 1: GDAL Import Error
```bash
# Solution: Set GDAL environment variables
export GDAL_DATA=$(gdal-config --datadir)
export PROJ_LIB=$(find $(conda info --base) -name proj.db -exec dirname {} \; | head -n 1)
```

### Issue 2: CUDA/GPU Not Available
```bash
# Solution: Train on CPU (slower but works)
python scripts/train_model.py --phase baseline --device cpu
```

### Issue 3: Out of Memory
```bash
# Solution: Reduce batch size
python scripts/train_model.py --phase baseline --batch-size 4
```

### Issue 4: Missing LISFLOOD-FP
```bash
# Solution: Use synthetic labels for testing
python scripts/generate_synthetic_labels.py --help
```

## 🎯 Quick Test: Verify Installation

```bash
# Test data acquisition
python -c "from src.data.sources.usgs_3dep import USGS3DEPDownloader; print('✅ Data acquisition ready')"

# Test preprocessing
python -c "from src.ml.data.real_data_preprocessing import RealDataPreprocessor; print('✅ Preprocessing ready')"

# Test ML models
python -c "from src.ml.models.flood_unet import FloodUNet; print('✅ ML models ready')"

# Test validation
python -c "from src.validation.pipeline_validator import PipelineValidator; print('✅ Validation ready')"
```

## 📊 Performance Expectations

| Stage | Time | Resources |
|-------|------|-----------|
| Data Download | 5-10 min | 2GB bandwidth |
| Preprocessing | 10-15 min | 4GB RAM |
| Simulation (optional) | 30-60 min | 2GB RAM |
| ML Training (100 epochs) | 2-4 hours | GPU: 30min, CPU: 4hrs |
| Inference | 2-5 min | 2GB RAM |

## 🚀 Next Steps

1. **Extend to New Regions**: Modify `config/nashville_demo_config.yaml` for other cities
2. **Improve Model**: Move to Phase 2 (multiscale) with `--phase multiscale`
3. **Production API**: Deploy trained model with FastAPI
4. **Cross-Validation**: Test generalization with multiple cities

## 📚 Additional Resources

- [ML Implementation Guide](docs/ML_IMPLEMENTATION.md)
- [Data Acquisition Docs](docs/data_acquisition.md)
- [Preprocessing Guide](docs/preprocessing_integration.md)
- [APPROACH.md](docs/APPROACH.md) - Original research specification

## 💡 Tips for Success

1. **Start Small**: Use `--debug` flag to test with dummy data first
2. **Monitor Resources**: Use `htop` or Activity Monitor to watch memory usage
3. **Use Checkpoints**: Training automatically saves checkpoints for resuming
4. **Validate Often**: Run validation scripts after each major step
5. **Check Logs**: Detailed logs are in `outputs/baseline/*.log`

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review log files in `outputs/baseline/`
3. Try with `--debug` flag for verbose output
4. Open an issue on GitHub with error messages

---

**Ready to start?** Follow the steps above to get your production ML flood prediction system running!