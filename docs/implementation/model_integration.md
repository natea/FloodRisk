# Model Integration Strategy: Combining UNOSAT U-Net and Rapid FloodModelling CNN

## Executive Summary

This document outlines the integration of two proven flood modeling approaches:
1. **UNOSAT U-Net**: SAR-based flood segmentation with 92% Dice coefficient
2. **Rapid FloodModelling CNN**: LISFLOOD-FP surrogate with 97% accuracy

Our approach combines the spatial learning capabilities of U-Net with the physics-based training methodology of the CNN surrogate model.

## Architecture Comparison

### UNOSAT U-Net (Repository 1)
```python
Architecture: Encoder-Decoder with Skip Connections
- Input: SAR imagery (single channel)
- Encoder: ResNet-34 backbone (pretrained on ImageNet)
- Decoder: Upsampling with skip connections
- Output: Binary flood mask
- Filter depths: [12, 24, 48, 96, 192]
- Activation: Softmax for multi-class segmentation
- Loss: Weighted cross-entropy
- Performance: 0.92 Dice, 0.91-0.92 recall
```

### Rapid FloodModelling CNN (Repository 2)
```python
Architecture: Sequential CNN + Dense Layers
- Input: Time-series upstream flow (27 features)
- Conv layers: 2 layers (32, 128 filters)
- Dense layers: 3 layers (32, 256, 512 neurons)
- Output: 581,061 flood depths (full domain)
- Activation: Linear (regression)
- Loss: MSE
- Performance: <0.3m error for 97% of cells
```

## Integration Strategy for Nashville FloodRisk

### 1. Hybrid Architecture Design

We'll create a hybrid model that leverages strengths from both approaches:

```python
class NashvilleFloodNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # U-Net backbone for spatial processing (from UNOSAT)
        self.encoder = ResNetEncoder(
            backbone='resnet34',
            pretrained=True,
            input_channels=4  # DEM, rainfall, land use, soil
        )
        
        # Multi-scale feature extraction
        self.multiscale_blocks = nn.ModuleList([
            ConvBlock(64, 128, scale=1),   # 256x256
            ConvBlock(128, 256, scale=2),  # 512x512
            ConvBlock(256, 512, scale=4)   # 1024x1024
        ])
        
        # Temporal processing (inspired by Rapid CNN)
        self.temporal_encoder = nn.LSTM(
            input_size=27,  # Rainfall time series
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = AdaptiveFusion()
        
        # Decoder with physics constraints
        self.decoder = PhysicsInformedDecoder(
            feature_channels=512,
            output_channels=1,  # Flood depth
            skip_connections=True
        )
```

### 2. Data Processing Pipeline

Combining approaches from both repositories:

```python
class HybridDataProcessor:
    def __init__(self):
        # From UNOSAT: Tile-based processing
        self.tile_size = 256
        self.overlap = 32
        
        # From Rapid CNN: LISFLOOD-FP integration
        self.lisflood_parser = LISFLOODParser()
        self.threshold = 0.3  # Minimum flood depth
        
    def process_training_data(self):
        # 1. Generate LISFLOOD-FP simulations (Repo 2 approach)
        simulations = self.run_lisflood_scenarios()
        
        # 2. Create tiles (Repo 1 approach)
        tiles = self.create_tiles(simulations)
        
        # 3. Apply augmentation (Repo 1)
        augmented = self.augment_tiles(tiles)
        
        # 4. Add temporal features (Repo 2)
        with_temporal = self.add_rainfall_series(augmented)
        
        return with_temporal
```

### 3. Training Strategy

```python
class HybridTrainer:
    def __init__(self, model):
        self.model = model
        
        # Combined loss function
        self.spatial_loss = WeightedBCE()  # From UNOSAT
        self.depth_loss = nn.MSELoss()     # From Rapid CNN
        self.physics_loss = MassConservationLoss()
        
        # Optimizer (both repos used Adam)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Learning rate schedule (from UNOSAT)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-2,
            epochs=100
        )
    
    def combined_loss(self, pred, target):
        # Multi-task loss
        extent_loss = self.spatial_loss(pred > 0, target > 0)
        depth_loss = self.depth_loss(pred, target)
        physics_loss = self.physics_loss(pred)
        
        return extent_loss + depth_loss + 0.1 * physics_loss
```

## Implementation Roadmap

### Phase 1: Model Architecture (Days 1-2)
- [ ] Implement U-Net encoder with ResNet-34 backbone
- [ ] Adapt input layer for 4 channels (DEM, rainfall, land use, soil)
- [ ] Create multi-scale processing modules
- [ ] Build physics-informed decoder

### Phase 2: Data Pipeline (Days 3-4)
- [ ] Set up LISFLOOD-FP simulation wrapper
- [ ] Implement tile extraction with overlap
- [ ] Create augmentation pipeline
- [ ] Add temporal rainfall encoding

### Phase 3: Training (Days 5-6)
- [ ] Configure hybrid loss function
- [ ] Implement training loop with validation
- [ ] Set up checkpointing and monitoring
- [ ] Run hyperparameter optimization

### Phase 4: Validation (Day 7)
- [ ] Compare with LISFLOOD-FP baseline
- [ ] Validate against NFIP claims
- [ ] Test on 2010 Nashville flood event
- [ ] Generate performance metrics

## Key Innovations

### From UNOSAT Repository:
1. **Pretrained Backbone**: ResNet-34 encoder initialization
2. **Tile-based Training**: Efficient memory usage with 256x256 patches
3. **Data Augmentation**: Rotation, flipping, scaling for robustness
4. **Weighted Loss**: Handle class imbalance (flood vs non-flood)

### From Rapid CNN Repository:
1. **LISFLOOD-FP Integration**: Physics-based training data
2. **Temporal Features**: Antecedent rainfall conditions (t-1 to t-8)
3. **Domain-specific Output**: Full spatial coverage in single pass
4. **Threshold Processing**: 0.3m minimum depth for flood classification

### Our Additions:
1. **Multi-scale Context**: 3 resolution levels for capturing drainage
2. **Physics Constraints**: Mass conservation in loss function
3. **NFIP Validation**: Real-world claims for ground truth
4. **Nashville-specific**: NOAA Atlas 14 rainfall scenarios

## Performance Targets

Based on the reference models:
- **Spatial Accuracy**: >0.90 Dice coefficient (UNOSAT achieved 0.92)
- **Depth Accuracy**: <0.5m RMSE (Rapid CNN achieved <0.3m for 97%)
- **Processing Speed**: <10 seconds for 10km² (100x faster than LISFLOOD-FP)
- **NFIP Correlation**: >0.7 R² with insurance claims

## Code Reuse Strategy

### Direct Reuse:
```python
# From UNOSAT: U-Net architecture
from reference_models.UNOSAT.naive_segmentation.UNet import model as unet_base

# Adapt for our inputs
class FloodUNet(nn.Module):
    def __init__(self):
        self.base = unet_base(
            input_shape=(256, 256, 4),  # Our channels
            classes=2,  # Flood/no-flood
            filter_depth=(64, 128, 256, 512, 1024)
        )
```

### Modified Reuse:
```python
# From Rapid CNN: Data preprocessing
from reference_models.RapidFloodCNN.InunMod_v1 import data_pre_process

# Adapt for Nashville data
class NashvillePreprocessor:
    def __init__(self):
        self.base_processor = data_pre_process
        self.nashville_dem = self.load_nashville_dem()
        
    def process(self):
        # Use base processing logic
        base_data = self.base_processor()
        # Add Nashville-specific features
        return self.add_nashville_features(base_data)
```

## Testing Strategy

### Unit Tests:
- Model architecture components
- Data preprocessing pipeline
- Loss function calculations
- Physics constraint validation

### Integration Tests:
- End-to-end training pipeline
- LISFLOOD-FP comparison
- API endpoint validation
- Performance benchmarks

### System Tests:
- Full Nashville prediction
- Multi-scenario validation
- Production deployment
- Load testing

## Deployment Configuration

```yaml
# docker-compose.yml additions
services:
  flood-model:
    build:
      context: .
      dockerfile: Dockerfile.model
    environment:
      - MODEL_WEIGHTS=/models/nashville_flood_v1.pth
      - BATCH_SIZE=8
      - DEVICE=cuda
    volumes:
      - ./models:/models
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Conclusion

By combining the proven U-Net architecture from UNOSAT with the physics-based training approach from Rapid FloodModelling CNN, we create a superior model that:
1. Learns spatial flood patterns effectively (U-Net strength)
2. Respects physical constraints (LISFLOOD-FP training)
3. Handles temporal dynamics (rainfall time series)
4. Validates against real-world data (NFIP claims)

This hybrid approach leverages the best of both repositories while adding Nashville-specific enhancements for production deployment.