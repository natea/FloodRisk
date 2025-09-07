# Code Adaptation Guide: UNOSAT and Rapid CNN Integration

## Overview

This guide provides specific code adaptations from both reference repositories for our Nashville flood risk model.

## 1. UNOSAT U-Net Adaptation

### Original UNOSAT Architecture (Keras)
```python
# From reference_models/UNOSAT/naive_segmentation/UNet.py
def model(input_shape=(64,64,3), classes=3, kernel_size=3, 
          filter_depth=(12,24,48,96,192)):
    img_input = Input(shape=input_shape)
    # Encoder with MaxPooling
    # Decoder with UpSampling and Concatenate
    return Model(img_input, output)
```

### Our PyTorch Adaptation
```python
# src/models/adapted_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptedUNet(nn.Module):
    """U-Net adapted from UNOSAT for DEM+Rainfall input"""
    
    def __init__(self, input_channels=4, output_channels=1):
        super().__init__()
        
        # Adapted filter depths from UNOSAT
        self.filter_depth = [64, 128, 256, 512, 1024]  # Increased for complexity
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(input_channels, self.filter_depth[0])
        self.enc2 = self.conv_block(self.filter_depth[0], self.filter_depth[1])
        self.enc3 = self.conv_block(self.filter_depth[1], self.filter_depth[2])
        self.enc4 = self.conv_block(self.filter_depth[2], self.filter_depth[3])
        
        # Bridge
        self.bridge = self.conv_block(self.filter_depth[3], self.filter_depth[4])
        
        # Decoder (upsampling) with skip connections
        self.dec4 = self.upconv_block(self.filter_depth[4], self.filter_depth[3])
        self.dec3 = self.upconv_block(self.filter_depth[3]*2, self.filter_depth[2])
        self.dec2 = self.upconv_block(self.filter_depth[2]*2, self.filter_depth[1])
        self.dec1 = self.upconv_block(self.filter_depth[1]*2, self.filter_depth[0])
        
        # Output layer
        self.final = nn.Conv2d(self.filter_depth[0], output_channels, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bridge
        b = self.bridge(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output
        out = self.final(d1)
        return out
```

## 2. Rapid CNN Data Processing Adaptation

### Original Rapid CNN Preprocessing
```python
# From reference_models/RapidFloodCNN/InunMod_v1.py
def data_pre_process():
    # Process LISFLOOD-FP outputs
    Y[Y<0.3] = 0  # Threshold at 0.3m
    # Create lagged features for temporal dynamics
    df['Upstream1-1'] = df['Upstream1'].shift(1)
    # ... up to shift(8)
```

### Our Adapted Preprocessing
```python
# src/preprocessing/lisflood_processor.py
import numpy as np
import rasterio as rio
import pandas as pd
from pathlib import Path

class LISFLOODProcessor:
    """Process LISFLOOD-FP outputs for training"""
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold  # From Rapid CNN
        self.lag_steps = 8  # Temporal features from Rapid CNN
        
    def process_flood_depths(self, lisflood_dir):
        """Process LISFLOOD-FP .wd files"""
        wd_files = sorted(Path(lisflood_dir).glob('*.wd'))
        
        # Skip initialization files (from Rapid CNN)
        wd_files = wd_files[8:]  # Skip first 2 hours (8 * 15min)
        
        flood_maps = []
        for wd_file in wd_files:
            with rio.open(wd_file) as src:
                depth = src.read(1)
                # Apply threshold from Rapid CNN
                depth[depth < self.threshold] = 0
                flood_maps.append(depth)
                
        return np.stack(flood_maps)
    
    def create_temporal_features(self, rainfall_df):
        """Create lagged rainfall features"""
        features = rainfall_df.copy()
        
        # Add lagged features (from Rapid CNN approach)
        for location in ['nashville_airport', 'cumberland_river', 'mill_creek']:
            for lag in range(1, self.lag_steps + 1):
                features[f'{location}_t-{lag}'] = features[location].shift(lag)
        
        # Add cumulative features
        for window in [3, 6, 12, 24]:  # hours
            features[f'cumsum_{window}h'] = (
                features[['nashville_airport', 'cumberland_river', 'mill_creek']]
                .rolling(window=window*4)  # 15-min intervals
                .sum()
                .mean(axis=1)
            )
        
        return features.dropna()
```

## 3. Training Pipeline Integration

### Combined Training Approach
```python
# src/training/hybrid_trainer.py
import torch
from torch.utils.data import DataLoader
import mlflow

class HybridTrainer:
    """Combines UNOSAT training strategy with Rapid CNN data generation"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Loss functions combining both approaches
        self.extent_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([10.0])  # UNOSAT class weighting
        )
        self.depth_loss = nn.MSELoss()  # Rapid CNN regression loss
        
        # Optimizer (both repos used Adam)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # UNOSAT's one-cycle policy
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Inputs: DEM + Rainfall (spatial + temporal)
            dem = batch['dem']
            rainfall = batch['rainfall']
            temporal = batch['temporal_features']
            target = batch['flood_depth']
            
            # Combine inputs
            spatial_input = torch.cat([dem, rainfall], dim=1)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_depth = self.model(spatial_input, temporal)
            
            # Combined loss (UNOSAT + Rapid CNN)
            extent_pred = (pred_depth > 0).float()
            extent_target = (target > 0).float()
            
            loss_extent = self.extent_loss(extent_pred, extent_target)
            loss_depth = self.depth_loss(pred_depth[target > 0], 
                                        target[target > 0])
            
            # Physics-informed regularization
            loss_physics = self.compute_physics_loss(pred_depth, dem)
            
            total = loss_extent + loss_depth + 0.1 * loss_physics
            
            # Backward pass
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += total.item()
            
            # Log to MLflow (modern addition)
            mlflow.log_metrics({
                'loss_extent': loss_extent.item(),
                'loss_depth': loss_depth.item(),
                'loss_physics': loss_physics.item()
            })
            
        return total_loss / len(train_loader)
    
    def compute_physics_loss(self, pred_depth, dem):
        """Physics constraints not in original repos"""
        # Water should flow downhill
        grad_x = torch.gradient(pred_depth, dim=-1)[0]
        grad_y = torch.gradient(pred_depth, dim=-2)[0]
        
        dem_grad_x = torch.gradient(dem, dim=-1)[0]
        dem_grad_y = torch.gradient(dem, dim=-2)[0]
        
        # Penalize uphill flow
        uphill_flow = torch.relu(grad_x * dem_grad_x + grad_y * dem_grad_y)
        
        return uphill_flow.mean()
```

## 4. Validation Metrics Integration

### Combined Validation Approach
```python
# src/validation/hybrid_metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix

class HybridMetrics:
    """Combines metrics from both repositories"""
    
    def __init__(self):
        # UNOSAT metrics
        self.dice_scores = []
        self.iou_scores = []
        
        # Rapid CNN metrics
        self.depth_errors = []
        self.cell_accuracies = []
        
    def compute_unosat_metrics(self, pred, target):
        """Spatial metrics from UNOSAT"""
        pred_binary = pred > 0
        target_binary = target > 0
        
        intersection = (pred_binary & target_binary).sum()
        union = (pred_binary | target_binary).sum()
        
        # Dice coefficient (UNOSAT primary metric)
        dice = 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)
        
        # IoU
        iou = intersection / (union + 1e-8)
        
        # Precision/Recall
        tn, fp, fn, tp = confusion_matrix(
            target_binary.flatten(), 
            pred_binary.flatten()
        ).ravel()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }
    
    def compute_rapid_cnn_metrics(self, pred, target):
        """Depth accuracy from Rapid CNN"""
        # Only evaluate where flooding occurs
        mask = target > 0
        
        if mask.sum() == 0:
            return {'rmse': 0, 'mae': 0, 'accuracy_03m': 1.0}
        
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # RMSE and MAE
        rmse = np.sqrt(((pred_masked - target_masked) ** 2).mean())
        mae = np.abs(pred_masked - target_masked).mean()
        
        # Percentage within 0.3m (Rapid CNN metric)
        within_threshold = np.abs(pred_masked - target_masked) < 0.3
        accuracy_03m = within_threshold.mean()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'accuracy_03m': accuracy_03m  # 97% target from paper
        }
```

## 5. Deployment Integration

### Production API combining both approaches
```python
# src/api/hybrid_inference.py
from fastapi import FastAPI, UploadFile
import torch
import numpy as np

app = FastAPI(title="Nashville Flood Risk API")

class FloodPredictor:
    def __init__(self):
        # Load adapted U-Net model
        self.model = AdaptedUNet.load_from_checkpoint('models/best.ckpt')
        self.model.eval()
        
        # Preprocessing from both repos
        self.lisflood_processor = LISFLOODProcessor()
        self.tile_size = 256  # UNOSAT tiling
        self.overlap = 32
        
    @torch.no_grad()
    def predict(self, dem, rainfall):
        """Inference combining both approaches"""
        
        # Create tiles (UNOSAT approach)
        tiles = self.create_tiles(dem, rainfall)
        predictions = []
        
        for tile in tiles:
            # Normalize (UNOSAT uses ImageNet normalization)
            tile_norm = self.normalize(tile)
            
            # Predict
            pred = self.model(tile_norm)
            
            # Apply threshold (Rapid CNN)
            pred[pred < 0.3] = 0
            
            predictions.append(pred)
        
        # Stitch tiles back
        full_prediction = self.stitch_tiles(predictions)
        
        return full_prediction

@app.post("/predict")
async def predict_flood(
    dem_file: UploadFile,
    rainfall_scenario: str = "100yr"
):
    # Process inputs
    dem = load_dem(dem_file)
    rainfall = get_rainfall_scenario(rainfall_scenario)
    
    # Run prediction
    flood_depth = predictor.predict(dem, rainfall)
    
    # Return GeoJSON (modern addition)
    return create_geojson_response(flood_depth)
```

## Performance Comparison

| Metric | UNOSAT U-Net | Rapid CNN | Our Hybrid |
|--------|--------------|-----------|------------|
| Architecture | Encoder-Decoder | Conv+Dense | Multi-scale U-Net |
| Input | SAR (1 channel) | Flow series (27 features) | DEM+Rainfall (4 channels) |
| Output | Binary mask | 581K depths | Continuous depth map |
| Dice Score | 0.92 | N/A | 0.94 (target) |
| Depth RMSE | N/A | <0.3m | <0.4m (target) |
| Inference Time | ~1s | ~0.5s | <5s (target) |
| Training Data | Manual labels | LISFLOOD-FP | LISFLOOD-FP + NFIP |

## Key Takeaways

1. **Architecture**: Use U-Net for spatial processing (UNOSAT strength)
2. **Data Generation**: Use LISFLOOD-FP for training data (Rapid CNN approach)
3. **Preprocessing**: Apply 0.3m threshold and temporal features
4. **Training**: Combine classification and regression losses
5. **Validation**: Use both spatial (Dice) and depth (RMSE) metrics

This hybrid approach leverages the proven components from both repositories while adding modern enhancements for production deployment.