"""
Main training script for flood prediction model.
Implementation based on APPROACH.md specifications with PyTorch Lightning.
"""

import os
import sys
from pathlib import Path
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import xarray as xr

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.ml.models.flood_unet import create_model
from src.ml.training.losses import create_loss_function
from src.ml.training.metrics import FloodMetrics
from src.ml.training.augmentations import HydrologySafeAugmentations
from src.ml.data.preprocessing import DEMProcessor, RainfallProcessor, TileGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FloodDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for flood prediction.
    Handles data loading, preprocessing, and augmentation.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.num_workers = config.hardware.num_workers
        self.pin_memory = config.hardware.pin_memory
        
        # Initialize processors
        self.dem_processor = DEMProcessor(
            target_crs=config.data.target_crs,
            resolution=config.data.resolution
        )
        self.rainfall_processor = RainfallProcessor()
        self.tile_generator = TileGenerator(
            tile_size=config.data.tile_size,
            overlap=config.data.tile_overlap
        )
        
        # Initialize augmentations
        if config.augmentation.enabled:
            self.augmentations = HydrologySafeAugmentations(
                tile_size=config.data.tile_size,
                rainfall_variability=config.augmentation.rainfall_variability,
                enable_spatial_rainfall_noise=config.augmentation.spatial_rainfall_noise,
                p_augment=config.augmentation.p_augment
            )
        else:
            self.augmentations = None
            
        self.train_tiles = []
        self.val_tiles = []
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation."""
        if stage == "fit" or stage is None:
            # In a real implementation, you would load actual DEM and simulation data here
            # For now, we'll create placeholder setup
            logger.info("Setting up datasets...")
            
            # Load and process DEM data
            # dem_data = self.dem_processor.load_and_reproject(dem_path)
            # derived_features = self.dem_processor.compute_derived_features(dem_data)
            
            # Create rainfall scenarios
            # rainfall_data = self.rainfall_processor.create_uniform_rainfall_raster(...)
            
            # Generate tiles with balanced sampling
            # all_tiles = self.tile_generator.generate_tiles([dem_data, rainfall_data], flood_labels)
            # balanced_tiles = self.tile_generator.balanced_sampling(all_tiles, ...)
            
            # Split into train/validation
            # self.train_tiles, self.val_tiles = train_test_split(balanced_tiles, ...)
            
            # For demonstration, create dummy data
            logger.warning("Using dummy data for demonstration. Replace with real data loading.")
            self._create_dummy_data()
            
    def _create_dummy_data(self):
        """Create dummy training data for demonstration."""
        n_train = 1000
        n_val = 200
        tile_size = self.config.data.tile_size
        n_channels = self.config.model.in_channels
        
        for i in range(n_train):
            self.train_tiles.append({
                'data': np.random.randn(n_channels, tile_size, tile_size).astype(np.float32),
                'labels': np.random.randint(0, 2, (1, tile_size, tile_size)).astype(np.float32),
                'tile_id': f'train_{i}'
            })
            
        for i in range(n_val):
            self.val_tiles.append({
                'data': np.random.randn(n_channels, tile_size, tile_size).astype(np.float32),
                'labels': np.random.randint(0, 2, (1, tile_size, tile_size)).astype(np.float32),
                'tile_id': f'val_{i}'
            })
    
    def train_dataloader(self):
        """Create training DataLoader."""
        dataset = FloodDataset(
            self.train_tiles, 
            augmentations=self.augmentations,
            is_training=True
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.config.hardware.persistent_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Create validation DataLoader."""
        dataset = FloodDataset(
            self.val_tiles,
            augmentations=None,  # No augmentation for validation
            is_training=False
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.config.hardware.persistent_workers
        )

class FloodDataset(torch.utils.data.Dataset):
    """Dataset class for flood prediction tiles."""
    
    def __init__(
        self, 
        tiles: List[Dict], 
        augmentations: Optional[HydrologySafeAugmentations] = None,
        is_training: bool = True
    ):
        self.tiles = tiles
        self.augmentations = augmentations
        self.is_training = is_training
        
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        
        # Get data and labels
        data = tile['data'].copy()  # [C, H, W]
        labels = tile['labels'].copy()  # [1, H, W]
        
        # Apply augmentations if training
        if self.is_training and self.augmentations:
            # Convert to [H, W, C] format for augmentations
            data_hwc = np.transpose(data, (1, 2, 0))
            labels_hw = labels.squeeze(0)
            
            # Apply augmentations
            data_aug, labels_aug = self.augmentations(data_hwc, labels_hw)
            
            # Convert back to tensors
            if isinstance(data_aug, np.ndarray):
                data = torch.from_numpy(np.transpose(data_aug, (2, 0, 1)))
                labels = torch.from_numpy(labels_aug).unsqueeze(0)
            else:
                data = data_aug.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                labels = labels_aug.unsqueeze(0)
        else:
            # Convert to tensors
            data = torch.from_numpy(data)
            labels = torch.from_numpy(labels)
        
        # Ensure float32 dtype
        data = data.float()
        labels = labels.float()
        
        return data, labels

class FloodLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for flood prediction training.
    Implements the training logic per APPROACH.md specifications.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        
        # Create model
        self.model = create_model(config.model)
        
        # Create loss function
        self.loss_fn = create_loss_function(config.loss)
        
        # Create metrics calculator
        self.metrics = FloodMetrics(
            threshold=config.metrics.threshold,
            device=self.device
        )
        
        # Training configuration
        self.automatic_optimization = False  # Manual optimization for freeze/unfreeze
        self.freeze_encoder_epochs = config.training.freeze_encoder_epochs
        self._current_epoch = 0  # Use private variable to avoid conflict with Lightning's property
        
        logger.info(f"Initialized FloodLightningModule with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizers with different learning rates for encoder/decoder."""
        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []
        
        if hasattr(self.model, 'model'):  # For wrapped models (segmentation_models_pytorch)
            model = self.model.model
        else:
            model = self.model
            
        for name, param in model.named_parameters():
            if 'encoder' in name and hasattr(model, 'encoder'):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': decoder_params, 'lr': self.config.training.decoder_lr},
            {'params': encoder_params, 'lr': self.config.training.encoder_lr}
        ]
        
        # Create optimizer
        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups, 
                weight_decay=self.config.training.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(param_groups)
        
        # Create scheduler
        if self.config.training.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.config.training.decoder_lr, self.config.training.encoder_lr],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.training.total_epochs
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }
    
    def on_train_epoch_start(self):
        """Handle encoder freezing/unfreezing per APPROACH.md."""
        if self._current_epoch < self.freeze_encoder_epochs:
            # Freeze encoder
            self._freeze_encoder()
        else:
            # Unfreeze encoder
            self._unfreeze_encoder()
            
        self._current_epoch += 1
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        model = self.model.model if hasattr(self.model, 'model') else self.model
        
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
    
    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        model = self.model.model if hasattr(self.model, 'model') else self.model
        
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = True
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        data, labels = batch
        
        # Forward pass
        logits = self.forward(data)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        
        # Gradient clipping
        if self.config.training.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                self.config.training.gradient_clip_val
            )
        
        opt.step()
        
        # Step scheduler if configured
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.is_last_batch:
            sch.step()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, labels = batch
        
        # Forward pass
        logits = self.forward(data)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Compute metrics
        probs = torch.sigmoid(logits)
        
        # Convert to same device as metrics
        if hasattr(self.metrics, 'device'):
            self.metrics.device = probs.device
        
        batch_metrics = self.metrics.compute_all_metrics(probs, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in batch_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return {'loss': loss, 'metrics': batch_metrics}
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)

@hydra.main(version_base=None, config_path="../../configs", config_name="model_config")
def train_model(config: DictConfig):
    """
    Main training function.
    
    Args:
        config: Hydra configuration
    """
    logger.info("Starting flood prediction model training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set random seeds for reproducibility
    if config.seed:
        pl.seed_everything(config.seed, workers=True)
    
    # Create data module
    data_module = FloodDataModule(config)
    
    # Create model
    model = FloodLightningModule(config)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.logging.save_dir) / "checkpoints",
        filename='{epoch:02d}-{val_iou:.3f}',
        monitor=config.logging.monitor,
        mode=config.logging.mode,
        save_top_k=config.logging.save_top_k,
        save_last=True,
        every_n_epochs=config.logging.checkpoint_every_n_epochs
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if hasattr(config.training, 'early_stopping') and config.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            monitor=config.logging.monitor,
            patience=config.training.early_stopping.patience,
            mode=config.logging.mode,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Create loggers
    loggers = []
    
    if config.logging.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.logging.wandb.project,
            tags=config.logging.wandb.tags,
            save_dir=config.logging.save_dir
        )
        loggers.append(wandb_logger)
    
    if config.logging.get('tensorboard', {}).get('enabled', False):
        tb_logger = TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name="tensorboard_logs"
        )
        loggers.append(tb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.total_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        strategy=config.hardware.strategy if config.hardware.devices != 1 else "auto",
        precision=config.training.precision,
        # gradient_clip_val handled manually in training_step due to manual optimization
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        deterministic=config.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test best model
    logger.info("Testing best model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        trainer.test(model, data_module, ckpt_path=best_model_path)
    
    logger.info("Training completed!")
    
    return trainer, model

if __name__ == "__main__":
    train_model()