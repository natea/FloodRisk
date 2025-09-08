#!/usr/bin/env python3
"""
Simplified training script that works without GDAL dependencies.
"""

import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(n_samples=100, tile_size=512, n_channels=2):
    """Create dummy training data for testing."""
    data = []
    for i in range(n_samples):
        # Random input (DEM + rainfall)
        x = torch.randn(n_channels, tile_size, tile_size)
        # Random binary labels
        y = torch.randint(0, 2, (1, tile_size, tile_size)).float()
        data.append((x, y))
    return data

def train_simple_model():
    """Train a simple model with dummy data."""
    logger.info("Starting simplified training (no GDAL required)...")
    
    # Create dummy data
    logger.info("Creating dummy training data...")
    train_data = create_dummy_data(100)
    val_data = create_dummy_data(20)
    
    # Create simple UNet model
    logger.info("Creating model...")
    try:
        from src.ml.models.flood_unet import FloodUNet
        model = FloodUNet(in_channels=2, num_classes=1)
    except ImportError:
        logger.warning("Could not import FloodUNet, using simple Conv2d")
        model = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1)
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    logger.info("Training for 10 epochs...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in train_data[:10]:  # Train on subset for speed
            optimizer.zero_grad()
            pred = model(x.unsqueeze(0))
            loss = criterion(pred, y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        logger.info(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "simple_model.pth")
    logger.info(f"Model saved to {output_dir / 'simple_model.pth'}")
    
    logger.info("Training complete!")
    logger.info("\nNext steps:")
    logger.info("1. Install GDAL for real data processing")
    logger.info("2. Download real DEM data with scripts/data_acquisition/download_nashville_data.py")
    logger.info("3. Run full training with scripts/train_model.py")

if __name__ == "__main__":
    train_simple_model()
