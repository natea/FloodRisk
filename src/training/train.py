"""
Training script for FloodCNN model with physics-informed constraints.
Features:
- Adam optimizer with learning rate scheduling
- Combined MSE loss and physics regularization
- Data loading and augmentation
- Model checkpointing and monitoring
- Validation and evaluation metrics
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.flood_cnn import create_flood_cnn_model, MultiScaleUNet


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodDataset(Dataset):
    """Dataset class for flood prediction data."""

    def __init__(
        self,
        data_dir: str,
        input_channels: List[str] = ["dem", "landuse", "soil", "rainfall"],
        target_channel: str = "flood_depth",
        transform: Optional[callable] = None,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.input_channels = input_channels
        self.target_channel = target_channel
        self.transform = transform
        self.normalize = normalize

        # Load data file paths
        self.data_files = self._load_data_files()

        # Compute normalization statistics if needed
        if self.normalize:
            self.stats = self._compute_stats()

    def _load_data_files(self) -> List[Dict[str, str]]:
        """Load data file paths."""
        # This is a placeholder - implement based on your data structure
        data_files = []

        # Example implementation for synthetic data
        # Replace with actual data loading logic
        for i in range(1000):  # Assume 1000 samples
            sample = {}
            for channel in self.input_channels + [self.target_channel]:
                sample[channel] = str(self.data_dir / f"sample_{i}_{channel}.npy")
            data_files.append(sample)

        return data_files

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics."""
        stats = {}

        # For demonstration, using default values
        # In practice, compute from training data
        for channel in self.input_channels + [self.target_channel]:
            if channel == "dem":
                stats[channel] = {"mean": 100.0, "std": 50.0}
            elif channel == "landuse":
                stats[channel] = {"mean": 5.0, "std": 3.0}
            elif channel == "soil":
                stats[channel] = {"mean": 0.3, "std": 0.2}
            elif channel == "rainfall":
                stats[channel] = {"mean": 10.0, "std": 15.0}
            else:  # flood_depth
                stats[channel] = {"mean": 0.5, "std": 1.0}

        return stats

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a data sample."""
        sample_files = self.data_files[idx]

        # Load input channels
        inputs = []
        for channel in self.input_channels:
            # For demonstration, generate synthetic data
            # Replace with actual data loading
            if os.path.exists(sample_files[channel]):
                data = np.load(sample_files[channel])
            else:
                # Generate synthetic data
                data = self._generate_synthetic_data(channel)

            if self.normalize:
                stats = self.stats[channel]
                data = (data - stats["mean"]) / stats["std"]

            inputs.append(data)

        inputs = np.stack(inputs, axis=0)  # Shape: (C, H, W)

        # Load target
        if os.path.exists(sample_files[self.target_channel]):
            target = np.load(sample_files[self.target_channel])
        else:
            target = self._generate_synthetic_target()

        if self.normalize:
            stats = self.stats[self.target_channel]
            target = (target - stats["mean"]) / stats["std"]

        target = np.expand_dims(target, axis=0)  # Shape: (1, H, W)

        # Extract rainfall for physics constraints
        rainfall_idx = (
            self.input_channels.index("rainfall")
            if "rainfall" in self.input_channels
            else 0
        )
        rainfall = np.expand_dims(inputs[rainfall_idx], axis=0)

        # Apply transforms
        if self.transform:
            inputs, target, rainfall = self.transform(inputs, target, rainfall)

        return (
            torch.from_numpy(inputs).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(rainfall).float(),
        )

    def _generate_synthetic_data(
        self, channel: str, size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """Generate synthetic data for demonstration."""
        if channel == "dem":
            # Generate elevation data with hills and valleys
            x = np.linspace(0, 10, size[0])
            y = np.linspace(0, 10, size[1])
            X, Y = np.meshgrid(x, y)
            data = (
                100
                + 50 * np.sin(0.5 * X) * np.cos(0.5 * Y)
                + np.random.normal(0, 5, size)
            )

        elif channel == "landuse":
            # Generate land use categories
            data = np.random.randint(1, 10, size).astype(np.float32)

        elif channel == "soil":
            # Generate soil permeability
            data = 0.1 + 0.4 * np.random.random(size)

        elif channel == "rainfall":
            # Generate rainfall with spatial correlation
            base = np.random.random(size)
            smoothed = np.zeros_like(base)
            kernel_size = 5
            for i in range(kernel_size, size[0] - kernel_size):
                for j in range(kernel_size, size[1] - kernel_size):
                    smoothed[i, j] = np.mean(
                        base[
                            i - kernel_size : i + kernel_size + 1,
                            j - kernel_size : j + kernel_size + 1,
                        ]
                    )
            data = 5 + 20 * smoothed

        else:
            data = np.random.random(size)

        return data.astype(np.float32)

    def _generate_synthetic_target(
        self, size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """Generate synthetic flood depth target."""
        # Simple flood simulation: deeper in low elevation areas with high rainfall
        dem = self._generate_synthetic_data("dem", size)
        rainfall = self._generate_synthetic_data("rainfall", size)

        # Normalize inputs
        dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
        rainfall_norm = (rainfall - rainfall.min()) / (rainfall.max() - rainfall.min())

        # Flood depth inversely related to elevation, positively to rainfall
        flood_depth = np.maximum(
            0, rainfall_norm * 2.0 - dem_norm * 1.5 + 0.1 * np.random.random(size)
        )

        return flood_depth.astype(np.float32)


class FloodTrainer:
    """Trainer class for FloodCNN model."""

    def __init__(
        self,
        model: MultiScaleUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Setup learning rate scheduler
        if config["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
            )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_physics": [],
            "val_physics": [],
            "learning_rate": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {"total": 0.0, "mse": 0.0, "physics": 0.0, "mae": 0.0}

        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (inputs, targets, rainfall) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            rainfall = rainfall.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions, physics_loss = self.model(inputs, rainfall)

            # Compute losses
            mse_loss = self.mse_loss(predictions, targets)
            mae_loss = self.mae_loss(predictions, targets)

            # Combined loss with physics regularization
            total_loss = mse_loss + self.config["physics_weight"] * physics_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.optimizer.step()

            # Update metrics
            epoch_losses["total"] += total_loss.item()
            epoch_losses["mse"] += mse_loss.item()
            epoch_losses["physics"] += physics_loss.item()
            epoch_losses["mae"] += mae_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{total_loss.item():.4f}",
                    "MSE": f"{mse_loss.item():.4f}",
                    "Physics": f"{physics_loss.item():.4f}",
                }
            )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_losses = {"total": 0.0, "mse": 0.0, "physics": 0.0, "mae": 0.0}

        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets, rainfall in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                rainfall = rainfall.to(self.device)

                # Forward pass
                predictions, physics_loss = self.model(inputs, rainfall)

                # Compute losses
                mse_loss = self.mse_loss(predictions, targets)
                mae_loss = self.mae_loss(predictions, targets)

                total_loss = mse_loss + self.config["physics_weight"] * physics_loss

                # Update metrics
                epoch_losses["total"] += total_loss.item()
                epoch_losses["mse"] += mse_loss.item()
                epoch_losses["physics"] += physics_loss.item()
                epoch_losses["mae"] += mae_loss.item()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = (
            Path(self.config["checkpoint_dir"]) / f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = Path(self.config["checkpoint_dir"]) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"New best model saved with validation loss: {self.best_val_loss:.6f}"
            )

    def plot_training_curves(self) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Total loss
        axes[0, 0].plot(self.history["train_loss"], label="Train")
        axes[0, 0].plot(self.history["val_loss"], label="Validation")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MSE loss
        axes[0, 1].plot(self.history["train_mse"], label="Train")
        axes[0, 1].plot(self.history["val_mse"], label="Validation")
        axes[0, 1].set_title("MSE Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Physics loss
        axes[1, 0].plot(self.history["train_physics"], label="Train")
        axes[1, 0].plot(self.history["val_physics"], label="Validation")
        axes[1, 0].set_title("Physics Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Physics Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot(self.history["learning_rate"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            Path(self.config["checkpoint_dir"]) / "training_curves.png", dpi=150
        )
        plt.close()

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(1, self.config["epochs"] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['epochs']}")

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate_epoch()

            # Update learning rate
            if self.config["scheduler"] == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics["total"])

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["train_physics"].append(train_metrics["physics"])
            self.history["val_physics"].append(val_metrics["physics"])
            self.history["learning_rate"].append(current_lr)

            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['total']:.6f} | "
                f"Val Loss: {val_metrics['total']:.6f} | "
                f"LR: {current_lr:.2e}"
            )
            logger.info(
                f"Train MSE: {train_metrics['mse']:.6f} | "
                f"Val MSE: {val_metrics['mse']:.6f}"
            )
            logger.info(
                f"Train Physics: {train_metrics['physics']:.6f} | "
                f"Val Physics: {val_metrics['physics']:.6f}"
            )

            # Check for best model
            is_best = val_metrics["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["total"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            if epoch % self.config["save_freq"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Plot training curves
            if epoch % 10 == 0:
                self.plot_training_curves()

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        # Final plots
        self.plot_training_curves()


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    # Create dataset
    dataset = FloodDataset(
        data_dir=config["data_dir"],
        input_channels=config["input_channels"],
        target_channel=config["target_channel"],
        normalize=config["normalize"],
    )

    # Split dataset
    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train FloodCNN model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.json",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flood_data",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    # Default configuration
    default_config = {
        "data_dir": args.data_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "input_channels": ["dem", "landuse", "soil", "rainfall"],
        "target_channel": "flood_depth",
        "normalize": True,
        "train_split": 0.8,
        "batch_size": 8,
        "num_workers": 4,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "physics_weight": 0.1,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "min_lr": 1e-6,
        "early_stopping_patience": 20,
        "save_freq": 5,
        "seed": 42,
        "model_config": {
            "input_channels": 4,
            "output_channels": 1,
            "base_channels": 64,
            "scales": [1, 2, 4],
            "dropout": 0.1,
            "grid_spacing": 1.0,
        },
    }

    # Load configuration if exists
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        # Update with any missing defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        logger.info(f"Using default configuration. Consider creating {args.config}")

    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Save configuration
    with open(Path(config["checkpoint_dir"]) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Set random seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_flood_cnn_model(**config["model_config"]).to(device)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)

    # Create trainer
    trainer = FloodTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
