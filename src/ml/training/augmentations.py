"""
Hydrology-safe augmentations for flood prediction training.
Implementation based on APPROACH.md specifications for rainfall domain randomization.
"""

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class HydrologySafeAugmentations:
    """
    Augmentations that preserve hydrological flow pathways.
    Based on APPROACH.md: 90° rotations, flips, small translations, rainfall randomization.
    Avoids elastic/affine warps that distort flow pathways.
    """

    def __init__(
        self,
        tile_size: int = 512,
        rainfall_variability: float = 0.15,  # ±15% per APPROACH.md
        enable_spatial_rainfall_noise: bool = True,
        p_augment: float = 0.8,
    ):
        """
        Initialize hydrology-safe augmentations.

        Args:
            tile_size: Size of input tiles
            rainfall_variability: Rainfall scaling variability (±10-15%)
            enable_spatial_rainfall_noise: Whether to add spatial gradients/noise to rainfall
            p_augment: Probability of applying augmentations
        """
        self.tile_size = tile_size
        self.rainfall_variability = rainfall_variability
        self.enable_spatial_rainfall_noise = enable_spatial_rainfall_noise
        self.p_augment = p_augment

        # Create albumentations pipeline for geometric transforms
        self.geometric_transforms = A.Compose(
            [
                # 90-degree rotations (preserves flow patterns)
                A.RandomRotate90(p=0.5),
                # Flips (preserves flow patterns)
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                # Small translations (preserves relative flow)
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # Small translations only
                    scale_limit=0.0,  # No scaling to avoid distortion
                    rotate_limit=0,  # No arbitrary rotations
                    p=0.3,
                ),
                # Optional: small crops with padding to maintain size
                A.RandomCrop(
                    height=int(tile_size * 0.95), width=int(tile_size * 0.95), p=0.2
                ),
                A.Resize(tile_size, tile_size, p=1.0),  # Ensure consistent size
            ],
            p=self.p_augment,
        )

        logger.info(
            f"Initialized hydrology-safe augmentations with {rainfall_variability*100:.1f}% rainfall variability"
        )

    def apply_rainfall_randomization(
        self, rainfall_channel: np.ndarray, dem_channel: np.ndarray
    ) -> np.ndarray:
        """
        Apply rainfall domain randomization as specified in APPROACH.md.
        ±10-15% scaling; small spatial gradients/noise only on rainfall layer.

        Args:
            rainfall_channel: Rainfall data [H, W]
            dem_channel: DEM data for spatial reference [H, W]

        Returns:
            Randomized rainfall channel
        """
        rainfall_aug = rainfall_channel.copy()

        # Global scaling (±10-15%)
        scale_factor = np.random.uniform(
            1.0 - self.rainfall_variability, 1.0 + self.rainfall_variability
        )
        rainfall_aug = rainfall_aug * scale_factor

        # Optional spatial gradients/noise
        if self.enable_spatial_rainfall_noise:
            # Create smooth spatial gradient
            h, w = rainfall_channel.shape

            # Random gradient direction
            grad_strength = np.random.uniform(0.0, 0.1)  # Up to 10% spatial variation
            if grad_strength > 0:
                # Create gradient field
                y_grad = np.random.uniform(-grad_strength, grad_strength)
                x_grad = np.random.uniform(-grad_strength, grad_strength)

                y_coords = np.linspace(-1, 1, h)
                x_coords = np.linspace(-1, 1, w)
                Y, X = np.meshgrid(y_coords, x_coords, indexing="ij")

                gradient_field = 1.0 + (Y * y_grad + X * x_grad)

                # Apply gradient
                rainfall_aug = rainfall_aug * gradient_field

            # Add small amount of smooth noise
            noise_strength = np.random.uniform(0.0, 0.05)  # Up to 5% noise
            if noise_strength > 0:
                noise = np.random.normal(1.0, noise_strength, rainfall_channel.shape)
                # Smooth the noise to avoid unrealistic spatial patterns
                from scipy import ndimage

                noise = ndimage.gaussian_filter(noise, sigma=1.0)
                rainfall_aug = rainfall_aug * noise

        # Ensure non-negative rainfall
        rainfall_aug = np.maximum(rainfall_aug, 0.0)

        return rainfall_aug

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        dem_channel_idx: int = 0,
        rainfall_channel_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply hydrology-safe augmentations to image and mask.

        Args:
            image: Input image [C, H, W] or [H, W, C]
            mask: Ground truth mask [H, W]
            dem_channel_idx: Index of DEM channel
            rainfall_channel_idx: Index of rainfall channel

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Handle different input formats
        if len(image.shape) == 3:
            if image.shape[0] <= 6:  # Assume [C, H, W]
                image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]

        # Separate channels for individual processing
        dem_channel = image[:, :, dem_channel_idx].copy()
        rainfall_channel = image[:, :, rainfall_channel_idx].copy()

        # Apply rainfall randomization (before geometric transforms)
        rainfall_augmented = self.apply_rainfall_randomization(
            rainfall_channel, dem_channel
        )

        # Reconstruct image with augmented rainfall
        image_aug = image.copy()
        image_aug[:, :, rainfall_channel_idx] = rainfall_augmented

        # Apply geometric transformations
        if mask is not None:
            # Apply geometric transforms to both image and mask
            transformed = self.geometric_transforms(image=image_aug, mask=mask)
            return transformed["image"], transformed["mask"]
        else:
            # Apply to image only
            transformed = self.geometric_transforms(image=image_aug)
            return transformed["image"], None

    def create_training_transforms(self) -> A.Compose:
        """
        Create complete training augmentation pipeline.

        Returns:
            Albumentations composition for training
        """
        return A.Compose(
            [
                # Custom rainfall randomization will be applied separately
                # Geometric transforms (hydrology-safe)
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.0, rotate_limit=0, p=0.3
                ),
                # Normalization and tensor conversion
                A.Normalize(
                    mean=0.0, std=1.0, p=1.0
                ),  # Will be overridden by custom normalization
                ToTensorV2(p=1.0),
            ],
            p=self.p_augment,
        )

    def create_validation_transforms(self) -> A.Compose:
        """
        Create validation transforms (no augmentation, just normalization).

        Returns:
            Albumentations composition for validation
        """
        return A.Compose(
            [
                A.Normalize(mean=0.0, std=1.0, p=1.0),  # Will be overridden
                ToTensorV2(p=1.0),
            ]
        )


class MultiReturnPeriodAugmentation:
    """
    Mix different return periods during training as mentioned in APPROACH.md.
    Train jointly on 100-yr and 500-yr events; optionally include sub-design negatives.
    """

    def __init__(
        self, return_periods: Dict[str, float] = None, weights: Dict[str, float] = None
    ):
        """
        Initialize multi-return period augmentation.

        Args:
            return_periods: Dictionary mapping period names to rainfall depths
            weights: Sampling weights for each return period
        """
        if return_periods is None:
            # Default return periods from APPROACH.md
            self.return_periods = {
                "100yr": 150.0,  # Example values - should be from NOAA Atlas 14
                "500yr": 200.0,
                "10yr": 75.0,  # Sub-design for negatives
                "25yr": 100.0,  # Sub-design for negatives
            }
        else:
            self.return_periods = return_periods

        if weights is None:
            # Default weights: emphasize design storms, some sub-design
            self.weights = {"100yr": 0.4, "500yr": 0.4, "10yr": 0.1, "25yr": 0.1}
        else:
            self.weights = weights

        # Create sampling probabilities
        periods = list(self.return_periods.keys())
        self.periods = periods
        self.probabilities = [self.weights[p] for p in periods]

        logger.info(
            f"Initialized multi-return period augmentation with periods: {periods}"
        )

    def sample_return_period(self) -> Tuple[str, float]:
        """
        Sample a return period for training.

        Returns:
            Tuple of (period_name, rainfall_depth_mm)
        """
        period = np.random.choice(self.periods, p=self.probabilities)
        rainfall_depth = self.return_periods[period]

        return period, rainfall_depth

    def generate_rainfall_scenarios(
        self, base_tile: Dict[str, Any], n_scenarios: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple rainfall scenarios from base tile.

        Args:
            base_tile: Base tile with DEM and other features
            n_scenarios: Number of rainfall scenarios to generate

        Returns:
            List of tile dictionaries with different rainfall scenarios
        """
        scenarios = []

        for _ in range(n_scenarios):
            scenario = base_tile.copy()
            period, rainfall_depth = self.sample_return_period()

            # Update rainfall channel (assuming uniform rainfall)
            rainfall_shape = scenario["dem"].shape  # Assume DEM shape
            scenario["rainfall"] = np.full(
                rainfall_shape, rainfall_depth, dtype=np.float32
            )
            scenario["return_period"] = period
            scenario["rainfall_depth_mm"] = rainfall_depth

            scenarios.append(scenario)

        return scenarios
