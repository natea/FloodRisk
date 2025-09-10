"""
Loss functions for flood prediction training.
Implementation of losses specified in APPROACH.md: BCE + Dice, Focal, Tversky.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss.
    As specified in APPROACH.md: Loss = 0.5 * BCE + 0.5 * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize BCE + Dice loss.

        Args:
            bce_weight: Weight for BCE component
            dice_weight: Weight for Dice component
            smooth: Smoothing factor for Dice
            pos_weight: Positive class weight for BCE (for class imbalance)
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: Model predictions [B, 1, H, W]
            targets: Ground truth labels [B, 1, H, W]

        Returns:
            Combined loss scalar
        """
        # BCE loss
        bce_loss = self.bce(logits, targets)

        # Dice loss
        probs = torch.sigmoid(logits)
        dice_loss = self._dice_loss(probs, targets)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Mentioned in APPROACH.md for heavy imbalance scenarios.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (γ=2 per APPROACH.md)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [B, 1, H, W]
            targets: Ground truth labels [B, 1, H, W]

        Returns:
            Focal loss
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance.
    Mentioned in APPROACH.md with α=0.5, β=0.7 for flood prediction.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.7, smooth: float = 1.0):
        """
        Initialize Tversky Loss.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.

        Args:
            logits: Model predictions [B, 1, H, W]
            targets: Ground truth labels [B, 1, H, W]

        Returns:
            Tversky loss
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky_index


class BCETverskyLoss(nn.Module):
    """
    Combined BCE and Tversky loss as alternative to BCE + Dice.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.7,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize combined BCE + Tversky loss.

        Args:
            bce_weight: Weight for BCE component
            tversky_weight: Weight for Tversky component
            alpha: Tversky alpha parameter
            beta: Tversky beta parameter
            pos_weight: Positive class weight for BCE
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        bce_loss = self.bce(logits, targets)
        tversky_loss = self.tversky(logits, targets)

        total_loss = self.bce_weight * bce_loss + self.tversky_weight * tversky_loss

        return total_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss for class imbalance.
    Mentioned in APPROACH.md with w_flood ≈ 2-4× w_dry.
    """

    def __init__(self, flood_weight: float = 3.0):
        """
        Initialize weighted BCE loss.

        Args:
            flood_weight: Weight for flood class (2-4x per APPROACH.md)
        """
        super().__init__()
        # pos_weight should be tensor for BCEWithLogitsLoss
        self.pos_weight = torch.tensor([flood_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss."""
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def create_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function based on configuration.

    Args:
        config: Loss configuration dictionary

    Returns:
        Loss function instance
    """
    loss_type = config.get("type", "bce_dice")

    if loss_type == "bce_dice":
        pos_weight = None
        if "flood_weight" in config:
            pos_weight = torch.tensor([config["flood_weight"]])

        loss_fn = BCEDiceLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            smooth=config.get("smooth", 1.0),
            pos_weight=pos_weight,
        )

    elif loss_type == "focal":
        loss_fn = FocalLoss(
            alpha=config.get("alpha", 1.0),
            gamma=config.get("gamma", 2.0),
            reduction=config.get("reduction", "mean"),
        )

    elif loss_type == "tversky":
        loss_fn = TverskyLoss(
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.7),
            smooth=config.get("smooth", 1.0),
        )

    elif loss_type == "bce_tversky":
        pos_weight = None
        if "flood_weight" in config:
            pos_weight = torch.tensor([config["flood_weight"]])

        loss_fn = BCETverskyLoss(
            bce_weight=config.get("bce_weight", 0.5),
            tversky_weight=config.get("tversky_weight", 0.5),
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.7),
            pos_weight=pos_weight,
        )

    elif loss_type == "weighted_bce":
        loss_fn = WeightedBCELoss(flood_weight=config.get("flood_weight", 3.0))

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    logger.info(f"Created {loss_type} loss function")
    return loss_fn
