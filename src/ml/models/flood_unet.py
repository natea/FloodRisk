"""
UNet-ResNet34 model for flood extent prediction.
Implementation based on APPROACH.md specifications using segmentation_models.pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FloodUNet(nn.Module):
    """
    UNet with ResNet-34 encoder for flood extent prediction.

    Based on APPROACH.md specifications:
    - UNet encoder-decoder with skip connections
    - ResNet-34 pretrained encoder adapted for input channels
    - Output logits at input resolution for binary flood prediction
    """

    def __init__(
        self,
        in_channels: int = 2,  # [DEM, Rain] baseline
        num_classes: int = 1,  # Binary flood/no-flood
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        dropout_rate: float = 0.2,
        use_fpn: bool = False,
        **kwargs,
    ):
        """
        Initialize FloodUNet model.

        Args:
            in_channels: Number of input channels (2-6 per APPROACH.md)
            num_classes: Number of output classes (1 for binary)
            encoder_name: Encoder architecture (resnet34/resnet50)
            encoder_weights: Pretrained weights (imagenet/None)
            dropout_rate: Dropout rate for regularization
            use_fpn: Whether to use Feature Pyramid Network head
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_fpn = use_fpn
        self.dropout_rate = dropout_rate

        logger.info(
            f"Initializing FloodUNet: {encoder_name}, {in_channels} channels, FPN={use_fpn}"
        )

        if use_fpn:
            # UNet with FPN head for multi-scale context
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs,
            )
        else:
            # Standard UNet
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs,
            )

        # Add dropout for uncertainty estimation (MC Dropout)
        if dropout_rate > 0:
            self._add_dropout_to_decoder()

        # Initialize weights for new conv layers if adapting from pretrained
        if encoder_weights == "imagenet" and in_channels != 3:
            self._adapt_first_layer()

    def _adapt_first_layer(self):
        """Adapt first convolutional layer for different input channels."""
        logger.info(f"Adapting first conv layer for {self.in_channels} input channels")

        encoder = self.model.encoder

        if hasattr(encoder, "conv1"):
            # ResNet-style encoder
            old_conv = encoder.conv1
            new_conv = nn.Conv2d(
                self.in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Initialize new channels
            with torch.no_grad():
                if self.in_channels < 3:
                    # Fewer channels: take mean of RGB
                    new_conv.weight[:, : self.in_channels] = old_conv.weight[
                        :, : self.in_channels
                    ]
                else:
                    # More channels: replicate existing weights
                    new_conv.weight[:, :3] = old_conv.weight
                    for i in range(3, self.in_channels):
                        new_conv.weight[:, i : i + 1] = old_conv.weight[
                            :, 0:1
                        ]  # Replicate R channel

            encoder.conv1 = new_conv

    def _add_dropout_to_decoder(self):
        """Add dropout layers to decoder for MC Dropout uncertainty estimation."""

        def add_dropout(module):
            # Create a list of children to avoid modifying dictionary during iteration
            children_list = list(module.named_children())
            for name, child in children_list:
                if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                    # Add dropout after conv layers
                    setattr(module, name + "_dropout", nn.Dropout2d(self.dropout_rate))
                else:
                    add_dropout(child)

        if hasattr(self.model, "decoder"):
            add_dropout(self.model.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Logits tensor [B, 1, H, W]
        """
        return self.model(x)

    def predict_proba(self, x: torch.Tensor, mc_samples: int = 10) -> torch.Tensor:
        """
        Predict with uncertainty using MC Dropout.

        Args:
            x: Input tensor
            mc_samples: Number of MC samples for uncertainty

        Returns:
            Tuple of (mean_probs, uncertainty)
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(x)
                probs = torch.sigmoid(logits)
                predictions.append(probs)

        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_probs, uncertainty


class DualStreamFloodUNet(nn.Module):
    """
    Dual-stream UNet for multi-scale context as mentioned in APPROACH.md.
    Processes high-res tile + coarser wide-area context.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        context_scale: int = 4,  # Context is 4x larger area, downsampled
        **kwargs,
    ):
        """
        Initialize dual-stream model.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            encoder_name: Encoder architecture
            context_scale: Scale factor for context stream
        """
        super().__init__()

        self.context_scale = context_scale

        # High-resolution stream
        self.high_res_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=32,  # Feature map output
            **kwargs,
        )

        # Context stream (processes larger area at lower resolution)
        self.context_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=32,  # Feature map output
            **kwargs,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with high-res tile and context.

        Args:
            x: High-resolution tile [B, C, H, W]
            context: Context tile [B, C, H//scale, W//scale]

        Returns:
            Prediction logits [B, 1, H, W]
        """
        # Process high-res stream
        high_res_features = self.high_res_stream(x)

        # Process context stream
        context_features = self.context_stream(context)

        # Upsample context features to match high-res
        context_features = F.interpolate(
            context_features,
            size=high_res_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Fuse features
        fused = torch.cat([high_res_features, context_features], dim=1)
        output = self.fusion(fused)

        return output


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create flood prediction model based on config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_type = config.get("type", "unet")

    if model_type == "unet":
        model = FloodUNet(
            in_channels=config.get("in_channels", 2),
            num_classes=config.get("num_classes", 1),
            encoder_name=config.get("encoder_name", "resnet34"),
            encoder_weights=config.get("encoder_weights", "imagenet"),
            dropout_rate=config.get("dropout_rate", 0.2),
            use_fpn=config.get("use_fpn", False),
        )
    elif model_type == "dual_stream":
        model = DualStreamFloodUNet(
            in_channels=config.get("in_channels", 2),
            num_classes=config.get("num_classes", 1),
            encoder_name=config.get("encoder_name", "resnet34"),
            context_scale=config.get("context_scale", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(
        f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    return model
