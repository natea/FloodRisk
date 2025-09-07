"""
Multi-scale U-Net CNN model for flood depth prediction with physics-informed constraints.
Features:
- Multi-scale input processing (3 scales)
- Physics-informed mass conservation constraints
- Rainfall attention mechanism
- Skip connections and feature fusion
- Flood depth prediction output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class AttentionBlock(nn.Module):
    """Spatial attention mechanism for rainfall features."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False).to(x.device)(spatial_concat)
        )
        
        return x * channel_att * spatial_att


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FeatureFusion(nn.Module):
    """Feature fusion module for combining multi-scale features."""
    
    def __init__(self, channels: List[int], output_channels: int):
        super(FeatureFusion, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, output_channels, kernel_size=1) for ch in channels
        ])
        self.fusion_conv = nn.Conv2d(
            output_channels * len(channels), output_channels, kernel_size=3, padding=1
        )
        self.norm = nn.BatchNorm2d(output_channels)
        
    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        aligned_features = []
        
        for i, (feat, conv) in enumerate(zip(features, self.convs)):
            # Resize to target size
            resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(conv(resized))
        
        # Concatenate and fuse
        fused = torch.cat(aligned_features, dim=1)
        return F.relu(self.norm(self.fusion_conv(fused)))


class PhysicsConstraintModule(nn.Module):
    """Physics-informed constraint module for mass conservation."""
    
    def __init__(self, grid_spacing: float = 1.0):
        super(PhysicsConstraintModule, self).__init__()
        self.grid_spacing = grid_spacing
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
    def compute_gradients(self, flood_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial gradients using Sobel operators."""
        grad_x = F.conv2d(flood_depth, self.sobel_x, padding=1) / (8 * self.grid_spacing)
        grad_y = F.conv2d(flood_depth, self.sobel_y, padding=1) / (8 * self.grid_spacing)
        return grad_x, grad_y
        
    def mass_conservation_loss(
        self, 
        flood_depth: torch.Tensor, 
        rainfall: torch.Tensor, 
        infiltration: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mass conservation constraint loss."""
        # Compute temporal derivative approximation
        # For simplicity, using spatial gradients as proxy for flow divergence
        grad_x, grad_y = self.compute_gradients(flood_depth)
        divergence = grad_x + grad_y
        
        # Mass conservation: ∂h/∂t + ∇·(vh) = P - I
        # Simplified: divergence ≈ rainfall - infiltration
        if infiltration is None:
            infiltration = torch.zeros_like(rainfall)
            
        mass_balance = divergence - (rainfall - infiltration)
        return torch.mean(mass_balance ** 2)


class MultiScaleUNet(nn.Module):
    """Multi-scale U-Net for flood depth prediction with physics constraints."""
    
    def __init__(
        self,
        input_channels: int = 4,  # DEM, land use, soil, rainfall
        output_channels: int = 1,  # Flood depth
        base_channels: int = 64,
        scales: List[int] = [1, 2, 4],  # Multi-scale factors
        dropout: float = 0.1,
        grid_spacing: float = 1.0
    ):
        super(MultiScaleUNet, self).__init__()
        
        self.scales = scales
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Multi-scale encoders
        self.encoders = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        
        for scale in scales:
            encoder_layers = nn.ModuleList()
            pool_layers = nn.ModuleList()
            
            # Encoder path for each scale
            in_ch = input_channels
            for i, ch in enumerate([base_channels, base_channels*2, base_channels*4, base_channels*8]):
                encoder_layers.append(ConvBlock(in_ch, ch, dropout))
                if i < 3:  # No pooling for the last layer
                    pool_layers.append(nn.MaxPool2d(2))
                in_ch = ch
                
            self.encoders[f'scale_{scale}'] = encoder_layers
            self.pools[f'scale_{scale}'] = pool_layers
        
        # Feature fusion modules
        self.fusion_modules = nn.ModuleDict({
            'level_1': FeatureFusion([base_channels] * len(scales), base_channels),
            'level_2': FeatureFusion([base_channels*2] * len(scales), base_channels*2),
            'level_3': FeatureFusion([base_channels*4] * len(scales), base_channels*4),
            'level_4': FeatureFusion([base_channels*8] * len(scales), base_channels*8),
        })
        
        # Rainfall attention mechanism
        self.rainfall_attention = AttentionBlock(base_channels*8)
        
        # Decoder path
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2),
            ConvBlock(base_channels*8, base_channels*4, dropout),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2),
            ConvBlock(base_channels*4, base_channels*2, dropout),
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2),
            ConvBlock(base_channels*2, base_channels, dropout),
        ])
        
        # Output layer
        self.output_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.output_activation = nn.ReLU()  # Flood depth should be non-negative
        
        # Physics constraint module
        self.physics_module = PhysicsConstraintModule(grid_spacing)
        
    def forward(
        self, 
        x: torch.Tensor, 
        rainfall: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            rainfall: Rainfall data for physics constraints (B, 1, H, W)
            
        Returns:
            flood_depth: Predicted flood depth (B, 1, H, W)
            physics_loss: Mass conservation constraint loss
        """
        batch_size, _, height, width = x.shape
        
        # Multi-scale feature extraction
        scale_features = {f'level_{i+1}': [] for i in range(4)}
        
        for scale in self.scales:
            # Scale input if needed
            if scale > 1:
                scaled_x = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            else:
                scaled_x = x
                
            # Encoder path
            features = []
            current = scaled_x
            
            encoder_layers = self.encoders[f'scale_{scale}']
            pool_layers = self.pools[f'scale_{scale}']
            
            for i, (conv_block, pool) in enumerate(zip(encoder_layers[:-1], pool_layers)):
                current = conv_block(current)
                features.append(current)
                current = pool(current)
                
            # Last encoder layer (no pooling)
            current = encoder_layers[-1](current)
            features.append(current)
            
            # Store features for fusion
            for j, feat in enumerate(features):
                scale_features[f'level_{j+1}'].append(feat)
        
        # Feature fusion at each level
        fused_features = []
        target_sizes = [(height//8, width//8), (height//4, width//4), 
                       (height//2, width//2), (height, width)]
        
        for i, level in enumerate(['level_4', 'level_3', 'level_2', 'level_1']):
            fused = self.fusion_modules[level](
                scale_features[level], 
                target_sizes[i]
            )
            fused_features.append(fused)
        
        # Apply rainfall attention to bottleneck features
        fused_features[0] = self.rainfall_attention(fused_features[0])
        
        # Decoder path with skip connections
        current = fused_features[0]  # Start from bottleneck
        
        # Decoder blocks
        for i in range(0, len(self.decoder), 2):
            # Upsampling
            current = self.decoder[i](current)
            
            # Skip connection
            skip_idx = (i // 2) + 1
            if skip_idx < len(fused_features):
                skip_features = fused_features[skip_idx]
                # Ensure spatial dimensions match
                if current.shape[-2:] != skip_features.shape[-2:]:
                    current = F.interpolate(
                        current, 
                        size=skip_features.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                current = torch.cat([current, skip_features], dim=1)
            
            # Convolution block
            current = self.decoder[i + 1](current)
        
        # Output prediction
        flood_depth = self.output_activation(self.output_conv(current))
        
        # Compute physics constraint loss
        physics_loss = torch.tensor(0.0, device=x.device)
        if rainfall is not None and self.training:
            physics_loss = self.physics_module.mass_conservation_loss(
                flood_depth, rainfall
            )
        
        return flood_depth, physics_loss
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_flood_cnn_model(
    input_channels: int = 4,
    output_channels: int = 1,
    base_channels: int = 64,
    scales: List[int] = [1, 2, 4],
    dropout: float = 0.1,
    grid_spacing: float = 1.0
) -> MultiScaleUNet:
    """
    Factory function to create FloodCNN model.
    
    Args:
        input_channels: Number of input channels (DEM, land use, soil, rainfall)
        output_channels: Number of output channels (flood depth)
        base_channels: Base number of channels in the network
        scales: Multi-scale factors for input processing
        dropout: Dropout rate for regularization
        grid_spacing: Spatial grid spacing for physics constraints
        
    Returns:
        MultiScaleUNet model instance
    """
    model = MultiScaleUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        scales=scales,
        dropout=dropout,
        grid_spacing=grid_spacing
    )
    
    print(f"Created FloodCNN model with {model.get_num_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_flood_cnn_model().to(device)
    
    # Test input
    batch_size = 2
    height, width = 256, 256
    input_channels = 4
    
    x = torch.randn(batch_size, input_channels, height, width).to(device)
    rainfall = torch.randn(batch_size, 1, height, width).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        flood_depth, physics_loss = model(x, rainfall)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {flood_depth.shape}")
    print(f"Physics loss: {physics_loss.item():.6f}")
    print(f"Output range: [{flood_depth.min().item():.3f}, {flood_depth.max().item():.3f}]")