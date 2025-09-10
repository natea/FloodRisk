"""
Unit tests for FloodCNN model architecture.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, Mock
import numpy as np

from src.models.flood_cnn import (
    PhysicsInformedLoss,
    MultiScaleEncoder,
    AttentionDecoder,
    RainfallScaling,
    FloodCNN,
)


class TestPhysicsInformedLoss:
    """Test suite for PhysicsInformedLoss."""

    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = PhysicsInformedLoss(lambda_physics=1.5, lambda_mass=0.8)
        assert loss_fn.lambda_physics == 1.5
        assert loss_fn.lambda_mass == 0.8
        assert isinstance(loss_fn.mse_loss, nn.MSELoss)

    def test_mass_conservation_loss(self, sample_flood_target):
        """Test mass conservation loss calculation."""
        loss_fn = PhysicsInformedLoss()
        batch_size, channels, height, width = sample_flood_target.shape

        # Create mock elevation data
        elevation = torch.rand(batch_size, 1, height, width) * 100

        # Calculate mass conservation loss
        mass_loss = loss_fn.mass_conservation_loss(sample_flood_target, elevation)

        assert isinstance(mass_loss, torch.Tensor)
        assert mass_loss.dim() == 0  # Scalar tensor
        assert mass_loss >= 0  # Loss should be non-negative

    def test_forward_pass(self, sample_flood_target):
        """Test forward pass of physics-informed loss."""
        loss_fn = PhysicsInformedLoss()
        batch_size, channels, height, width = sample_flood_target.shape

        # Create predictions and targets
        pred = sample_flood_target + torch.randn_like(sample_flood_target) * 0.1
        target = sample_flood_target
        elevation = torch.rand(batch_size, 1, height, width) * 100

        # Calculate total loss
        total_loss = loss_fn(pred, target, elevation)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss >= 0


class TestMultiScaleEncoder:
    """Test suite for MultiScaleEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = MultiScaleEncoder(
            in_channels=[1, 1, 1, 4],  # 3 elevation scales + terrain features
            base_filters=32,
        )

        # Check that encoders were created for each scale
        assert len(encoder.scale_encoders) == 4
        assert encoder.fusion_conv is not None

    def test_forward_pass(self, sample_flood_cnn_input):
        """Test encoder forward pass."""
        encoder = MultiScaleEncoder(in_channels=[1, 1, 1, 4], base_filters=32)

        inputs = [
            sample_flood_cnn_input["elevation_256m"],
            sample_flood_cnn_input["elevation_512m"],
            sample_flood_cnn_input["elevation_1024m"],
            sample_flood_cnn_input["terrain_features"],
        ]

        features = encoder(inputs)

        # Check output shapes and types
        assert isinstance(features, list)
        assert len(features) > 0
        for feat in features:
            assert isinstance(feat, torch.Tensor)
            assert feat.dim() == 4  # Batch, Channel, Height, Width


class TestAttentionDecoder:
    """Test suite for AttentionDecoder."""

    def test_initialization(self):
        """Test decoder initialization."""
        decoder = AttentionDecoder(
            feature_channels=[512, 256, 128, 64], output_channels=1
        )

        assert decoder.output_conv is not None
        assert len(decoder.decoder_blocks) > 0

    def test_forward_pass(self):
        """Test decoder forward pass."""
        decoder = AttentionDecoder(feature_channels=[128, 64, 32], output_channels=1)

        # Create mock feature maps
        features = [
            torch.rand(2, 128, 32, 32),  # Smallest feature map
            torch.rand(2, 64, 64, 64),  # Medium feature map
            torch.rand(2, 32, 128, 128),  # Largest feature map
        ]

        output = decoder(features)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 1, 128, 128)  # Match largest input


class TestRainfallScaling:
    """Test suite for RainfallScaling."""

    def test_initialization(self):
        """Test rainfall scaling initialization."""
        scaling = RainfallScaling(feature_dim=256, rainfall_dim=1)

        assert scaling.rainfall_encoder is not None
        assert scaling.scaling_factor is not None

    def test_forward_pass(self):
        """Test rainfall scaling forward pass."""
        scaling = RainfallScaling(feature_dim=64, rainfall_dim=1)

        # Create mock inputs
        features = torch.rand(2, 64, 32, 32)
        rainfall = torch.rand(2, 1)

        scaled_features = scaling(features, rainfall)

        assert isinstance(scaled_features, torch.Tensor)
        assert scaled_features.shape == features.shape


class TestFloodCNN:
    """Test suite for complete FloodCNN model."""

    def test_initialization(self):
        """Test model initialization."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=32
        )

        assert model.encoder is not None
        assert model.decoder is not None
        assert model.rainfall_scaling is not None

    def test_forward_pass(self, sample_flood_cnn_input, pytorch_device):
        """Test complete model forward pass."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=32
        )
        model.to(pytorch_device)

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}

        # Forward pass
        outputs = model(
            elevation_256m=inputs["elevation_256m"],
            elevation_512m=inputs["elevation_512m"],
            elevation_1024m=inputs["elevation_1024m"],
            terrain_features=inputs["terrain_features"],
            rainfall=inputs["rainfall"],
        )

        assert isinstance(outputs, dict)
        assert "flood_depth" in outputs
        assert outputs["flood_depth"].shape == (2, 1, 256, 256)

    def test_model_training_step(
        self, sample_flood_cnn_input, sample_flood_target, pytorch_device
    ):
        """Test model training with loss calculation."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1],
            terrain_channels=4,
            base_filters=16,  # Smaller for faster testing
        )
        model.to(pytorch_device)

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}
        target = sample_flood_target.to(pytorch_device)

        # Create loss function
        loss_fn = PhysicsInformedLoss()

        # Forward pass
        model.train()
        outputs = model(**inputs)

        # Calculate loss
        loss = loss_fn(outputs["flood_depth"], target, inputs["elevation_256m"])

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss > 0

    @pytest.mark.slow
    def test_model_gradient_flow(
        self, sample_flood_cnn_input, sample_flood_target, pytorch_device
    ):
        """Test that gradients flow through the model properly."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=16
        )
        model.to(pytorch_device)

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}
        target = sample_flood_target.to(pytorch_device)

        # Enable gradients
        for param in model.parameters():
            param.requires_grad_(True)

        # Forward and backward pass
        model.train()
        outputs = model(**inputs)
        loss = nn.MSELoss()(outputs["flood_depth"], target)
        loss.backward()

        # Check that gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_model_inference_mode(self, sample_flood_cnn_input, pytorch_device):
        """Test model in inference mode."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=16
        )
        model.to(pytorch_device)
        model.eval()

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert isinstance(outputs, dict)
        assert "flood_depth" in outputs
        assert not outputs["flood_depth"].requires_grad

    def test_model_output_constraints(self, sample_flood_cnn_input, pytorch_device):
        """Test that model outputs satisfy physical constraints."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=16
        )
        model.to(pytorch_device)
        model.eval()

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        flood_depth = outputs["flood_depth"]

        # Check physical constraints
        assert torch.all(flood_depth >= 0), "Flood depth should be non-negative"
        assert torch.all(torch.isfinite(flood_depth)), "Flood depth should be finite"

    def test_model_with_different_input_sizes(self, pytorch_device):
        """Test model with different input sizes."""
        model = FloodCNN(
            elevation_channels=[1, 1, 1], terrain_channels=4, base_filters=16
        )
        model.to(pytorch_device)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            inputs = {
                "elevation_256m": torch.rand(batch_size, 1, 256, 256).to(
                    pytorch_device
                ),
                "elevation_512m": torch.rand(batch_size, 1, 128, 128).to(
                    pytorch_device
                ),
                "elevation_1024m": torch.rand(batch_size, 1, 64, 64).to(pytorch_device),
                "terrain_features": torch.rand(batch_size, 4, 256, 256).to(
                    pytorch_device
                ),
                "rainfall": torch.rand(batch_size, 1).to(pytorch_device),
            }

            outputs = model(**inputs)
            assert outputs["flood_depth"].shape == (batch_size, 1, 256, 256)


@pytest.mark.integration
class TestFloodCNNIntegration:
    """Integration tests for FloodCNN with realistic scenarios."""

    def test_end_to_end_prediction(self, sample_flood_cnn_input, pytorch_device):
        """Test end-to-end flood prediction workflow."""
        model = FloodCNN()
        model.to(pytorch_device)

        # Move inputs to device
        inputs = {k: v.to(pytorch_device) for k, v in sample_flood_cnn_input.items()}

        # Prediction
        model.eval()
        with torch.no_grad():
            predictions = model(**inputs)

        # Validation
        assert "flood_depth" in predictions
        flood_depth = predictions["flood_depth"]

        # Check output properties
        assert flood_depth.shape[0] == inputs["elevation_256m"].shape[0]  # Batch size
        assert flood_depth.shape[1] == 1  # Single channel (depth)
        assert torch.all(flood_depth >= 0)  # Non-negative depths
        assert torch.all(flood_depth <= 50)  # Reasonable maximum depth (50m)
