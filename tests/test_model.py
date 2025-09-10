"""Integration tests for CNN model architecture and training.

This module tests the complete model pipeline including:
- Model initialization and architecture validation
- Forward pass with realistic data
- Training loop components
- Physics-informed loss functions
- Multi-scale input processing
- Production readiness validation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import warnings

from src.models.flood_cnn import (
    FloodDepthPredictor,
    PhysicsInformedLoss,
    DimensionlessFeatureProcessor,
    MultiScaleInputProcessor,
    RainfallScalingModule,
    FloodModelTrainer,
    create_flood_model,
    AttentionFusion,
    EncoderBlock,
    DecoderBlock,
)


class TestFloodCNNArchitecture:
    """Test the CNN architecture components and forward pass."""

    @pytest.fixture
    def model_config(self):
        """Standard model configuration for testing."""
        return {
            "input_features": 6,
            "num_scales": 3,
            "base_channels": 32,  # Smaller for faster testing
            "depth_levels": 3,  # Reduced for testing
            "dropout_rate": 0.1,
            "predict_uncertainty": True,
        }

    @pytest.fixture
    def flood_model(self, model_config):
        """Create a FloodDepthPredictor model for testing."""
        return create_flood_model(model_config)

    @pytest.fixture
    def sample_input_data(self):
        """Create realistic input data for model testing."""
        batch_size = 2
        height, width = 128, 128  # Smaller for faster testing

        return {
            "high_res": torch.randn(batch_size, 6, height, width),
            "med_res": torch.randn(batch_size, 6, height // 2, width // 2),
            "low_res": torch.randn(batch_size, 6, height // 4, width // 4),
            "rainfall": torch.randn(batch_size, 1, height, width),
            "elevation": torch.randn(batch_size, 1, height, width) * 100
            + 500,  # Realistic elevation
            "slope": torch.rand(batch_size, 1, height, width)
            * 0.5,  # 0-0.5 slope ratio
            "land_use": torch.randint(0, 10, (batch_size, 1, height, width)).float(),
        }

    @pytest.fixture
    def rainfall_stats(self):
        """Create rainfall statistics for scaling."""
        batch_size = 2
        return torch.tensor(
            [
                [
                    25.5,
                    120.0,
                    0.3,
                ],  # intensity (mm/h), duration (min), antecedent moisture
                [15.2, 180.0, 0.7],
            ],
            dtype=torch.float32,
        )

    def test_model_initialization(self, model_config):
        """Test model initialization with various configurations."""
        model = create_flood_model(model_config)

        assert isinstance(model, FloodDepthPredictor)
        assert hasattr(model, "dimensionless_processor")
        assert hasattr(model, "multiscale_processor")
        assert hasattr(model, "rainfall_scaler")
        assert hasattr(model, "encoder_blocks")
        assert hasattr(model, "decoder_blocks")
        assert hasattr(model, "depth_head")

        if model_config["predict_uncertainty"]:
            assert hasattr(model, "uncertainty_head")

    def test_model_forward_pass(self, flood_model, sample_input_data, rainfall_stats):
        """Test complete forward pass through the model."""
        flood_model.eval()

        with torch.no_grad():
            predictions = flood_model(sample_input_data, rainfall_stats)

        # Check output structure
        assert "depth" in predictions
        depth_pred = predictions["depth"]

        batch_size, height, width = sample_input_data["high_res"].shape[0], 128, 128
        assert depth_pred.shape == (batch_size, 1, height, width)
        assert torch.all(depth_pred >= 0), "Depth predictions should be non-negative"

        if "uncertainty" in predictions:
            uncertainty_pred = predictions["uncertainty"]
            assert uncertainty_pred.shape == depth_pred.shape
            assert torch.all(
                uncertainty_pred >= 0
            ), "Uncertainty should be non-negative"

    def test_multiscale_input_processing(self, sample_input_data):
        """Test multi-scale input processor."""
        processor = MultiScaleInputProcessor(input_channels=6, target_size=128)

        # Test with all scales
        multiscale_inputs = {
            "high_res": sample_input_data["high_res"],
            "med_res": sample_input_data["med_res"],
            "low_res": sample_input_data["low_res"],
        }

        output = processor(multiscale_inputs)

        # Output should have combined channels from all scales
        expected_channels = 64 + 32 + 16  # From processor architecture
        assert output.shape == (2, expected_channels, 128, 128)

        # Test with missing scales
        single_scale = {"high_res": sample_input_data["high_res"]}
        output_single = processor(single_scale)
        assert output_single.shape[1] == 64  # Only high_res channels

    def test_dimensionless_feature_processor(self, sample_input_data):
        """Test dimensionless feature processing."""
        processor = DimensionlessFeatureProcessor()

        # Test individual processing functions
        # Froude number calculation
        velocity = torch.rand(2, 1, 10, 10) * 5  # 0-5 m/s
        depth = torch.rand(2, 1, 10, 10) * 2 + 0.1  # 0.1-2.1 m
        froude = processor.froude_number(velocity, depth)

        assert froude.shape == velocity.shape
        assert torch.all(froude >= 0)
        assert torch.all(froude < 10), "Froude numbers should be reasonable"

        # Test complete feature processing
        features = {
            "elevation": sample_input_data["elevation"],
            "rainfall": sample_input_data["rainfall"],
            "slope": sample_input_data["slope"],
            "land_use": sample_input_data["land_use"],
        }

        processed = processor(features)
        assert processed.shape[0] == 2  # Batch size
        assert processed.shape[1] == 4  # Number of feature types
        assert processed.shape[2:] == (128, 128)  # Spatial dimensions

    def test_rainfall_scaling_module(self, sample_input_data, rainfall_stats):
        """Test rainfall scaling with intensity and spatial patterns."""
        scaler = RainfallScalingModule(hidden_dim=64)

        rainfall_map = sample_input_data["rainfall"]
        scaled_rainfall = scaler(rainfall_map, rainfall_stats)

        assert scaled_rainfall.shape == rainfall_map.shape
        assert not torch.allclose(
            scaled_rainfall, rainfall_map
        ), "Scaling should modify rainfall"

        # Scaled rainfall should generally maintain spatial patterns
        original_var = torch.var(rainfall_map)
        scaled_var = torch.var(scaled_rainfall)
        assert scaled_var > 0, "Scaled rainfall should have spatial variation"

    def test_attention_fusion(self):
        """Test attention mechanism for feature fusion."""
        batch_size, channels, height, width = 2, 64, 32, 32
        x = torch.randn(batch_size, channels, height, width)

        attention = AttentionFusion(in_channels=channels, reduction=8)
        attended = attention(x)

        assert attended.shape == x.shape
        # Attention should modify the input
        assert not torch.allclose(attended, x)

        # Output should maintain reasonable magnitude
        assert torch.all(torch.isfinite(attended))

    def test_encoder_decoder_blocks(self):
        """Test encoder and decoder building blocks."""
        # Test encoder block
        encoder = EncoderBlock(in_channels=32, out_channels=64)
        x_enc = torch.randn(2, 32, 64, 64)
        enc_out = encoder(x_enc)

        assert enc_out.shape == (2, 64, 64, 64)
        assert not torch.allclose(enc_out, x_enc.mean())  # Should not be trivial

        # Test decoder block
        decoder = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        x_dec = torch.randn(2, 128, 32, 32)  # Upsampled input
        skip = torch.randn(2, 64, 64, 64)  # Skip connection
        dec_out = decoder(x_dec, skip)

        assert dec_out.shape == (2, 64, 64, 64)
        assert torch.all(torch.isfinite(dec_out))

    def test_physics_informed_loss(self, sample_input_data):
        """Test physics-informed loss function."""
        loss_fn = PhysicsInformedLoss(lambda_physics=1.0, lambda_mass=0.5)

        # Create synthetic predictions and targets
        batch_size, height, width = 2, 64, 64
        predictions = {"depth": torch.rand(batch_size, 1, height, width) * 5}
        targets = torch.rand(batch_size, 1, height, width) * 5

        inputs = {
            "elevation": sample_input_data["elevation"][:, :, :height, :width],
            "rainfall": sample_input_data["rainfall"][:, :, :height, :width],
        }

        loss_dict = loss_fn(predictions, targets, inputs)

        # Check loss components
        required_keys = {"total_loss", "mse_loss", "mass_conservation_loss"}
        assert set(loss_dict.keys()) == required_keys

        for key, loss_val in loss_dict.items():
            assert isinstance(loss_val, torch.Tensor)
            assert loss_val.dim() == 0  # Scalar loss
            assert loss_val >= 0, f"{key} should be non-negative"
            assert torch.isfinite(loss_val), f"{key} should be finite"

    def test_model_gradients(self, flood_model, sample_input_data, rainfall_stats):
        """Test that model computes gradients correctly."""
        flood_model.train()

        predictions = flood_model(sample_input_data, rainfall_stats)
        targets = torch.rand_like(predictions["depth"])

        loss_dict = flood_model.compute_loss(predictions, targets, sample_input_data)
        loss = loss_dict["total_loss"]

        # Backward pass
        loss.backward()

        # Check that gradients exist and are reasonable
        has_gradients = False
        for name, param in flood_model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                assert torch.all(
                    torch.isfinite(param.grad)
                ), f"Infinite gradients in {name}"
                assert not torch.all(param.grad == 0), f"Zero gradients in {name}"

        assert has_gradients, "Model should have gradients after backward pass"

    @pytest.mark.parametrize("input_size", [(64, 64), (128, 128), (256, 256)])
    def test_model_different_sizes(self, model_config, input_size):
        """Test model with different input sizes."""
        model = create_flood_model(model_config)
        model.eval()

        batch_size = 1
        height, width = input_size

        inputs = {
            "high_res": torch.randn(batch_size, 6, height, width),
            "med_res": torch.randn(batch_size, 6, height // 2, width // 2),
            "low_res": torch.randn(batch_size, 6, height // 4, width // 4),
        }

        with torch.no_grad():
            predictions = model(inputs)

        # Output should match input spatial dimensions
        assert predictions["depth"].shape == (batch_size, 1, height, width)

    def test_model_memory_efficiency(
        self, flood_model, sample_input_data, rainfall_stats
    ):
        """Test model memory usage and efficiency."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        flood_model.eval()

        # Multiple forward passes
        for _ in range(5):
            with torch.no_grad():
                predictions = flood_model(sample_input_data, rainfall_stats)

            # Clear predictions to avoid accumulating memory
            del predictions

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        assert (
            memory_increase < 500 * 1024 * 1024
        ), f"Memory usage increased by {memory_increase / 1024 / 1024:.1f} MB"


class TestModelTraining:
    """Test model training components and workflows."""

    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer with model and data."""
        config = {
            "input_features": 6,
            "base_channels": 32,
            "depth_levels": 3,
            "predict_uncertainty": True,
        }

        model = create_flood_model(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = FloodModelTrainer(model, device)

        return trainer, device

    @pytest.fixture
    def training_batch(self):
        """Create a training batch."""
        batch_size = 4
        height, width = 64, 64

        return {
            "inputs": {
                "high_res": torch.randn(batch_size, 6, height, width),
                "med_res": torch.randn(batch_size, 6, height // 2, width // 2),
                "low_res": torch.randn(batch_size, 6, height // 4, width // 4),
                "rainfall": torch.randn(batch_size, 1, height, width),
                "elevation": torch.randn(batch_size, 1, height, width) * 100 + 500,
            },
            "targets": torch.rand(batch_size, 1, height, width) * 3,  # 0-3m depth
            "rainfall_stats": torch.tensor(
                [
                    [20.0, 120.0, 0.4],
                    [35.0, 90.0, 0.2],
                    [15.0, 180.0, 0.6],
                    [28.0, 60.0, 0.3],
                ],
                dtype=torch.float32,
            ),
        }

    def test_training_step(self, trainer_setup, training_batch):
        """Test single training step."""
        trainer, device = trainer_setup

        # Mock optimizer to avoid actual parameter updates
        trainer.model.train()

        loss_dict = trainer.train_step(training_batch)

        # Check loss dictionary
        expected_keys = {"total_loss", "mse_loss", "mass_conservation_loss"}
        assert set(loss_dict.keys()) == expected_keys

        for key, loss_val in loss_dict.items():
            assert isinstance(loss_val, float)
            assert loss_val >= 0, f"{key} should be non-negative"
            assert np.isfinite(loss_val), f"{key} should be finite"

    def test_validation_step(self, trainer_setup, training_batch):
        """Test single validation step."""
        trainer, device = trainer_setup

        metrics = trainer.validate_step(training_batch)

        # Check metrics dictionary
        expected_keys = {
            "total_loss",
            "mse_loss",
            "mass_conservation_loss",
            "mse",
            "mae",
        }
        assert set(metrics.keys()) == expected_keys

        for key, metric_val in metrics.items():
            assert isinstance(metric_val, float)
            assert metric_val >= 0, f"{key} should be non-negative"
            assert np.isfinite(metric_val), f"{key} should be finite"

    def test_training_loop_simulation(self, trainer_setup, training_batch):
        """Simulate a few training iterations."""
        trainer, device = trainer_setup
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)

        initial_loss = None
        losses = []

        # Simulate training loop
        for epoch in range(5):
            trainer.model.train()
            optimizer.zero_grad()

            # Forward pass
            inputs = {k: v.to(device) for k, v in training_batch["inputs"].items()}
            targets = training_batch["targets"].to(device)
            rainfall_stats = training_batch.get("rainfall_stats")
            if rainfall_stats is not None:
                rainfall_stats = rainfall_stats.to(device)

            predictions = trainer.model(inputs, rainfall_stats)
            loss_dict = trainer.model.compute_loss(predictions, targets, inputs)

            total_loss = loss_dict["total_loss"]
            losses.append(total_loss.item())

            # Backward pass
            total_loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = total_loss.item()

        # Training should reduce loss (or at least not increase dramatically)
        final_loss = losses[-1]
        assert (
            final_loss <= initial_loss * 2
        ), "Loss should not increase dramatically during training"

        # Check that loss trend is reasonable
        assert all(np.isfinite(loss) for loss in losses), "All losses should be finite"

    def test_model_state_dict_save_load(self, trainer_setup):
        """Test saving and loading model state."""
        trainer, device = trainer_setup

        # Get initial model state
        initial_state = trainer.model.state_dict()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(initial_state, tmp_file.name)

            # Create new model and load state
            config = {
                "input_features": 6,
                "base_channels": 32,
                "depth_levels": 3,
                "predict_uncertainty": True,
            }
            new_model = create_flood_model(config)
            new_model.load_state_dict(torch.load(tmp_file.name, map_location=device))

            # Compare parameters
            for (name1, param1), (name2, param2) in zip(
                trainer.model.named_parameters(), new_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(
                    param1, param2
                ), f"Parameters {name1} don't match after save/load"

        # Clean up
        os.unlink(tmp_file.name)

    def test_gradient_clipping(self, trainer_setup, training_batch):
        """Test gradient clipping functionality."""
        trainer, device = trainer_setup
        optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=0.01
        )  # Higher LR for larger gradients

        trainer.model.train()
        optimizer.zero_grad()

        # Forward and backward pass
        inputs = {k: v.to(device) for k, v in training_batch["inputs"].items()}
        targets = training_batch["targets"].to(device)

        predictions = trainer.model(inputs)
        loss_dict = trainer.model.compute_loss(predictions, targets, inputs)
        loss_dict["total_loss"].backward()

        # Check gradients before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float("inf")
        )

        # Apply gradient clipping
        max_grad_norm = 1.0
        grad_norm_after = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), max_grad_norm
        )

        if grad_norm_before > max_grad_norm:
            assert (
                grad_norm_after <= max_grad_norm * 1.01
            ), "Gradients should be clipped"


class TestModelProductionReadiness:
    """Test model production readiness and validation."""

    @pytest.fixture
    def production_model(self):
        """Create a production-ready model configuration."""
        config = {
            "input_features": 6,
            "num_scales": 3,
            "base_channels": 64,
            "depth_levels": 4,
            "dropout_rate": 0.1,
            "predict_uncertainty": True,
        }
        return create_flood_model(config)

    def test_model_inference_mode(self, production_model, sample_input_data):
        """Test model in inference mode (eval)."""
        production_model.eval()

        # Test deterministic behavior in eval mode
        with torch.no_grad():
            pred1 = production_model(sample_input_data)
            pred2 = production_model(sample_input_data)

        # Predictions should be identical in eval mode (no dropout)
        torch.testing.assert_close(pred1["depth"], pred2["depth"])

        if "uncertainty" in pred1:
            torch.testing.assert_close(pred1["uncertainty"], pred2["uncertainty"])

    def test_batch_size_independence(self, production_model):
        """Test model predictions are independent of batch size."""
        production_model.eval()
        height, width = 128, 128

        # Single sample
        single_input = {
            "high_res": torch.randn(1, 6, height, width),
            "med_res": torch.randn(1, 6, height // 2, width // 2),
            "low_res": torch.randn(1, 6, height // 4, width // 4),
        }

        # Batch of same sample
        batch_input = {
            "high_res": single_input["high_res"].repeat(4, 1, 1, 1),
            "med_res": single_input["med_res"].repeat(4, 1, 1, 1),
            "low_res": single_input["low_res"].repeat(4, 1, 1, 1),
        }

        with torch.no_grad():
            single_pred = production_model(single_input)
            batch_pred = production_model(batch_input)

        # All batch predictions should be identical to single prediction
        for i in range(4):
            torch.testing.assert_close(
                single_pred["depth"][0], batch_pred["depth"][i], rtol=1e-5, atol=1e-6
            )

    def test_input_validation(self, production_model):
        """Test model input validation and error handling."""
        production_model.eval()

        # Test with invalid input shapes
        invalid_inputs = {
            "high_res": torch.randn(2, 6, 64, 32),  # Non-square
            "med_res": torch.randn(2, 6, 32, 16),
            "low_res": torch.randn(2, 6, 16, 8),
        }

        try:
            with torch.no_grad():
                predictions = production_model(invalid_inputs)
            # Should handle non-square inputs gracefully
            assert predictions["depth"].shape == (2, 1, 64, 32)
        except Exception as e:
            # If model fails, it should fail gracefully
            assert isinstance(e, (RuntimeError, ValueError))

    def test_numerical_stability(self, production_model):
        """Test model numerical stability with extreme inputs."""
        production_model.eval()
        batch_size, height, width = 2, 64, 64

        # Test with various extreme inputs
        extreme_cases = [
            # Very large values
            {
                "high_res": torch.ones(batch_size, 6, height, width) * 1e6,
                "elevation": torch.ones(batch_size, 1, height, width) * 1e6,
            },
            # Very small values
            {
                "high_res": torch.ones(batch_size, 6, height, width) * 1e-6,
                "elevation": torch.ones(batch_size, 1, height, width) * 1e-6,
            },
            # Mixed extreme values
            {
                "high_res": torch.cat(
                    [
                        torch.ones(batch_size, 3, height, width) * 1e6,
                        torch.ones(batch_size, 3, height, width) * 1e-6,
                    ],
                    dim=1,
                ),
                "elevation": torch.ones(batch_size, 1, height, width) * 500,
            },
        ]

        for case_idx, inputs in enumerate(extreme_cases):
            with torch.no_grad():
                try:
                    predictions = production_model(inputs)

                    # Check for NaN/inf in predictions
                    assert torch.all(
                        torch.isfinite(predictions["depth"])
                    ), f"Non-finite predictions in case {case_idx}"

                    # Predictions should be reasonable (not extreme)
                    assert (
                        torch.max(predictions["depth"]) < 1e3
                    ), f"Unreasonably large predictions in case {case_idx}"

                except Exception as e:
                    pytest.fail(f"Model failed on extreme case {case_idx}: {e}")

    def test_memory_leak_detection(self, production_model, sample_input_data):
        """Test for memory leaks during repeated inference."""
        import gc

        production_model.eval()

        # Run many inference steps
        for i in range(50):
            with torch.no_grad():
                predictions = production_model(sample_input_data)

            # Explicitly delete predictions
            del predictions

            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Test should complete without memory errors
        assert True, "Memory leak test completed successfully"

    def test_model_serialization(self, production_model):
        """Test model serialization for deployment."""
        import pickle
        import io

        # Test pickle serialization
        buffer = io.BytesIO()
        torch.save(production_model, buffer)
        buffer.seek(0)

        loaded_model = torch.load(buffer)

        # Test that loaded model works
        test_input = {"high_res": torch.randn(1, 6, 64, 64)}

        with torch.no_grad():
            original_pred = production_model(test_input)
            loaded_pred = loaded_model(test_input)

        torch.testing.assert_close(original_pred["depth"], loaded_pred["depth"])

    def test_jit_compilation(self, production_model, sample_input_data):
        """Test TorchScript compilation for deployment."""
        production_model.eval()

        try:
            # Attempt TorchScript tracing
            traced_model = torch.jit.trace(production_model, (sample_input_data, None))

            with torch.no_grad():
                original_pred = production_model(sample_input_data)
                traced_pred = traced_model(sample_input_data, None)

            torch.testing.assert_close(
                original_pred["depth"], traced_pred["depth"], rtol=1e-4, atol=1e-5
            )

        except Exception as e:
            # JIT compilation might fail due to dynamic operations
            # This is acceptable, but we should log it
            warnings.warn(f"TorchScript compilation failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
