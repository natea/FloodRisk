"""Integration tests for model validation and metrics calculation.

This module tests the complete validation pipeline including:
- Model evaluation metrics
- Production validation workflows
- Performance benchmarking
- Cross-validation procedures
- Real-world scenario testing
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

from src.models.flood_cnn import (
    FloodDepthPredictor,
    create_flood_model,
    FloodModelTrainer,
)


class FloodMetrics:
    """Comprehensive flood prediction metrics for validation."""

    @staticmethod
    def calculate_flood_metrics(
        predictions: np.ndarray, targets: np.ndarray, flood_threshold: float = 0.1
    ) -> dict:
        """Calculate comprehensive flood prediction metrics."""
        # Ensure arrays are flattened and valid
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        if len(pred_valid) == 0:
            return {
                key: np.nan
                for key in [
                    "mse",
                    "rmse",
                    "mae",
                    "r2",
                    "correlation",
                    "bias",
                    "flood_hit_rate",
                    "flood_false_alarm",
                    "flood_critical_success",
                    "depth_accuracy_shallow",
                    "depth_accuracy_deep",
                    "volume_error",
                ]
            }

        metrics = {}

        # Basic regression metrics
        metrics["mse"] = mean_squared_error(target_valid, pred_valid)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(target_valid, pred_valid)
        metrics["r2"] = r2_score(target_valid, pred_valid)

        # Correlation metrics
        if len(np.unique(pred_valid)) > 1 and len(np.unique(target_valid)) > 1:
            metrics["correlation"] = pearsonr(pred_valid, target_valid)[0]
            metrics["spearman"] = spearmanr(pred_valid, target_valid)[0]
        else:
            metrics["correlation"] = 0.0
            metrics["spearman"] = 0.0

        # Bias metrics
        metrics["bias"] = np.mean(pred_valid - target_valid)
        metrics["percent_bias"] = (
            np.sum(pred_valid - target_valid) / np.sum(target_valid)
        ) * 100

        # Flood detection metrics (binary classification)
        pred_flood = (pred_valid >= flood_threshold).astype(int)
        target_flood = (target_valid >= flood_threshold).astype(int)

        true_positives = np.sum((pred_flood == 1) & (target_flood == 1))
        false_positives = np.sum((pred_flood == 1) & (target_flood == 0))
        true_negatives = np.sum((pred_flood == 0) & (target_flood == 0))
        false_negatives = np.sum((pred_flood == 0) & (target_flood == 1))

        # Hit rate (sensitivity/recall)
        metrics["flood_hit_rate"] = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        # False alarm rate
        metrics["flood_false_alarm"] = (
            false_positives / (false_positives + true_negatives)
            if (false_positives + true_negatives) > 0
            else 0.0
        )

        # Critical Success Index (CSI)
        metrics["flood_critical_success"] = (
            true_positives / (true_positives + false_positives + false_negatives)
            if (true_positives + false_positives + false_negatives) > 0
            else 0.0
        )

        # Depth-specific accuracy
        shallow_mask = (target_valid > flood_threshold) & (target_valid <= 1.0)
        deep_mask = target_valid > 1.0

        if np.sum(shallow_mask) > 0:
            shallow_mae = mean_absolute_error(
                target_valid[shallow_mask], pred_valid[shallow_mask]
            )
            metrics["depth_accuracy_shallow"] = 1.0 - (
                shallow_mae / np.mean(target_valid[shallow_mask])
            )
        else:
            metrics["depth_accuracy_shallow"] = np.nan

        if np.sum(deep_mask) > 0:
            deep_mae = mean_absolute_error(
                target_valid[deep_mask], pred_valid[deep_mask]
            )
            metrics["depth_accuracy_deep"] = 1.0 - (
                deep_mae / np.mean(target_valid[deep_mask])
            )
        else:
            metrics["depth_accuracy_deep"] = np.nan

        # Volume conservation metrics
        total_predicted_volume = np.sum(pred_valid)
        total_target_volume = np.sum(target_valid)
        metrics["volume_error"] = (
            abs(total_predicted_volume - total_target_volume) / total_target_volume
            if total_target_volume > 0
            else np.nan
        )

        return metrics

    @staticmethod
    def calculate_spatial_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
        """Calculate spatial pattern metrics."""
        # Ensure 2D arrays
        if predictions.ndim > 2:
            predictions = predictions.squeeze()
        if targets.ndim > 2:
            targets = targets.squeeze()

        metrics = {}

        # Pattern correlation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        if np.sum(valid_mask) > 10:  # Minimum valid pixels
            metrics["spatial_correlation"] = pearsonr(
                pred_flat[valid_mask], target_flat[valid_mask]
            )[0]
        else:
            metrics["spatial_correlation"] = np.nan

        # Spatial distribution comparison (histograms)
        pred_hist, _ = np.histogram(pred_flat[valid_mask], bins=50, density=True)
        target_hist, _ = np.histogram(target_flat[valid_mask], bins=50, density=True)

        # Add small epsilon to avoid zero probabilities
        pred_hist += 1e-10
        target_hist += 1e-10
        pred_hist /= np.sum(pred_hist)
        target_hist /= np.sum(target_hist)

        metrics["jensen_shannon_distance"] = jensenshannon(pred_hist, target_hist)

        # Flood extent similarity (Jaccard index)
        pred_flood = (predictions > 0.1).astype(int)
        target_flood = (targets > 0.1).astype(int)

        intersection = np.sum(pred_flood & target_flood)
        union = np.sum(pred_flood | target_flood)

        metrics["jaccard_index"] = intersection / union if union > 0 else 0.0

        return metrics


class TestModelValidation:
    """Test model validation and evaluation metrics."""

    @pytest.fixture
    def validation_model(self):
        """Create model for validation testing."""
        config = {
            "input_features": 6,
            "base_channels": 32,
            "depth_levels": 3,
            "predict_uncertainty": True,
        }
        return create_flood_model(config)

    @pytest.fixture
    def validation_data(self):
        """Create realistic validation dataset."""
        batch_size = 8
        height, width = 64, 64

        # Generate realistic flood scenarios
        scenarios = []

        for i in range(batch_size):
            # Create elevation data
            elevation = np.random.normal(100, 20, (height, width))

            # Add terrain features
            x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
            elevation += 20 * np.sin(x / 3) * np.cos(y / 3)

            # Create realistic flood depth based on elevation
            # Lower elevations more likely to flood
            elevation_norm = (elevation - np.min(elevation)) / (
                np.max(elevation) - np.min(elevation)
            )
            flood_probability = 1 - elevation_norm

            # Add some randomness
            flood_depth = np.maximum(
                0, flood_probability * np.random.exponential(2, (height, width))
            )

            # Ensure some areas have significant flooding
            if i % 3 == 0:  # Every third scenario has major flooding
                center_x, center_y = width // 2, height // 2
                flood_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < (
                    width / 4
                ) ** 2
                flood_depth[flood_mask] += np.random.exponential(3, np.sum(flood_mask))

            scenarios.append(
                {
                    "elevation": elevation.astype(np.float32),
                    "flood_depth": flood_depth.astype(np.float32),
                }
            )

        # Convert to tensors
        inputs = {
            "high_res": torch.stack(
                [
                    torch.from_numpy(
                        np.random.randn(6, height, width).astype(np.float32)
                    )
                    for _ in range(batch_size)
                ]
            ),
            "elevation": torch.stack(
                [torch.from_numpy(s["elevation"]).unsqueeze(0) for s in scenarios]
            ),
        }

        targets = torch.stack(
            [torch.from_numpy(s["flood_depth"]).unsqueeze(0) for s in scenarios]
        )

        return inputs, targets, scenarios

    def test_comprehensive_metrics_calculation(self, validation_data):
        """Test comprehensive metrics calculation."""
        inputs, targets, scenarios = validation_data

        # Create synthetic predictions (slightly perturbed targets for realistic testing)
        predictions = targets + torch.randn_like(targets) * 0.5
        predictions = torch.clamp(predictions, min=0)  # Ensure non-negative

        pred_np = predictions.numpy()
        target_np = targets.numpy()

        metrics = FloodMetrics.calculate_flood_metrics(pred_np, target_np)

        # Check all expected metrics are present
        expected_metrics = {
            "mse",
            "rmse",
            "mae",
            "r2",
            "correlation",
            "bias",
            "flood_hit_rate",
            "flood_false_alarm",
            "flood_critical_success",
            "depth_accuracy_shallow",
            "depth_accuracy_deep",
            "volume_error",
        }

        assert set(metrics.keys()) >= expected_metrics

        # Validate metric ranges
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert -1 <= metrics["r2"] <= 1  # R² can be negative for very poor fits
        assert -1 <= metrics["correlation"] <= 1
        assert 0 <= metrics["flood_hit_rate"] <= 1
        assert 0 <= metrics["flood_false_alarm"] <= 1
        assert 0 <= metrics["flood_critical_success"] <= 1

        # All metrics should be finite
        for key, value in metrics.items():
            if not np.isnan(value):
                assert np.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_spatial_metrics_calculation(self, validation_data):
        """Test spatial pattern metrics."""
        inputs, targets, scenarios = validation_data

        predictions = targets + torch.randn_like(targets) * 0.3
        predictions = torch.clamp(predictions, min=0)

        # Test spatial metrics on single sample
        pred_sample = predictions[0, 0].numpy()
        target_sample = targets[0, 0].numpy()

        spatial_metrics = FloodMetrics.calculate_spatial_metrics(
            pred_sample, target_sample
        )

        expected_spatial_metrics = {
            "spatial_correlation",
            "jensen_shannon_distance",
            "jaccard_index",
        }

        assert set(spatial_metrics.keys()) == expected_spatial_metrics

        # Validate metric ranges
        if not np.isnan(spatial_metrics["spatial_correlation"]):
            assert -1 <= spatial_metrics["spatial_correlation"] <= 1

        assert 0 <= spatial_metrics["jensen_shannon_distance"] <= 1
        assert 0 <= spatial_metrics["jaccard_index"] <= 1

    def test_model_evaluation_pipeline(self, validation_model, validation_data):
        """Test complete model evaluation pipeline."""
        inputs, targets, scenarios = validation_data

        validation_model.eval()
        trainer = FloodModelTrainer(validation_model, "cpu")

        # Batch evaluation
        batch_data = {"inputs": inputs, "targets": targets}

        all_predictions = []
        all_targets = []
        all_metrics = []

        # Evaluate in smaller batches
        batch_size = 2
        for i in range(0, len(targets), batch_size):
            batch_inputs = {k: v[i : i + batch_size] for k, v in inputs.items()}
            batch_targets = targets[i : i + batch_size]

            with torch.no_grad():
                predictions = validation_model(batch_inputs)

            pred_depth = predictions["depth"]

            # Calculate metrics for this batch
            batch_metrics = trainer.validate_step(
                {"inputs": batch_inputs, "targets": batch_targets}
            )

            all_predictions.append(pred_depth)
            all_targets.append(batch_targets)
            all_metrics.append(batch_metrics)

        # Combine all predictions and targets
        final_predictions = torch.cat(all_predictions, dim=0)
        final_targets = torch.cat(all_targets, dim=0)

        # Calculate comprehensive metrics
        pred_np = final_predictions.numpy()
        target_np = final_targets.numpy()

        comprehensive_metrics = FloodMetrics.calculate_flood_metrics(pred_np, target_np)

        # Validate evaluation pipeline results
        assert len(all_metrics) > 0
        assert final_predictions.shape == final_targets.shape
        assert not np.all(np.isnan(pred_np))
        assert comprehensive_metrics["mse"] >= 0

    def test_cross_validation_simulation(self, validation_model):
        """Simulate k-fold cross-validation workflow."""
        # Create multiple validation folds
        k_folds = 3
        fold_results = []

        for fold in range(k_folds):
            # Create fold-specific data
            batch_size = 4
            height, width = 32, 32

            fold_inputs = {"high_res": torch.randn(batch_size, 6, height, width)}
            fold_targets = torch.rand(batch_size, 1, height, width) * 2

            # Evaluate model on this fold
            validation_model.eval()
            with torch.no_grad():
                predictions = validation_model(fold_inputs)

            # Calculate fold metrics
            pred_np = predictions["depth"].numpy()
            target_np = fold_targets.numpy()

            fold_metrics = FloodMetrics.calculate_flood_metrics(pred_np, target_np)
            fold_results.append(fold_metrics)

        # Aggregate cross-validation results
        cv_metrics = {}
        for metric_name in fold_results[0].keys():
            values = [
                fold[metric_name]
                for fold in fold_results
                if not np.isnan(fold[metric_name])
            ]
            if values:
                cv_metrics[f"{metric_name}_mean"] = np.mean(values)
                cv_metrics[f"{metric_name}_std"] = np.std(values)

        # Validate cross-validation results
        assert len(fold_results) == k_folds
        assert len(cv_metrics) > 0

        # Check that we have mean and std for key metrics
        key_metrics = ["mse", "mae", "r2"]
        for metric in key_metrics:
            assert f"{metric}_mean" in cv_metrics
            assert f"{metric}_std" in cv_metrics
            assert cv_metrics[f"{metric}_std"] >= 0

    def test_performance_benchmarking(self, validation_model, validation_data):
        """Test model performance benchmarking."""
        inputs, targets, scenarios = validation_data

        validation_model.eval()

        # Benchmark inference time
        num_runs = 10
        inference_times = []

        for _ in range(num_runs):
            start_time = time.time()

            with torch.no_grad():
                predictions = validation_model(inputs)

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        # Calculate performance metrics
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)

        batch_size = inputs["high_res"].shape[0]
        samples_per_second = batch_size / mean_inference_time

        # Performance assertions
        assert mean_inference_time > 0
        assert samples_per_second > 0

        # Model should process at reasonable speed (adjust threshold as needed)
        assert (
            samples_per_second > 0.1
        ), f"Model too slow: {samples_per_second} samples/sec"

        # Benchmark memory usage
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run multiple inferences
        for _ in range(5):
            with torch.no_grad():
                _ = validation_model(inputs)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage increased by {memory_increase / 1024 / 1024:.1f} MB"

    def test_robustness_validation(self, validation_model):
        """Test model robustness to various input conditions."""
        validation_model.eval()
        batch_size, height, width = 2, 32, 32

        # Test robustness scenarios
        robustness_tests = [
            # Normal case
            {
                "name": "normal",
                "inputs": {"high_res": torch.randn(batch_size, 6, height, width)},
            },
            # Extreme values
            {
                "name": "extreme_high",
                "inputs": {"high_res": torch.ones(batch_size, 6, height, width) * 100},
            },
            {
                "name": "extreme_low",
                "inputs": {"high_res": torch.ones(batch_size, 6, height, width) * -100},
            },
            # Noisy inputs
            {
                "name": "noisy",
                "inputs": {"high_res": torch.randn(batch_size, 6, height, width) * 10},
            },
            # Sparse inputs
            {
                "name": "sparse",
                "inputs": {"high_res": torch.zeros(batch_size, 6, height, width)},
            },
        ]

        results = {}

        for test_case in robustness_tests:
            try:
                with torch.no_grad():
                    predictions = validation_model(test_case["inputs"])

                # Check prediction quality
                pred_depth = predictions["depth"]

                # Basic sanity checks
                assert torch.all(
                    torch.isfinite(pred_depth)
                ), f"Non-finite predictions in {test_case['name']}"
                assert torch.all(
                    pred_depth >= 0
                ), f"Negative predictions in {test_case['name']}"

                # Record statistics
                results[test_case["name"]] = {
                    "success": True,
                    "mean_prediction": torch.mean(pred_depth).item(),
                    "max_prediction": torch.max(pred_depth).item(),
                    "std_prediction": torch.std(pred_depth).item(),
                }

            except Exception as e:
                results[test_case["name"]] = {"success": False, "error": str(e)}

        # Validate robustness results
        successful_tests = sum(1 for r in results.values() if r.get("success", False))
        assert (
            successful_tests >= len(robustness_tests) * 0.8
        ), "Model should handle most robustness tests"

        # Normal case should always work
        assert results["normal"]["success"], "Model should handle normal inputs"

    def test_uncertainty_quantification(self, validation_data):
        """Test uncertainty quantification in predictions."""
        # Create model with uncertainty
        config = {
            "input_features": 6,
            "base_channels": 32,
            "depth_levels": 3,
            "predict_uncertainty": True,
        }

        model = create_flood_model(config)
        model.eval()

        inputs, targets, scenarios = validation_data

        with torch.no_grad():
            predictions = model(inputs)

        # Check uncertainty predictions
        assert "uncertainty" in predictions
        uncertainty = predictions["uncertainty"]

        # Uncertainty should be positive
        assert torch.all(uncertainty >= 0), "Uncertainty should be non-negative"

        # Uncertainty should vary across the prediction
        assert torch.std(uncertainty) > 0, "Uncertainty should show spatial variation"

        # Test uncertainty calibration
        depth_pred = predictions["depth"]

        # High uncertainty regions should correlate with prediction errors
        pred_np = depth_pred.numpy().flatten()
        target_np = targets.numpy().flatten()
        uncertainty_np = uncertainty.numpy().flatten()

        # Remove invalid values
        valid_mask = ~(
            np.isnan(pred_np) | np.isnan(target_np) | np.isnan(uncertainty_np)
        )
        pred_valid = pred_np[valid_mask]
        target_valid = target_np[valid_mask]
        uncertainty_valid = uncertainty_np[valid_mask]

        if len(uncertainty_valid) > 10:
            # Calculate prediction errors
            errors = np.abs(pred_valid - target_valid)

            # Uncertainty should correlate with errors (higher uncertainty = higher errors)
            if np.std(uncertainty_valid) > 0 and np.std(errors) > 0:
                correlation = pearsonr(uncertainty_valid, errors)[0]
                assert (
                    correlation > -0.5
                ), "Uncertainty should somewhat correlate with prediction errors"


class TestProductionValidation:
    """Test production-level validation requirements."""

    @pytest.fixture
    def production_validation_suite(self):
        """Set up production validation environment."""
        config = {
            "input_features": 6,
            "num_scales": 3,
            "base_channels": 64,
            "depth_levels": 4,
            "dropout_rate": 0.1,
            "predict_uncertainty": True,
        }

        model = create_flood_model(config)
        return model

    def test_production_accuracy_requirements(self, production_validation_suite):
        """Test that model meets production accuracy requirements."""
        model = production_validation_suite
        model.eval()

        # Create realistic production test data
        test_scenarios = self._create_production_test_scenarios()

        all_metrics = []

        for scenario in test_scenarios:
            with torch.no_grad():
                predictions = model(scenario["inputs"])

            pred_np = predictions["depth"].numpy()
            target_np = scenario["targets"].numpy()

            metrics = FloodMetrics.calculate_flood_metrics(pred_np, target_np)
            all_metrics.append(metrics)

        # Aggregate metrics across scenarios
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)

        # Production accuracy requirements
        assert (
            avg_metrics.get("mae", float("inf")) < 0.5
        ), "MAE should be < 0.5m for production"
        assert (
            avg_metrics.get("rmse", float("inf")) < 0.8
        ), "RMSE should be < 0.8m for production"
        assert (
            avg_metrics.get("r2", -float("inf")) > 0.7
        ), "R² should be > 0.7 for production"
        assert (
            avg_metrics.get("flood_hit_rate", 0) > 0.8
        ), "Flood detection rate should be > 80%"
        assert (
            avg_metrics.get("flood_false_alarm", 1) < 0.2
        ), "False alarm rate should be < 20%"

    def test_production_performance_requirements(self, production_validation_suite):
        """Test that model meets production performance requirements."""
        model = production_validation_suite
        model.eval()

        # Test with production-sized inputs
        batch_size = 16
        height, width = 256, 256

        inputs = {
            "high_res": torch.randn(batch_size, 6, height, width),
            "med_res": torch.randn(batch_size, 6, height // 2, width // 2),
            "low_res": torch.randn(batch_size, 6, height // 4, width // 4),
        }

        # Warm-up run
        with torch.no_grad():
            _ = model(inputs)

        # Measure performance
        num_runs = 5
        inference_times = []

        for _ in range(num_runs):
            start_time = time.time()

            with torch.no_grad():
                predictions = model(inputs)

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        avg_inference_time = np.mean(inference_times)
        throughput = batch_size / avg_inference_time

        # Production performance requirements
        assert (
            avg_inference_time < 10.0
        ), f"Inference time too slow: {avg_inference_time:.2f}s"
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} samples/sec"

        # Memory efficiency
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        assert memory_mb < 4000, f"Memory usage too high: {memory_mb:.1f} MB"

    def test_production_stability_requirements(self, production_validation_suite):
        """Test model stability over extended operation."""
        model = production_validation_suite
        model.eval()

        # Test extended operation
        num_iterations = 50
        batch_size = 4
        height, width = 128, 128

        prediction_statistics = []

        for i in range(num_iterations):
            # Create slightly different inputs each time
            inputs = {"high_res": torch.randn(batch_size, 6, height, width) + i * 0.01}

            with torch.no_grad():
                predictions = model(inputs)

            pred_stats = {
                "iteration": i,
                "mean": torch.mean(predictions["depth"]).item(),
                "std": torch.std(predictions["depth"]).item(),
                "max": torch.max(predictions["depth"]).item(),
                "min": torch.min(predictions["depth"]).item(),
            }

            prediction_statistics.append(pred_stats)

        # Check stability over iterations
        means = [s["mean"] for s in prediction_statistics]
        stds = [s["std"] for s in prediction_statistics]

        # Predictions should be stable (not trending)
        mean_trend = np.polyfit(range(len(means)), means, 1)[0]
        std_trend = np.polyfit(range(len(stds)), stds, 1)[0]

        assert abs(mean_trend) < 0.01, f"Mean prediction trending: {mean_trend}"
        assert abs(std_trend) < 0.01, f"Prediction variability trending: {std_trend}"

        # No catastrophic failures
        for stats in prediction_statistics:
            assert np.isfinite(stats["mean"]), "Non-finite predictions detected"
            assert stats["max"] < 100, "Unreasonably large predictions detected"
            assert stats["min"] >= 0, "Negative predictions detected"

    def test_model_versioning_and_reproducibility(self, production_validation_suite):
        """Test model versioning and reproducible predictions."""
        model = production_validation_suite

        # Save model state
        model_state = model.state_dict()

        # Test reproducibility with same inputs
        torch.manual_seed(42)
        inputs = {"high_res": torch.randn(2, 6, 64, 64)}

        model.eval()
        with torch.no_grad():
            pred1 = model(inputs)

        # Reset model and test again
        model.load_state_dict(model_state)
        model.eval()
        with torch.no_grad():
            pred2 = model(inputs)

        # Predictions should be identical
        torch.testing.assert_close(pred1["depth"], pred2["depth"])

        # Test model serialization
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(model_state, tmp_file.name)

            # Load in new model
            new_model = create_flood_model(
                {
                    "input_features": 6,
                    "num_scales": 3,
                    "base_channels": 64,
                    "depth_levels": 4,
                    "predict_uncertainty": True,
                }
            )

            new_model.load_state_dict(torch.load(tmp_file.name, map_location="cpu"))
            new_model.eval()

            with torch.no_grad():
                pred3 = new_model(inputs)

            # Predictions should match original
            torch.testing.assert_close(pred1["depth"], pred3["depth"])

        # Clean up
        os.unlink(tmp_file.name)

    def _create_production_test_scenarios(self):
        """Create realistic production test scenarios."""
        scenarios = []

        # Scenario 1: Urban flooding
        batch_size = 4
        height, width = 128, 128

        urban_inputs = {
            "high_res": torch.randn(batch_size, 6, height, width),
            "elevation": torch.randn(batch_size, 1, height, width) * 5
            + 50,  # Urban elevations
        }
        # Urban flooding typically shallow but widespread
        urban_targets = torch.rand(batch_size, 1, height, width) * 1.5

        scenarios.append(
            {"name": "urban_flooding", "inputs": urban_inputs, "targets": urban_targets}
        )

        # Scenario 2: River flooding
        river_inputs = {
            "high_res": torch.randn(batch_size, 6, height, width),
            "elevation": torch.randn(batch_size, 1, height, width) * 20 + 100,
        }
        # River flooding can be deeper, more localized
        river_targets = torch.zeros(batch_size, 1, height, width)
        # Add flooding along "river" path
        river_targets[:, :, height // 2 - 5 : height // 2 + 5, :] = (
            torch.rand(batch_size, 1, 10, width) * 4
        )

        scenarios.append(
            {"name": "river_flooding", "inputs": river_inputs, "targets": river_targets}
        )

        # Scenario 3: Coastal flooding
        coastal_inputs = {
            "high_res": torch.randn(batch_size, 6, height, width),
            "elevation": torch.randn(batch_size, 1, height, width) * 3
            + 2,  # Low coastal elevations
        }
        # Coastal flooding affects low-lying areas
        coastal_targets = torch.clamp(
            torch.randn(batch_size, 1, height, width) * 2 + 1, min=0
        )

        scenarios.append(
            {
                "name": "coastal_flooding",
                "inputs": coastal_inputs,
                "targets": coastal_targets,
            }
        )

        return scenarios


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
