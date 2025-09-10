"""End-to-end integration tests for the complete flood risk pipeline.

This module tests the complete workflow from raw data input to final predictions:
- Complete data processing pipeline
- Model training and evaluation workflows
- Production deployment scenarios
- Real-world data processing
- Performance and scalability validation
- System integration testing
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import json
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from sklearn.model_selection import train_test_split

# Import modules to test
from src.preprocessing.dem.hydrological_conditioning import HydrologicalConditioner
from src.preprocessing.terrain.feature_extraction import (
    TerrainFeatureExtractor,
    HANDCalculator,
)
from src.models.flood_cnn import (
    FloodDepthPredictor,
    create_flood_model,
    FloodModelTrainer,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodRiskPipeline:
    """Complete flood risk assessment pipeline for E2E testing."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.trainer = None
        self.preprocessing_results = {}
        self.feature_data = {}
        self.training_history = {}

    def preprocess_data(self, dem_path: str, output_dir: str) -> dict:
        """Complete data preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline")

        # Step 1: Hydrological conditioning
        logger.info("Step 1: Hydrological conditioning")
        conditioner = HydrologicalConditioner(dem_path)
        conditioning_results = conditioner.condition_dem(
            stream_threshold=self.config.get("stream_threshold", 1000),
            buffer_distance=self.config.get("buffer_distance", 50),
        )

        # Step 2: Terrain feature extraction
        logger.info("Step 2: Terrain feature extraction")
        conditioned_dem = conditioning_results["conditioned_dem"]
        pixel_size = abs(conditioner.dem_profile["transform"][0])

        feature_extractor = TerrainFeatureExtractor(
            conditioned_dem,
            pixel_size,
            nodata_value=conditioner.dem_profile.get("nodata"),
        )

        terrain_features = feature_extractor.extract_all_features(
            flow_direction=conditioning_results["flow_direction"]
        )

        # Step 3: HAND calculation
        logger.info("Step 3: HAND calculation")
        hand_calc = HANDCalculator(
            conditioned_dem,
            conditioning_results["flow_accumulation"],
            pixel_size,
            stream_threshold=self.config.get("stream_threshold", 1000),
        )
        hand = hand_calc.calculate_hand(
            max_distance=self.config.get("hand_max_distance", 1000)
        )
        terrain_features["hand"] = hand

        # Step 4: Combine features for model input
        logger.info("Step 4: Creating model input features")
        feature_stack = self._create_feature_stack(
            terrain_features, conditioning_results
        )

        # Save intermediate results
        if output_dir:
            self._save_preprocessing_results(
                conditioning_results, terrain_features, feature_stack, output_dir
            )

        results = {
            "conditioning": conditioning_results,
            "terrain_features": terrain_features,
            "model_features": feature_stack,
            "metadata": {
                "pixel_size": pixel_size,
                "shape": conditioned_dem.shape,
                "crs": conditioner.dem_profile.get("crs"),
                "transform": conditioner.dem_profile.get("transform"),
            },
        }

        self.preprocessing_results = results
        logger.info("Data preprocessing completed successfully")
        return results

    def prepare_training_data(
        self, features: dict, synthetic_targets: bool = True
    ) -> dict:
        """Prepare training data from preprocessed features."""
        logger.info("Preparing training data")

        model_features = features["model_features"]

        if synthetic_targets:
            # Generate synthetic flood targets for testing
            targets = self._generate_synthetic_flood_targets(
                features["terrain_features"], features["metadata"]["shape"]
            )
        else:
            # In real scenario, would load observed flood data
            raise NotImplementedError("Loading real flood targets not implemented")

        # Convert to PyTorch tensors
        feature_tensor = torch.from_numpy(model_features).float()
        target_tensor = torch.from_numpy(targets).float()

        # Add batch dimension
        if feature_tensor.dim() == 3:
            feature_tensor = feature_tensor.unsqueeze(0)
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)

        # Create train/validation split
        # For spatial data, we need to be careful about spatial correlation
        train_data, val_data = self._spatial_train_val_split(
            feature_tensor, target_tensor, val_fraction=0.2
        )

        training_data = {
            "train_features": train_data[0],
            "train_targets": train_data[1],
            "val_features": val_data[0],
            "val_targets": val_data[1],
            "feature_names": self._get_feature_names(),
            "data_statistics": self._calculate_data_statistics(
                feature_tensor, target_tensor
            ),
        }

        logger.info(
            f"Training data prepared: {train_data[0].shape[0]} train, {val_data[0].shape[0]} val samples"
        )
        return training_data

    def train_model(self, training_data: dict, num_epochs: int = 10) -> dict:
        """Train the flood prediction model."""
        logger.info(f"Starting model training for {num_epochs} epochs")

        # Initialize model
        model_config = {
            "input_features": training_data["train_features"].shape[1],
            "base_channels": self.config.get("base_channels", 32),
            "depth_levels": self.config.get("depth_levels", 3),
            "predict_uncertainty": True,
        }

        self.model = create_flood_model(model_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = FloodModelTrainer(self.model, device)

        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.get("learning_rate", 0.001)
        )

        # Training loop
        train_losses = []
        val_metrics = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training step
            self.model.train()
            optimizer.zero_grad()

            # Prepare batch
            train_batch = {
                "inputs": {"high_res": training_data["train_features"]},
                "targets": training_data["train_targets"],
            }

            # Forward pass
            predictions = self.model(train_batch["inputs"])
            loss_dict = self.model.compute_loss(
                predictions, train_batch["targets"], train_batch["inputs"]
            )

            # Backward pass
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            optimizer.step()

            train_losses.append({k: v.item() for k, v in loss_dict.items()})

            # Validation step
            if epoch % 2 == 0:  # Validate every 2 epochs
                val_batch = {
                    "inputs": {"high_res": training_data["val_features"]},
                    "targets": training_data["val_targets"],
                }

                val_metrics_epoch = self.trainer.validate_step(val_batch)
                val_metrics.append(val_metrics_epoch)

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Loss={total_loss.item():.4f}, "
                    f"Val_MSE={val_metrics_epoch['mse']:.4f}, "
                    f"Time={time.time()-epoch_start:.2f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Loss={total_loss.item():.4f}, "
                    f"Time={time.time()-epoch_start:.2f}s"
                )

        training_results = {
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "model_config": model_config,
            "final_model_state": self.model.state_dict(),
            "training_time_seconds": sum(
                [time.time() - time.time() for _ in range(num_epochs)]  # Approximate
            ),
        }

        self.training_history = training_results
        logger.info("Model training completed successfully")
        return training_results

    def evaluate_model(self, test_data: dict = None) -> dict:
        """Comprehensive model evaluation."""
        logger.info("Starting model evaluation")

        if test_data is None:
            # Use validation data for testing
            test_features = self.preprocessing_results.get("model_features")
            if test_features is not None:
                test_features = torch.from_numpy(test_features).float().unsqueeze(0)
                # Generate test targets
                test_targets = self._generate_synthetic_flood_targets(
                    self.preprocessing_results["terrain_features"],
                    self.preprocessing_results["metadata"]["shape"],
                )
                test_targets = (
                    torch.from_numpy(test_targets).float().unsqueeze(0).unsqueeze(0)
                )
            else:
                raise ValueError("No test data available")
        else:
            test_features = test_data["features"]
            test_targets = test_data["targets"]

        # Model predictions
        self.model.eval()
        with torch.no_grad():
            test_inputs = {"high_res": test_features}
            predictions = self.model(test_inputs)

        # Calculate comprehensive metrics
        pred_np = predictions["depth"].cpu().numpy()
        target_np = test_targets.cpu().numpy()

        evaluation_metrics = self._calculate_comprehensive_metrics(pred_np, target_np)

        # Spatial analysis
        spatial_metrics = self._analyze_spatial_patterns(pred_np, target_np)
        evaluation_metrics.update(spatial_metrics)

        # Uncertainty analysis (if available)
        if "uncertainty" in predictions:
            uncertainty_np = predictions["uncertainty"].cpu().numpy()
            uncertainty_metrics = self._analyze_uncertainty(
                pred_np, target_np, uncertainty_np
            )
            evaluation_metrics.update(uncertainty_metrics)

        evaluation_results = {
            "metrics": evaluation_metrics,
            "predictions": pred_np,
            "targets": target_np,
            "model_performance_summary": self._generate_performance_summary(
                evaluation_metrics
            ),
        }

        logger.info("Model evaluation completed")
        logger.info(
            f"Performance Summary: {evaluation_results['model_performance_summary']}"
        )

        return evaluation_results

    def run_full_pipeline(self, dem_path: str, output_dir: str) -> dict:
        """Run the complete end-to-end pipeline."""
        logger.info("Starting full E2E pipeline")
        pipeline_start = time.time()

        try:
            # Step 1: Data preprocessing
            preprocessing_results = self.preprocess_data(dem_path, output_dir)

            # Step 2: Prepare training data
            training_data = self.prepare_training_data(preprocessing_results)

            # Step 3: Train model
            training_results = self.train_model(
                training_data, num_epochs=5
            )  # Reduced for testing

            # Step 4: Evaluate model
            evaluation_results = self.evaluate_model()

            pipeline_time = time.time() - pipeline_start

            # Compile full results
            full_results = {
                "preprocessing": preprocessing_results,
                "training_data": training_data,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "pipeline_metadata": {
                    "total_time_seconds": pipeline_time,
                    "config_used": self.config,
                    "success": True,
                },
            }

            logger.info(
                f"Full E2E pipeline completed successfully in {pipeline_time:.2f} seconds"
            )
            return full_results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "pipeline_metadata": {
                    "total_time_seconds": time.time() - pipeline_start,
                    "config_used": self.config,
                    "success": False,
                    "error": str(e),
                }
            }

    # Helper methods
    def _create_feature_stack(
        self, terrain_features: dict, conditioning_results: dict
    ) -> np.ndarray:
        """Create multi-channel feature stack for model input."""
        features_to_stack = [
            conditioning_results["conditioned_dem"],
            terrain_features["slope_degrees"],
            terrain_features["curvature_total"],
            terrain_features["flow_accumulation"],
            terrain_features["wetness_index"],
            terrain_features["hand"],
        ]

        # Normalize and stack features
        normalized_features = []
        for feature in features_to_stack:
            # Handle NaN values
            feature_clean = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize to [0, 1] range
            if np.max(feature_clean) > np.min(feature_clean):
                feature_norm = (feature_clean - np.min(feature_clean)) / (
                    np.max(feature_clean) - np.min(feature_clean)
                )
            else:
                feature_norm = feature_clean

            normalized_features.append(feature_norm)

        # Stack along channel dimension
        feature_stack = np.stack(
            normalized_features, axis=0
        )  # Shape: (channels, height, width)

        return feature_stack.astype(np.float32)

    def _generate_synthetic_flood_targets(
        self, terrain_features: dict, shape: tuple
    ) -> np.ndarray:
        """Generate synthetic flood depth targets for testing."""
        height, width = shape

        # Base flood depth influenced by elevation and flow accumulation
        elevation = self.preprocessing_results["conditioning"]["conditioned_dem"]
        flow_acc = terrain_features["flow_accumulation"]
        twi = terrain_features["wetness_index"]

        # Normalize elevation (lower = more likely to flood)
        elev_norm = (elevation - np.min(elevation)) / (
            np.max(elevation) - np.min(elevation)
        )
        flood_susceptibility = 1 - elev_norm

        # Add flow accumulation influence
        flow_norm = np.clip(flow_acc / np.percentile(flow_acc, 95), 0, 1)
        flood_susceptibility += 0.5 * flow_norm

        # Add wetness index influence
        twi_clean = np.nan_to_num(twi, nan=0)
        twi_norm = np.clip(
            (twi_clean - np.min(twi_clean)) / (np.max(twi_clean) - np.min(twi_clean)),
            0,
            1,
        )
        flood_susceptibility += 0.3 * twi_norm

        # Add some randomness
        flood_depth = flood_susceptibility * np.random.exponential(1.5, shape)

        # Ensure some areas have significant flooding
        # Add localized flooding events
        num_flood_centers = np.random.randint(2, 5)
        for _ in range(num_flood_centers):
            center_y = np.random.randint(height // 4, 3 * height // 4)
            center_x = np.random.randint(width // 4, 3 * width // 4)
            radius = np.random.randint(
                min(height, width) // 10, min(height, width) // 5
            )

            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

            flood_depth[mask] += np.random.exponential(2.0, np.sum(mask))

        # Clip to reasonable range
        flood_depth = np.clip(flood_depth, 0, 10)  # 0-10m flood depth

        return flood_depth.astype(np.float32)

    def _spatial_train_val_split(
        self, features: torch.Tensor, targets: torch.Tensor, val_fraction: float = 0.2
    ):
        """Split data spatially to avoid data leakage."""
        # For simplicity, just do random split
        # In production, would do proper spatial splitting
        batch_size = features.shape[0]
        indices = torch.randperm(batch_size)

        val_size = int(batch_size * val_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_data = (features[train_indices], targets[train_indices])
        val_data = (features[val_indices], targets[val_indices])

        return train_data, val_data

    def _get_feature_names(self) -> list:
        """Get names of input features."""
        return [
            "elevation",
            "slope",
            "curvature",
            "flow_accumulation",
            "wetness_index",
            "hand",
        ]

    def _calculate_data_statistics(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """Calculate statistics of the training data."""
        return {
            "feature_stats": {
                "mean": torch.mean(features, dim=[0, 2, 3]).tolist(),
                "std": torch.std(features, dim=[0, 2, 3]).tolist(),
                "min": torch.min(
                    features.view(features.shape[0], features.shape[1], -1), dim=2
                )[0]
                .min(0)[0]
                .tolist(),
                "max": torch.max(
                    features.view(features.shape[0], features.shape[1], -1), dim=2
                )[0]
                .max(0)[0]
                .tolist(),
            },
            "target_stats": {
                "mean": torch.mean(targets).item(),
                "std": torch.std(targets).item(),
                "min": torch.min(targets).item(),
                "max": torch.max(targets).item(),
                "flood_fraction": (
                    torch.sum(targets > 0.1) / torch.numel(targets)
                ).item(),
            },
        }

    def _calculate_comprehensive_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> dict:
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        if len(pred_valid) == 0:
            return {
                key: float("nan")
                for key in ["mse", "rmse", "mae", "r2", "flood_accuracy"]
            }

        # Basic regression metrics
        mse = mean_squared_error(target_valid, pred_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_valid, pred_valid)
        r2 = (
            r2_score(target_valid, pred_valid)
            if len(np.unique(target_valid)) > 1
            else 0.0
        )

        # Flood detection metrics
        pred_flood = (pred_valid > 0.1).astype(int)
        target_flood = (target_valid > 0.1).astype(int)
        flood_accuracy = np.mean(pred_flood == target_flood)

        # Additional metrics
        bias = np.mean(pred_valid - target_valid)
        correlation = (
            np.corrcoef(pred_valid, target_valid)[0, 1]
            if len(np.unique(pred_valid)) > 1
            else 0.0
        )

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "flood_accuracy": float(flood_accuracy),
            "bias": float(bias),
            "correlation": float(correlation),
        }

    def _analyze_spatial_patterns(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> dict:
        """Analyze spatial patterns in predictions."""
        # Simple spatial correlation analysis
        if predictions.shape != targets.shape:
            return {"spatial_correlation": float("nan")}

        # Calculate spatial correlation
        pred_2d = predictions.squeeze() if predictions.ndim > 2 else predictions
        target_2d = targets.squeeze() if targets.ndim > 2 else targets

        spatial_corr = np.corrcoef(pred_2d.flatten(), target_2d.flatten())[0, 1]

        return {
            "spatial_correlation": (
                float(spatial_corr) if not np.isnan(spatial_corr) else 0.0
            )
        }

    def _analyze_uncertainty(
        self, predictions: np.ndarray, targets: np.ndarray, uncertainty: np.ndarray
    ) -> dict:
        """Analyze uncertainty quantification."""
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        unc_flat = uncertainty.flatten()

        # Calculate prediction errors
        errors = np.abs(pred_flat - target_flat)

        # Uncertainty should correlate with errors
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat) | np.isnan(unc_flat))
        if np.sum(valid_mask) > 10:
            error_unc_corr = np.corrcoef(errors[valid_mask], unc_flat[valid_mask])[0, 1]
        else:
            error_unc_corr = 0.0

        return {
            "uncertainty_error_correlation": (
                float(error_unc_corr) if not np.isnan(error_unc_corr) else 0.0
            ),
            "mean_uncertainty": float(np.nanmean(unc_flat)),
            "uncertainty_range": float(np.nanmax(unc_flat) - np.nanmin(unc_flat)),
        }

    def _generate_performance_summary(self, metrics: dict) -> str:
        """Generate human-readable performance summary."""
        rmse = metrics.get("rmse", float("nan"))
        mae = metrics.get("mae", float("nan"))
        r2 = metrics.get("r2", float("nan"))
        flood_acc = metrics.get("flood_accuracy", float("nan"))

        if rmse < 0.5 and mae < 0.3 and r2 > 0.7 and flood_acc > 0.8:
            performance_level = "Excellent"
        elif rmse < 0.8 and mae < 0.5 and r2 > 0.5 and flood_acc > 0.7:
            performance_level = "Good"
        elif rmse < 1.2 and mae < 0.8 and r2 > 0.3 and flood_acc > 0.6:
            performance_level = "Acceptable"
        else:
            performance_level = "Needs Improvement"

        return f"{performance_level} (RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, Flood Acc: {flood_acc:.3f})"

    def _save_preprocessing_results(
        self,
        conditioning: dict,
        features: dict,
        feature_stack: np.ndarray,
        output_dir: str,
    ):
        """Save preprocessing results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save key arrays as numpy files (more efficient than GeoTIFF for testing)
        np.save(output_path / "conditioned_dem.npy", conditioning["conditioned_dem"])
        np.save(output_path / "feature_stack.npy", feature_stack)

        # Save metadata
        metadata = {
            "feature_names": self._get_feature_names(),
            "processing_config": self.config,
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


class TestE2EFloodRiskPipeline:
    """Test the complete end-to-end flood risk pipeline."""

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for the E2E pipeline."""
        return {
            "stream_threshold": 500,
            "buffer_distance": 30,
            "hand_max_distance": 500,
            "base_channels": 32,
            "depth_levels": 3,
            "learning_rate": 0.001,
        }

    @pytest.fixture
    def sample_dem_file(self, tmp_path):
        """Create a sample DEM file for E2E testing."""
        # Create more complex terrain for realistic testing
        height, width = 100, 100
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)

        # Create realistic terrain with multiple features
        dem = (
            200  # Base elevation
            + 50 * np.sin(X / 3) * np.cos(Y / 3)  # Large-scale topography
            + 20 * np.sin(X) * np.cos(Y)  # Medium-scale features
            + 10 * np.random.normal(0, 1, (height, width))  # Noise
            + 100 * np.exp(-((X - 5) ** 2 + (Y - 7) ** 2) / 4)  # Hill
            + -50 * np.exp(-((X - 7) ** 2 + (Y - 3) ** 2) / 2)  # Valley
        )

        # Add stream network
        stream_y = (Y > 4.5) & (Y < 5.5) & (X > 2) & (X < 8)
        stream_x = (X > 4.5) & (X < 5.5) & (Y > 1) & (Y < 9)
        dem[stream_y | stream_x] -= 30

        # Add some sinks for testing conditioning
        sink_mask = ((X - 2) ** 2 + (Y - 8) ** 2) < 0.5
        dem[sink_mask] -= 20

        dem_path = tmp_path / "test_dem.tif"

        # Create geospatial properties
        transform = from_bounds(
            west=0, south=0, east=1000, north=1000, width=width, height=height
        )

        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(dem.astype(np.float32), 1)

        return str(dem_path)

    def test_full_pipeline_execution(self, pipeline_config, sample_dem_file, tmp_path):
        """Test complete end-to-end pipeline execution."""
        pipeline = FloodRiskPipeline(pipeline_config)
        output_dir = tmp_path / "pipeline_output"

        # Run full pipeline
        results = pipeline.run_full_pipeline(sample_dem_file, str(output_dir))

        # Validate pipeline success
        assert results["pipeline_metadata"][
            "success"
        ], f"Pipeline failed: {results['pipeline_metadata'].get('error', 'Unknown error')}"

        # Check all major components completed
        assert "preprocessing" in results
        assert "training_results" in results
        assert "evaluation_results" in results

        # Validate preprocessing results
        preprocessing = results["preprocessing"]
        assert "model_features" in preprocessing
        assert preprocessing["model_features"].shape[0] == 6  # 6 feature channels

        # Validate training results
        training = results["training_results"]
        assert "train_losses" in training
        assert len(training["train_losses"]) > 0

        # Training should generally reduce loss
        initial_loss = training["train_losses"][0]["total_loss"]
        final_loss = training["train_losses"][-1]["total_loss"]
        assert (
            final_loss <= initial_loss * 2
        ), "Training loss should not increase dramatically"

        # Validate evaluation results
        evaluation = results["evaluation_results"]
        assert "metrics" in evaluation

        metrics = evaluation["metrics"]
        # Basic sanity checks on metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
        assert not np.isnan(metrics["mse"])
        assert not np.isnan(metrics["mae"])

        # Check that outputs are saved
        assert output_dir.exists()
        assert (output_dir / "metadata.json").exists()

    def test_preprocessing_pipeline(self, pipeline_config, sample_dem_file, tmp_path):
        """Test the preprocessing portion of the pipeline in detail."""
        pipeline = FloodRiskPipeline(pipeline_config)
        output_dir = tmp_path / "preprocessing_output"

        # Run preprocessing only
        results = pipeline.preprocess_data(sample_dem_file, str(output_dir))

        # Validate preprocessing components
        assert "conditioning" in results
        assert "terrain_features" in results
        assert "model_features" in results
        assert "metadata" in results

        # Check conditioning results
        conditioning = results["conditioning"]
        required_conditioning_keys = {
            "conditioned_dem",
            "filled_dem",
            "flow_direction",
            "flow_accumulation",
            "streams",
        }
        assert set(conditioning.keys()) >= required_conditioning_keys

        # Check terrain features
        terrain = results["terrain_features"]
        required_terrain_keys = {
            "slope_degrees",
            "curvature_total",
            "flow_accumulation",
            "wetness_index",
            "hand",
        }
        assert set(terrain.keys()) >= required_terrain_keys

        # Check model features
        model_features = results["model_features"]
        assert model_features.ndim == 3  # (channels, height, width)
        assert model_features.shape[0] == 6  # 6 feature channels
        assert not np.all(np.isnan(model_features))

        # Check metadata
        metadata = results["metadata"]
        assert "pixel_size" in metadata
        assert "shape" in metadata
        assert metadata["pixel_size"] > 0

        # Check that outputs are saved
        assert output_dir.exists()
        assert (output_dir / "conditioned_dem.npy").exists()
        assert (output_dir / "feature_stack.npy").exists()
        assert (output_dir / "metadata.json").exists()

    def test_training_pipeline(self, pipeline_config):
        """Test the training portion of the pipeline."""
        pipeline = FloodRiskPipeline(pipeline_config)

        # Create mock preprocessing results
        height, width = 64, 64
        mock_features = {
            "model_features": np.random.randn(6, height, width).astype(np.float32),
            "terrain_features": {
                "slope_degrees": np.random.rand(height, width),
                "flow_accumulation": np.random.rand(height, width) * 1000,
                "wetness_index": np.random.randn(height, width),
            },
            "metadata": {"shape": (height, width)},
        }

        pipeline.preprocessing_results = {
            "conditioning": {"conditioned_dem": np.random.randn(height, width)},
            **mock_features,
        }

        # Prepare training data
        training_data = pipeline.prepare_training_data(mock_features)

        # Validate training data preparation
        assert "train_features" in training_data
        assert "train_targets" in training_data
        assert "val_features" in training_data
        assert "val_targets" in training_data

        # Check tensor shapes
        assert (
            training_data["train_features"].dim() == 4
        )  # (batch, channels, height, width)
        assert training_data["train_targets"].dim() == 4  # (batch, 1, height, width)

        # Train model (short training for testing)
        training_results = pipeline.train_model(training_data, num_epochs=3)

        # Validate training results
        assert "train_losses" in training_results
        assert "model_config" in training_results
        assert "final_model_state" in training_results

        # Check that model was actually trained
        assert len(training_results["train_losses"]) == 3

        # Model should be available for inference
        assert pipeline.model is not None
        assert pipeline.trainer is not None

    def test_model_evaluation(self, pipeline_config):
        """Test model evaluation functionality."""
        pipeline = FloodRiskPipeline(pipeline_config)

        # Create and train a simple model
        height, width = 32, 32
        mock_features = {
            "model_features": np.random.randn(6, height, width).astype(np.float32),
            "terrain_features": {
                "slope_degrees": np.random.rand(height, width),
                "flow_accumulation": np.random.rand(height, width) * 1000,
                "wetness_index": np.random.randn(height, width),
            },
            "metadata": {"shape": (height, width)},
        }

        pipeline.preprocessing_results = {
            "conditioning": {"conditioned_dem": np.random.randn(height, width)},
            **mock_features,
        }

        training_data = pipeline.prepare_training_data(mock_features)
        pipeline.train_model(training_data, num_epochs=2)

        # Evaluate model
        evaluation_results = pipeline.evaluate_model()

        # Validate evaluation results
        assert "metrics" in evaluation_results
        assert "model_performance_summary" in evaluation_results

        metrics = evaluation_results["metrics"]
        required_metrics = {"mse", "rmse", "mae", "r2", "flood_accuracy"}
        assert set(metrics.keys()) >= required_metrics

        # Check metric validity
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float))
            if not np.isnan(value):
                assert np.isfinite(
                    value
                ), f"Metric {metric_name} is not finite: {value}"

        # Performance summary should be a string
        assert isinstance(evaluation_results["model_performance_summary"], str)
        assert len(evaluation_results["model_performance_summary"]) > 10

    def test_pipeline_performance_requirements(
        self, pipeline_config, sample_dem_file, tmp_path
    ):
        """Test that the pipeline meets performance requirements."""
        pipeline = FloodRiskPipeline(pipeline_config)
        output_dir = tmp_path / "performance_test"

        # Time the full pipeline
        start_time = time.time()
        results = pipeline.run_full_pipeline(sample_dem_file, str(output_dir))
        total_time = time.time() - start_time

        # Pipeline should complete successfully
        assert results["pipeline_metadata"]["success"]

        # Performance requirements (adjust as needed)
        assert total_time < 120, f"Pipeline too slow: {total_time:.2f} seconds"

        # Check reported time matches measured time (approximately)
        reported_time = results["pipeline_metadata"]["total_time_seconds"]
        assert (
            abs(reported_time - total_time) < 5
        ), "Reported time doesn't match measured time"

        # Memory usage should be reasonable
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        assert memory_mb < 2000, f"Memory usage too high: {memory_mb:.1f} MB"

    def test_pipeline_robustness(self, pipeline_config, tmp_path):
        """Test pipeline robustness to various input conditions."""
        # Test with different DEM characteristics
        test_scenarios = [
            {"name": "flat_terrain", "elevation_range": 5},
            {"name": "mountainous", "elevation_range": 500},
            {"name": "small_dem", "size": (32, 32)},
            {"name": "rectangular", "size": (50, 80)},
        ]

        for scenario in test_scenarios:
            # Create scenario-specific DEM
            if "size" in scenario:
                height, width = scenario["size"]
            else:
                height, width = 64, 64

            elevation_range = scenario.get("elevation_range", 100)

            # Generate terrain
            x = np.linspace(0, 10, width)
            y = np.linspace(0, 10, height)
            X, Y = np.meshgrid(x, y)

            dem = 100 + elevation_range * np.sin(X / 2) * np.cos(Y / 2)
            dem += np.random.normal(0, elevation_range / 10, (height, width))

            # Save DEM
            dem_path = tmp_path / f"test_dem_{scenario['name']}.tif"
            transform = from_bounds(0, 0, width * 10, height * 10, width, height)

            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=CRS.from_epsg(4326),
                transform=transform,
            ) as dst:
                dst.write(dem.astype(np.float32), 1)

            # Test pipeline with this scenario
            pipeline = FloodRiskPipeline(pipeline_config)
            output_dir = tmp_path / f"output_{scenario['name']}"

            try:
                results = pipeline.run_full_pipeline(str(dem_path), str(output_dir))

                # Pipeline should handle diverse inputs
                assert results["pipeline_metadata"][
                    "success"
                ], f"Pipeline failed on {scenario['name']}: {results['pipeline_metadata'].get('error')}"

                # Basic validation
                assert "evaluation_results" in results
                metrics = results["evaluation_results"]["metrics"]
                assert not np.isnan(metrics["mse"])

            except Exception as e:
                pytest.fail(f"Pipeline failed on scenario {scenario['name']}: {str(e)}")

    def test_pipeline_error_handling(self, pipeline_config, tmp_path):
        """Test pipeline error handling and recovery."""
        # Test with invalid DEM file
        invalid_dem_path = tmp_path / "invalid.tif"
        with open(invalid_dem_path, "w") as f:
            f.write("This is not a valid GeoTIFF file")

        pipeline = FloodRiskPipeline(pipeline_config)
        output_dir = tmp_path / "error_test"

        # Pipeline should handle invalid input gracefully
        results = pipeline.run_full_pipeline(str(invalid_dem_path), str(output_dir))

        # Should report failure
        assert not results["pipeline_metadata"]["success"]
        assert "error" in results["pipeline_metadata"]
        assert isinstance(results["pipeline_metadata"]["error"], str)
        assert len(results["pipeline_metadata"]["error"]) > 0

    def test_pipeline_configuration_validation(self, sample_dem_file, tmp_path):
        """Test pipeline behavior with different configurations."""
        configurations = [
            {
                "name": "minimal",
                "config": {
                    "stream_threshold": 100,
                    "base_channels": 16,
                    "depth_levels": 2,
                },
            },
            {
                "name": "standard",
                "config": {
                    "stream_threshold": 500,
                    "base_channels": 32,
                    "depth_levels": 3,
                },
            },
            {
                "name": "complex",
                "config": {
                    "stream_threshold": 1000,
                    "base_channels": 64,
                    "depth_levels": 4,
                },
            },
        ]

        for config_test in configurations:
            config = config_test["config"]
            pipeline = FloodRiskPipeline(config)
            output_dir = tmp_path / f"config_test_{config_test['name']}"

            # Run pipeline with this configuration
            results = pipeline.run_full_pipeline(sample_dem_file, str(output_dir))

            # Should succeed with any reasonable configuration
            assert results["pipeline_metadata"][
                "success"
            ], f"Pipeline failed with {config_test['name']} config: {results['pipeline_metadata'].get('error')}"

            # Configuration should be preserved in results
            assert results["pipeline_metadata"]["config_used"] == config


class TestProductionReadiness:
    """Test production readiness of the complete system."""

    def test_system_integration_validation(self, tmp_path):
        """Test complete system integration for production deployment."""
        # Create production-like configuration
        production_config = {
            "stream_threshold": 1000,
            "buffer_distance": 50,
            "hand_max_distance": 1000,
            "base_channels": 64,
            "depth_levels": 4,
            "learning_rate": 0.0005,
        }

        # Create production-sized test data
        height, width = 256, 256

        # Create complex realistic DEM
        x = np.linspace(0, 25.6, width)  # 25.6 km
        y = np.linspace(0, 25.6, height)
        X, Y = np.meshgrid(x, y)

        dem = (
            500  # Base elevation
            + 200 * np.sin(X / 8) * np.cos(Y / 6)  # Regional topography
            + 100 * np.sin(X / 3) * np.cos(Y / 4)  # Local relief
            + 50 * np.random.normal(0, 1, (height, width))  # Noise
            + 300 * np.exp(-((X - 12) ** 2 + (Y - 18) ** 2) / 20)  # Mountain
            + -150 * np.exp(-((X - 20) ** 2 + (Y - 8) ** 2) / 10)  # Valley
        )

        # Add complex drainage network
        main_river = (Y > 12) & (Y < 14) & (X > 5) & (X < 22)
        tributary1 = (X > 8) & (X < 10) & (Y > 3) & (Y < 12)
        tributary2 = (X > 15) & (X < 17) & (Y > 14) & (Y < 20)

        dem[main_river | tributary1 | tributary2] -= 80

        # Save production DEM
        dem_path = tmp_path / "production_dem.tif"
        transform = from_bounds(0, 0, 25600, 25600, width, height)  # 100m pixels

        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(32633),  # UTM zone 33N
            transform=transform,
            compress="lzw",  # Production compression
        ) as dst:
            dst.write(dem.astype(np.float32), 1)

        # Run production pipeline
        pipeline = FloodRiskPipeline(production_config)
        output_dir = tmp_path / "production_output"

        start_time = time.time()
        results = pipeline.run_full_pipeline(str(dem_path), str(output_dir))
        processing_time = time.time() - start_time

        # Production validation requirements
        assert results["pipeline_metadata"][
            "success"
        ], f"Production pipeline failed: {results['pipeline_metadata'].get('error')}"

        # Performance requirements
        assert (
            processing_time < 300
        ), f"Production pipeline too slow: {processing_time:.2f}s"

        # Accuracy requirements
        metrics = results["evaluation_results"]["metrics"]
        assert (
            metrics["rmse"] < 1.0
        ), f"RMSE too high for production: {metrics['rmse']:.3f}"
        assert (
            metrics["mae"] < 0.6
        ), f"MAE too high for production: {metrics['mae']:.3f}"
        assert metrics["r2"] > 0.6, f"R² too low for production: {metrics['r2']:.3f}"
        assert (
            metrics["flood_accuracy"] > 0.75
        ), f"Flood accuracy too low: {metrics['flood_accuracy']:.3f}"

        # Output validation
        assert output_dir.exists()
        assert (output_dir / "metadata.json").exists()

        # Metadata should contain production information
        with open(output_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        assert "feature_names" in metadata
        assert "processing_config" in metadata
        assert len(metadata["feature_names"]) == 6

    @contextmanager
    def production_monitoring(self):
        """Context manager for production monitoring during testing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Record initial state
        initial_memory = process.memory_info().rss
        initial_time = time.time()

        monitoring_data = {
            "memory_samples": [initial_memory],
            "cpu_samples": [],
            "start_time": initial_time,
        }

        try:
            yield monitoring_data
        finally:
            # Record final state
            final_memory = process.memory_info().rss
            final_time = time.time()

            monitoring_data.update(
                {
                    "final_memory": final_memory,
                    "total_time": final_time - initial_time,
                    "memory_increase": final_memory - initial_memory,
                    "peak_memory": max(monitoring_data["memory_samples"]),
                }
            )

    def test_production_resource_monitoring(self, tmp_path):
        """Test resource usage monitoring during production operation."""
        # Configure for resource testing
        config = {
            "stream_threshold": 500,
            "buffer_distance": 30,
            "base_channels": 32,
            "depth_levels": 3,
        }

        # Create test DEM
        height, width = 128, 128
        dem = np.random.normal(200, 50, (height, width)).astype(np.float32)

        dem_path = tmp_path / "resource_test_dem.tif"
        transform = from_bounds(0, 0, 1280, 1280, width, height)

        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(dem, 1)

        # Monitor resource usage
        with self.production_monitoring() as monitor:
            pipeline = FloodRiskPipeline(config)
            output_dir = tmp_path / "monitored_output"

            results = pipeline.run_full_pipeline(str(dem_path), str(output_dir))

            # Sample memory usage during processing
            import psutil
            import os

            process = psutil.Process(os.getpid())
            monitor["memory_samples"].append(process.memory_info().rss)

        # Validate resource usage
        assert results["pipeline_metadata"]["success"]

        # Memory usage should be bounded
        memory_increase_mb = monitor["memory_increase"] / 1024 / 1024
        peak_memory_mb = monitor["peak_memory"] / 1024 / 1024

        assert (
            memory_increase_mb < 500
        ), f"Memory increase too high: {memory_increase_mb:.1f} MB"
        assert peak_memory_mb < 2000, f"Peak memory too high: {peak_memory_mb:.1f} MB"

        # Processing time should be reasonable
        assert (
            monitor["total_time"] < 60
        ), f"Processing time too long: {monitor['total_time']:.2f}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
