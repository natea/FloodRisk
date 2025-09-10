"""Data normalization and dimensionless feature computation module.

This module provides functionality for local normalization of terrain and
hydrological features to create dimensionless variables suitable for
machine learning applications.
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter, gaussian_filter
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class NormalizationParams:
    """Parameters for data normalization."""

    method: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentiles: Optional[Tuple[float, float]] = None
    local_window_size: Optional[int] = None
    epsilon: float = 1e-8


class DataNormalizer:
    """Normalizes data using various methods for ML applications.

    This class provides multiple normalization strategies including global,
    local, and robust normalization methods suitable for terrain and
    hydrological data.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize data normalizer.

        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon
        self.normalization_params = {}

    def z_score_normalize(
        self, data: np.ndarray, feature_name: str = "unnamed", use_robust: bool = False
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply Z-score normalization (standardization).

        Args:
            data: Input data array
            feature_name: Name of the feature for parameter storage
            use_robust: Use median and MAD instead of mean and std

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        # Handle masked arrays
        if hasattr(data, "mask"):
            valid_data = data.compressed()
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            warnings.warn(f"No valid data for feature {feature_name}")
            return data.copy(), NormalizationParams("z_score")

        if use_robust:
            # Robust normalization using median and MAD
            center = np.median(valid_data)
            scale = stats.median_abs_deviation(valid_data, scale="normal")

            if scale < self.epsilon:
                scale = 1.0
                warnings.warn(
                    f"Very small MAD for feature {feature_name}, using scale=1.0"
                )

            normalized = (data - center) / scale

            params = NormalizationParams(
                method="z_score_robust",
                mean=float(center),
                std=float(scale),
                epsilon=self.epsilon,
            )
        else:
            # Standard normalization using mean and std
            mean = np.mean(valid_data)
            std = np.std(valid_data)

            if std < self.epsilon:
                std = 1.0
                warnings.warn(
                    f"Very small std for feature {feature_name}, using std=1.0"
                )

            normalized = (data - mean) / std

            params = NormalizationParams(
                method="z_score", mean=float(mean), std=float(std), epsilon=self.epsilon
            )

        self.normalization_params[feature_name] = params
        return normalized, params

    def min_max_normalize(
        self,
        data: np.ndarray,
        feature_name: str = "unnamed",
        target_range: Tuple[float, float] = (0, 1),
        percentile_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply min-max normalization.

        Args:
            data: Input data array
            feature_name: Name of the feature for parameter storage
            target_range: Target range for normalized values
            percentile_range: Use percentiles instead of min/max for robustness

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        # Handle masked arrays
        if hasattr(data, "mask"):
            valid_data = data.compressed()
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            warnings.warn(f"No valid data for feature {feature_name}")
            return data.copy(), NormalizationParams("min_max")

        if percentile_range is not None:
            # Use percentiles for robust normalization
            min_val = np.percentile(valid_data, percentile_range[0])
            max_val = np.percentile(valid_data, percentile_range[1])
        else:
            # Use actual min/max
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)

        data_range = max_val - min_val

        if data_range < self.epsilon:
            warnings.warn(f"Very small range for feature {feature_name}")
            normalized = np.full_like(data, target_range[0])
        else:
            # Scale to [0, 1] first, then to target range
            normalized_01 = (data - min_val) / data_range
            target_range_size = target_range[1] - target_range[0]
            normalized = normalized_01 * target_range_size + target_range[0]

        params = NormalizationParams(
            method="min_max",
            min_val=float(min_val),
            max_val=float(max_val),
            percentiles=percentile_range,
            epsilon=self.epsilon,
        )

        self.normalization_params[feature_name] = params
        return normalized, params

    def local_normalize(
        self,
        data: np.ndarray,
        window_size: int,
        feature_name: str = "unnamed",
        method: str = "z_score",
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply local (moving window) normalization.

        Args:
            data: Input 2D data array
            window_size: Size of local window
            feature_name: Name of the feature for parameter storage
            method: Normalization method ('z_score' or 'min_max')

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        if data.ndim != 2:
            raise ValueError("Local normalization requires 2D data")

        if method not in ["z_score", "min_max"]:
            raise ValueError("Method must be 'z_score' or 'min_max'")

        # Calculate local statistics using uniform filter (moving average)
        local_mean = uniform_filter(
            data.astype(np.float64), size=window_size, mode="reflect"
        )

        if method == "z_score":
            # Calculate local standard deviation
            local_var = (
                uniform_filter(
                    data.astype(np.float64) ** 2, size=window_size, mode="reflect"
                )
                - local_mean**2
            )
            local_std = np.sqrt(np.maximum(local_var, self.epsilon))

            # Apply local z-score normalization
            normalized = (data - local_mean) / local_std

        else:  # min_max
            # Calculate local min and max using erosion and dilation
            from scipy.ndimage import minimum_filter, maximum_filter

            local_min = minimum_filter(data, size=window_size, mode="reflect")
            local_max = maximum_filter(data, size=window_size, mode="reflect")

            local_range = local_max - local_min
            local_range = np.maximum(local_range, self.epsilon)

            # Apply local min-max normalization
            normalized = (data - local_min) / local_range

        params = NormalizationParams(
            method=f"local_{method}",
            local_window_size=window_size,
            epsilon=self.epsilon,
        )

        self.normalization_params[feature_name] = params
        return normalized, params

    def quantile_normalize(
        self, data: np.ndarray, feature_name: str = "unnamed", n_quantiles: int = 1000
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply quantile normalization (uniform distribution).

        Args:
            data: Input data array
            feature_name: Name of the feature for parameter storage
            n_quantiles: Number of quantiles to use

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        # Handle masked arrays
        if hasattr(data, "mask"):
            valid_data = data.compressed()
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            warnings.warn(f"No valid data for feature {feature_name}")
            return data.copy(), NormalizationParams("quantile")

        # Calculate quantiles
        quantiles = np.linspace(0, 100, n_quantiles)
        reference_quantiles = np.percentile(valid_data, quantiles)

        # Apply quantile transform
        normalized = np.zeros_like(data, dtype=np.float64)

        for i in range(len(data.flat)):
            value = data.flat[i]
            if np.isnan(value) or (hasattr(data, "mask") and data.mask.flat[i]):
                normalized.flat[i] = value
            else:
                # Find quantile rank
                rank = np.searchsorted(reference_quantiles, value) / len(
                    reference_quantiles
                )
                normalized.flat[i] = rank

        params = NormalizationParams(method="quantile", epsilon=self.epsilon)

        self.normalization_params[feature_name] = params
        return normalized, params

    def log_normalize(
        self,
        data: np.ndarray,
        feature_name: str = "unnamed",
        offset: Optional[float] = None,
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply log normalization for skewed data.

        Args:
            data: Input data array
            feature_name: Name of the feature for parameter storage
            offset: Offset to add before taking log (auto-calculated if None)

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        # Handle masked arrays
        if hasattr(data, "mask"):
            valid_data = data.compressed()
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            warnings.warn(f"No valid data for feature {feature_name}")
            return data.copy(), NormalizationParams("log")

        # Calculate offset if not provided
        if offset is None:
            min_val = np.min(valid_data)
            if min_val <= 0:
                offset = -min_val + self.epsilon
            else:
                offset = 0

        # Apply log transform
        log_data = np.log(data + offset)

        # Apply z-score normalization to log-transformed data
        normalized, _ = self.z_score_normalize(log_data, f"{feature_name}_log")

        params = NormalizationParams(
            method="log", min_val=float(offset), epsilon=self.epsilon
        )

        self.normalization_params[feature_name] = params
        return normalized, params

    def power_normalize(
        self,
        data: np.ndarray,
        feature_name: str = "unnamed",
        power: Optional[float] = None,
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Apply power transformation (Box-Cox like) for skewed data.

        Args:
            data: Input data array (must be positive)
            feature_name: Name of the feature for parameter storage
            power: Power parameter (auto-estimated if None)

        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        # Handle masked arrays
        if hasattr(data, "mask"):
            valid_data = data.compressed()
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            warnings.warn(f"No valid data for feature {feature_name}")
            return data.copy(), NormalizationParams("power")

        # Ensure positive values
        min_val = np.min(valid_data)
        if min_val <= 0:
            offset = -min_val + self.epsilon
            data_shifted = data + offset
        else:
            offset = 0
            data_shifted = data

        # Estimate optimal power if not provided
        if power is None:
            # Use Yeo-Johnson estimation for power parameter
            try:
                _, power = stats.yeojohnson(valid_data + offset)
            except:
                power = 0.5  # Default to square root transform

        # Apply power transform
        if abs(power) < self.epsilon:
            # Power ~ 0, use log transform
            transformed = np.log(data_shifted)
        else:
            transformed = np.power(data_shifted, power)

        # Apply z-score normalization
        normalized, _ = self.z_score_normalize(transformed, f"{feature_name}_power")

        params = NormalizationParams(
            method="power",
            min_val=float(offset),
            std=float(power),  # Store power in std field
            epsilon=self.epsilon,
        )

        self.normalization_params[feature_name] = params
        return normalized, params

    def compute_dimensionless_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute dimensionless terrain and hydrological features.

        Args:
            features: Dictionary of input features

        Returns:
            Dictionary of dimensionless features
        """
        dimensionless = {}

        # Topographic Position Index (TPI) - already dimensionless
        if "elevation" in features:
            elevation = features["elevation"]

            # Local relief (dimensionless when normalized by local elevation)
            if elevation.ndim == 2:
                local_mean = uniform_filter(elevation.astype(np.float64), size=9)
                tpi = elevation - local_mean
                dimensionless["tpi"] = tpi

                # Relative elevation
                local_min = uniform_filter(
                    elevation.astype(np.float64), size=15, mode="minimum"
                )
                local_max = uniform_filter(
                    elevation.astype(np.float64), size=15, mode="maximum"
                )
                local_range = local_max - local_min
                local_range = np.maximum(local_range, self.epsilon)

                relative_elevation = (elevation - local_min) / local_range
                dimensionless["relative_elevation"] = relative_elevation

        # Slope position (dimensionless)
        if "slope" in features and "elevation" in features:
            slope = features["slope"]
            elevation = features["elevation"]

            if slope.ndim == 2 and elevation.ndim == 2:
                # Local slope position
                local_slope_mean = uniform_filter(slope.astype(np.float64), size=9)
                slope_position = (slope - local_slope_mean) / (
                    local_slope_mean + self.epsilon
                )
                dimensionless["slope_position"] = slope_position

        # Curvature ratios (dimensionless)
        if "profile_curvature" in features and "planform_curvature" in features:
            prof_curv = features["profile_curvature"]
            plan_curv = features["planform_curvature"]

            # Curvature ratio
            curvature_magnitude = np.sqrt(prof_curv**2 + plan_curv**2) + self.epsilon
            curvature_ratio = prof_curv / curvature_magnitude
            dimensionless["curvature_ratio"] = curvature_ratio

        # Wetness contrast (dimensionless)
        if "twi" in features:
            twi = features["twi"]

            if twi.ndim == 2:
                local_twi_mean = uniform_filter(twi.astype(np.float64), size=9)
                twi_contrast = (twi - local_twi_mean) / (local_twi_mean + self.epsilon)
                dimensionless["twi_contrast"] = twi_contrast

        # HAND normalization (dimensionless when normalized by local relief)
        if "hand" in features and "elevation" in features:
            hand = features["hand"]
            elevation = features["elevation"]

            if hand.ndim == 2 and elevation.ndim == 2:
                local_relief = uniform_filter(
                    elevation.astype(np.float64), size=15, mode="maximum"
                ) - uniform_filter(
                    elevation.astype(np.float64), size=15, mode="minimum"
                )
                local_relief = np.maximum(local_relief, self.epsilon)

                normalized_hand = hand / local_relief
                dimensionless["normalized_hand"] = normalized_hand

        # Flow accumulation contrast (dimensionless)
        if "flow_accumulation" in features:
            flow_acc = features["flow_accumulation"]

            if flow_acc.ndim == 2:
                # Log transform first to handle large range
                log_flow_acc = np.log(flow_acc + 1)
                local_log_mean = uniform_filter(log_flow_acc.astype(np.float64), size=9)
                flow_contrast = (log_flow_acc - local_log_mean) / (
                    local_log_mean + self.epsilon
                )
                dimensionless["flow_contrast"] = flow_contrast

        # Roughness contrast (dimensionless)
        if "roughness" in features:
            roughness = features["roughness"]

            if roughness.ndim == 2:
                local_roughness_mean = uniform_filter(
                    roughness.astype(np.float64), size=9
                )
                roughness_contrast = (roughness - local_roughness_mean) / (
                    local_roughness_mean + self.epsilon
                )
                dimensionless["roughness_contrast"] = roughness_contrast

        # Aspect uniformity (dimensionless)
        if "aspect" in features:
            aspect = features["aspect"]

            if aspect.ndim == 2:
                # Convert aspect to unit vectors and calculate local uniformity
                aspect_rad = np.radians(aspect)
                cos_aspect = np.cos(aspect_rad)
                sin_aspect = np.sin(aspect_rad)

                local_cos_mean = uniform_filter(cos_aspect.astype(np.float64), size=9)
                local_sin_mean = uniform_filter(sin_aspect.astype(np.float64), size=9)

                # Vector magnitude (measure of local aspect uniformity)
                aspect_uniformity = np.sqrt(local_cos_mean**2 + local_sin_mean**2)
                dimensionless["aspect_uniformity"] = aspect_uniformity

        return dimensionless

    def normalize_feature_set(
        self, features: Dict[str, np.ndarray], normalization_config: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """Normalize a set of features using specified methods.

        Args:
            features: Dictionary of input features
            normalization_config: Dictionary mapping feature names to normalization methods

        Returns:
            Dictionary of normalized features
        """
        normalized_features = {}

        for feature_name, data in features.items():
            if feature_name in normalization_config:
                method = normalization_config[feature_name]

                try:
                    if method == "z_score":
                        normalized, _ = self.z_score_normalize(data, feature_name)
                    elif method == "z_score_robust":
                        normalized, _ = self.z_score_normalize(
                            data, feature_name, use_robust=True
                        )
                    elif method == "min_max":
                        normalized, _ = self.min_max_normalize(data, feature_name)
                    elif method == "min_max_robust":
                        normalized, _ = self.min_max_normalize(
                            data, feature_name, percentile_range=(5, 95)
                        )
                    elif method == "log":
                        normalized, _ = self.log_normalize(data, feature_name)
                    elif method == "power":
                        normalized, _ = self.power_normalize(data, feature_name)
                    elif method == "quantile":
                        normalized, _ = self.quantile_normalize(data, feature_name)
                    elif method.startswith("local_"):
                        window_size = 9  # Default window size
                        local_method = method.replace("local_", "")
                        normalized, _ = self.local_normalize(
                            data, window_size, feature_name, local_method
                        )
                    else:
                        warnings.warn(
                            f"Unknown normalization method '{method}' for feature '{feature_name}'"
                        )
                        normalized = data.copy()

                    normalized_features[feature_name] = normalized

                except Exception as e:
                    warnings.warn(
                        f"Failed to normalize feature '{feature_name}': {str(e)}"
                    )
                    normalized_features[feature_name] = data.copy()
            else:
                # No normalization specified, keep original
                normalized_features[feature_name] = data.copy()

        return normalized_features

    def apply_normalization(
        self, data: np.ndarray, params: NormalizationParams
    ) -> np.ndarray:
        """Apply normalization using stored parameters.

        Args:
            data: Input data to normalize
            params: Normalization parameters

        Returns:
            Normalized data
        """
        if params.method == "z_score" or params.method == "z_score_robust":
            return (data - params.mean) / params.std

        elif params.method == "min_max":
            data_range = params.max_val - params.min_val
            if data_range < params.epsilon:
                return np.zeros_like(data)
            return (data - params.min_val) / data_range

        elif params.method == "log":
            log_data = np.log(data + params.min_val)
            # Apply stored z-score parameters if available
            if params.mean is not None and params.std is not None:
                return (log_data - params.mean) / params.std
            return log_data

        elif params.method == "power":
            data_shifted = data + params.min_val
            power = params.std  # Power stored in std field

            if abs(power) < params.epsilon:
                transformed = np.log(data_shifted)
            else:
                transformed = np.power(data_shifted, power)

            # Apply stored z-score parameters if available
            if params.mean is not None:
                return (transformed - params.mean) / (params.std or 1.0)
            return transformed

        else:
            warnings.warn(f"Cannot apply normalization method: {params.method}")
            return data.copy()

    def save_normalization_params(self, output_path: Union[str, Path]) -> None:
        """Save normalization parameters to file.

        Args:
            output_path: Path to save parameters
        """
        output_path = Path(output_path)

        # Convert parameters to serializable format
        serializable_params = {}

        for feature_name, params in self.normalization_params.items():
            serializable_params[feature_name] = {
                "method": params.method,
                "mean": params.mean,
                "std": params.std,
                "min_val": params.min_val,
                "max_val": params.max_val,
                "percentiles": params.percentiles,
                "local_window_size": params.local_window_size,
                "epsilon": params.epsilon,
            }

        with open(output_path, "w") as f:
            json.dump(serializable_params, f, indent=2)

    def load_normalization_params(self, input_path: Union[str, Path]) -> None:
        """Load normalization parameters from file.

        Args:
            input_path: Path to load parameters from
        """
        input_path = Path(input_path)

        with open(input_path, "r") as f:
            serializable_params = json.load(f)

        # Convert back to NormalizationParams objects
        self.normalization_params = {}

        for feature_name, params_dict in serializable_params.items():
            self.normalization_params[feature_name] = NormalizationParams(
                method=params_dict["method"],
                mean=params_dict.get("mean"),
                std=params_dict.get("std"),
                min_val=params_dict.get("min_val"),
                max_val=params_dict.get("max_val"),
                percentiles=(
                    tuple(params_dict["percentiles"])
                    if params_dict.get("percentiles")
                    else None
                ),
                local_window_size=params_dict.get("local_window_size"),
                epsilon=params_dict.get("epsilon", 1e-8),
            )

    def get_feature_statistics(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for features before and after normalization.

        Args:
            features: Dictionary of feature arrays

        Returns:
            Dictionary of feature statistics
        """
        statistics = {}

        for feature_name, data in features.items():
            # Handle masked arrays
            if hasattr(data, "mask"):
                valid_data = data.compressed()
            else:
                valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                statistics[feature_name] = {
                    "count": len(valid_data),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "median": float(np.median(valid_data)),
                    "skewness": float(stats.skew(valid_data)),
                    "kurtosis": float(stats.kurtosis(valid_data)),
                    "q25": float(np.percentile(valid_data, 25)),
                    "q75": float(np.percentile(valid_data, 75)),
                }
            else:
                statistics[feature_name] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                }

        return statistics
