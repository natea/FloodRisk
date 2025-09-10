"""
Validation Metrics for Flood Risk Models

Implements comprehensive metrics for evaluating flood risk predictions:
- Intersection over Union (IoU)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Critical Success Index (CSI)
- F1 Score and related classification metrics
- Bias, Nash-Sutcliffe Efficiency, and hydrological metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricError(Exception):
    """Custom exception for metric calculation errors"""

    pass


class BaseMetric(ABC):
    """Abstract base class for all metrics"""

    @abstractmethod
    def calculate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Union[float, Dict]:
        """Calculate the metric"""
        pass

    def _validate_inputs(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> None:
        """Validate input arrays"""
        if predictions.shape != observations.shape:
            raise MetricError(
                f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}"
            )

        if predictions.size == 0 or observations.size == 0:
            raise MetricError("Empty arrays provided")

        if np.any(np.isnan(predictions)) or np.any(np.isnan(observations)):
            warnings.warn("NaN values detected in input arrays")


class IoUCalculator(BaseMetric):
    """
    Intersection over Union (IoU) Calculator for flood extent validation

    IoU = Area of Overlap / Area of Union
    Perfect score = 1.0, Worst score = 0.0
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize IoU calculator

        Args:
            threshold: Binary classification threshold for continuous predictions
        """
        self.threshold = threshold
        logger.info(f"IoU Calculator initialized with threshold={threshold}")

    def calculate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate IoU for flood extent

        Args:
            predictions: Predicted flood depths/probabilities
            observations: Observed flood depths/binary flood extent

        Returns:
            Dictionary with IoU score and components
        """
        try:
            self._validate_inputs(predictions, observations)

            # Binarize predictions if continuous
            if predictions.dtype != bool and np.max(predictions) > 1.0:
                pred_binary = predictions > self.threshold
            else:
                pred_binary = predictions.astype(bool)

            # Binarize observations if continuous
            if observations.dtype != bool and np.max(observations) > 1.0:
                obs_binary = observations > self.threshold
            else:
                obs_binary = observations.astype(bool)

            # Calculate intersection and union
            intersection = np.logical_and(pred_binary, obs_binary)
            union = np.logical_or(pred_binary, obs_binary)

            intersection_area = np.sum(intersection)
            union_area = np.sum(union)

            # Handle edge case where union is zero
            if union_area == 0:
                iou_score = 1.0 if intersection_area == 0 else 0.0
                logger.warning(
                    "Union area is zero - no flood extent in either prediction or observation"
                )
            else:
                iou_score = intersection_area / union_area

            result = {
                "iou": float(iou_score),
                "intersection_area": int(intersection_area),
                "union_area": int(union_area),
                "predicted_area": int(np.sum(pred_binary)),
                "observed_area": int(np.sum(obs_binary)),
            }

            logger.info(f"IoU calculated: {iou_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating IoU: {e}")
            raise MetricError(f"IoU calculation failed: {e}")


class RegressionMetrics(BaseMetric):
    """
    Regression metrics for continuous flood depth validation

    Includes: MAE, RMSE, Bias, R², Nash-Sutcliffe Efficiency
    """

    def calculate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics

        Args:
            predictions: Predicted flood depths
            observations: Observed flood depths

        Returns:
            Dictionary with all regression metrics
        """
        try:
            self._validate_inputs(predictions, observations)

            # Remove NaN values
            mask = ~(np.isnan(predictions) | np.isnan(observations))
            pred_clean = predictions[mask]
            obs_clean = observations[mask]

            if len(pred_clean) == 0:
                raise MetricError("No valid data points after removing NaN values")

            # Calculate basic metrics
            mae = self._calculate_mae(pred_clean, obs_clean)
            rmse = self._calculate_rmse(pred_clean, obs_clean)
            bias = self._calculate_bias(pred_clean, obs_clean)
            r_squared = self._calculate_r_squared(pred_clean, obs_clean)
            nse = self._calculate_nse(pred_clean, obs_clean)

            # Additional metrics
            mape = self._calculate_mape(pred_clean, obs_clean)
            pbias = self._calculate_pbias(pred_clean, obs_clean)
            kge = self._calculate_kge(pred_clean, obs_clean)

            result = {
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "r_squared": r_squared,
                "nse": nse,
                "mape": mape,
                "pbias": pbias,
                "kge": kge,
                "n_samples": len(pred_clean),
            }

            logger.info(
                f"Regression metrics calculated: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r_squared:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            raise MetricError(f"Regression metrics calculation failed: {e}")

    def _calculate_mae(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(predictions - observations)))

    def _calculate_rmse(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Root Mean Square Error"""
        return float(np.sqrt(np.mean((predictions - observations) ** 2)))

    def _calculate_bias(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Mean Bias"""
        return float(np.mean(predictions - observations))

    def _calculate_r_squared(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Coefficient of Determination (R²)"""
        ss_res = np.sum((observations - predictions) ** 2)
        ss_tot = np.sum((observations - np.mean(observations)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1 - (ss_res / ss_tot))

    def _calculate_nse(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Nash-Sutcliffe Efficiency"""
        numerator = np.sum((observations - predictions) ** 2)
        denominator = np.sum((observations - np.mean(observations)) ** 2)

        if denominator == 0:
            return 1.0 if numerator == 0 else -np.inf

        return float(1 - (numerator / denominator))

    def _calculate_mape(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = observations != 0
        if not np.any(mask):
            return np.inf

        return float(
            np.mean(
                np.abs((observations[mask] - predictions[mask]) / observations[mask])
                * 100
            )
        )

    def _calculate_pbias(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Percent Bias"""
        sum_obs = np.sum(observations)
        if sum_obs == 0:
            return np.inf if np.sum(predictions) != 0 else 0.0

        return float(100 * np.sum(predictions - observations) / sum_obs)

    def _calculate_kge(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> float:
        """Kling-Gupta Efficiency"""
        # Correlation coefficient
        r = np.corrcoef(predictions, observations)[0, 1]
        if np.isnan(r):
            r = 0.0

        # Bias ratio
        alpha = (
            np.std(predictions) / np.std(observations)
            if np.std(observations) != 0
            else 1.0
        )

        # Variability ratio
        beta = (
            np.mean(predictions) / np.mean(observations)
            if np.mean(observations) != 0
            else 1.0
        )

        # KGE calculation
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return float(kge)


class ClassificationMetrics(BaseMetric):
    """
    Classification metrics for binary flood/no-flood validation

    Includes: Precision, Recall, F1, Accuracy, Specificity, False Alarm Rate
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize classification metrics calculator

        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
        logger.info(f"Classification Metrics initialized with threshold={threshold}")

    def calculate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics

        Args:
            predictions: Predicted flood probabilities or depths
            observations: Binary flood observations

        Returns:
            Dictionary with all classification metrics
        """
        try:
            self._validate_inputs(predictions, observations)

            # Binarize predictions
            pred_binary = (predictions > self.threshold).astype(int)
            obs_binary = observations.astype(int)

            # Calculate confusion matrix
            tp, fp, tn, fn = self._calculate_confusion_matrix(pred_binary, obs_binary)

            # Calculate metrics
            precision = self._safe_divide(tp, tp + fp)
            recall = self._safe_divide(tp, tp + fn)
            f1 = self._safe_divide(2 * precision * recall, precision + recall)
            accuracy = self._safe_divide(tp + tn, tp + fp + tn + fn)
            specificity = self._safe_divide(tn, tn + fp)
            far = self._safe_divide(fp, tp + fp)  # False Alarm Rate
            pod = recall  # Probability of Detection (same as recall)

            # Additional metrics
            balanced_accuracy = (recall + specificity) / 2
            mcc = self._calculate_mcc(
                tp, fp, tn, fn
            )  # Matthews Correlation Coefficient

            result = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "specificity": specificity,
                "false_alarm_rate": far,
                "probability_of_detection": pod,
                "balanced_accuracy": balanced_accuracy,
                "matthews_correlation": mcc,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }

            logger.info(
                f"Classification metrics calculated: F1={f1:.4f}, Accuracy={accuracy:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            raise MetricError(f"Classification metrics calculation failed: {e}")

    def _calculate_confusion_matrix(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix components"""
        tp = np.sum((predictions == 1) & (observations == 1))
        fp = np.sum((predictions == 1) & (observations == 0))
        tn = np.sum((predictions == 0) & (observations == 0))
        fn = np.sum((predictions == 0) & (observations == 1))

        return tp, fp, tn, fn

    def _safe_divide(
        self, numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """Safe division with default value"""
        return float(numerator / denominator if denominator != 0 else default)

    def _calculate_mcc(self, tp: int, fp: int, tn: int, fn: int) -> float:
        """Matthews Correlation Coefficient"""
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return self._safe_divide(numerator, denominator)


class CriticalSuccessIndex(BaseMetric):
    """
    Critical Success Index (CSI) / Threat Score

    CSI = True Positives / (True Positives + False Positives + False Negatives)
    Commonly used in meteorology and flood forecasting
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize CSI calculator

        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
        logger.info(f"CSI Calculator initialized with threshold={threshold}")

    def calculate(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Critical Success Index

        Args:
            predictions: Predicted flood probabilities or depths
            observations: Binary flood observations

        Returns:
            Dictionary with CSI and related metrics
        """
        try:
            self._validate_inputs(predictions, observations)

            # Binarize predictions
            pred_binary = (predictions > self.threshold).astype(int)
            obs_binary = observations.astype(int)

            # Calculate confusion matrix components
            tp = np.sum((pred_binary == 1) & (obs_binary == 1))
            fp = np.sum((pred_binary == 1) & (obs_binary == 0))
            fn = np.sum((pred_binary == 0) & (obs_binary == 1))

            # Calculate CSI
            denominator = tp + fp + fn
            if denominator == 0:
                csi = 1.0 if tp == 0 else 0.0
                logger.warning(
                    "No positive predictions or observations for CSI calculation"
                )
            else:
                csi = tp / denominator

            # Calculate related metrics
            bias_score = (tp + fp) / (tp + fn) if (tp + fn) > 0 else np.inf

            result = {
                "csi": float(csi),
                "threat_score": float(csi),  # CSI is also known as threat score
                "bias_score": float(bias_score),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

            logger.info(f"CSI calculated: {csi:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating CSI: {e}")
            raise MetricError(f"CSI calculation failed: {e}")


class MetricsCalculator:
    """
    Main metrics calculator that orchestrates all individual metric calculations
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize comprehensive metrics calculator

        Args:
            threshold: Default threshold for binary classification metrics
        """
        self.threshold = threshold
        self.iou_calc = IoUCalculator(threshold)
        self.regression_calc = RegressionMetrics()
        self.classification_calc = ClassificationMetrics(threshold)
        self.csi_calc = CriticalSuccessIndex(threshold)

        logger.info(f"MetricsCalculator initialized with threshold={threshold}")

    def calculate_all_metrics(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Calculate all available metrics

        Args:
            predictions: Model predictions
            observations: Ground truth observations

        Returns:
            Dictionary with all metric categories and their results
        """
        try:
            logger.info("Starting comprehensive metrics calculation")

            results = {}

            # IoU metrics
            results["iou"] = self.iou_calc.calculate(predictions, observations)

            # Regression metrics (for continuous data)
            try:
                results["regression"] = self.regression_calc.calculate(
                    predictions, observations
                )
            except MetricError as e:
                logger.warning(f"Regression metrics failed: {e}")
                results["regression"] = {"error": str(e)}

            # Classification metrics
            try:
                results["classification"] = self.classification_calc.calculate(
                    predictions, observations
                )
            except MetricError as e:
                logger.warning(f"Classification metrics failed: {e}")
                results["classification"] = {"error": str(e)}

            # CSI metrics
            results["csi"] = self.csi_calc.calculate(predictions, observations)

            logger.info("Comprehensive metrics calculation completed")
            return results

        except Exception as e:
            logger.error(f"Error in comprehensive metrics calculation: {e}")
            raise MetricError(f"Comprehensive metrics calculation failed: {e}")

    def calculate_iou(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """Calculate IoU metrics only"""
        return self.iou_calc.calculate(predictions, observations)

    def calculate_regression_metrics(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics only"""
        return self.regression_calc.calculate(predictions, observations)

    def calculate_classification_metrics(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics only"""
        return self.classification_calc.calculate(predictions, observations)

    def calculate_csi(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> Dict[str, float]:
        """Calculate CSI metrics only"""
        return self.csi_calc.calculate(predictions, observations)

    def get_metrics_summary(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a summary DataFrame of key metrics

        Args:
            results: Results from calculate_all_metrics

        Returns:
            DataFrame with summary metrics
        """
        summary_data = []

        # Extract key metrics from each category
        key_metrics = {
            "iou": ["iou"],
            "regression": ["mae", "rmse", "r_squared", "nse"],
            "classification": ["f1_score", "accuracy", "precision", "recall"],
            "csi": ["csi", "bias_score"],
        }

        for category, metrics in key_metrics.items():
            if category in results and "error" not in results[category]:
                for metric in metrics:
                    if metric in results[category]:
                        summary_data.append(
                            {
                                "Metric": f"{category.upper()}_{metric.upper()}",
                                "Value": results[category][metric],
                                "Category": category.capitalize(),
                            }
                        )

        return pd.DataFrame(summary_data)


# Utility functions for data preprocessing
def preprocess_flood_data(
    predictions: np.ndarray,
    observations: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess flood data for validation

    Args:
        predictions: Raw predictions
        observations: Raw observations
        min_depth: Minimum depth to consider as flood
        max_depth: Maximum depth to clip values

    Returns:
        Preprocessed predictions and observations
    """
    logger.info(
        f"Preprocessing flood data with min_depth={min_depth}, max_depth={max_depth}"
    )

    # Clip negative values
    predictions = np.maximum(predictions, 0)
    observations = np.maximum(observations, 0)

    # Apply minimum depth threshold
    predictions[predictions < min_depth] = 0
    observations[observations < min_depth] = 0

    # Apply maximum depth clipping if specified
    if max_depth is not None:
        predictions = np.minimum(predictions, max_depth)
        observations = np.minimum(observations, max_depth)

    logger.info("Data preprocessing completed")
    return predictions, observations


def calculate_flood_extent_metrics(
    depth_predictions: np.ndarray,
    depth_observations: np.ndarray,
    depth_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Calculate flood extent specific metrics

    Args:
        depth_predictions: Predicted flood depths
        depth_observations: Observed flood depths
        depth_threshold: Threshold to determine flood extent

    Returns:
        Dictionary with flood extent metrics
    """
    logger.info(f"Calculating flood extent metrics with threshold={depth_threshold}")

    # Convert depths to binary flood extent
    pred_extent = (depth_predictions > depth_threshold).astype(int)
    obs_extent = (depth_observations > depth_threshold).astype(int)

    # Calculate metrics
    calc = MetricsCalculator()
    iou_results = calc.calculate_iou(pred_extent, obs_extent)
    class_results = calc.calculate_classification_metrics(pred_extent, obs_extent)
    csi_results = calc.calculate_csi(pred_extent, obs_extent)

    # Combine results
    extent_metrics = {
        **iou_results,
        **{
            k: v
            for k, v in class_results.items()
            if k in ["f1_score", "precision", "recall", "accuracy"]
        },
        **{k: v for k, v in csi_results.items() if k in ["csi", "threat_score"]},
    }

    logger.info("Flood extent metrics calculation completed")
    return extent_metrics
