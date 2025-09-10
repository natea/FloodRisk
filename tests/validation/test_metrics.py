"""
Test suite for validation metrics module

Tests all metric calculations including:
- IoU (Intersection over Union)
- Classification metrics (Precision, Recall, F1, Accuracy)
- Regression metrics (MAE, RMSE, R², NSE)
- Critical Success Index (CSI)
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import Mock, patch

# Import the modules to test
from src.validation.metrics import (
    MetricsCalculator,
    IoUCalculator,
    RegressionMetrics,
    ClassificationMetrics,
    CriticalSuccessIndex,
    MetricError,
    preprocess_flood_data,
    calculate_flood_extent_metrics,
)


class TestIoUCalculator:
    """Test cases for IoU Calculator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.iou_calc = IoUCalculator(threshold=0.5)

    def test_perfect_iou(self):
        """Test IoU calculation with perfect overlap"""
        predictions = np.array([[1, 1, 0], [0, 1, 1]])
        observations = np.array([[1, 1, 0], [0, 1, 1]])

        result = self.iou_calc.calculate(predictions, observations)

        assert result["iou"] == 1.0
        assert result["intersection_area"] == 4
        assert result["union_area"] == 4
        assert result["predicted_area"] == 4
        assert result["observed_area"] == 4

    def test_no_overlap_iou(self):
        """Test IoU calculation with no overlap"""
        predictions = np.array([[1, 1, 0], [0, 0, 0]])
        observations = np.array([[0, 0, 0], [0, 1, 1]])

        result = self.iou_calc.calculate(predictions, observations)

        assert result["iou"] == 0.0
        assert result["intersection_area"] == 0
        assert result["union_area"] == 4
        assert result["predicted_area"] == 2
        assert result["observed_area"] == 2

    def test_partial_overlap_iou(self):
        """Test IoU calculation with partial overlap"""
        predictions = np.array([[1, 1, 0], [1, 0, 0]])
        observations = np.array([[1, 0, 0], [1, 1, 0]])

        result = self.iou_calc.calculate(predictions, observations)

        # Intersection: 2 cells, Union: 4 cells
        expected_iou = 2.0 / 4.0
        assert result["iou"] == expected_iou
        assert result["intersection_area"] == 2
        assert result["union_area"] == 4

    def test_continuous_data_iou(self):
        """Test IoU with continuous depth data"""
        predictions = np.array([[0.8, 0.6, 0.1], [0.3, 0.9, 0.7]])
        observations = np.array([[0.7, 0.2, 0.0], [0.1, 0.8, 0.6]])

        result = self.iou_calc.calculate(predictions, observations)

        # With threshold 0.5: pred=[1,1,0,0,1,1], obs=[1,0,0,0,1,1]
        # Intersection: 3, Union: 4
        expected_iou = 3.0 / 4.0
        assert result["iou"] == expected_iou

    def test_empty_arrays_iou(self):
        """Test IoU with empty arrays"""
        predictions = np.array([])
        observations = np.array([])

        with pytest.raises(MetricError):
            self.iou_calc.calculate(predictions, observations)

    def test_shape_mismatch_iou(self):
        """Test IoU with mismatched array shapes"""
        predictions = np.array([[1, 0]])
        observations = np.array([[1], [0]])

        with pytest.raises(MetricError):
            self.iou_calc.calculate(predictions, observations)

    def test_zero_union_case(self):
        """Test IoU when both arrays are all zeros"""
        predictions = np.zeros((3, 3))
        observations = np.zeros((3, 3))

        result = self.iou_calc.calculate(predictions, observations)

        assert result["iou"] == 1.0  # Perfect match when both are empty
        assert result["union_area"] == 0


class TestRegressionMetrics:
    """Test cases for Regression Metrics"""

    def setup_method(self):
        """Setup test fixtures"""
        self.reg_calc = RegressionMetrics()

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        observations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = self.reg_calc.calculate(predictions, observations)

        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["bias"] == 0.0
        assert result["r_squared"] == 1.0
        assert result["nse"] == 1.0
        assert result["mape"] == 0.0
        assert result["pbias"] == 0.0
        assert result["kge"] == 1.0

    def test_constant_predictions(self):
        """Test metrics with constant predictions"""
        predictions = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        observations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = self.reg_calc.calculate(predictions, observations)

        expected_mae = np.mean([2.0, 1.0, 0.0, 1.0, 2.0])  # = 1.2
        expected_rmse = np.sqrt(np.mean([4.0, 1.0, 0.0, 1.0, 4.0]))  # = sqrt(2)

        assert result["mae"] == expected_mae
        assert result["rmse"] == expected_rmse
        assert result["bias"] == 0.0  # Mean of predictions = mean of observations

    def test_systematic_bias(self):
        """Test metrics with systematic bias"""
        predictions = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        observations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = self.reg_calc.calculate(predictions, observations)

        assert result["bias"] == 1.0  # Systematic overestimation
        assert result["mae"] == 1.0
        assert result["rmse"] == 1.0

    def test_nan_handling(self):
        """Test handling of NaN values"""
        predictions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        observations = np.array([1.1, np.nan, 3.0, 4.1, 5.1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.reg_calc.calculate(predictions, observations)

        # Should only use valid pairs: (1.0, 1.1), (4.0, 4.1), (5.0, 5.1)
        assert result["n_samples"] == 3
        assert "mae" in result
        assert "rmse" in result

    def test_zero_observations(self):
        """Test handling of zero observations for MAPE"""
        predictions = np.array([1.0, 2.0, 3.0])
        observations = np.array([0.0, 0.0, 0.0])

        result = self.reg_calc.calculate(predictions, observations)

        assert result["mape"] == np.inf  # Division by zero
        assert result["pbias"] == np.inf

    def test_nse_edge_cases(self):
        """Test Nash-Sutcliffe Efficiency edge cases"""
        # Case where variance of observations is zero
        predictions = np.array([1.0, 1.0, 1.0])
        observations = np.array([2.0, 2.0, 2.0])

        result = self.reg_calc.calculate(predictions, observations)

        # When observations have no variance, NSE should be 1.0 if predictions are perfect
        # or -inf if predictions are wrong
        assert result["nse"] == -np.inf


class TestClassificationMetrics:
    """Test cases for Classification Metrics"""

    def setup_method(self):
        """Setup test fixtures"""
        self.class_calc = ClassificationMetrics(threshold=0.5)

    def test_perfect_classification(self):
        """Test metrics with perfect classification"""
        predictions = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        observations = np.array([0, 1, 1, 0, 1])

        result = self.class_calc.calculate(predictions, observations)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["accuracy"] == 1.0
        assert result["specificity"] == 1.0
        assert result["false_alarm_rate"] == 0.0

    def test_all_negative_predictions(self):
        """Test with all negative predictions"""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.1])
        observations = np.array([0, 0, 1, 1, 0])

        result = self.class_calc.calculate(predictions, observations)

        # All predictions are 0, observations have 2 positives
        assert result["precision"] == 0.0  # No true positives
        assert result["recall"] == 0.0  # Missed all positives
        assert result["f1_score"] == 0.0
        assert result["false_negatives"] == 2
        assert result["true_negatives"] == 3

    def test_all_positive_predictions(self):
        """Test with all positive predictions"""
        predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.9])
        observations = np.array([1, 1, 0, 0, 1])

        result = self.class_calc.calculate(predictions, observations)

        # All predictions are 1, observations have 3 positives and 2 negatives
        assert result["recall"] == 1.0  # Found all positives
        assert (
            result["false_positives"] == 2
        )  # But also predicted negatives as positive
        assert result["precision"] == 3.0 / 5.0  # 3 TP out of 5 predictions

    def test_balanced_classification(self):
        """Test with balanced dataset"""
        predictions = np.array([0.8, 0.3, 0.7, 0.2])
        observations = np.array([1, 0, 1, 0])

        result = self.class_calc.calculate(predictions, observations)

        # Perfect classification: TP=2, TN=2, FP=0, FN=0
        assert result["accuracy"] == 1.0
        assert result["balanced_accuracy"] == 1.0
        assert result["matthews_correlation"] == 1.0

    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation"""
        predictions = np.array([0.8, 0.3, 0.7, 0.8, 0.2])
        observations = np.array([1, 1, 0, 1, 0])

        result = self.class_calc.calculate(predictions, observations)

        # pred_binary = [1, 0, 1, 1, 0]
        # obs_binary  = [1, 1, 0, 1, 0]
        # TP: 2 (indices 0, 3), FP: 1 (index 2), TN: 1 (index 4), FN: 1 (index 1)

        assert result["true_positives"] == 2
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["false_negatives"] == 1

    def test_mcc_calculation(self):
        """Test Matthews Correlation Coefficient calculation"""
        # Test case with known MCC value
        predictions = np.array([1, 1, 0, 0])
        observations = np.array([1, 0, 1, 0])

        result = self.class_calc.calculate(predictions, observations)

        # TP=1, FP=1, TN=1, FN=1
        # MCC = (1*1 - 1*1) / sqrt((1+1)*(1+1)*(1+1)*(1+1)) = 0/4 = 0
        assert result["matthews_correlation"] == 0.0


class TestCriticalSuccessIndex:
    """Test cases for Critical Success Index"""

    def setup_method(self):
        """Setup test fixtures"""
        self.csi_calc = CriticalSuccessIndex(threshold=0.5)

    def test_perfect_csi(self):
        """Test CSI with perfect predictions"""
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        observations = np.array([1, 0, 1, 0])

        result = self.csi_calc.calculate(predictions, observations)

        assert result["csi"] == 1.0
        assert result["threat_score"] == 1.0  # CSI == Threat Score
        assert result["bias_score"] == 1.0

    def test_no_events_csi(self):
        """Test CSI when no events occur"""
        predictions = np.array([0.1, 0.2, 0.1, 0.3])
        observations = np.array([0, 0, 0, 0])

        result = self.csi_calc.calculate(predictions, observations)

        # No true positives, no false positives, no false negatives
        assert result["csi"] == 1.0  # Perfect score when no events
        assert result["bias_score"] == np.inf  # Division by zero

    def test_bias_calculation(self):
        """Test bias score calculation"""
        predictions = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
        observations = np.array([1, 1, 0, 0, 0])

        result = self.csi_calc.calculate(predictions, observations)

        # pred_binary = [1, 1, 1, 0, 0] -> 3 predicted events
        # obs_binary = [1, 1, 0, 0, 0] -> 2 observed events
        # Bias = (TP + FP) / (TP + FN) = 3 / 2 = 1.5

        assert result["bias_score"] == 1.5  # Over-forecasting bias

    def test_csi_with_misses_only(self):
        """Test CSI when all events are missed"""
        predictions = np.array([0.1, 0.2, 0.1])
        observations = np.array([1, 1, 1])

        result = self.csi_calc.calculate(predictions, observations)

        # TP=0, FP=0, FN=3
        # CSI = 0 / (0 + 0 + 3) = 0
        assert result["csi"] == 0.0
        assert result["false_negatives"] == 3


class TestMetricsCalculator:
    """Test cases for the main MetricsCalculator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.calc = MetricsCalculator(threshold=0.5)

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics together"""
        predictions = np.array([[0.8, 0.3, 0.1], [0.7, 0.9, 0.2]])
        observations = np.array([[0.9, 0.1, 0.0], [0.6, 0.8, 0.0]])

        results = self.calc.calculate_all_metrics(predictions, observations)

        # Check that all metric categories are present
        assert "iou" in results
        assert "regression" in results
        assert "classification" in results
        assert "csi" in results

        # Check that each category has expected keys
        assert "iou" in results["iou"]
        assert "mae" in results["regression"]
        assert "f1_score" in results["classification"]
        assert "csi" in results["csi"]

    def test_individual_metric_methods(self):
        """Test individual metric calculation methods"""
        predictions = np.array([0.8, 0.3, 0.7, 0.2])
        observations = np.array([0.9, 0.1, 0.6, 0.0])

        iou_result = self.calc.calculate_iou(predictions, observations)
        reg_result = self.calc.calculate_regression_metrics(predictions, observations)
        class_result = self.calc.calculate_classification_metrics(
            predictions, observations
        )
        csi_result = self.calc.calculate_csi(predictions, observations)

        assert isinstance(iou_result, dict)
        assert isinstance(reg_result, dict)
        assert isinstance(class_result, dict)
        assert isinstance(csi_result, dict)

        assert "iou" in iou_result
        assert "mae" in reg_result
        assert "f1_score" in class_result
        assert "csi" in csi_result

    def test_get_metrics_summary(self):
        """Test metrics summary DataFrame generation"""
        predictions = np.array([[0.8, 0.3], [0.7, 0.9]])
        observations = np.array([[0.9, 0.1], [0.6, 0.8]])

        results = self.calc.calculate_all_metrics(predictions, observations)
        summary_df = self.calc.get_metrics_summary(results)

        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert "Metric" in summary_df.columns
        assert "Value" in summary_df.columns
        assert "Category" in summary_df.columns

        # Check that key metrics are present
        metric_names = summary_df["Metric"].tolist()
        assert any("IOU" in name for name in metric_names)
        assert any("F1_SCORE" in name for name in metric_names)


class TestUtilityFunctions:
    """Test cases for utility functions"""

    def test_preprocess_flood_data(self):
        """Test flood data preprocessing"""
        predictions = np.array([[-0.5, 0.005, 1.2], [0.02, 100.0, 0.5]])
        observations = np.array([[-0.1, 0.0, 1.0], [0.01, 95.0, 0.6]])

        pred_clean, obs_clean = preprocess_flood_data(
            predictions, observations, min_depth=0.01, max_depth=50.0
        )

        # Check negative values are clipped
        assert np.all(pred_clean >= 0)
        assert np.all(obs_clean >= 0)

        # Check minimum threshold applied
        assert pred_clean[0, 1] == 0.0  # 0.005 < 0.01
        assert obs_clean[0, 1] == 0.0  # 0.0 < 0.01

        # Check maximum clipping
        assert pred_clean[1, 1] == 50.0  # 100.0 clipped to 50.0
        assert obs_clean[1, 1] == 50.0  # 95.0 clipped to 50.0

    def test_calculate_flood_extent_metrics(self):
        """Test flood extent specific metrics calculation"""
        depth_predictions = np.array([[0.05, 0.0, 0.3], [0.02, 0.15, 0.0]])
        depth_observations = np.array([[0.08, 0.0, 0.25], [0.0, 0.12, 0.0]])

        extent_metrics = calculate_flood_extent_metrics(
            depth_predictions, depth_observations, depth_threshold=0.01
        )

        # Check that flood extent metrics are calculated
        assert "iou" in extent_metrics
        assert "f1_score" in extent_metrics
        assert "precision" in extent_metrics
        assert "recall" in extent_metrics
        assert "csi" in extent_metrics

        # Check values are reasonable
        assert 0 <= extent_metrics["iou"] <= 1
        assert 0 <= extent_metrics["f1_score"] <= 1


class TestErrorHandling:
    """Test cases for error handling and edge cases"""

    def test_empty_array_handling(self):
        """Test handling of empty arrays"""
        calc = MetricsCalculator()

        with pytest.raises(MetricError):
            calc.calculate_iou(np.array([]), np.array([]))

    def test_shape_mismatch_handling(self):
        """Test handling of shape mismatches"""
        calc = MetricsCalculator()

        predictions = np.array([[1, 2]])
        observations = np.array([[1], [2]])

        with pytest.raises(MetricError):
            calc.calculate_iou(predictions, observations)

    def test_nan_warning_handling(self):
        """Test that NaN warnings are properly issued"""
        calc = MetricsCalculator()

        predictions = np.array([1.0, np.nan, 3.0])
        observations = np.array([1.1, 2.0, np.nan])

        with pytest.warns(UserWarning):
            result = calc.calculate_regression_metrics(predictions, observations)
            assert isinstance(result, dict)

    def test_all_nan_handling(self):
        """Test handling when all values are NaN"""
        reg_calc = RegressionMetrics()

        predictions = np.array([np.nan, np.nan, np.nan])
        observations = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(MetricError):
            reg_calc.calculate(predictions, observations)


@pytest.fixture
def sample_flood_data():
    """Fixture providing sample flood data for testing"""
    np.random.seed(42)  # For reproducible results

    # Create synthetic flood depth data
    predictions = np.random.lognormal(0, 0.5, (10, 10))
    observations = predictions + np.random.normal(0, 0.1, (10, 10))

    # Add some areas with no flooding
    predictions[0:3, 0:3] = 0
    observations[0:3, 0:3] = 0

    # Ensure no negative values
    predictions = np.maximum(predictions, 0)
    observations = np.maximum(observations, 0)

    return predictions, observations


def test_integration_with_sample_data(sample_flood_data):
    """Integration test with realistic sample data"""
    predictions, observations = sample_flood_data

    calc = MetricsCalculator(threshold=0.01)
    results = calc.calculate_all_metrics(predictions, observations)

    # Verify all metrics are calculated
    assert all(
        category in results
        for category in ["iou", "regression", "classification", "csi"]
    )

    # Verify results are reasonable
    assert 0 <= results["iou"]["iou"] <= 1
    assert results["regression"]["mae"] >= 0
    assert results["regression"]["rmse"] >= results["regression"]["mae"]
    assert 0 <= results["classification"]["accuracy"] <= 1
    assert 0 <= results["csi"]["csi"] <= 1

    # Test summary generation
    summary = calc.get_metrics_summary(results)
    assert len(summary) > 0
    assert all(col in summary.columns for col in ["Metric", "Value", "Category"])


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
