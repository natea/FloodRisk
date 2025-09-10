"""Quality assurance tools for flood risk preprocessing pipeline.

This module provides comprehensive QA functionality including:
- Data validation and consistency checks
- Spatial alignment verification
- Statistical quality metrics
- Visualization tools for QA
- Performance monitoring
"""

import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QAMetrics:
    """Container for QA metrics."""

    metric_name: str
    value: float
    status: str  # 'pass', 'warning', 'fail'
    threshold: Optional[float] = None
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PreprocessingQA:
    """Comprehensive QA system for preprocessing pipeline."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize QA system.

        Args:
            config: QA configuration dictionary
        """
        self.config = config or self._default_config()
        self.metrics_history = []
        self.reports = []

    def _default_config(self) -> Dict[str, Any]:
        """Default QA configuration."""
        return {
            "thresholds": {
                "nodata_percent_max": 10.0,
                "correlation_min": 0.1,
                "alignment_tolerance_m": 1.0,
                "elevation_range_valid": (0, 5000),
                "slope_range_valid": (0, 90),
                "flow_acc_min": 1.0,
                "hand_range_valid": (0, 1000),
                "spatial_resolution_tolerance": 0.1,
            },
            "visualization": {
                "figsize": (12, 8),
                "dpi": 300,
                "save_plots": True,
                "plot_format": "png",
            },
            "reporting": {
                "save_detailed": True,
                "save_summary": True,
                "include_plots": True,
            },
        }

    def run_comprehensive_qa(
        self,
        processed_data: Dict[str, xr.DataArray],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive QA on processed data.

        Args:
            processed_data: Dictionary of processed data arrays
            output_dir: Optional directory for QA outputs

        Returns:
            QA results dictionary
        """
        logger.info("Running comprehensive preprocessing QA")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        qa_results = {
            "timestamp": datetime.now(),
            "data_validation": {},
            "spatial_validation": {},
            "statistical_validation": {},
            "consistency_validation": {},
            "performance_metrics": {},
            "overall_status": "unknown",
        }

        # Data validation
        logger.info("Running data validation checks")
        qa_results["data_validation"] = self._validate_data_quality(processed_data)

        # Spatial validation
        logger.info("Running spatial validation checks")
        qa_results["spatial_validation"] = self._validate_spatial_consistency(
            processed_data
        )

        # Statistical validation
        logger.info("Running statistical validation checks")
        qa_results["statistical_validation"] = self._validate_statistics(processed_data)

        # Consistency validation
        logger.info("Running consistency validation checks")
        qa_results["consistency_validation"] = self._validate_consistency(
            processed_data
        )

        # Performance metrics
        logger.info("Computing performance metrics")
        qa_results["performance_metrics"] = self._compute_performance_metrics(
            processed_data
        )

        # Determine overall status
        qa_results["overall_status"] = self._determine_overall_status(qa_results)

        # Generate visualizations
        if output_dir and self.config["visualization"]["save_plots"]:
            logger.info("Generating QA visualizations")
            self._generate_qa_visualizations(processed_data, qa_results, output_dir)

        # Generate reports
        if output_dir:
            logger.info("Generating QA reports")
            self._generate_qa_reports(qa_results, output_dir)

        # Store metrics
        self._store_metrics(qa_results)

        logger.info(f"QA completed with overall status: {qa_results['overall_status']}")
        return qa_results

    def _validate_data_quality(self, data: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        """Validate data quality metrics."""
        validation = {}

        for name, array in data.items():
            if not isinstance(array, xr.DataArray):
                continue

            array_validation = {}

            # Check for NaN/NoData
            if hasattr(array, "mask"):
                nodata_count = np.sum(array.mask)
            else:
                nodata_count = np.sum(np.isnan(array.values))

            total_pixels = array.size
            nodata_percent = (nodata_count / total_pixels) * 100

            array_validation["nodata_percent"] = nodata_percent
            array_validation["nodata_acceptable"] = (
                nodata_percent <= self.config["thresholds"]["nodata_percent_max"]
            )

            # Check data ranges
            valid_data = array.values[~np.isnan(array.values)]
            if len(valid_data) > 0:
                array_validation["min_value"] = float(np.min(valid_data))
                array_validation["max_value"] = float(np.max(valid_data))
                array_validation["mean_value"] = float(np.mean(valid_data))
                array_validation["std_value"] = float(np.std(valid_data))

                # Range validation based on data type
                range_valid = self._validate_data_range(name, valid_data)
                array_validation["range_valid"] = range_valid
            else:
                array_validation["range_valid"] = False
                array_validation["error"] = "No valid data found"

            validation[name] = array_validation

        return validation

    def _validate_spatial_consistency(
        self, data: Dict[str, xr.DataArray]
    ) -> Dict[str, Any]:
        """Validate spatial consistency across arrays."""
        validation = {}

        arrays = [v for v in data.values() if isinstance(v, xr.DataArray)]
        if len(arrays) < 2:
            validation["error"] = "Insufficient arrays for spatial validation"
            return validation

        reference = arrays[0]
        validation["reference_shape"] = reference.shape
        validation["reference_crs"] = (
            str(reference.rio.crs) if hasattr(reference, "rio") else "unknown"
        )

        alignment_results = []
        resolution_results = []

        for i, array in enumerate(arrays[1:], 1):
            # Check shape alignment
            shapes_match = array.shape == reference.shape
            alignment_results.append(shapes_match)

            # Check coordinate alignment
            if hasattr(array, "x") and hasattr(reference, "x"):
                x_aligned = np.allclose(
                    array.x.values,
                    reference.x.values,
                    atol=self.config["thresholds"]["alignment_tolerance_m"],
                )
                y_aligned = np.allclose(
                    array.y.values,
                    reference.y.values,
                    atol=self.config["thresholds"]["alignment_tolerance_m"],
                )
                coord_aligned = x_aligned and y_aligned
            else:
                coord_aligned = False

            # Check resolution consistency
            if hasattr(array, "rio") and hasattr(reference, "rio"):
                try:
                    ref_res = reference.rio.resolution()
                    arr_res = array.rio.resolution()
                    res_match = (
                        abs(ref_res[0] - arr_res[0])
                        < self.config["thresholds"]["spatial_resolution_tolerance"]
                        and abs(ref_res[1] - arr_res[1])
                        < self.config["thresholds"]["spatial_resolution_tolerance"]
                    )
                    resolution_results.append(res_match)
                except:
                    resolution_results.append(False)
            else:
                resolution_results.append(False)

        validation["all_shapes_aligned"] = all(alignment_results)
        validation["all_resolutions_match"] = all(resolution_results)
        validation["spatial_consistency"] = (
            validation["all_shapes_aligned"] and validation["all_resolutions_match"]
        )

        return validation

    def _validate_statistics(self, data: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        """Validate statistical properties of data."""
        validation = {}

        for name, array in data.items():
            if not isinstance(array, xr.DataArray):
                continue

            valid_data = array.values[~np.isnan(array.values)]
            if len(valid_data) == 0:
                continue

            stats = {
                "count": len(valid_data),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "q25": float(np.percentile(valid_data, 25)),
                "q50": float(np.percentile(valid_data, 50)),
                "q75": float(np.percentile(valid_data, 75)),
            }

            # Statistical tests
            stats["finite_values"] = np.all(np.isfinite(valid_data))
            stats["reasonable_range"] = self._check_reasonable_range(name, valid_data)
            stats["distribution_ok"] = self._check_distribution(valid_data)

            validation[name] = stats

        return validation

    def _validate_consistency(self, data: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        """Validate consistency between related datasets."""
        validation = {}

        # Check DEM vs derived features consistency
        if "dem" in data and "slope_degrees" in data:
            validation["dem_slope_consistent"] = self._check_dem_slope_consistency(
                data["dem"], data["slope_degrees"]
            )

        # Check flow accumulation vs streams consistency
        if "flow_accumulation" in data and "streams" in data:
            validation["flow_streams_consistent"] = (
                self._check_flow_streams_consistency(
                    data["flow_accumulation"], data["streams"]
                )
            )

        # Check HAND vs elevation consistency
        if "hand" in data and "dem" in data:
            validation["hand_dem_consistent"] = self._check_hand_dem_consistency(
                data["hand"], data["dem"]
            )

        # Cross-correlation analysis
        validation["correlations"] = self._compute_feature_correlations(data)

        return validation

    def _compute_performance_metrics(
        self, data: Dict[str, xr.DataArray]
    ) -> Dict[str, Any]:
        """Compute performance and efficiency metrics."""
        metrics = {}

        total_pixels = 0
        total_memory = 0

        for name, array in data.items():
            if isinstance(array, xr.DataArray):
                pixels = array.size
                memory = array.nbytes

                total_pixels += pixels
                total_memory += memory

                metrics[f"{name}_pixels"] = pixels
                metrics[f"{name}_memory_mb"] = memory / 1024 / 1024

        metrics["total_pixels"] = total_pixels
        metrics["total_memory_mb"] = total_memory / 1024 / 1024
        metrics["avg_memory_per_pixel"] = (
            total_memory / total_pixels if total_pixels > 0 else 0
        )

        return metrics

    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall QA status."""
        critical_failures = []
        warnings = []

        # Check data validation
        data_val = results.get("data_validation", {})
        for name, metrics in data_val.items():
            if isinstance(metrics, dict):
                if not metrics.get("nodata_acceptable", True):
                    critical_failures.append(f"High NoData percentage in {name}")
                if not metrics.get("range_valid", True):
                    warnings.append(f"Data range issues in {name}")

        # Check spatial validation
        spatial_val = results.get("spatial_validation", {})
        if not spatial_val.get("spatial_consistency", True):
            critical_failures.append("Spatial alignment issues")

        # Determine status
        if critical_failures:
            return "fail"
        elif warnings:
            return "warning"
        else:
            return "pass"

    def _validate_data_range(self, name: str, data: np.ndarray) -> bool:
        """Validate data range for specific data types."""
        thresholds = self.config["thresholds"]

        if "elevation" in name.lower() or name == "dem":
            min_val, max_val = thresholds["elevation_range_valid"]
            return min_val <= np.min(data) and np.max(data) <= max_val
        elif "slope" in name.lower():
            min_val, max_val = thresholds["slope_range_valid"]
            return min_val <= np.min(data) and np.max(data) <= max_val
        elif "flow" in name.lower():
            return np.min(data) >= thresholds["flow_acc_min"]
        elif "hand" in name.lower():
            min_val, max_val = thresholds["hand_range_valid"]
            return min_val <= np.min(data) and np.max(data) <= max_val
        else:
            return True  # No specific validation for other types

    def _check_reasonable_range(self, name: str, data: np.ndarray) -> bool:
        """Check if data values are in reasonable range."""
        return not (
            np.any(np.isinf(data)) or np.any(data < -1e10) or np.any(data > 1e10)
        )

    def _check_distribution(self, data: np.ndarray) -> bool:
        """Check if data distribution is reasonable."""
        if len(data) < 10:
            return False

        # Check for reasonable standard deviation
        std = np.std(data)
        mean = np.mean(data)

        if std == 0:  # Constant values
            return False

        # Check coefficient of variation is reasonable
        cv = std / abs(mean) if mean != 0 else float("inf")
        return cv < 100  # Arbitrary threshold for reasonable variation

    def _check_dem_slope_consistency(
        self, dem: xr.DataArray, slope: xr.DataArray
    ) -> bool:
        """Check consistency between DEM and slope."""
        # Compute simple slope from DEM
        dem_data = dem.values
        gy, gx = np.gradient(dem_data)
        computed_slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

        # Compare with provided slope
        correlation = np.corrcoef(computed_slope.flat, slope.values.flat)[0, 1]
        return correlation > self.config["thresholds"]["correlation_min"]

    def _check_flow_streams_consistency(
        self, flow_acc: xr.DataArray, streams: xr.DataArray
    ) -> bool:
        """Check consistency between flow accumulation and streams."""
        # Streams should correspond to high flow accumulation
        stream_pixels = streams.values > 0
        if not np.any(stream_pixels):
            return False

        # Average flow accumulation in stream areas should be higher
        stream_flow = np.mean(flow_acc.values[stream_pixels])
        non_stream_flow = np.mean(flow_acc.values[~stream_pixels])

        return stream_flow > non_stream_flow

    def _check_hand_dem_consistency(
        self, hand: xr.DataArray, dem: xr.DataArray
    ) -> bool:
        """Check consistency between HAND and DEM."""
        # HAND should be non-negative and less than elevation differences
        hand_data = hand.values
        valid_hand = hand_data[~np.isnan(hand_data)]

        if len(valid_hand) == 0:
            return False

        # Check all HAND values are non-negative
        return np.all(valid_hand >= 0) and np.all(
            valid_hand <= 1000
        )  # Reasonable upper bound

    def _compute_feature_correlations(
        self, data: Dict[str, xr.DataArray]
    ) -> Dict[str, float]:
        """Compute correlations between features."""
        correlations = {}

        # Convert to arrays for correlation
        feature_data = {}
        for name, array in data.items():
            if isinstance(array, xr.DataArray):
                valid_data = array.values[~np.isnan(array.values)]
                if len(valid_data) > 100:  # Need sufficient data
                    feature_data[name] = array.values.flatten()

        # Compute pairwise correlations
        feature_names = list(feature_data.keys())
        for i, name1 in enumerate(feature_names):
            for name2 in feature_names[i + 1 :]:
                try:
                    data1 = feature_data[name1]
                    data2 = feature_data[name2]

                    # Handle different array sizes by taking minimum
                    min_size = min(len(data1), len(data2))
                    corr = np.corrcoef(data1[:min_size], data2[:min_size])[0, 1]

                    if not np.isnan(corr):
                        correlations[f"{name1}_vs_{name2}"] = float(corr)
                except:
                    continue

        return correlations

    def _generate_qa_visualizations(
        self, data: Dict[str, xr.DataArray], results: Dict[str, Any], output_dir: Path
    ):
        """Generate QA visualization plots."""
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Data overview plots
        self._plot_data_overview(data, viz_dir)

        # Statistical summaries
        self._plot_statistical_summaries(results["statistical_validation"], viz_dir)

        # Correlation heatmap
        if "correlations" in results["consistency_validation"]:
            self._plot_correlation_heatmap(
                results["consistency_validation"]["correlations"], viz_dir
            )

        # Quality metrics dashboard
        self._plot_quality_dashboard(results, viz_dir)

    def _plot_data_overview(self, data: Dict[str, xr.DataArray], output_dir: Path):
        """Plot overview of all data layers."""
        arrays = [v for k, v in data.items() if isinstance(v, xr.DataArray)]
        if not arrays:
            return

        n_arrays = len(arrays)
        cols = min(4, n_arrays)
        rows = (n_arrays + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, (name, array) in enumerate(data.items()):
            if isinstance(array, xr.DataArray) and i < len(axes):
                im = axes[i].imshow(array.values, cmap="viridis", aspect="auto")
                axes[i].set_title(f"{name}\nShape: {array.shape}")
                axes[i].set_xlabel("X")
                axes[i].set_ylabel("Y")
                plt.colorbar(im, ax=axes[i])

        # Hide unused subplots
        for i in range(len(arrays), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            output_dir / "data_overview.png",
            dpi=self.config["visualization"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    def _plot_statistical_summaries(self, stats: Dict[str, Dict], output_dir: Path):
        """Plot statistical summaries."""
        if not stats:
            return

        # Create summary DataFrame
        summary_data = []
        for name, metrics in stats.items():
            if isinstance(metrics, dict):
                summary_data.append(
                    {
                        "Dataset": name,
                        "Mean": metrics.get("mean", 0),
                        "Std": metrics.get("std", 0),
                        "Min": metrics.get("min", 0),
                        "Max": metrics.get("max", 0),
                        "Count": metrics.get("count", 0),
                    }
                )

        if not summary_data:
            return

        df = pd.DataFrame(summary_data)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean and std
        axes[0, 0].bar(df["Dataset"], df["Mean"])
        axes[0, 0].set_title("Mean Values")
        axes[0, 0].tick_params(axis="x", rotation=45)

        axes[0, 1].bar(df["Dataset"], df["Std"])
        axes[0, 1].set_title("Standard Deviation")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Min and max
        axes[1, 0].bar(df["Dataset"], df["Min"])
        axes[1, 0].set_title("Minimum Values")
        axes[1, 0].tick_params(axis="x", rotation=45)

        axes[1, 1].bar(df["Dataset"], df["Max"])
        axes[1, 1].set_title("Maximum Values")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_dir / "statistical_summary.png",
            dpi=self.config["visualization"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    def _plot_correlation_heatmap(
        self, correlations: Dict[str, float], output_dir: Path
    ):
        """Plot correlation heatmap."""
        if not correlations:
            return

        # Parse correlation pairs
        corr_data = []
        feature_names = set()

        for pair, corr in correlations.items():
            if "_vs_" in pair:
                name1, name2 = pair.split("_vs_")
                corr_data.append((name1, name2, corr))
                feature_names.update([name1, name2])

        if not corr_data:
            return

        # Create correlation matrix
        features = sorted(feature_names)
        corr_matrix = np.eye(len(features))

        for name1, name2, corr in corr_data:
            i, j = features.index(name1), features.index(name2)
            corr_matrix[i, j] = corr_matrix[j, i] = corr

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=features,
            yticklabels=features,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            output_dir / "correlation_heatmap.png",
            dpi=self.config["visualization"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    def _plot_quality_dashboard(self, results: Dict[str, Any], output_dir: Path):
        """Plot quality metrics dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Overall status pie chart
        status_counts = {"Pass": 0, "Warning": 0, "Fail": 0}

        # Count status from data validation
        for name, metrics in results["data_validation"].items():
            if isinstance(metrics, dict):
                if not metrics.get("nodata_acceptable", True) or not metrics.get(
                    "range_valid", True
                ):
                    status_counts["Fail"] += 1
                else:
                    status_counts["Pass"] += 1

        # Pie chart
        status_data = [v for v in status_counts.values() if v > 0]
        status_labels = [k for k, v in status_counts.items() if v > 0]

        if status_data:
            axes[0, 0].pie(status_data, labels=status_labels, autopct="%1.1f%%")
            axes[0, 0].set_title("QA Status Distribution")

        # NoData percentages
        nodata_data = []
        nodata_names = []

        for name, metrics in results["data_validation"].items():
            if isinstance(metrics, dict) and "nodata_percent" in metrics:
                nodata_data.append(metrics["nodata_percent"])
                nodata_names.append(name)

        if nodata_data:
            axes[0, 1].bar(range(len(nodata_names)), nodata_data)
            axes[0, 1].set_xticks(range(len(nodata_names)))
            axes[0, 1].set_xticklabels(nodata_names, rotation=45)
            axes[0, 1].set_title("NoData Percentage by Dataset")
            axes[0, 1].set_ylabel("Percentage")

        # Memory usage
        perf_metrics = results.get("performance_metrics", {})
        memory_data = []
        memory_names = []

        for key, value in perf_metrics.items():
            if key.endswith("_memory_mb"):
                memory_data.append(value)
                memory_names.append(key.replace("_memory_mb", ""))

        if memory_data:
            axes[1, 0].bar(range(len(memory_names)), memory_data)
            axes[1, 0].set_xticks(range(len(memory_names)))
            axes[1, 0].set_xticklabels(memory_names, rotation=45)
            axes[1, 0].set_title("Memory Usage by Dataset")
            axes[1, 0].set_ylabel("Memory (MB)")

        # Summary text
        status_text = f"Overall Status: {results['overall_status'].upper()}\n"
        status_text += (
            f"Total Memory: {perf_metrics.get('total_memory_mb', 0):.1f} MB\n"
        )
        status_text += f"Total Pixels: {perf_metrics.get('total_pixels', 0):,}\n"
        status_text += f"Spatial Consistency: {results['spatial_validation'].get('spatial_consistency', 'Unknown')}"

        axes[1, 1].text(
            0.1,
            0.5,
            status_text,
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment="center",
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("QA Summary")

        plt.tight_layout()
        plt.savefig(
            output_dir / "quality_dashboard.png",
            dpi=self.config["visualization"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    def _generate_qa_reports(self, results: Dict[str, Any], output_dir: Path):
        """Generate QA reports."""
        # Detailed JSON report
        detailed_file = output_dir / "qa_detailed_report.json"
        with open(detailed_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Summary report
        summary_file = output_dir / "qa_summary_report.txt"
        with open(summary_file, "w") as f:
            f.write("FloodRisk Preprocessing QA Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Overall Status: {results['overall_status'].upper()}\n\n")

            # Data validation summary
            f.write("Data Validation:\n")
            f.write("-" * 16 + "\n")
            for name, metrics in results["data_validation"].items():
                if isinstance(metrics, dict):
                    f.write(f"  {name}:\n")
                    f.write(f"    NoData %: {metrics.get('nodata_percent', 0):.2f}%\n")
                    f.write(
                        f"    Range Valid: {metrics.get('range_valid', 'Unknown')}\n"
                    )
                    f.write(
                        f"    Min/Max: {metrics.get('min_value', 'N/A')}/{metrics.get('max_value', 'N/A')}\n\n"
                    )

            # Spatial validation summary
            f.write("Spatial Validation:\n")
            f.write("-" * 18 + "\n")
            spatial = results["spatial_validation"]
            f.write(
                f"  Shapes Aligned: {spatial.get('all_shapes_aligned', 'Unknown')}\n"
            )
            f.write(
                f"  Resolutions Match: {spatial.get('all_resolutions_match', 'Unknown')}\n"
            )
            f.write(
                f"  Overall Consistent: {spatial.get('spatial_consistency', 'Unknown')}\n\n"
            )

            # Performance summary
            f.write("Performance Metrics:\n")
            f.write("-" * 19 + "\n")
            perf = results["performance_metrics"]
            f.write(f"  Total Memory: {perf.get('total_memory_mb', 0):.1f} MB\n")
            f.write(f"  Total Pixels: {perf.get('total_pixels', 0):,}\n")

    def _store_metrics(self, results: Dict[str, Any]):
        """Store metrics in history."""
        timestamp = results["timestamp"]
        overall_status = results["overall_status"]

        # Create summary metrics
        metric = QAMetrics(
            metric_name="overall_qa_status",
            value=(
                1.0
                if overall_status == "pass"
                else 0.5 if overall_status == "warning" else 0.0
            ),
            status=overall_status,
            message=f"QA completed with status: {overall_status}",
            timestamp=timestamp,
        )

        self.metrics_history.append(metric)

        # Store individual metrics
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for name, value in category_results.items():
                    if isinstance(value, (int, float)):
                        metric = QAMetrics(
                            metric_name=f"{category}_{name}",
                            value=float(value),
                            status="pass",  # Default, could be refined
                            timestamp=timestamp,
                        )
                        self.metrics_history.append(metric)


def run_qa_pipeline(
    processed_data: Dict[str, xr.DataArray],
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to run QA pipeline."""

    # Load config if provided
    config = None
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    # Initialize QA system
    qa_system = PreprocessingQA(config)

    # Run comprehensive QA
    results = qa_system.run_comprehensive_qa(processed_data, output_dir)

    return results
