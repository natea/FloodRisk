"""Validation framework for LISFLOOD-FP simulation results.

This module provides comprehensive validation and quality control for
flood simulation outputs, ensuring reliability for ML training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Threshold values for simulation validation."""

    # Physical constraints
    min_reasonable_depth_m: float = 0.01
    max_reasonable_depth_m: float = 50.0

    # Flood extent constraints
    min_flood_fraction: float = 0.0001  # 0.01%
    max_flood_fraction: float = 0.5  # 50%

    # Temporal constraints
    min_simulation_time_s: float = 10.0  # Too fast suggests failure
    max_simulation_time_s: float = 86400.0  # 24 hours max

    # Spatial constraints
    min_flooded_area_m2: float = 100.0  # 100 m² minimum
    max_flooded_area_m2: float = 1e9  # 1000 km² maximum

    # Data quality
    max_nan_fraction: float = 0.01  # 1% NaN values max
    max_negative_fraction: float = 0.01  # 1% negative depths max

    # Performance thresholds
    target_success_rate: float = 0.9  # 90% success rate for batches

    # Mass conservation (if applicable)
    mass_conservation_tolerance: float = 0.1  # 10% tolerance


@dataclass
class ValidationResult:
    """Result of simulation validation."""

    simulation_id: str
    status: str = "unknown"  # passed, failed, warning
    overall_score: float = 0.0  # 0-100 score

    # Individual check results
    physical_checks: Dict = field(default_factory=dict)
    spatial_checks: Dict = field(default_factory=dict)
    temporal_checks: Dict = field(default_factory=dict)
    data_quality_checks: Dict = field(default_factory=dict)

    # Issues found
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validator_version: str = "1.0.0"


class SimulationValidator:
    """Comprehensive validator for flood simulation results."""

    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        """Initialize validator with thresholds.

        Args:
            thresholds: Validation threshold configuration
        """
        self.thresholds = thresholds or ValidationThresholds()
        logger.info("SimulationValidator initialized")

    def validate_single_simulation(
        self,
        simulation_result: Dict,
        depth_data: Optional[np.ndarray] = None,
        flood_extent: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """Validate a single simulation result.

        Args:
            simulation_result: Simulation result dictionary
            depth_data: Optional depth array for detailed validation
            flood_extent: Optional flood extent array

        Returns:
            ValidationResult with comprehensive assessment
        """
        simulation_id = simulation_result.get("simulation_id", "unknown")
        result = ValidationResult(simulation_id=simulation_id)

        logger.info(f"Validating simulation: {simulation_id}")

        try:
            # Basic status check
            if simulation_result.get("status") != "success":
                result.status = "failed"
                result.errors.append(
                    f"Simulation failed: {simulation_result.get('error', 'Unknown error')}"
                )
                return result

            # Extract statistics if available
            statistics = simulation_result.get("outputs", {}).get("statistics", {})

            # Physical validation
            result.physical_checks = self._validate_physical_realism(
                statistics, depth_data
            )

            # Spatial validation
            result.spatial_checks = self._validate_spatial_characteristics(
                statistics, flood_extent
            )

            # Temporal validation
            result.temporal_checks = self._validate_temporal_characteristics(
                simulation_result
            )

            # Data quality validation
            result.data_quality_checks = self._validate_data_quality(
                statistics, depth_data
            )

            # Compile overall assessment
            result = self._compile_validation_result(result)

            logger.info(
                f"Validation completed for {simulation_id}: {result.status} (score: {result.overall_score:.1f})"
            )

        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Validation failed for {simulation_id}: {e}")

        return result

    def validate_batch_results(
        self, simulation_results: List[Dict], detailed_validation: bool = True
    ) -> Dict:
        """Validate a batch of simulation results.

        Args:
            simulation_results: List of simulation result dictionaries
            detailed_validation: Whether to perform detailed validation

        Returns:
            Batch validation summary
        """
        logger.info(f"Validating batch of {len(simulation_results)} simulations")

        batch_start_time = datetime.now()
        individual_results = []

        # Validate each simulation
        for sim_result in simulation_results:
            if detailed_validation:
                # Load depth data for detailed validation
                depth_data = self._load_depth_data_if_available(sim_result)
                flood_extent = self._load_extent_data_if_available(sim_result)

                validation = self.validate_single_simulation(
                    sim_result, depth_data, flood_extent
                )
            else:
                # Quick validation without loading data
                validation = self.validate_single_simulation(sim_result)

            individual_results.append(validation)

        # Aggregate batch statistics
        batch_summary = self._create_batch_validation_summary(
            individual_results, simulation_results, batch_start_time
        )

        logger.info(
            f"Batch validation completed: {batch_summary['summary']['passed_count']}/{len(simulation_results)} passed"
        )

        return batch_summary

    def _validate_physical_realism(
        self, statistics: Dict, depth_data: Optional[np.ndarray] = None
    ) -> Dict:
        """Validate physical realism of flood depths."""
        checks = {
            "depth_range_check": {"status": "unknown", "details": {}},
            "depth_distribution_check": {"status": "unknown", "details": {}},
            "mass_conservation_check": {"status": "unknown", "details": {}},
        }

        # Check depth ranges
        max_depth = statistics.get("max_depth_m", 0)
        mean_depth = statistics.get("mean_depth_flooded_m", 0)

        if max_depth < self.thresholds.min_reasonable_depth_m:
            checks["depth_range_check"]["status"] = "warning"
            checks["depth_range_check"]["details"][
                "issue"
            ] = f"Very shallow max depth: {max_depth:.3f}m"
        elif max_depth > self.thresholds.max_reasonable_depth_m:
            checks["depth_range_check"]["status"] = "failed"
            checks["depth_range_check"]["details"][
                "issue"
            ] = f"Unrealistic max depth: {max_depth:.1f}m"
        else:
            checks["depth_range_check"]["status"] = "passed"

        checks["depth_range_check"]["details"].update(
            {
                "max_depth_m": max_depth,
                "mean_depth_m": mean_depth,
                "threshold_min": self.thresholds.min_reasonable_depth_m,
                "threshold_max": self.thresholds.max_reasonable_depth_m,
            }
        )

        # Check depth distribution
        if depth_data is not None:
            self._validate_depth_distribution(
                depth_data, checks["depth_distribution_check"]
            )
        else:
            checks["depth_distribution_check"]["status"] = "skipped"
            checks["depth_distribution_check"]["details"] = {
                "reason": "No depth data provided"
            }

        # Mass conservation check (simplified)
        flood_fraction = statistics.get("flood_fraction", 0)
        if flood_fraction > 0.4:  # More than 40% flooded suggests issues
            checks["mass_conservation_check"]["status"] = "warning"
            checks["mass_conservation_check"]["details"][
                "issue"
            ] = f"High flood fraction: {flood_fraction:.2%}"
        elif flood_fraction < 0.0001:  # Less than 0.01% suggests no flooding
            checks["mass_conservation_check"]["status"] = "warning"
            checks["mass_conservation_check"]["details"][
                "issue"
            ] = f"Very low flood fraction: {flood_fraction:.4%}"
        else:
            checks["mass_conservation_check"]["status"] = "passed"

        return checks

    def _validate_depth_distribution(self, depth_data: np.ndarray, check_result: Dict):
        """Validate depth data distribution."""
        try:
            # Check for invalid values
            nan_fraction = np.sum(np.isnan(depth_data)) / depth_data.size
            inf_fraction = np.sum(np.isinf(depth_data)) / depth_data.size
            negative_fraction = np.sum(depth_data < 0) / depth_data.size

            issues = []

            if nan_fraction > self.thresholds.max_nan_fraction:
                issues.append(f"High NaN fraction: {nan_fraction:.3%}")

            if inf_fraction > 0:
                issues.append(f"Infinite values found: {inf_fraction:.3%}")

            if negative_fraction > self.thresholds.max_negative_fraction:
                issues.append(f"High negative value fraction: {negative_fraction:.3%}")

            # Statistical checks
            percentiles = (
                np.percentile(depth_data[depth_data > 0], [50, 90, 95, 99])
                if np.any(depth_data > 0)
                else np.zeros(4)
            )

            # Check for reasonable distribution
            if (
                len(percentiles) >= 2 and percentiles[1] > percentiles[0] * 100
            ):  # P90 >> P50
                issues.append("Highly skewed depth distribution")

            check_result["details"] = {
                "nan_fraction": nan_fraction,
                "negative_fraction": negative_fraction,
                "inf_fraction": inf_fraction,
                "percentiles": {
                    "p50": float(percentiles[0]),
                    "p90": float(percentiles[1]),
                    "p95": float(percentiles[2]),
                    "p99": float(percentiles[3]),
                },
            }

            if issues:
                check_result["status"] = (
                    "failed"
                    if any(["NaN" in issue or "Infinite" in issue for issue in issues])
                    else "warning"
                )
                check_result["details"]["issues"] = issues
            else:
                check_result["status"] = "passed"

        except Exception as e:
            check_result["status"] = "failed"
            check_result["details"] = {"error": str(e)}

    def _validate_spatial_characteristics(
        self, statistics: Dict, flood_extent: Optional[np.ndarray] = None
    ) -> Dict:
        """Validate spatial characteristics of flooding."""
        checks = {
            "flood_fraction_check": {"status": "unknown", "details": {}},
            "flood_area_check": {"status": "unknown", "details": {}},
            "spatial_continuity_check": {"status": "unknown", "details": {}},
        }

        # Flood fraction check
        flood_fraction = statistics.get("flood_fraction", 0)

        if flood_fraction < self.thresholds.min_flood_fraction:
            checks["flood_fraction_check"]["status"] = "warning"
            checks["flood_fraction_check"]["details"][
                "issue"
            ] = f"Low flood fraction: {flood_fraction:.4%}"
        elif flood_fraction > self.thresholds.max_flood_fraction:
            checks["flood_fraction_check"]["status"] = "failed"
            checks["flood_fraction_check"]["details"][
                "issue"
            ] = f"Excessive flood fraction: {flood_fraction:.2%}"
        else:
            checks["flood_fraction_check"]["status"] = "passed"

        checks["flood_fraction_check"]["details"].update(
            {
                "flood_fraction": flood_fraction,
                "threshold_min": self.thresholds.min_flood_fraction,
                "threshold_max": self.thresholds.max_flood_fraction,
            }
        )

        # Flood area check
        flooded_area = statistics.get("flooded_area_km2", 0) * 1e6  # Convert to m²

        if flooded_area < self.thresholds.min_flooded_area_m2:
            checks["flood_area_check"]["status"] = "warning"
            checks["flood_area_check"]["details"][
                "issue"
            ] = f"Small flooded area: {flooded_area/1e6:.3f} km²"
        elif flooded_area > self.thresholds.max_flooded_area_m2:
            checks["flood_area_check"]["status"] = "failed"
            checks["flood_area_check"]["details"][
                "issue"
            ] = f"Excessive flooded area: {flooded_area/1e6:.1f} km²"
        else:
            checks["flood_area_check"]["status"] = "passed"

        checks["flood_area_check"]["details"].update(
            {
                "flooded_area_km2": flooded_area / 1e6,
                "threshold_min_km2": self.thresholds.min_flooded_area_m2 / 1e6,
                "threshold_max_km2": self.thresholds.max_flooded_area_m2 / 1e6,
            }
        )

        # Spatial continuity check
        if flood_extent is not None:
            self._validate_spatial_continuity(
                flood_extent, checks["spatial_continuity_check"]
            )
        else:
            checks["spatial_continuity_check"]["status"] = "skipped"
            checks["spatial_continuity_check"]["details"] = {
                "reason": "No flood extent data provided"
            }

        return checks

    def _validate_spatial_continuity(
        self, flood_extent: np.ndarray, check_result: Dict
    ):
        """Validate spatial continuity of flood extent."""
        try:
            from scipy import ndimage

            # Count connected components
            labeled, num_components = ndimage.label(flood_extent)
            component_sizes = [
                np.sum(labeled == i) for i in range(1, num_components + 1)
            ]

            total_flooded = np.sum(flood_extent)

            if total_flooded == 0:
                check_result["status"] = "warning"
                check_result["details"] = {"issue": "No flooded areas found"}
                return

            # Calculate fragmentation metrics
            largest_component = max(component_sizes) if component_sizes else 0
            connectivity_ratio = (
                largest_component / total_flooded if total_flooded > 0 else 0
            )

            # Check for excessive fragmentation
            small_components = sum(
                1 for size in component_sizes if size < 10
            )  # Less than 10 pixels

            issues = []

            if connectivity_ratio < 0.5 and num_components > 20:
                issues.append(
                    f"Highly fragmented flooding: {num_components} components"
                )

            if small_components > num_components * 0.8:
                issues.append(
                    f"Many small isolated areas: {small_components}/{num_components}"
                )

            check_result["details"] = {
                "num_components": num_components,
                "largest_component_size": largest_component,
                "connectivity_ratio": connectivity_ratio,
                "small_components": small_components,
                "component_sizes_stats": {
                    "mean": float(np.mean(component_sizes)) if component_sizes else 0,
                    "max": float(np.max(component_sizes)) if component_sizes else 0,
                    "min": float(np.min(component_sizes)) if component_sizes else 0,
                },
            }

            if issues:
                check_result["status"] = "warning"
                check_result["details"]["issues"] = issues
            else:
                check_result["status"] = "passed"

        except ImportError:
            check_result["status"] = "skipped"
            check_result["details"] = {
                "reason": "scipy not available for spatial analysis"
            }
        except Exception as e:
            check_result["status"] = "failed"
            check_result["details"] = {"error": str(e)}

    def _validate_temporal_characteristics(self, simulation_result: Dict) -> Dict:
        """Validate temporal characteristics of simulation."""
        checks = {
            "runtime_check": {"status": "unknown", "details": {}},
            "convergence_check": {"status": "unknown", "details": {}},
        }

        # Runtime validation
        runtime = simulation_result.get("runtime_seconds", 0)

        if runtime < self.thresholds.min_simulation_time_s:
            checks["runtime_check"]["status"] = "warning"
            checks["runtime_check"]["details"][
                "issue"
            ] = f"Very fast simulation: {runtime:.1f}s"
        elif runtime > self.thresholds.max_simulation_time_s:
            checks["runtime_check"]["status"] = "warning"
            checks["runtime_check"]["details"][
                "issue"
            ] = f"Long simulation: {runtime/3600:.1f}h"
        else:
            checks["runtime_check"]["status"] = "passed"

        checks["runtime_check"]["details"].update(
            {
                "runtime_seconds": runtime,
                "runtime_hours": runtime / 3600,
                "threshold_min_s": self.thresholds.min_simulation_time_s,
                "threshold_max_s": self.thresholds.max_simulation_time_s,
            }
        )

        # Convergence check (placeholder - would need more detailed output)
        checks["convergence_check"]["status"] = "skipped"
        checks["convergence_check"]["details"] = {
            "reason": "Convergence data not available"
        }

        return checks

    def _validate_data_quality(
        self, statistics: Dict, depth_data: Optional[np.ndarray] = None
    ) -> Dict:
        """Validate overall data quality."""
        checks = {
            "completeness_check": {"status": "unknown", "details": {}},
            "consistency_check": {"status": "unknown", "details": {}},
        }

        # Completeness check
        required_stats = [
            "total_pixels",
            "flooded_pixels",
            "max_depth_m",
            "flood_fraction",
        ]
        missing_stats = [stat for stat in required_stats if stat not in statistics]

        if missing_stats:
            checks["completeness_check"]["status"] = "failed"
            checks["completeness_check"]["details"][
                "issue"
            ] = f"Missing statistics: {missing_stats}"
        else:
            checks["completeness_check"]["status"] = "passed"

        checks["completeness_check"]["details"]["required_stats"] = required_stats
        checks["completeness_check"]["details"]["missing_stats"] = missing_stats

        # Consistency check
        total_pixels = statistics.get("total_pixels", 0)
        flooded_pixels = statistics.get("flooded_pixels", 0)
        calculated_fraction = flooded_pixels / total_pixels if total_pixels > 0 else 0
        reported_fraction = statistics.get("flood_fraction", 0)

        fraction_difference = abs(calculated_fraction - reported_fraction)

        if fraction_difference > 0.001:  # 0.1% tolerance
            checks["consistency_check"]["status"] = "warning"
            checks["consistency_check"]["details"][
                "issue"
            ] = f"Inconsistent flood fraction: calculated={calculated_fraction:.4f}, reported={reported_fraction:.4f}"
        else:
            checks["consistency_check"]["status"] = "passed"

        checks["consistency_check"]["details"].update(
            {
                "calculated_fraction": calculated_fraction,
                "reported_fraction": reported_fraction,
                "difference": fraction_difference,
            }
        )

        return checks

    def _compile_validation_result(self, result: ValidationResult) -> ValidationResult:
        """Compile overall validation result from individual checks."""
        all_checks = [
            result.physical_checks,
            result.spatial_checks,
            result.temporal_checks,
            result.data_quality_checks,
        ]

        # Collect errors and warnings
        for check_group in all_checks:
            for check_name, check_data in check_group.items():
                status = check_data.get("status", "unknown")
                details = check_data.get("details", {})

                if status == "failed":
                    error_msg = (
                        details.get("issue")
                        or details.get("error")
                        or f"{check_name} failed"
                    )
                    result.errors.append(f"{check_name}: {error_msg}")
                elif status == "warning":
                    warning_msg = details.get("issue") or f"{check_name} warning"
                    result.warnings.append(f"{check_name}: {warning_msg}")

        # Determine overall status
        if result.errors:
            result.status = "failed"
        elif result.warnings:
            result.status = "warning"
        else:
            result.status = "passed"

        # Calculate score (0-100)
        total_checks = sum(len(check_group) for check_group in all_checks)
        passed_checks = 0
        warning_checks = 0

        for check_group in all_checks:
            for check_data in check_group.values():
                status = check_data.get("status", "unknown")
                if status == "passed":
                    passed_checks += 1
                elif status == "warning":
                    warning_checks += 1

        if total_checks > 0:
            result.overall_score = (
                (passed_checks + 0.5 * warning_checks) / total_checks * 100
            )

        return result

    def _create_batch_validation_summary(
        self,
        individual_results: List[ValidationResult],
        simulation_results: List[Dict],
        start_time: datetime,
    ) -> Dict:
        """Create comprehensive batch validation summary."""
        end_time = datetime.now()

        # Count results by status
        passed_count = sum(1 for r in individual_results if r.status == "passed")
        warning_count = sum(1 for r in individual_results if r.status == "warning")
        failed_count = sum(1 for r in individual_results if r.status == "failed")

        # Calculate average score
        scores = [r.overall_score for r in individual_results]
        avg_score = np.mean(scores) if scores else 0

        # Identify common issues
        all_errors = []
        all_warnings = []

        for result in individual_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        # Count issue frequencies
        error_frequency = {}
        for error in all_errors:
            error_type = error.split(":")[0]  # Get check name
            error_frequency[error_type] = error_frequency.get(error_type, 0) + 1

        warning_frequency = {}
        for warning in all_warnings:
            warning_type = warning.split(":")[0]  # Get check name
            warning_frequency[warning_type] = warning_frequency.get(warning_type, 0) + 1

        summary = {
            "validation_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "validator_version": (
                    individual_results[0].validator_version
                    if individual_results
                    else "unknown"
                ),
            },
            "summary": {
                "total_simulations": len(simulation_results),
                "validated_simulations": len(individual_results),
                "passed_count": passed_count,
                "warning_count": warning_count,
                "failed_count": failed_count,
                "success_rate": (
                    passed_count / len(individual_results) if individual_results else 0
                ),
                "average_score": avg_score,
            },
            "quality_assessment": {
                "meets_target_success_rate": (
                    passed_count / len(individual_results) if individual_results else 0
                )
                >= self.thresholds.target_success_rate,
                "common_errors": dict(
                    sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)
                ),
                "common_warnings": dict(
                    sorted(warning_frequency.items(), key=lambda x: x[1], reverse=True)
                ),
            },
            "individual_results": [
                {
                    "simulation_id": r.simulation_id,
                    "status": r.status,
                    "score": r.overall_score,
                    "error_count": len(r.errors),
                    "warning_count": len(r.warnings),
                }
                for r in individual_results
            ],
            "detailed_results": individual_results,  # Full validation results
        }

        return summary

    def _load_depth_data_if_available(
        self, simulation_result: Dict
    ) -> Optional[np.ndarray]:
        """Load depth data from simulation result if available."""
        try:
            outputs = simulation_result.get("outputs", {})
            depth_file = outputs.get("depth_file") or outputs.get("depth_numpy")

            if depth_file and Path(depth_file).exists():
                if depth_file.endswith(".npy"):
                    return np.load(depth_file)
                else:
                    # Try to load as binary float32 (LISFLOOD-FP format)
                    data = np.fromfile(depth_file, dtype=np.float32)
                    # This is a simplified approach - real implementation would need proper shape inference
                    size = int(np.sqrt(data.size))
                    if size * size == data.size:
                        return data.reshape(size, size)

        except Exception as e:
            logger.warning(f"Could not load depth data: {e}")

        return None

    def _load_extent_data_if_available(
        self, simulation_result: Dict
    ) -> Optional[np.ndarray]:
        """Load flood extent data from simulation result if available."""
        try:
            outputs = simulation_result.get("outputs", {})
            extent_file = outputs.get("extent_file") or outputs.get("extent_numpy")

            if (
                extent_file
                and Path(extent_file).exists()
                and extent_file.endswith(".npy")
            ):
                return np.load(extent_file)

        except Exception as e:
            logger.warning(f"Could not load extent data: {e}")

        return None
