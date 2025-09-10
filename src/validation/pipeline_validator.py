"""
End-to-End Pipeline Validation Framework

Comprehensive data quality validation for the FloodRisk ML pipeline:
- DEM quality validation (elevation range, void detection, spatial continuity)
- Rainfall data validation (value range, coverage, temporal consistency)
- Simulation results validation (physical plausibility, convergence)
- Spatial consistency validation (CRS, extent alignment, resolution matching)
- Tile quality validation (flood/dry balance, spatial coverage, edge effects)
- ML pipeline integration validation
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
import xarray as xr
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""

    component: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    score: float
    details: Dict[str, Any]
    issues: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "component": self.component,
            "status": self.status,
            "score": self.score,
            "details": self.details,
            "issues": self.issues,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


class BaseValidator(ABC):
    """Abstract base class for all validators"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Perform validation and return results"""
        pass

    def _create_result(
        self,
        component: str,
        status: str,
        score: float,
        details: Dict[str, Any],
        issues: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> ValidationResult:
        """Create standardized validation result"""
        return ValidationResult(
            component=component,
            status=status,
            score=score,
            details=details,
            issues=issues or [],
            timestamp=datetime.now(),
            metadata=metadata,
        )


class DEMValidator(BaseValidator):
    """
    DEM Quality Validator

    Validates:
    - Elevation range and distribution
    - Void/NoData detection
    - Spatial continuity and smoothness
    - Resolution consistency
    - Coordinate system validity
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.elevation_bounds = config.get("elevation_bounds", (-500, 9000))  # meters
        self.void_threshold = config.get("void_threshold", 0.05)  # 5% max voids
        self.smoothness_threshold = config.get(
            "smoothness_threshold", 100
        )  # max elevation change

    def validate(self, dem_path: Union[str, Path], **kwargs) -> ValidationResult:
        """Validate DEM quality"""
        try:
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1, masked=True)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds

            issues = []
            details = {
                "file_path": str(dem_path),
                "shape": dem_data.shape,
                "resolution": (abs(transform.a), abs(transform.e)),
                "bounds": bounds,
                "crs": str(crs) if crs else None,
            }

            score = 1.0

            # 1. Elevation range validation
            valid_data = dem_data.compressed()  # Remove masked values
            if len(valid_data) == 0:
                issues.append("DEM contains no valid elevation data")
                score = 0.0
            else:
                min_elev = float(np.min(valid_data))
                max_elev = float(np.max(valid_data))
                mean_elev = float(np.mean(valid_data))
                std_elev = float(np.std(valid_data))

                details.update(
                    {
                        "elevation_stats": {
                            "min": min_elev,
                            "max": max_elev,
                            "mean": mean_elev,
                            "std": std_elev,
                            "range": max_elev - min_elev,
                        }
                    }
                )

                # Check elevation bounds
                if min_elev < self.elevation_bounds[0]:
                    issues.append(
                        f"Minimum elevation ({min_elev:.2f}m) below expected range"
                    )
                    score -= 0.2

                if max_elev > self.elevation_bounds[1]:
                    issues.append(
                        f"Maximum elevation ({max_elev:.2f}m) above expected range"
                    )
                    score -= 0.2

            # 2. Void detection
            total_pixels = dem_data.size
            void_pixels = np.sum(dem_data.mask) if hasattr(dem_data, "mask") else 0
            void_percentage = void_pixels / total_pixels if total_pixels > 0 else 1.0

            details["void_analysis"] = {
                "void_pixels": int(void_pixels),
                "total_pixels": int(total_pixels),
                "void_percentage": float(void_percentage),
            }

            if void_percentage > self.void_threshold:
                issues.append(
                    f"High void percentage ({void_percentage*100:.2f}%) exceeds threshold"
                )
                score -= 0.3

            # 3. Spatial continuity check
            if len(valid_data) > 0:
                gradient_x = np.gradient(valid_data.reshape(dem_data.shape), axis=1)
                gradient_y = np.gradient(valid_data.reshape(dem_data.shape), axis=0)
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

                max_gradient = float(np.nanmax(gradient_magnitude))
                mean_gradient = float(np.nanmean(gradient_magnitude))

                details["spatial_analysis"] = {
                    "max_gradient": max_gradient,
                    "mean_gradient": mean_gradient,
                    "smoothness_score": 1.0 / (1.0 + mean_gradient),
                }

                if max_gradient > self.smoothness_threshold:
                    issues.append(
                        f"High elevation gradients detected (max: {max_gradient:.2f}m/pixel)"
                    )
                    score -= 0.2

            # 4. CRS validation
            if crs is None:
                issues.append("No coordinate reference system defined")
                score -= 0.3
            elif not crs.is_valid:
                issues.append(f"Invalid coordinate reference system: {crs}")
                score -= 0.3

            # Determine status
            if score >= 0.8:
                status = "PASS"
            elif score >= 0.6:
                status = "WARN"
            else:
                status = "FAIL"

            self.logger.info(f"DEM validation completed: {status} (score: {score:.3f})")
            return self._create_result(
                "DEM_Quality", status, max(0.0, score), details, issues
            )

        except Exception as e:
            self.logger.error(f"DEM validation failed: {e}")
            return self._create_result(
                "DEM_Quality",
                "FAIL",
                0.0,
                {"error": str(e)},
                [f"Validation failed: {e}"],
            )


class RainfallValidator(BaseValidator):
    """
    Rainfall Data Validator

    Validates:
    - Value range and distribution
    - Spatial coverage and resolution
    - Temporal consistency
    - Missing data patterns
    - Physical plausibility
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_intensity = config.get("max_intensity", 500)  # mm/hr
        self.min_coverage = config.get("min_coverage", 0.95)  # 95% spatial coverage
        self.missing_data_threshold = config.get(
            "missing_data_threshold", 0.1
        )  # 10% max missing

    def validate(
        self, rainfall_data: Union[str, Path, np.ndarray, xr.DataArray], **kwargs
    ) -> ValidationResult:
        """Validate rainfall data quality"""
        try:
            # Load data if path provided
            if isinstance(rainfall_data, (str, Path)):
                if str(rainfall_data).endswith(".nc"):
                    rainfall_data = xr.open_dataarray(rainfall_data)
                else:
                    with rasterio.open(rainfall_data) as src:
                        rainfall_data = src.read(masked=True)

            issues = []
            details = {}
            score = 1.0

            # Convert to numpy for analysis
            if isinstance(rainfall_data, xr.DataArray):
                data_array = rainfall_data.values
                has_time = "time" in rainfall_data.dims
                details["has_temporal_dimension"] = has_time
                details["dimensions"] = list(rainfall_data.dims)
                details["shape"] = rainfall_data.shape
            else:
                data_array = rainfall_data
                has_time = len(data_array.shape) > 2
                details["has_temporal_dimension"] = has_time
                details["shape"] = data_array.shape

            # Handle masked arrays
            if hasattr(data_array, "mask"):
                valid_data = data_array.compressed()
            else:
                valid_data = data_array[~np.isnan(data_array)]

            if len(valid_data) == 0:
                issues.append("No valid rainfall data found")
                score = 0.0
            else:
                # 1. Value range validation
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                mean_val = float(np.mean(valid_data))
                std_val = float(np.std(valid_data))

                details["value_stats"] = {
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "std": std_val,
                    "percentiles": {
                        "25": float(np.percentile(valid_data, 25)),
                        "50": float(np.percentile(valid_data, 50)),
                        "75": float(np.percentile(valid_data, 75)),
                        "95": float(np.percentile(valid_data, 95)),
                        "99": float(np.percentile(valid_data, 99)),
                    },
                }

                # Check for negative values
                if min_val < 0:
                    issues.append(
                        f"Negative rainfall values detected (min: {min_val:.3f})"
                    )
                    score -= 0.2

                # Check for extreme values
                if max_val > self.max_intensity:
                    issues.append(
                        f"Extreme rainfall intensity ({max_val:.2f} mm/hr) exceeds threshold"
                    )
                    score -= 0.2

                # 2. Missing data analysis
                total_pixels = data_array.size
                valid_pixels = len(valid_data)
                missing_percentage = (total_pixels - valid_pixels) / total_pixels

                details["coverage_analysis"] = {
                    "total_pixels": int(total_pixels),
                    "valid_pixels": int(valid_pixels),
                    "missing_percentage": float(missing_percentage),
                    "coverage_percentage": float(1 - missing_percentage),
                }

                if missing_percentage > self.missing_data_threshold:
                    issues.append(
                        f"High missing data percentage ({missing_percentage*100:.2f}%)"
                    )
                    score -= 0.3

                # 3. Spatial pattern analysis
                if data_array.ndim >= 2:
                    # Check for spatial patterns that might indicate data issues
                    if data_array.ndim == 3:  # Time series
                        spatial_data = np.nanmean(data_array, axis=0)
                    else:
                        spatial_data = data_array

                    # Calculate spatial statistics
                    if not np.all(np.isnan(spatial_data)):
                        spatial_std = float(np.nanstd(spatial_data))
                        spatial_mean = float(np.nanmean(spatial_data))

                        details["spatial_analysis"] = {
                            "spatial_mean": spatial_mean,
                            "spatial_std": spatial_std,
                            "coefficient_of_variation": (
                                spatial_std / spatial_mean
                                if spatial_mean > 0
                                else np.inf
                            ),
                        }

                # 4. Temporal consistency (if time dimension exists)
                if has_time and data_array.ndim == 3:
                    temporal_means = np.nanmean(data_array, axis=(1, 2))
                    temporal_consistency = (
                        1.0 - (np.nanstd(temporal_means) / np.nanmean(temporal_means))
                        if np.nanmean(temporal_means) > 0
                        else 0.0
                    )

                    details["temporal_analysis"] = {
                        "temporal_consistency": float(temporal_consistency),
                        "time_steps": data_array.shape[0],
                        "temporal_mean": float(np.nanmean(temporal_means)),
                        "temporal_std": float(np.nanstd(temporal_means)),
                    }

                    if temporal_consistency < 0.5:
                        issues.append(
                            f"Poor temporal consistency ({temporal_consistency:.3f})"
                        )
                        score -= 0.2

            # Determine status
            if score >= 0.8:
                status = "PASS"
            elif score >= 0.6:
                status = "WARN"
            else:
                status = "FAIL"

            self.logger.info(
                f"Rainfall validation completed: {status} (score: {score:.3f})"
            )
            return self._create_result(
                "Rainfall_Quality", status, max(0.0, score), details, issues
            )

        except Exception as e:
            self.logger.error(f"Rainfall validation failed: {e}")
            return self._create_result(
                "Rainfall_Quality",
                "FAIL",
                0.0,
                {"error": str(e)},
                [f"Validation failed: {e}"],
            )


class SpatialConsistencyValidator(BaseValidator):
    """
    Spatial Consistency Validator

    Validates:
    - CRS consistency across datasets
    - Extent alignment
    - Resolution matching
    - Spatial overlap
    - Projection accuracy
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tolerance = config.get(
            "spatial_tolerance", 0.1
        )  # tolerance in coordinate units
        self.min_overlap = config.get("min_overlap", 0.95)  # 95% minimum overlap

    def validate(self, datasets: List[Dict[str, Any]], **kwargs) -> ValidationResult:
        """
        Validate spatial consistency across multiple datasets

        Args:
            datasets: List of dataset info dicts with keys: 'path', 'type', 'name'
        """
        try:
            if len(datasets) < 2:
                return self._create_result(
                    "Spatial_Consistency",
                    "WARN",
                    0.5,
                    {
                        "message": "Need at least 2 datasets for spatial consistency check"
                    },
                    ["Insufficient datasets for comparison"],
                )

            dataset_info = []
            issues = []
            score = 1.0

            # Extract spatial info from each dataset
            for dataset in datasets:
                try:
                    with rasterio.open(dataset["path"]) as src:
                        info = {
                            "name": dataset.get("name", dataset["path"]),
                            "type": dataset.get("type", "unknown"),
                            "crs": src.crs,
                            "bounds": src.bounds,
                            "transform": src.transform,
                            "shape": src.shape,
                            "resolution": (abs(src.transform.a), abs(src.transform.e)),
                        }
                        dataset_info.append(info)
                except Exception as e:
                    issues.append(
                        f"Could not read spatial info from {dataset['path']}: {e}"
                    )
                    score -= 0.3

            if len(dataset_info) < 2:
                return self._create_result(
                    "Spatial_Consistency",
                    "FAIL",
                    0.0,
                    {"error": "Could not read spatial info from datasets"},
                    issues,
                )

            # Compare CRS
            reference_crs = dataset_info[0]["crs"]
            crs_consistent = True

            for i, info in enumerate(dataset_info[1:], 1):
                if info["crs"] != reference_crs:
                    issues.append(
                        f"CRS mismatch: {dataset_info[0]['name']} ({reference_crs}) vs {info['name']} ({info['crs']})"
                    )
                    crs_consistent = False
                    score -= 0.4

            # Compare spatial extents and resolutions
            extent_analysis = self._analyze_spatial_extents(dataset_info)
            resolution_analysis = self._analyze_resolutions(dataset_info)

            # Check spatial overlap
            overlap_analysis = self._calculate_spatial_overlap(dataset_info)
            if overlap_analysis["min_overlap"] < self.min_overlap:
                issues.append(
                    f"Insufficient spatial overlap ({overlap_analysis['min_overlap']:.3f}) below threshold"
                )
                score -= 0.3

            details = {
                "datasets": [
                    {
                        "name": info["name"],
                        "crs": str(info["crs"]),
                        "bounds": info["bounds"],
                        "shape": info["shape"],
                        "resolution": info["resolution"],
                    }
                    for info in dataset_info
                ],
                "crs_consistent": crs_consistent,
                "reference_crs": str(reference_crs),
                "extent_analysis": extent_analysis,
                "resolution_analysis": resolution_analysis,
                "overlap_analysis": overlap_analysis,
            }

            # Add resolution consistency check
            if not resolution_analysis["consistent"]:
                issues.append(
                    f"Resolution inconsistency detected: {resolution_analysis['details']}"
                )
                score -= 0.2

            # Determine status
            if score >= 0.8:
                status = "PASS"
            elif score >= 0.6:
                status = "WARN"
            else:
                status = "FAIL"

            self.logger.info(
                f"Spatial consistency validation completed: {status} (score: {score:.3f})"
            )
            return self._create_result(
                "Spatial_Consistency", status, max(0.0, score), details, issues
            )

        except Exception as e:
            self.logger.error(f"Spatial consistency validation failed: {e}")
            return self._create_result(
                "Spatial_Consistency",
                "FAIL",
                0.0,
                {"error": str(e)},
                [f"Validation failed: {e}"],
            )

    def _analyze_spatial_extents(self, dataset_info: List[Dict]) -> Dict[str, Any]:
        """Analyze spatial extents across datasets"""
        bounds_list = [info["bounds"] for info in dataset_info]

        # Calculate union and intersection bounds
        min_left = min(b.left for b in bounds_list)
        max_right = max(b.right for b in bounds_list)
        min_bottom = min(b.bottom for b in bounds_list)
        max_top = max(b.top for b in bounds_list)

        max_left = max(b.left for b in bounds_list)
        min_right = min(b.right for b in bounds_list)
        max_bottom = max(b.bottom for b in bounds_list)
        min_top = min(b.top for b in bounds_list)

        union_area = (max_right - min_left) * (max_top - min_bottom)
        intersection_area = max(0, (min_right - max_left)) * max(
            0, (min_top - max_bottom)
        )

        return {
            "union_bounds": (min_left, min_bottom, max_right, max_top),
            "intersection_bounds": (max_left, max_bottom, min_right, min_top),
            "union_area": union_area,
            "intersection_area": intersection_area,
            "overlap_ratio": intersection_area / union_area if union_area > 0 else 0,
        }

    def _analyze_resolutions(self, dataset_info: List[Dict]) -> Dict[str, Any]:
        """Analyze resolution consistency"""
        resolutions = [info["resolution"] for info in dataset_info]
        reference_res = resolutions[0]

        consistent = True
        details = []

        for i, res in enumerate(resolutions[1:], 1):
            x_diff = abs(res[0] - reference_res[0])
            y_diff = abs(res[1] - reference_res[1])

            if x_diff > self.tolerance or y_diff > self.tolerance:
                consistent = False
                details.append(
                    f"{dataset_info[i]['name']}: {res} vs reference {reference_res}"
                )

        return {
            "consistent": consistent,
            "reference_resolution": reference_res,
            "all_resolutions": resolutions,
            "details": details,
        }

    def _calculate_spatial_overlap(self, dataset_info: List[Dict]) -> Dict[str, Any]:
        """Calculate spatial overlap between datasets"""
        bounds_list = [info["bounds"] for info in dataset_info]
        overlaps = []

        for i in range(len(bounds_list)):
            for j in range(i + 1, len(bounds_list)):
                b1, b2 = bounds_list[i], bounds_list[j]

                # Calculate intersection
                left = max(b1.left, b2.left)
                right = min(b1.right, b2.right)
                bottom = max(b1.bottom, b2.bottom)
                top = min(b1.top, b2.top)

                if left < right and bottom < top:
                    intersection_area = (right - left) * (top - bottom)
                    area1 = (b1.right - b1.left) * (b1.top - b1.bottom)
                    area2 = (b2.right - b2.left) * (b2.top - b2.bottom)

                    overlap1 = intersection_area / area1
                    overlap2 = intersection_area / area2

                    overlaps.append(
                        {
                            "dataset1": dataset_info[i]["name"],
                            "dataset2": dataset_info[j]["name"],
                            "overlap1": overlap1,
                            "overlap2": overlap2,
                            "mutual_overlap": min(overlap1, overlap2),
                        }
                    )
                else:
                    overlaps.append(
                        {
                            "dataset1": dataset_info[i]["name"],
                            "dataset2": dataset_info[j]["name"],
                            "overlap1": 0.0,
                            "overlap2": 0.0,
                            "mutual_overlap": 0.0,
                        }
                    )

        min_overlap = min(o["mutual_overlap"] for o in overlaps) if overlaps else 0.0
        mean_overlap = (
            np.mean([o["mutual_overlap"] for o in overlaps]) if overlaps else 0.0
        )

        return {
            "pairwise_overlaps": overlaps,
            "min_overlap": min_overlap,
            "mean_overlap": mean_overlap,
        }


class SimulationValidator(BaseValidator):
    """
    Simulation Results Validator

    Validates:
    - Physical plausibility of flood depths
    - Mass conservation
    - Convergence indicators
    - Boundary condition consistency
    - Result stability
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_depth = config.get("max_depth", 50.0)  # 50m max depth
        self.mass_conservation_tolerance = config.get(
            "mass_conservation_tolerance", 0.05
        )  # 5%
        self.convergence_threshold = config.get("convergence_threshold", 1e-6)

    def validate(
        self, simulation_results: Dict[str, Any], **kwargs
    ) -> ValidationResult:
        """Validate simulation results"""
        try:
            issues = []
            details = {}
            score = 1.0

            # Extract simulation data
            depths = simulation_results.get("depths")
            velocities = simulation_results.get("velocities")
            convergence_info = simulation_results.get("convergence", {})

            if depths is None:
                issues.append("No depth data provided in simulation results")
                score = 0.0
            else:
                # 1. Physical plausibility checks
                valid_depths = depths[~np.isnan(depths)]
                if len(valid_depths) > 0:
                    depth_stats = {
                        "min": float(np.min(valid_depths)),
                        "max": float(np.max(valid_depths)),
                        "mean": float(np.mean(valid_depths)),
                        "std": float(np.std(valid_depths)),
                        "flooded_fraction": float(
                            np.sum(valid_depths > 0.01) / len(valid_depths)
                        ),
                    }
                    details["depth_analysis"] = depth_stats

                    # Check for negative depths
                    if depth_stats["min"] < 0:
                        issues.append(
                            f"Negative flood depths detected (min: {depth_stats['min']:.3f}m)"
                        )
                        score -= 0.3

                    # Check for extreme depths
                    if depth_stats["max"] > self.max_depth:
                        issues.append(
                            f"Extreme flood depth ({depth_stats['max']:.2f}m) exceeds threshold"
                        )
                        score -= 0.2

                    # Check for unrealistic spatial patterns
                    if depths.ndim >= 2:
                        gradient_check = self._check_spatial_gradients(depths)
                        details["gradient_analysis"] = gradient_check

                        if gradient_check["extreme_gradients"] > 0.01:  # 1% of pixels
                            issues.append(
                                f"High number of extreme gradients ({gradient_check['extreme_gradients']*100:.2f}%)"
                            )
                            score -= 0.2

                # 2. Mass conservation check
                if "inflow" in simulation_results and "outflow" in simulation_results:
                    mass_balance = self._check_mass_conservation(
                        simulation_results["inflow"],
                        simulation_results["outflow"],
                        depths,
                    )
                    details["mass_conservation"] = mass_balance

                    if (
                        mass_balance["conservation_error"]
                        > self.mass_conservation_tolerance
                    ):
                        issues.append(
                            f"Poor mass conservation (error: {mass_balance['conservation_error']*100:.2f}%)"
                        )
                        score -= 0.3

                # 3. Convergence analysis
                if convergence_info:
                    convergence_analysis = self._analyze_convergence(convergence_info)
                    details["convergence_analysis"] = convergence_analysis

                    if not convergence_analysis["converged"]:
                        issues.append("Simulation did not converge properly")
                        score -= 0.4
                    elif (
                        convergence_analysis["iterations"]
                        > convergence_analysis.get("max_iterations", 1000) * 0.9
                    ):
                        issues.append(
                            "Simulation required excessive iterations to converge"
                        )
                        score -= 0.1

                # 4. Velocity validation (if available)
                if velocities is not None:
                    velocity_analysis = self._validate_velocities(velocities, depths)
                    details["velocity_analysis"] = velocity_analysis

                    if velocity_analysis["max_velocity"] > 20.0:  # 20 m/s threshold
                        issues.append(
                            f"Extreme velocities detected (max: {velocity_analysis['max_velocity']:.2f} m/s)"
                        )
                        score -= 0.2

            # Determine status
            if score >= 0.8:
                status = "PASS"
            elif score >= 0.6:
                status = "WARN"
            else:
                status = "FAIL"

            self.logger.info(
                f"Simulation validation completed: {status} (score: {score:.3f})"
            )
            return self._create_result(
                "Simulation_Quality", status, max(0.0, score), details, issues
            )

        except Exception as e:
            self.logger.error(f"Simulation validation failed: {e}")
            return self._create_result(
                "Simulation_Quality",
                "FAIL",
                0.0,
                {"error": str(e)},
                [f"Validation failed: {e}"],
            )

    def _check_spatial_gradients(self, depths: np.ndarray) -> Dict[str, Any]:
        """Check for unrealistic spatial gradients in flood depths"""
        if depths.ndim != 2:
            return {"error": "Depth data must be 2D for gradient analysis"}

        # Calculate gradients
        grad_x, grad_y = np.gradient(depths)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Remove NaN values
        valid_gradients = gradient_magnitude[~np.isnan(gradient_magnitude)]

        if len(valid_gradients) == 0:
            return {"error": "No valid gradients calculated"}

        # Define extreme gradient threshold (e.g., >10m depth change per pixel)
        extreme_threshold = 10.0
        extreme_gradients = np.sum(valid_gradients > extreme_threshold) / len(
            valid_gradients
        )

        return {
            "max_gradient": float(np.max(valid_gradients)),
            "mean_gradient": float(np.mean(valid_gradients)),
            "std_gradient": float(np.std(valid_gradients)),
            "extreme_gradients": float(extreme_gradients),
            "extreme_threshold": extreme_threshold,
        }

    def _check_mass_conservation(
        self, inflow: float, outflow: float, depths: np.ndarray
    ) -> Dict[str, Any]:
        """Check mass conservation in simulation"""
        # Calculate stored volume
        valid_depths = depths[~np.isnan(depths) & (depths > 0)]
        stored_volume = float(
            np.sum(valid_depths)
        )  # Simplified, assumes unit cell area

        # Mass balance: inflow = outflow + storage change
        total_input = inflow
        total_output = outflow + stored_volume

        conservation_error = abs(total_input - total_output) / max(total_input, 1e-10)

        return {
            "inflow": float(inflow),
            "outflow": float(outflow),
            "stored_volume": stored_volume,
            "conservation_error": float(conservation_error),
            "balance_check": (
                "PASS"
                if conservation_error < self.mass_conservation_tolerance
                else "FAIL"
            ),
        }

    def _analyze_convergence(self, convergence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze simulation convergence"""
        residuals = convergence_info.get("residuals", [])
        final_residual = convergence_info.get("final_residual", np.inf)
        iterations = convergence_info.get("iterations", 0)

        converged = final_residual < self.convergence_threshold

        analysis = {
            "converged": converged,
            "final_residual": float(final_residual),
            "iterations": int(iterations),
            "convergence_rate": None,
        }

        if len(residuals) > 1:
            # Calculate convergence rate
            log_residuals = np.log10(np.array(residuals) + 1e-15)
            if len(log_residuals) > 5:
                # Linear fit to log residuals
                x = np.arange(len(log_residuals))
                slope = np.polyfit(x, log_residuals, 1)[0]
                analysis["convergence_rate"] = float(slope)

        return analysis

    def _validate_velocities(
        self, velocities: np.ndarray, depths: np.ndarray
    ) -> Dict[str, Any]:
        """Validate velocity field"""
        # Assume velocities shape is (2, H, W) for u, v components
        if velocities.ndim == 3 and velocities.shape[0] == 2:
            u_vel = velocities[0]
            v_vel = velocities[1]
            velocity_magnitude = np.sqrt(u_vel**2 + v_vel**2)
        else:
            velocity_magnitude = velocities

        # Only consider velocities where there's water
        water_mask = depths > 0.01
        valid_velocities = velocity_magnitude[
            water_mask & ~np.isnan(velocity_magnitude)
        ]

        if len(valid_velocities) == 0:
            return {"error": "No valid velocities in flooded areas"}

        return {
            "max_velocity": float(np.max(valid_velocities)),
            "mean_velocity": float(np.mean(valid_velocities)),
            "std_velocity": float(np.std(valid_velocities)),
            "high_velocity_fraction": float(
                np.sum(valid_velocities > 5.0) / len(valid_velocities)
            ),
        }


class TileQualityValidator(BaseValidator):
    """
    Tile Quality Validator for ML training data

    Validates:
    - Flood/dry balance in tiles
    - Spatial coverage and distribution
    - Edge effects and artifacts
    - Class distribution
    - Tile size and overlap consistency
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_flood_ratio = config.get(
            "target_flood_ratio", (0.1, 0.9)
        )  # 10-90% flood coverage
        self.min_tiles = config.get("min_tiles", 100)
        self.edge_threshold = config.get("edge_threshold", 5)  # pixels from edge

    def validate(self, tiles_info: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate tile quality for ML training"""
        try:
            issues = []
            details = {}
            score = 1.0

            tiles_data = tiles_info.get("tiles", [])
            tile_metadata = tiles_info.get("metadata", {})

            if len(tiles_data) == 0:
                return self._create_result(
                    "Tile_Quality",
                    "FAIL",
                    0.0,
                    {"error": "No tiles provided for validation"},
                    ["No tiles found"],
                )

            # 1. Tile count validation
            num_tiles = len(tiles_data)
            details["tile_count"] = num_tiles

            if num_tiles < self.min_tiles:
                issues.append(
                    f"Insufficient tiles ({num_tiles}) below minimum threshold ({self.min_tiles})"
                )
                score -= 0.2

            # 2. Flood/dry balance analysis
            flood_ratios = []
            tile_sizes = []
            edge_issues = 0

            for i, tile in enumerate(tiles_data):
                if isinstance(tile, dict):
                    tile_data = tile.get("data")
                    if tile_data is None:
                        continue
                else:
                    tile_data = tile

                if tile_data is not None:
                    # Calculate flood ratio
                    flood_pixels = np.sum(tile_data > 0.01)  # Assuming depth data
                    total_pixels = tile_data.size
                    flood_ratio = flood_pixels / total_pixels if total_pixels > 0 else 0
                    flood_ratios.append(flood_ratio)

                    # Track tile sizes
                    tile_sizes.append(tile_data.shape)

                    # Check edge effects (simplified)
                    edge_sum = (
                        np.sum(tile_data[: self.edge_threshold, :])
                        + np.sum(tile_data[-self.edge_threshold :, :])
                        + np.sum(tile_data[:, : self.edge_threshold])
                        + np.sum(tile_data[:, -self.edge_threshold :])
                    )

                    center_area = tile_data[
                        self.edge_threshold : -self.edge_threshold,
                        self.edge_threshold : -self.edge_threshold,
                    ]
                    center_sum = np.sum(center_area) if center_area.size > 0 else 0

                    # Detect potential edge artifacts
                    if edge_sum > 0 and center_sum > 0:
                        edge_ratio = edge_sum / (edge_sum + center_sum)
                        if edge_ratio > 0.5:  # More flood at edges than center
                            edge_issues += 1

            # Analyze flood ratio distribution
            if flood_ratios:
                flood_ratios = np.array(flood_ratios)

                balance_analysis = {
                    "mean_flood_ratio": float(np.mean(flood_ratios)),
                    "std_flood_ratio": float(np.std(flood_ratios)),
                    "min_flood_ratio": float(np.min(flood_ratios)),
                    "max_flood_ratio": float(np.max(flood_ratios)),
                    "target_range": self.target_flood_ratio,
                    "tiles_in_range": int(
                        np.sum(
                            (flood_ratios >= self.target_flood_ratio[0])
                            & (flood_ratios <= self.target_flood_ratio[1])
                        )
                    ),
                    "all_dry_tiles": int(np.sum(flood_ratios == 0)),
                    "all_flooded_tiles": int(np.sum(flood_ratios == 1.0)),
                }
                details["flood_balance"] = balance_analysis

                # Check balance issues
                tiles_in_range_ratio = balance_analysis["tiles_in_range"] / len(
                    flood_ratios
                )
                if tiles_in_range_ratio < 0.7:  # Less than 70% in good range
                    issues.append(
                        f"Poor flood/dry balance: only {tiles_in_range_ratio*100:.1f}% of tiles in target range"
                    )
                    score -= 0.3

                # Check for too many extreme tiles
                extreme_tiles = (
                    balance_analysis["all_dry_tiles"]
                    + balance_analysis["all_flooded_tiles"]
                )
                if extreme_tiles / len(flood_ratios) > 0.3:  # More than 30% extreme
                    issues.append(
                        f"Too many extreme tiles ({extreme_tiles} all-dry or all-flooded)"
                    )
                    score -= 0.2

            # 3. Tile size consistency
            if tile_sizes:
                unique_sizes = list(set(tile_sizes))
                details["tile_sizes"] = {
                    "unique_sizes": unique_sizes,
                    "size_consistent": len(unique_sizes) == 1,
                    "most_common_size": max(set(tile_sizes), key=tile_sizes.count),
                }

                if len(unique_sizes) > 1:
                    issues.append(f"Inconsistent tile sizes detected: {unique_sizes}")
                    score -= 0.1

            # 4. Edge effects analysis
            details["edge_analysis"] = {
                "tiles_with_edge_issues": edge_issues,
                "edge_issue_ratio": edge_issues / len(tiles_data) if tiles_data else 0,
            }

            if edge_issues > len(tiles_data) * 0.1:  # More than 10% with edge issues
                issues.append(
                    f"High number of tiles with potential edge artifacts ({edge_issues})"
                )
                score -= 0.2

            # 5. Metadata validation
            if tile_metadata:
                details["metadata_validation"] = self._validate_tile_metadata(
                    tile_metadata
                )

            # Determine status
            if score >= 0.8:
                status = "PASS"
            elif score >= 0.6:
                status = "WARN"
            else:
                status = "FAIL"

            self.logger.info(
                f"Tile quality validation completed: {status} (score: {score:.3f})"
            )
            return self._create_result(
                "Tile_Quality", status, max(0.0, score), details, issues
            )

        except Exception as e:
            self.logger.error(f"Tile quality validation failed: {e}")
            return self._create_result(
                "Tile_Quality",
                "FAIL",
                0.0,
                {"error": str(e)},
                [f"Validation failed: {e}"],
            )

    def _validate_tile_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tile metadata"""
        required_fields = ["tile_size", "overlap", "projection", "bounds"]
        validation_results = {}

        for field in required_fields:
            validation_results[f"{field}_present"] = field in metadata

        # Additional metadata checks
        if "tile_size" in metadata:
            tile_size = metadata["tile_size"]
            validation_results["tile_size_valid"] = isinstance(
                tile_size, (int, tuple, list)
            )

        if "overlap" in metadata:
            overlap = metadata["overlap"]
            validation_results["overlap_valid"] = (
                isinstance(overlap, (int, float)) and 0 <= overlap < 1
            )

        return validation_results


class PipelineValidator:
    """
    Main Pipeline Validator orchestrating all validation components
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize validators
        self.dem_validator = DEMValidator(self.config.get("dem", {}))
        self.rainfall_validator = RainfallValidator(self.config.get("rainfall", {}))
        self.spatial_validator = SpatialConsistencyValidator(
            self.config.get("spatial", {})
        )
        self.simulation_validator = SimulationValidator(
            self.config.get("simulation", {})
        )
        self.tile_validator = TileQualityValidator(self.config.get("tiles", {}))

        self.results = []

    def validate_full_pipeline(
        self, pipeline_data: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """
        Run complete pipeline validation

        Args:
            pipeline_data: Dictionary containing all pipeline components to validate
                - dem_path: Path to DEM file
                - rainfall_data: Rainfall data or path
                - simulation_results: Simulation output
                - tiles_info: Tile generation results
                - spatial_datasets: List of datasets for spatial consistency

        Returns:
            Dictionary of validation results by component
        """
        self.logger.info("Starting full pipeline validation")
        validation_results = {}

        # 1. DEM Validation
        if "dem_path" in pipeline_data:
            try:
                dem_result = self.dem_validator.validate(pipeline_data["dem_path"])
                validation_results["dem"] = dem_result
                self.results.append(dem_result)
                self.logger.info(f"DEM validation: {dem_result.status}")
            except Exception as e:
                self.logger.error(f"DEM validation failed: {e}")

        # 2. Rainfall Validation
        if "rainfall_data" in pipeline_data:
            try:
                rainfall_result = self.rainfall_validator.validate(
                    pipeline_data["rainfall_data"]
                )
                validation_results["rainfall"] = rainfall_result
                self.results.append(rainfall_result)
                self.logger.info(f"Rainfall validation: {rainfall_result.status}")
            except Exception as e:
                self.logger.error(f"Rainfall validation failed: {e}")

        # 3. Spatial Consistency Validation
        if "spatial_datasets" in pipeline_data:
            try:
                spatial_result = self.spatial_validator.validate(
                    pipeline_data["spatial_datasets"]
                )
                validation_results["spatial_consistency"] = spatial_result
                self.results.append(spatial_result)
                self.logger.info(
                    f"Spatial consistency validation: {spatial_result.status}"
                )
            except Exception as e:
                self.logger.error(f"Spatial consistency validation failed: {e}")

        # 4. Simulation Validation
        if "simulation_results" in pipeline_data:
            try:
                sim_result = self.simulation_validator.validate(
                    pipeline_data["simulation_results"]
                )
                validation_results["simulation"] = sim_result
                self.results.append(sim_result)
                self.logger.info(f"Simulation validation: {sim_result.status}")
            except Exception as e:
                self.logger.error(f"Simulation validation failed: {e}")

        # 5. Tile Quality Validation
        if "tiles_info" in pipeline_data:
            try:
                tile_result = self.tile_validator.validate(pipeline_data["tiles_info"])
                validation_results["tiles"] = tile_result
                self.results.append(tile_result)
                self.logger.info(f"Tile quality validation: {tile_result.status}")
            except Exception as e:
                self.logger.error(f"Tile quality validation failed: {e}")

        self.logger.info("Full pipeline validation completed")
        return validation_results

    def generate_validation_report(
        self, output_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report

        Args:
            output_path: Optional path to save report as JSON

        Returns:
            Dictionary containing full validation report
        """
        if not self.results:
            self.logger.warning("No validation results available for report generation")
            return {}

        # Calculate overall scores
        scores = [r.score for r in self.results if r.score is not None]
        overall_score = np.mean(scores) if scores else 0.0

        # Count status distribution
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # Collect all issues
        all_issues = []
        for result in self.results:
            for issue in result.issues:
                all_issues.append(
                    {
                        "component": result.component,
                        "issue": issue,
                        "timestamp": result.timestamp,
                    }
                )

        report = {
            "validation_summary": {
                "overall_score": overall_score,
                "total_components": len(self.results),
                "status_distribution": status_counts,
                "validation_timestamp": datetime.now().isoformat(),
            },
            "component_results": [result.to_dict() for result in self.results],
            "issues_summary": all_issues,
            "recommendations": self._generate_recommendations(),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved to {output_path}")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        for result in self.results:
            if result.status == "FAIL":
                recommendations.append(
                    f"CRITICAL: Address issues in {result.component} before proceeding"
                )
            elif result.status == "WARN":
                recommendations.append(
                    f"WARNING: Review and potentially fix issues in {result.component}"
                )

            # Component-specific recommendations
            if result.component == "DEM_Quality" and result.score < 0.8:
                recommendations.append(
                    "Consider DEM preprocessing: void filling, smoothing, or elevation correction"
                )

            elif result.component == "Rainfall_Quality" and result.score < 0.8:
                recommendations.append(
                    "Review rainfall data quality: check for missing values, extreme outliers"
                )

            elif result.component == "Spatial_Consistency" and result.score < 0.8:
                recommendations.append(
                    "Ensure all datasets use consistent CRS and spatial extents"
                )

            elif result.component == "Simulation_Quality" and result.score < 0.8:
                recommendations.append(
                    "Check simulation convergence and physical plausibility of results"
                )

            elif result.component == "Tile_Quality" and result.score < 0.8:
                recommendations.append(
                    "Review tile generation: balance flood/dry ratios, check for edge effects"
                )

        if not recommendations:
            recommendations.append(
                "All validation checks passed successfully - pipeline ready for ML training"
            )

        return recommendations

    def get_pipeline_status(self) -> str:
        """Get overall pipeline validation status"""
        if not self.results:
            return "UNKNOWN"

        statuses = [r.status for r in self.results]

        if any(s == "FAIL" for s in statuses):
            return "FAIL"
        elif any(s == "WARN" for s in statuses):
            return "WARN"
        else:
            return "PASS"

    def clear_results(self):
        """Clear validation results"""
        self.results.clear()
        self.logger.info("Validation results cleared")
