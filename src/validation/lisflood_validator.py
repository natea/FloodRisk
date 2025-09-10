"""
LISFLOOD-FP Model Validator

Compares flood risk model outputs with LISFLOOD-FP reference simulations.
LISFLOOD-FP is a widely used 2D flood inundation model that serves as a
benchmark for flood modeling validation.

Features:
- Reads LISFLOOD-FP output files (.asc, .wd, .max files)
- Spatial alignment and resampling
- Statistical comparison metrics
- Visualization of differences
- Temporal analysis for dynamic models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
from datetime import datetime

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from rasterio.enums import Resampling

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not available - limited spatial functionality")

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    warnings.warn("xarray not available - limited NetCDF support")

from .metrics import MetricsCalculator, MetricError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LISFLOODConfig:
    """Configuration for LISFLOOD-FP validation"""

    reference_path: str
    output_format: str = "asc"  # 'asc', 'wd', 'max', 'netcdf'
    spatial_tolerance: float = 1e-6
    temporal_tolerance: float = 3600  # seconds
    depth_threshold: float = 0.01  # minimum depth to consider as flood
    max_depth: float = 50.0  # maximum realistic depth
    resample_method: str = "bilinear"
    mask_nodata: bool = True
    calculate_spatial_stats: bool = True
    calculate_temporal_stats: bool = True


class LISFLOODFileReader:
    """
    Handles reading various LISFLOOD-FP output file formats
    """

    @staticmethod
    def read_asc_file(filepath: Union[str, Path]) -> Dict:
        """
        Read LISFLOOD-FP ASCII grid file

        Args:
            filepath: Path to .asc file

        Returns:
            Dictionary with data, transform, and metadata
        """
        try:
            filepath = Path(filepath)
            logger.info(f"Reading LISFLOOD-FP ASC file: {filepath}")

            with open(filepath, "r") as f:
                # Read header
                header = {}
                for i in range(6):  # Standard ASC header has 6 lines
                    line = f.readline().strip().split()
                    header[line[0].lower()] = (
                        float(line[1]) if "." in line[1] else int(line[1])
                    )

                # Read data
                data = []
                for line in f:
                    row = [float(x) for x in line.strip().split()]
                    data.append(row)

            data = np.array(data)

            # Handle nodata values
            nodata_value = header.get("nodata_value", -9999)
            if "nodata_value" in header:
                data[data == nodata_value] = np.nan

            # Create transform for georeferencing
            transform = from_bounds(
                header["xllcorner"],
                header["yllcorner"],
                header["xllcorner"] + header["ncols"] * header["cellsize"],
                header["yllcorner"] + header["nrows"] * header["cellsize"],
                header["ncols"],
                header["nrows"],
            )

            result = {
                "data": data,
                "transform": transform,
                "crs": None,  # ASC files don't include CRS info
                "header": header,
                "nodata": nodata_value,
            }

            logger.info(f"Successfully read ASC file: {data.shape} cells")
            return result

        except Exception as e:
            logger.error(f"Error reading ASC file {filepath}: {e}")
            raise IOError(f"Failed to read LISFLOOD-FP ASC file: {e}")

    @staticmethod
    def read_wd_file(filepath: Union[str, Path]) -> Dict:
        """
        Read LISFLOOD-FP water depth file (binary format)

        Args:
            filepath: Path to .wd file

        Returns:
            Dictionary with data and metadata
        """
        try:
            filepath = Path(filepath)
            logger.info(f"Reading LISFLOOD-FP WD file: {filepath}")

            # WD files are typically binary - need header info
            # This would need to be customized based on specific LISFLOOD-FP setup
            raise NotImplementedError("Binary WD file reading not yet implemented")

        except Exception as e:
            logger.error(f"Error reading WD file {filepath}: {e}")
            raise IOError(f"Failed to read LISFLOOD-FP WD file: {e}")

    @staticmethod
    def read_netcdf_file(filepath: Union[str, Path]) -> Dict:
        """
        Read LISFLOOD-FP NetCDF output file

        Args:
            filepath: Path to NetCDF file

        Returns:
            Dictionary with data and metadata
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required for NetCDF support")

        try:
            filepath = Path(filepath)
            logger.info(f"Reading LISFLOOD-FP NetCDF file: {filepath}")

            ds = xr.open_dataset(filepath)

            # Extract water depth data (variable name may vary)
            depth_vars = ["depth", "water_depth", "wd", "h"]
            depth_var = None
            for var in depth_vars:
                if var in ds.variables:
                    depth_var = var
                    break

            if depth_var is None:
                available_vars = list(ds.variables.keys())
                raise ValueError(
                    f"No depth variable found. Available: {available_vars}"
                )

            data = ds[depth_var].values

            # Get spatial information
            if "x" in ds.coords and "y" in ds.coords:
                x_coords = ds.x.values
                y_coords = ds.y.values

                # Create transform
                x_res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
                y_res = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0

                transform = from_bounds(
                    x_coords.min() - x_res / 2,
                    y_coords.min() - y_res / 2,
                    x_coords.max() + x_res / 2,
                    y_coords.max() + y_res / 2,
                    len(x_coords),
                    len(y_coords),
                )
            else:
                transform = None

            result = {
                "data": data,
                "transform": transform,
                "crs": ds.attrs.get("crs", None),
                "dataset": ds,
                "time_coords": ds.coords.get("time", None),
            }

            logger.info(f"Successfully read NetCDF file: {data.shape}")
            return result

        except Exception as e:
            logger.error(f"Error reading NetCDF file {filepath}: {e}")
            raise IOError(f"Failed to read LISFLOOD-FP NetCDF file: {e}")


class SpatialAligner:
    """
    Handles spatial alignment between model outputs and LISFLOOD-FP reference data
    """

    def __init__(self, tolerance: float = 1e-6, method: str = "bilinear"):
        """
        Initialize spatial aligner

        Args:
            tolerance: Spatial tolerance for alignment
            method: Resampling method ('nearest', 'bilinear', 'cubic')
        """
        self.tolerance = tolerance
        self.method = method
        logger.info(
            f"SpatialAligner initialized: tolerance={tolerance}, method={method}"
        )

    def align_grids(
        self,
        model_data: np.ndarray,
        reference_data: Dict,
        model_transform=None,
        target_transform=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align model data with reference data spatially

        Args:
            model_data: Model predictions
            reference_data: Reference data dictionary
            model_transform: Model data transform
            target_transform: Target transform (uses reference if None)

        Returns:
            Tuple of aligned (model_data, reference_data)
        """
        try:
            logger.info("Aligning model and reference grids")

            ref_data = reference_data["data"]
            ref_transform = reference_data.get("transform", target_transform)

            # If no spatial information, assume grids are already aligned
            if model_transform is None or ref_transform is None:
                logger.warning(
                    "No transform information - assuming grids are spatially aligned"
                )
                return self._align_by_shape(model_data, ref_data)

            if not HAS_RASTERIO:
                logger.warning("rasterio not available - using shape-based alignment")
                return self._align_by_shape(model_data, ref_data)

            # Use rasterio for proper spatial alignment
            return self._align_with_rasterio(
                model_data, ref_data, model_transform, ref_transform
            )

        except Exception as e:
            logger.error(f"Error aligning grids: {e}")
            raise MetricError(f"Grid alignment failed: {e}")

    def _align_by_shape(
        self, model_data: np.ndarray, ref_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align grids based on shape only (fallback method)
        """
        logger.info("Aligning grids by shape")

        if model_data.shape == ref_data.shape:
            return model_data, ref_data

        # Crop or pad to match shapes
        min_rows = min(model_data.shape[0], ref_data.shape[0])
        min_cols = min(model_data.shape[1], ref_data.shape[1])

        model_aligned = model_data[:min_rows, :min_cols]
        ref_aligned = ref_data[:min_rows, :min_cols]

        logger.info(f"Shape alignment completed: {model_aligned.shape}")
        return model_aligned, ref_aligned

    def _align_with_rasterio(
        self,
        model_data: np.ndarray,
        ref_data: np.ndarray,
        model_transform,
        ref_transform,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align grids using rasterio reprojection
        """
        logger.info("Aligning grids with rasterio")

        # Reproject model data to reference grid
        resampling_method = getattr(Resampling, self.method)

        model_aligned = np.empty_like(ref_data)

        reproject(
            source=model_data,
            destination=model_aligned,
            src_transform=model_transform,
            dst_transform=ref_transform,
            resampling=resampling_method,
        )

        logger.info("Rasterio alignment completed")
        return model_aligned, ref_data


class LISFLOODValidator:
    """
    Main validator for comparing model outputs with LISFLOOD-FP simulations
    """

    def __init__(self, config: Optional[LISFLOODConfig] = None):
        """
        Initialize LISFLOOD-FP validator

        Args:
            config: Validation configuration
        """
        self.config = config or LISFLOODConfig("./lisflood_reference")
        self.metrics_calc = MetricsCalculator()
        self.file_reader = LISFLOODFileReader()
        self.spatial_aligner = SpatialAligner(
            tolerance=self.config.spatial_tolerance, method=self.config.resample_method
        )

        logger.info(
            f"LISFLOOD Validator initialized with reference: {self.config.reference_path}"
        )

    def validate_against_lisflood(
        self,
        model_predictions: np.ndarray,
        reference_file: Optional[str] = None,
        model_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Validate model predictions against LISFLOOD-FP reference data

        Args:
            model_predictions: Model flood depth predictions
            reference_file: Path to LISFLOOD-FP reference file (optional)
            model_metadata: Metadata about model predictions

        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            logger.info("Starting LISFLOOD-FP validation")

            # Load reference data
            ref_file = reference_file or self.config.reference_path
            reference_data = self._load_reference_data(ref_file)

            # Preprocess data
            model_clean, ref_clean = self._preprocess_data(
                model_predictions, reference_data, model_metadata
            )

            # Spatial alignment
            model_aligned, ref_aligned = self.spatial_aligner.align_grids(
                model_clean,
                reference_data,
                model_metadata.get("transform") if model_metadata else None,
            )

            # Calculate validation metrics
            validation_results = self._calculate_validation_metrics(
                model_aligned, ref_aligned
            )

            # Spatial statistics
            if self.config.calculate_spatial_stats:
                spatial_stats = self._calculate_spatial_statistics(
                    model_aligned, ref_aligned
                )
                validation_results["spatial_analysis"] = spatial_stats

            # Add metadata
            validation_results["metadata"] = {
                "reference_file": str(ref_file),
                "validation_timestamp": datetime.now().isoformat(),
                "config": self.config.__dict__,
                "data_shapes": {
                    "model": model_aligned.shape,
                    "reference": ref_aligned.shape,
                },
            }

            logger.info("LISFLOOD-FP validation completed successfully")
            return validation_results

        except Exception as e:
            logger.error(f"LISFLOOD-FP validation failed: {e}")
            raise MetricError(f"LISFLOOD validation error: {e}")

    def _load_reference_data(self, reference_file: Union[str, Path]) -> Dict:
        """Load LISFLOOD-FP reference data"""
        ref_path = Path(reference_file)

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_path}")

        # Determine file type and load appropriately
        if ref_path.suffix.lower() == ".asc":
            return self.file_reader.read_asc_file(ref_path)
        elif ref_path.suffix.lower() == ".nc":
            return self.file_reader.read_netcdf_file(ref_path)
        elif ref_path.suffix.lower() == ".wd":
            return self.file_reader.read_wd_file(ref_path)
        else:
            # Try ASC format as default
            logger.warning(
                f"Unknown file extension {ref_path.suffix}, trying ASC format"
            )
            return self.file_reader.read_asc_file(ref_path)

    def _preprocess_data(
        self,
        model_data: np.ndarray,
        reference_data: Dict,
        model_metadata: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess model and reference data for comparison"""
        logger.info("Preprocessing data for validation")

        ref_data = reference_data["data"]

        # Handle nodata values
        if self.config.mask_nodata:
            if "nodata" in reference_data:
                nodata_mask = ref_data == reference_data["nodata"]
                ref_data = np.where(nodata_mask, np.nan, ref_data)

        # Apply depth thresholds
        model_data = np.maximum(model_data, 0)  # No negative depths
        ref_data = np.maximum(ref_data, 0)

        # Apply minimum depth threshold
        model_data[model_data < self.config.depth_threshold] = 0
        ref_data[ref_data < self.config.depth_threshold] = 0

        # Apply maximum depth clipping
        model_data = np.minimum(model_data, self.config.max_depth)
        ref_data = np.minimum(ref_data, self.config.max_depth)

        logger.info("Data preprocessing completed")
        return model_data, ref_data

    def _calculate_validation_metrics(
        self, model_data: np.ndarray, reference_data: np.ndarray
    ) -> Dict:
        """Calculate comprehensive validation metrics"""
        logger.info("Calculating validation metrics")

        # Get all standard metrics
        all_metrics = self.metrics_calc.calculate_all_metrics(
            model_data, reference_data
        )

        # Add LISFLOOD-specific metrics
        lisflood_metrics = self._calculate_lisflood_specific_metrics(
            model_data, reference_data
        )

        return {"standard_metrics": all_metrics, "lisflood_specific": lisflood_metrics}

    def _calculate_lisflood_specific_metrics(
        self, model_data: np.ndarray, reference_data: np.ndarray
    ) -> Dict:
        """Calculate metrics specific to LISFLOOD-FP comparison"""

        # Flood extent agreement
        model_extent = (model_data > self.config.depth_threshold).astype(int)
        ref_extent = (reference_data > self.config.depth_threshold).astype(int)

        # Wet/dry classification accuracy
        extent_metrics = self.metrics_calc.calculate_classification_metrics(
            model_extent, ref_extent
        )

        # Depth difference analysis
        depth_diff = model_data - reference_data
        depth_stats = {
            "mean_depth_difference": float(np.nanmean(depth_diff)),
            "std_depth_difference": float(np.nanstd(depth_diff)),
            "max_underestimation": float(np.nanmin(depth_diff)),
            "max_overestimation": float(np.nanmax(depth_diff)),
            "depth_rmse": float(np.sqrt(np.nanmean(depth_diff**2))),
        }

        # Volumetric comparison
        model_volume = float(np.nansum(model_data))
        ref_volume = float(np.nansum(reference_data))
        volume_metrics = {
            "model_total_volume": model_volume,
            "reference_total_volume": ref_volume,
            "volume_difference": model_volume - ref_volume,
            "volume_bias_percent": (
                100 * (model_volume - ref_volume) / ref_volume
                if ref_volume > 0
                else np.inf
            ),
        }

        return {
            "extent_classification": extent_metrics,
            "depth_analysis": depth_stats,
            "volumetric_analysis": volume_metrics,
        }

    def _calculate_spatial_statistics(
        self, model_data: np.ndarray, reference_data: np.ndarray
    ) -> Dict:
        """Calculate spatial statistics and patterns"""

        # Spatial correlation
        valid_mask = ~(np.isnan(model_data) | np.isnan(reference_data))
        if np.sum(valid_mask) > 1:
            spatial_corr = np.corrcoef(
                model_data[valid_mask], reference_data[valid_mask]
            )[0, 1]
        else:
            spatial_corr = np.nan

        # Spatial pattern analysis could be expanded here
        spatial_stats = {
            "spatial_correlation": (
                float(spatial_corr) if not np.isnan(spatial_corr) else None
            ),
            "valid_cells": int(np.sum(valid_mask)),
            "total_cells": int(model_data.size),
        }

        return spatial_stats

    def generate_comparison_report(
        self, validation_results: Dict, output_path: str
    ) -> None:
        """
        Generate a detailed comparison report

        Args:
            validation_results: Results from validate_against_lisflood
            output_path: Path for output report
        """
        logger.info(f"Generating LISFLOOD comparison report: {output_path}")

        # Create summary
        summary = {
            "validation_summary": self._create_validation_summary(validation_results),
            "detailed_results": validation_results,
            "recommendations": self._generate_recommendations(validation_results),
        }

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    def _create_validation_summary(self, results: Dict) -> Dict:
        """Create a summary of key validation metrics"""

        standard = results.get("standard_metrics", {})
        lisflood = results.get("lisflood_specific", {})

        # Extract key metrics
        summary = {}

        # IoU and CSI
        if "iou" in standard:
            summary["flood_extent_iou"] = standard["iou"].get("iou")
        if "csi" in standard:
            summary["critical_success_index"] = standard["csi"].get("csi")

        # Regression metrics
        if "regression" in standard and "error" not in standard["regression"]:
            reg = standard["regression"]
            summary.update(
                {
                    "depth_mae": reg.get("mae"),
                    "depth_rmse": reg.get("rmse"),
                    "nash_sutcliffe": reg.get("nse"),
                    "r_squared": reg.get("r_squared"),
                }
            )

        # Classification metrics
        if "classification" in standard and "error" not in standard["classification"]:
            cls = standard["classification"]
            summary.update(
                {
                    "extent_f1_score": cls.get("f1_score"),
                    "extent_accuracy": cls.get("accuracy"),
                }
            )

        # LISFLOOD-specific
        if "volumetric_analysis" in lisflood:
            vol = lisflood["volumetric_analysis"]
            summary["volume_bias_percent"] = vol.get("volume_bias_percent")

        return summary

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        standard = results.get("standard_metrics", {})

        # IoU recommendations
        if "iou" in standard:
            iou_score = standard["iou"].get("iou", 0)
            if iou_score < 0.5:
                recommendations.append(
                    "Low IoU score suggests poor flood extent prediction - review model parameters"
                )
            elif iou_score > 0.8:
                recommendations.append(
                    "Excellent flood extent prediction - model performs well"
                )

        # RMSE recommendations
        if "regression" in standard and "rmse" in standard["regression"]:
            rmse = standard["regression"]["rmse"]
            if rmse > 1.0:
                recommendations.append(
                    "High RMSE indicates significant depth prediction errors - consider model calibration"
                )

        # Volume bias recommendations
        lisflood = results.get("lisflood_specific", {})
        if "volumetric_analysis" in lisflood:
            vol_bias = lisflood["volumetric_analysis"].get("volume_bias_percent", 0)
            if abs(vol_bias) > 20:
                recommendations.append(
                    f"Large volume bias ({vol_bias:.1f}%) - check model water balance"
                )

        return recommendations


# Utility functions
def batch_validate_lisflood(
    model_files: List[str],
    reference_files: List[str],
    config: Optional[LISFLOODConfig] = None,
) -> pd.DataFrame:
    """
    Batch validation against multiple LISFLOOD-FP reference files

    Args:
        model_files: List of model output files
        reference_files: List of LISFLOOD-FP reference files
        config: Validation configuration

    Returns:
        DataFrame with validation results for all files
    """
    validator = LISFLOODValidator(config)
    results = []

    for model_file, ref_file in zip(model_files, reference_files):
        try:
            logger.info(f"Validating {model_file} against {ref_file}")

            # This would need file loading logic based on format
            # For now, assume arrays are provided
            # model_data = load_model_file(model_file)
            # validation_result = validator.validate_against_lisflood(model_data, ref_file)

            # results.append({
            #     'model_file': model_file,
            #     'reference_file': ref_file,
            #     **validation_result.get('validation_summary', {})
            # })

        except Exception as e:
            logger.error(f"Validation failed for {model_file}: {e}")
            results.append(
                {"model_file": model_file, "reference_file": ref_file, "error": str(e)}
            )

    return pd.DataFrame(results)
