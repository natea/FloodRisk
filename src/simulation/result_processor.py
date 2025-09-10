"""Result processing for LISFLOOD-FP simulation outputs.

This module handles post-processing of LISFLOOD-FP simulation outputs,
converting depth rasters to binary flood extent maps suitable for ML training.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not available - GeoTIFF export disabled")

try:
    from scipy import ndimage
    from scipy.ndimage import binary_closing, binary_opening, label

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("scipy not available - morphological operations disabled")

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for result processing."""

    # Flood extent thresholds
    flood_depth_threshold_m: float = 0.05  # Minimum depth for flood mapping

    # Morphological processing
    remove_small_areas: bool = True
    min_flood_area_pixels: int = 4  # Remove areas smaller than this
    close_gaps: bool = True
    gap_closing_iterations: int = 2

    # Output formats
    save_numpy: bool = True
    save_geotiff: bool = True if HAS_RASTERIO else False
    save_statistics: bool = True

    # Quality control
    expected_flood_fraction_range: Tuple[float, float] = (0.001, 0.4)
    max_depth_threshold_m: float = 50.0  # Flag unrealistic depths

    # Coordinate system (for GeoTIFF output)
    crs: Optional[str] = "EPSG:3857"  # Web Mercator default
    pixel_size_m: float = 10.0  # DEM resolution


class ResultProcessor:
    """Processes LISFLOOD-FP simulation outputs for ML training."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize result processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()

        if not HAS_SCIPY and (self.config.remove_small_areas or self.config.close_gaps):
            logger.warning("scipy not available - disabling morphological operations")
            self.config.remove_small_areas = False
            self.config.close_gaps = False

        logger.info("ResultProcessor initialized")

    def process_simulation_output(
        self,
        depth_file: str,
        output_dir: str,
        simulation_id: str,
        dem_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict:
        """Process a single simulation output file.

        Args:
            depth_file: Path to LISFLOOD-FP depth output (.max file)
            output_dir: Directory for processed outputs
            simulation_id: Unique simulation identifier
            dem_bounds: Optional DEM bounds (minx, miny, maxx, maxy) for georeferencing

        Returns:
            Dictionary with processing results and output paths
        """
        logger.info(f"Processing simulation output: {simulation_id}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        try:
            # Load depth data
            depth_data = self._load_depth_data(depth_file)

            # Create flood extent
            flood_extent = self._create_flood_extent(depth_data)

            # Calculate statistics
            statistics = self._calculate_statistics(depth_data, flood_extent)

            # Quality control checks
            qc_results = self._quality_control_checks(
                depth_data, flood_extent, statistics
            )

            # Save outputs
            output_files = self._save_outputs(
                depth_data=depth_data,
                flood_extent=flood_extent,
                statistics=statistics,
                output_path=output_path,
                simulation_id=simulation_id,
                dem_bounds=dem_bounds,
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            result = {
                "simulation_id": simulation_id,
                "status": "success",
                "processing_time_seconds": processing_time,
                "input_file": depth_file,
                "output_files": output_files,
                "statistics": statistics,
                "quality_control": qc_results,
                "config": self.config.__dict__,
                "processed_at": end_time.isoformat(),
            }

            logger.info(
                f"Successfully processed {simulation_id}: "
                f"{statistics['flooded_pixels']}/{statistics['total_pixels']} pixels flooded "
                f"({statistics['flood_fraction']:.3%})"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process {simulation_id}: {e}")

            return {
                "simulation_id": simulation_id,
                "status": "failed",
                "error": str(e),
                "input_file": depth_file,
                "processed_at": datetime.now().isoformat(),
            }

    def process_batch_outputs(
        self,
        simulation_results: List[Dict],
        output_dir: str,
        dem_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict:
        """Process multiple simulation outputs in batch.

        Args:
            simulation_results: List of simulation result dictionaries
            output_dir: Base output directory
            dem_bounds: Optional DEM bounds for georeferencing

        Returns:
            Batch processing summary
        """
        logger.info(f"Processing batch of {len(simulation_results)} simulation outputs")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_results = []
        failed_count = 0

        for sim_result in simulation_results:
            if sim_result.get("status") != "success":
                logger.warning(
                    f"Skipping failed simulation: {sim_result.get('simulation_id', 'unknown')}"
                )
                continue

            # Extract simulation info
            simulation_id = sim_result["simulation_id"]
            depth_file = sim_result.get("outputs", {}).get("depth_file")

            if not depth_file or not Path(depth_file).exists():
                logger.warning(
                    f"Depth file not found for {simulation_id}: {depth_file}"
                )
                failed_count += 1
                continue

            # Process individual simulation
            result = self.process_simulation_output(
                depth_file=depth_file,
                output_dir=str(output_path),
                simulation_id=simulation_id,
                dem_bounds=dem_bounds,
            )

            # Add original simulation metadata
            result["original_simulation"] = sim_result
            processed_results.append(result)

            if result["status"] != "success":
                failed_count += 1

        # Generate batch summary
        successful_results = [r for r in processed_results if r["status"] == "success"]
        batch_stats = self._aggregate_batch_statistics(successful_results)

        batch_summary = {
            "total_simulations": len(simulation_results),
            "processed_simulations": len(processed_results),
            "successful_processing": len(successful_results),
            "failed_processing": failed_count,
            "batch_statistics": batch_stats,
            "output_directory": str(output_path),
            "processed_at": datetime.now().isoformat(),
        }

        # Save batch results
        self._save_batch_results(processed_results, batch_summary, output_path)

        logger.info(
            f"Batch processing completed: {len(successful_results)}/{len(processed_results)} successful"
        )

        return batch_summary

    def _load_depth_data(self, file_path: str) -> np.ndarray:
        """Load LISFLOOD-FP depth output file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Depth file not found: {file_path}")

        try:
            # LISFLOOD-FP .max files are binary float32
            data = np.fromfile(file_path, dtype=np.float32)

            if data.size == 0:
                raise ValueError("Empty depth file")

            # Try to infer grid dimensions
            grid_shape = self._infer_grid_shape(data.size)
            depth_array = data.reshape(grid_shape)

            # Handle invalid values
            depth_array = np.where(
                depth_array < 0, 0, depth_array
            )  # Remove negative depths
            depth_array = np.where(
                np.isnan(depth_array), 0, depth_array
            )  # Remove NaN values

            logger.debug(
                f"Loaded depth data: shape={depth_array.shape}, "
                f"min={np.min(depth_array):.3f}, max={np.max(depth_array):.3f}"
            )

            return depth_array

        except Exception as e:
            logger.error(f"Failed to load depth data from {file_path}: {e}")
            raise ValueError(f"Unable to parse LISFLOOD-FP output file: {file_path}")

    def _infer_grid_shape(self, total_size: int) -> Tuple[int, int]:
        """Infer 2D grid shape from total array size."""
        # Try to find factors close to square
        factors = []
        for i in range(1, int(np.sqrt(total_size)) + 1):
            if total_size % i == 0:
                factors.append((i, total_size // i))

        if not factors:
            raise ValueError(f"Cannot infer grid shape for size {total_size}")

        # Return the most square-like dimensions
        best_shape = min(factors, key=lambda x: abs(x[0] - x[1]))

        logger.debug(f"Inferred grid shape: {best_shape} from size {total_size}")
        return best_shape

    def _create_flood_extent(self, depth_data: np.ndarray) -> np.ndarray:
        """Create binary flood extent from depth data."""
        # Apply depth threshold
        flood_extent = depth_data >= self.config.flood_depth_threshold_m

        if not HAS_SCIPY:
            return flood_extent.astype(np.uint8)

        # Morphological processing
        if self.config.remove_small_areas:
            flood_extent = self._remove_small_areas(flood_extent)

        if self.config.close_gaps:
            flood_extent = self._close_gaps(flood_extent)

        return flood_extent.astype(np.uint8)

    def _remove_small_areas(self, binary_array: np.ndarray) -> np.ndarray:
        """Remove small isolated flood areas."""
        if not HAS_SCIPY:
            return binary_array

        labeled, num_features = label(binary_array)

        # Remove areas smaller than threshold
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < self.config.min_flood_area_pixels:
                binary_array[labeled == i] = False

        return binary_array

    def _close_gaps(self, binary_array: np.ndarray) -> np.ndarray:
        """Close small gaps in flood extent."""
        if not HAS_SCIPY:
            return binary_array

        # Use cross-shaped structuring element
        structure = ndimage.generate_binary_structure(2, 1)

        return binary_closing(
            binary_array,
            structure=structure,
            iterations=self.config.gap_closing_iterations,
        )

    def _calculate_statistics(
        self, depth_data: np.ndarray, flood_extent: np.ndarray
    ) -> Dict:
        """Calculate flood statistics."""
        total_pixels = depth_data.size
        flooded_pixels = np.sum(flood_extent)
        flood_fraction = flooded_pixels / total_pixels

        # Depth statistics
        max_depth = float(np.max(depth_data))
        mean_depth_all = float(np.mean(depth_data))

        if flooded_pixels > 0:
            mean_depth_flooded = float(np.mean(depth_data[flood_extent > 0]))
            depth_percentiles = np.percentile(
                depth_data[flood_extent > 0], [50, 75, 90, 95, 99]
            )
        else:
            mean_depth_flooded = 0.0
            depth_percentiles = np.zeros(5)

        statistics = {
            "total_pixels": int(total_pixels),
            "flooded_pixels": int(flooded_pixels),
            "flood_fraction": float(flood_fraction),
            "max_depth_m": max_depth,
            "mean_depth_all_m": mean_depth_all,
            "mean_depth_flooded_m": mean_depth_flooded,
            "depth_percentiles": {
                "p50": float(depth_percentiles[0]),
                "p75": float(depth_percentiles[1]),
                "p90": float(depth_percentiles[2]),
                "p95": float(depth_percentiles[3]),
                "p99": float(depth_percentiles[4]),
            },
            "depth_threshold_m": self.config.flood_depth_threshold_m,
            "grid_shape": list(depth_data.shape),
            "pixel_area_m2": self.config.pixel_size_m**2,
            "flooded_area_km2": float(
                flooded_pixels * (self.config.pixel_size_m**2) / 1e6
            ),
        }

        return statistics

    def _quality_control_checks(
        self, depth_data: np.ndarray, flood_extent: np.ndarray, statistics: Dict
    ) -> Dict:
        """Perform quality control checks on processed data."""
        qc = {"status": "passed", "warnings": [], "errors": []}

        flood_fraction = statistics["flood_fraction"]
        max_depth = statistics["max_depth_m"]

        # Check flood fraction is within expected range
        min_expected, max_expected = self.config.expected_flood_fraction_range

        if flood_fraction < min_expected:
            qc["warnings"].append(
                f"Low flood fraction: {flood_fraction:.4%} < {min_expected:.4%}"
            )
        elif flood_fraction > max_expected:
            qc["errors"].append(
                f"High flood fraction: {flood_fraction:.4%} > {max_expected:.4%}"
            )

        # Check for unrealistic depths
        if max_depth > self.config.max_depth_threshold_m:
            qc["errors"].append(
                f"Unrealistic max depth: {max_depth:.2f}m > {self.config.max_depth_threshold_m}m"
            )

        # Check for very shallow flooding
        if max_depth < self.config.flood_depth_threshold_m * 2:
            qc["warnings"].append(f"Very shallow flooding: max depth {max_depth:.3f}m")

        # Check data validity
        if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)):
            qc["errors"].append("Invalid values (NaN/Inf) found in depth data")

        if np.any(depth_data < 0):
            qc["warnings"].append("Negative depth values found")

        # Overall status
        if qc["errors"]:
            qc["status"] = "failed"
        elif qc["warnings"]:
            qc["status"] = "passed_with_warnings"

        return qc

    def _save_outputs(
        self,
        depth_data: np.ndarray,
        flood_extent: np.ndarray,
        statistics: Dict,
        output_path: Path,
        simulation_id: str,
        dem_bounds: Optional[Tuple[float, float, float, float]],
    ) -> Dict:
        """Save processed outputs to disk."""
        output_files = {}

        # Save NumPy arrays
        if self.config.save_numpy:
            extent_file = output_path / f"{simulation_id}_flood_extent.npy"
            depth_file = output_path / f"{simulation_id}_depth.npy"

            np.save(extent_file, flood_extent)
            np.save(depth_file, depth_data)

            output_files["extent_numpy"] = str(extent_file)
            output_files["depth_numpy"] = str(depth_file)

        # Save GeoTIFF files
        if self.config.save_geotiff and HAS_RASTERIO and dem_bounds:
            extent_tiff = output_path / f"{simulation_id}_flood_extent.tif"
            depth_tiff = output_path / f"{simulation_id}_depth.tif"

            self._save_geotiff(flood_extent.astype(np.uint8), extent_tiff, dem_bounds)
            self._save_geotiff(depth_data.astype(np.float32), depth_tiff, dem_bounds)

            output_files["extent_geotiff"] = str(extent_tiff)
            output_files["depth_geotiff"] = str(depth_tiff)

        # Save statistics
        if self.config.save_statistics:
            stats_file = output_path / f"{simulation_id}_statistics.json"
            with open(stats_file, "w") as f:
                json.dump(statistics, f, indent=2)
            output_files["statistics"] = str(stats_file)

        return output_files

    def _save_geotiff(
        self,
        data: np.ndarray,
        output_file: Path,
        bounds: Tuple[float, float, float, float],
    ):
        """Save array as GeoTIFF with spatial reference."""
        if not HAS_RASTERIO:
            return

        minx, miny, maxx, maxy = bounds
        height, width = data.shape

        # Calculate transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Determine data type and nodata value
        if data.dtype == np.uint8:
            dtype = rasterio.uint8
            nodata = 255
        else:
            dtype = rasterio.float32
            nodata = -9999.0

        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=self.config.crs,
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)

    def _aggregate_batch_statistics(self, successful_results: List[Dict]) -> Dict:
        """Aggregate statistics from batch processing."""
        if not successful_results:
            return {}

        # Extract statistics
        all_stats = [r["statistics"] for r in successful_results]

        # Aggregate key metrics
        flood_fractions = [s["flood_fraction"] for s in all_stats]
        max_depths = [s["max_depth_m"] for s in all_stats]
        flooded_areas = [s["flooded_area_km2"] for s in all_stats]

        aggregated = {
            "simulation_count": len(all_stats),
            "flood_fraction": {
                "mean": float(np.mean(flood_fractions)),
                "std": float(np.std(flood_fractions)),
                "min": float(np.min(flood_fractions)),
                "max": float(np.max(flood_fractions)),
                "median": float(np.median(flood_fractions)),
            },
            "max_depth_m": {
                "mean": float(np.mean(max_depths)),
                "std": float(np.std(max_depths)),
                "min": float(np.min(max_depths)),
                "max": float(np.max(max_depths)),
                "median": float(np.median(max_depths)),
            },
            "flooded_area_km2": {
                "mean": float(np.mean(flooded_areas)),
                "std": float(np.std(flooded_areas)),
                "min": float(np.min(flooded_areas)),
                "max": float(np.max(flooded_areas)),
                "median": float(np.median(flooded_areas)),
                "total": float(np.sum(flooded_areas)),
            },
        }

        return aggregated

    def _save_batch_results(
        self, results: List[Dict], summary: Dict, output_path: Path
    ):
        """Save batch processing results."""
        # Save detailed results
        results_file = output_path / "processing_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Create training data manifest
        successful_results = [r for r in results if r["status"] == "success"]
        training_manifest = self._create_training_manifest(successful_results)

        manifest_file = output_path / "training_data_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(training_manifest, f, indent=2)

        logger.info(f"Saved batch results to {output_path}")

    def _create_training_manifest(self, successful_results: List[Dict]) -> Dict:
        """Create manifest for ML training data."""
        manifest = {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(successful_results),
            "data_format": "numpy" if self.config.save_numpy else "geotiff",
            "flood_threshold_m": self.config.flood_depth_threshold_m,
            "samples": [],
        }

        for result in successful_results:
            sample = {
                "simulation_id": result["simulation_id"],
                "flood_extent_file": result["output_files"].get("extent_numpy")
                or result["output_files"].get("extent_geotiff"),
                "depth_file": result["output_files"].get("depth_numpy")
                or result["output_files"].get("depth_geotiff"),
                "statistics": result["statistics"],
                "return_period": result.get("original_simulation", {})
                .get("scenario", {})
                .get("return_period", {}),
                "hyetograph": result.get("original_simulation", {})
                .get("scenario", {})
                .get("hyetograph", {}),
            }
            manifest["samples"].append(sample)

        return manifest
