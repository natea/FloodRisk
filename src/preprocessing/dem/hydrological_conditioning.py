"""
Hydrological conditioning of Digital Elevation Models (DEMs) for flood risk modeling.

This module provides functions for DEM preprocessing including sink removal,
flow direction calculation, and hydrological conditioning to prepare elevation
data for accurate hydrological modeling.
"""

import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage import morphology
import richdem as rd
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HydrologicalConditioner:
    """Handles DEM preprocessing for hydrological modeling."""

    def __init__(self, dem_path: str, output_crs: str = "EPSG:3857"):
        """
        Initialize the hydrological conditioner.

        Args:
            dem_path: Path to input DEM file
            output_crs: Target coordinate reference system
        """
        self.dem_path = dem_path
        self.output_crs = output_crs
        self.dem_array = None
        self.dem_profile = None
        self._load_dem()

    def _load_dem(self) -> None:
        """Load DEM from file."""
        try:
            with rasterio.open(self.dem_path) as src:
                self.dem_array = src.read(1)
                self.dem_profile = src.profile
                logger.info(f"Loaded DEM: {self.dem_array.shape} pixels")
        except Exception as e:
            logger.error(f"Failed to load DEM: {e}")
            raise

    def fill_sinks(
        self, method: str = "planchon_darboux", epsilon: float = 0.01
    ) -> np.ndarray:
        """
        Remove sinks (depressions) from DEM using various algorithms.

        Args:
            method: Sink filling method ('planchon_darboux', 'wang_liu', 'wei')
            epsilon: Small increment for fill operations

        Returns:
            Sink-filled DEM array
        """
        logger.info(f"Filling sinks using {method} method")

        # Convert to RichDEM format
        rd_dem = rd.rdarray(self.dem_array, no_data=self.dem_profile.get("nodata"))

        if method == "planchon_darboux":
            filled_dem = rd.FillDepressions(rd_dem, epsilon=epsilon, in_place=False)
        elif method == "wang_liu":
            filled_dem = rd.FillDepressions(
                rd_dem, method="Wei2018", epsilon=epsilon, in_place=False
            )
        elif method == "wei":
            filled_dem = rd.FillDepressions(rd_dem, method="Wei2018", in_place=False)
        else:
            raise ValueError(f"Unknown sink filling method: {method}")

        logger.info("Sink filling completed")
        return np.array(filled_dem)

    def calculate_flow_direction(
        self, filled_dem: np.ndarray, method: str = "d8"
    ) -> np.ndarray:
        """
        Calculate flow direction from filled DEM.

        Args:
            filled_dem: Sink-filled DEM array
            method: Flow direction algorithm ('d8', 'dinf', 'mfd')

        Returns:
            Flow direction array
        """
        logger.info(f"Calculating flow direction using {method} method")

        rd_dem = rd.rdarray(filled_dem, no_data=self.dem_profile.get("nodata"))

        if method == "d8":
            flow_dir = rd.FlowDir(rd_dem, method="D8")
        elif method == "dinf":
            flow_dir = rd.FlowDir(rd_dem, method="Dinf")
        elif method == "mfd":
            flow_dir = rd.FlowDir(rd_dem, method="MFD")
        else:
            raise ValueError(f"Unknown flow direction method: {method}")

        logger.info("Flow direction calculation completed")
        return np.array(flow_dir)

    def calculate_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """
        Calculate flow accumulation from flow direction.

        Args:
            flow_dir: Flow direction array

        Returns:
            Flow accumulation array
        """
        logger.info("Calculating flow accumulation")

        rd_flow_dir = rd.rdarray(flow_dir, no_data=self.dem_profile.get("nodata"))
        flow_acc = rd.FlowAccumulation(rd_flow_dir, method="D8")

        logger.info("Flow accumulation calculation completed")
        return np.array(flow_acc)

    def condition_dem(
        self, stream_threshold: float = 1000, buffer_distance: float = 100
    ) -> Dict[str, np.ndarray]:
        """
        Complete hydrological conditioning workflow.

        Args:
            stream_threshold: Minimum flow accumulation to define streams
            buffer_distance: Buffer distance around streams for conditioning

        Returns:
            Dictionary with conditioned DEM and derived products
        """
        logger.info("Starting hydrological conditioning workflow")

        # Step 1: Fill sinks
        filled_dem = self.fill_sinks()

        # Step 2: Calculate flow direction
        flow_dir = self.calculate_flow_direction(filled_dem)

        # Step 3: Calculate flow accumulation
        flow_acc = self.calculate_flow_accumulation(flow_dir)

        # Step 4: Extract stream network
        streams = self._extract_streams(flow_acc, stream_threshold)

        # Step 5: Condition DEM along streams
        conditioned_dem = self._condition_along_streams(
            filled_dem, streams, buffer_distance
        )

        results = {
            "conditioned_dem": conditioned_dem,
            "filled_dem": filled_dem,
            "flow_direction": flow_dir,
            "flow_accumulation": flow_acc,
            "streams": streams,
        }

        logger.info("Hydrological conditioning completed")
        return results

    def _extract_streams(self, flow_acc: np.ndarray, threshold: float) -> np.ndarray:
        """Extract stream network from flow accumulation."""
        streams = flow_acc >= threshold

        # Clean up stream network
        streams = morphology.remove_small_objects(streams, min_size=100)
        streams = morphology.skeletonize(streams)

        return streams.astype(np.uint8)

    def _condition_along_streams(
        self, dem: np.ndarray, streams: np.ndarray, buffer_distance: float
    ) -> np.ndarray:
        """
        Condition DEM along stream channels to ensure proper drainage.

        Args:
            dem: Input DEM
            streams: Binary stream network
            buffer_distance: Buffer distance in meters

        Returns:
            Conditioned DEM
        """
        # Create buffer around streams
        pixel_size = abs(self.dem_profile["transform"][0])
        buffer_pixels = int(buffer_distance / pixel_size)

        stream_buffer = ndimage.binary_dilation(streams, iterations=buffer_pixels)

        # Apply breach conditioning
        conditioned_dem = dem.copy()

        # Smooth elevation along stream corridors
        for i in range(buffer_pixels):
            mask = ndimage.binary_dilation(streams, iterations=i + 1)
            smoothed = ndimage.gaussian_filter(dem, sigma=2)
            conditioned_dem = np.where(mask, smoothed, conditioned_dem)

        return conditioned_dem

    def save_results(self, results: Dict[str, np.ndarray], output_dir: str) -> None:
        """
        Save conditioning results to files.

        Args:
            results: Dictionary of arrays to save
            output_dir: Output directory path
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for name, array in results.items():
            output_path = os.path.join(output_dir, f"{name}.tif")

            with rasterio.open(
                output_path, "w", **self.dem_profile, dtype=array.dtype
            ) as dst:
                dst.write(array, 1)

            logger.info(f"Saved {name} to {output_path}")


def validate_dem_quality(dem: np.ndarray, flow_acc: np.ndarray) -> Dict[str, float]:
    """
    Validate DEM quality metrics.

    Args:
        dem: DEM array
        flow_acc: Flow accumulation array

    Returns:
        Quality metrics dictionary
    """
    metrics = {}

    # Check for remaining sinks
    sinks = _identify_sinks(dem)
    metrics["sink_count"] = np.sum(sinks)
    metrics["sink_percentage"] = (np.sum(sinks) / sinks.size) * 100

    # Stream network connectivity
    stream_network = flow_acc > np.percentile(flow_acc[flow_acc > 0], 95)
    metrics["stream_connectivity"] = _calculate_connectivity(stream_network)

    # Elevation statistics
    metrics["elevation_range"] = np.ptp(dem[~np.isnan(dem)])
    metrics["elevation_mean"] = np.nanmean(dem)
    metrics["elevation_std"] = np.nanstd(dem)

    return metrics


def _identify_sinks(dem: np.ndarray) -> np.ndarray:
    """Identify remaining sinks in DEM."""
    # Use morphological operations to find depressions
    h_minima = morphology.h_minima(dem, h=0.1)
    return h_minima


def _calculate_connectivity(network: np.ndarray) -> float:
    """Calculate network connectivity metric."""
    labeled_network, num_components = ndimage.label(network)

    if num_components == 0:
        return 0.0

    # Calculate relative size of largest component
    component_sizes = np.bincount(labeled_network.flat)[1:]  # Exclude background
    largest_component = np.max(component_sizes)
    total_pixels = np.sum(network)

    connectivity = largest_component / total_pixels if total_pixels > 0 else 0.0
    return connectivity
