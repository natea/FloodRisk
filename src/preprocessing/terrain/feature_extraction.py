"""
Terrain feature extraction for flood risk modeling.

This module extracts various terrain features from DEMs including slope,
curvature, flow accumulation, and Height Above Nearest Drainage (HAND).
"""

import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, feature
import richdem as rd
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class TerrainFeatureExtractor:
    """Extract terrain features from DEM data."""

    def __init__(
        self,
        dem_array: np.ndarray,
        pixel_size: float,
        nodata_value: Optional[float] = None,
    ):
        """
        Initialize terrain feature extractor.

        Args:
            dem_array: DEM elevation data
            pixel_size: Pixel resolution in meters
            nodata_value: NoData value in DEM
        """
        self.dem = dem_array
        self.pixel_size = pixel_size
        self.nodata = nodata_value
        self.mask = self._create_valid_mask()

    def _create_valid_mask(self) -> np.ndarray:
        """Create mask for valid (non-nodata) pixels."""
        if self.nodata is not None:
            return ~np.isclose(self.dem, self.nodata)
        return ~np.isnan(self.dem)

    def calculate_slope(
        self, method: str = "horn", units: str = "degrees"
    ) -> np.ndarray:
        """
        Calculate slope from DEM.

        Args:
            method: Slope calculation method ('horn', 'zevenbergen_thorne')
            units: Output units ('degrees', 'radians', 'percent')

        Returns:
            Slope array
        """
        logger.info(f"Calculating slope using {method} method")

        rd_dem = rd.rdarray(self.dem, no_data=self.nodata)

        if method == "horn":
            slope = rd.TerrainAttribute(rd_dem, attrib="slope_riserun")
        elif method == "zevenbergen_thorne":
            slope = rd.TerrainAttribute(rd_dem, attrib="slope_riserun", zscale=1.0)
        else:
            raise ValueError(f"Unknown slope method: {method}")

        slope_array = np.array(slope)

        # Convert units
        if units == "degrees":
            slope_array = np.arctan(slope_array) * 180 / np.pi
        elif units == "radians":
            slope_array = np.arctan(slope_array)
        elif units == "percent":
            slope_array = slope_array * 100

        # Apply mask
        slope_array = np.where(self.mask, slope_array, np.nan)

        logger.info("Slope calculation completed")
        return slope_array

    def calculate_aspect(self, method: str = "horn") -> np.ndarray:
        """
        Calculate aspect from DEM.

        Args:
            method: Aspect calculation method

        Returns:
            Aspect array in degrees (0-360)
        """
        logger.info(f"Calculating aspect using {method} method")

        rd_dem = rd.rdarray(self.dem, no_data=self.nodata)
        aspect = rd.TerrainAttribute(rd_dem, attrib="aspect")
        aspect_array = np.array(aspect)

        # Convert from radians to degrees and adjust range
        aspect_array = aspect_array * 180 / np.pi
        aspect_array = np.where(aspect_array < 0, aspect_array + 360, aspect_array)

        # Apply mask
        aspect_array = np.where(self.mask, aspect_array, np.nan)

        logger.info("Aspect calculation completed")
        return aspect_array

    def calculate_curvature(self, curvature_type: str = "total") -> np.ndarray:
        """
        Calculate terrain curvature.

        Args:
            curvature_type: Type of curvature ('total', 'plan', 'profile')

        Returns:
            Curvature array
        """
        logger.info(f"Calculating {curvature_type} curvature")

        rd_dem = rd.rdarray(self.dem, no_data=self.nodata)

        if curvature_type == "total":
            curvature = rd.TerrainAttribute(rd_dem, attrib="curvature")
        elif curvature_type == "plan":
            curvature = rd.TerrainAttribute(rd_dem, attrib="planform_curvature")
        elif curvature_type == "profile":
            curvature = rd.TerrainAttribute(rd_dem, attrib="profile_curvature")
        else:
            raise ValueError(f"Unknown curvature type: {curvature_type}")

        curvature_array = np.array(curvature)
        curvature_array = np.where(self.mask, curvature_array, np.nan)

        logger.info(f"{curvature_type.title()} curvature calculation completed")
        return curvature_array

    def calculate_roughness(self, window_size: int = 3) -> np.ndarray:
        """
        Calculate terrain roughness using standard deviation.

        Args:
            window_size: Size of moving window for roughness calculation

        Returns:
            Roughness array
        """
        logger.info(f"Calculating terrain roughness with window size {window_size}")

        # Use generic filter to calculate local standard deviation
        def local_std(values):
            return (
                np.std(values[~np.isnan(values)])
                if np.any(~np.isnan(values))
                else np.nan
            )

        roughness = ndimage.generic_filter(
            self.dem, local_std, size=window_size, mode="constant", cval=np.nan
        )

        roughness = np.where(self.mask, roughness, np.nan)

        logger.info("Roughness calculation completed")
        return roughness

    def calculate_tpi(self, window_size: int = 3) -> np.ndarray:
        """
        Calculate Topographic Position Index (TPI).

        Args:
            window_size: Size of neighborhood window

        Returns:
            TPI array
        """
        logger.info(f"Calculating TPI with window size {window_size}")

        # Calculate local mean elevation
        kernel = np.ones((window_size, window_size))
        kernel[window_size // 2, window_size // 2] = 0  # Exclude center pixel
        kernel = kernel / (window_size * window_size - 1)

        local_mean = ndimage.convolve(self.dem, kernel, mode="constant", cval=np.nan)

        # TPI is difference between elevation and local mean
        tpi = self.dem - local_mean
        tpi = np.where(self.mask, tpi, np.nan)

        logger.info("TPI calculation completed")
        return tpi

    def calculate_tri(self, window_size: int = 3) -> np.ndarray:
        """
        Calculate Terrain Ruggedness Index (TRI).

        Args:
            window_size: Size of neighborhood window

        Returns:
            TRI array
        """
        logger.info(f"Calculating TRI with window size {window_size}")

        # Calculate mean of absolute differences
        def tri_function(values):
            center_val = values[len(values) // 2]
            if np.isnan(center_val):
                return np.nan
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                return np.nan
            return np.mean(np.abs(valid_values - center_val))

        tri = ndimage.generic_filter(
            self.dem, tri_function, size=window_size, mode="constant", cval=np.nan
        )

        tri = np.where(self.mask, tri, np.nan)

        logger.info("TRI calculation completed")
        return tri

    def calculate_flow_accumulation(
        self, flow_direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate flow accumulation.

        Args:
            flow_direction: Pre-calculated flow direction array

        Returns:
            Flow accumulation array
        """
        logger.info("Calculating flow accumulation")

        rd_dem = rd.rdarray(self.dem, no_data=self.nodata)

        if flow_direction is None:
            # Calculate flow direction first
            flow_dir = rd.FlowDir(rd_dem, method="D8")
        else:
            flow_dir = rd.rdarray(flow_direction, no_data=self.nodata)

        flow_acc = rd.FlowAccumulation(flow_dir, method="D8")
        flow_acc_array = np.array(flow_acc)
        flow_acc_array = np.where(self.mask, flow_acc_array, np.nan)

        logger.info("Flow accumulation calculation completed")
        return flow_acc_array

    def calculate_wetness_index(
        self, flow_accumulation: np.ndarray, slope: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Topographic Wetness Index (TWI).

        Args:
            flow_accumulation: Flow accumulation array
            slope: Slope array in radians

        Returns:
            TWI array
        """
        logger.info("Calculating Topographic Wetness Index")

        # Convert flow accumulation to specific catchment area
        catchment_area = (flow_accumulation + 1) * self.pixel_size

        # Avoid division by zero in slope
        slope_safe = np.maximum(slope, 0.001)  # Minimum slope of 0.001 radians

        # Calculate TWI
        twi = np.log(catchment_area / np.tan(slope_safe))
        twi = np.where(self.mask, twi, np.nan)

        logger.info("TWI calculation completed")
        return twi

    def extract_all_features(
        self, flow_direction: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract all terrain features.

        Args:
            flow_direction: Pre-calculated flow direction array

        Returns:
            Dictionary containing all terrain features
        """
        logger.info("Extracting all terrain features")

        features = {}

        # Basic terrain attributes
        features["slope_degrees"] = self.calculate_slope(units="degrees")
        features["slope_radians"] = self.calculate_slope(units="radians")
        features["aspect"] = self.calculate_aspect()

        # Curvatures
        features["curvature_total"] = self.calculate_curvature("total")
        features["curvature_plan"] = self.calculate_curvature("plan")
        features["curvature_profile"] = self.calculate_curvature("profile")

        # Roughness measures
        features["roughness"] = self.calculate_roughness()
        features["tpi"] = self.calculate_tpi()
        features["tri"] = self.calculate_tri()

        # Hydrological features
        features["flow_accumulation"] = self.calculate_flow_accumulation(flow_direction)
        features["wetness_index"] = self.calculate_wetness_index(
            features["flow_accumulation"], features["slope_radians"]
        )

        logger.info("All terrain features extracted successfully")
        return features


class HANDCalculator:
    """Calculate Height Above Nearest Drainage (HAND)."""

    def __init__(
        self,
        dem: np.ndarray,
        flow_accumulation: np.ndarray,
        pixel_size: float,
        stream_threshold: float = 1000,
    ):
        """
        Initialize HAND calculator.

        Args:
            dem: DEM elevation data
            flow_accumulation: Flow accumulation array
            pixel_size: Pixel resolution in meters
            stream_threshold: Minimum flow accumulation to define streams
        """
        self.dem = dem
        self.flow_acc = flow_accumulation
        self.pixel_size = pixel_size
        self.stream_threshold = stream_threshold
        self.streams = self._identify_streams()

    def _identify_streams(self) -> np.ndarray:
        """Identify stream pixels from flow accumulation."""
        streams = self.flow_acc >= self.stream_threshold
        return streams

    def calculate_hand(self, max_distance: float = 5000) -> np.ndarray:
        """
        Calculate Height Above Nearest Drainage.

        Args:
            max_distance: Maximum search distance in meters

        Returns:
            HAND array
        """
        logger.info("Calculating Height Above Nearest Drainage (HAND)")

        max_pixels = int(max_distance / self.pixel_size)

        # Calculate distance to nearest stream
        stream_distance = ndimage.distance_transform_edt(~self.streams)

        # Initialize HAND array
        hand = np.full_like(self.dem, np.nan)

        # For each pixel, find nearest stream and calculate height difference
        rows, cols = np.where(stream_distance <= max_pixels)

        for row, col in zip(rows, cols):
            if np.isnan(self.dem[row, col]):
                continue

            # Find nearest stream pixel
            stream_row, stream_col = self._find_nearest_stream(row, col, max_pixels)

            if stream_row is not None and stream_col is not None:
                stream_elevation = self.dem[stream_row, stream_col]
                if not np.isnan(stream_elevation):
                    hand[row, col] = self.dem[row, col] - stream_elevation

        # Ensure HAND is non-negative
        hand = np.maximum(hand, 0)

        logger.info("HAND calculation completed")
        return hand

    def _find_nearest_stream(
        self, row: int, col: int, max_pixels: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find nearest stream pixel to given location."""
        min_distance = float("inf")
        nearest_row, nearest_col = None, None

        # Search in expanding square around pixel
        for distance in range(1, max_pixels + 1):
            found_stream = False

            # Check pixels at current distance
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    if abs(dr) != distance and abs(dc) != distance:
                        continue  # Only check perimeter of square

                    r, c = row + dr, col + dc

                    # Check bounds
                    if (
                        0 <= r < self.streams.shape[0]
                        and 0 <= c < self.streams.shape[1]
                        and self.streams[r, c]
                    ):

                        euclidean_dist = np.sqrt(dr * dr + dc * dc)
                        if euclidean_dist < min_distance:
                            min_distance = euclidean_dist
                            nearest_row, nearest_col = r, c
                            found_stream = True

            if found_stream:
                break

        return nearest_row, nearest_col


def calculate_terrain_derivatives(
    dem: np.ndarray, pixel_size: float
) -> Dict[str, np.ndarray]:
    """
    Calculate first and second derivatives of terrain.

    Args:
        dem: DEM elevation data
        pixel_size: Pixel resolution in meters

    Returns:
        Dictionary of derivative arrays
    """
    logger.info("Calculating terrain derivatives")

    derivatives = {}

    # First derivatives (gradients)
    grad_y, grad_x = np.gradient(dem, pixel_size)
    derivatives["gradient_x"] = grad_x
    derivatives["gradient_y"] = grad_y
    derivatives["gradient_magnitude"] = np.sqrt(grad_x**2 + grad_y**2)

    # Second derivatives (curvatures)
    grad_xx, grad_xy = np.gradient(grad_x, pixel_size)
    grad_yx, grad_yy = np.gradient(grad_y, pixel_size)

    derivatives["curvature_xx"] = grad_xx
    derivatives["curvature_yy"] = grad_yy
    derivatives["curvature_xy"] = grad_xy

    # Mean and Gaussian curvatures
    p = grad_x
    q = grad_y
    r = grad_xx
    s = grad_xy
    t = grad_yy

    # Mean curvature
    mean_curv = (r * (1 + q**2) - 2 * p * q * s + t * (1 + p**2)) / (
        2 * (1 + p**2 + q**2) ** (3 / 2)
    )
    derivatives["mean_curvature"] = mean_curv

    # Gaussian curvature
    gauss_curv = (r * t - s**2) / (1 + p**2 + q**2) ** 2
    derivatives["gaussian_curvature"] = gauss_curv

    logger.info("Terrain derivatives calculation completed")
    return derivatives
