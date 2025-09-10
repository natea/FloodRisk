"""Enhanced preprocessing pipeline for real-world flood prediction data.

This module bridges data acquisition outputs with the ML pipeline, providing
comprehensive preprocessing for USGS 3DEP DEMs and NOAA Atlas 14 rainfall data.

Features:
- Integration with existing preprocessing modules
- Advanced flow routing algorithms using RichDEM
- Robust spatial alignment and CRS handling
- Data validation and quality assurance
- Configurable processing pipelines
- Memory-efficient processing for large datasets
- Caching and incremental processing
"""

import logging
import numpy as np
import rasterio
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Any
from rasterio import features, warp, transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import warnings
import json
import hashlib
import pickle
from datetime import datetime
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import richdem as rd

    HAS_RICHDEM = True
except ImportError:
    warnings.warn(
        "RichDEM not available. Advanced flow routing will use scipy methods."
    )
    rd = None
    HAS_RICHDEM = False

# Import existing modules
try:
    from ...preprocessing.dem_processor import DEMProcessor as BaseDEMProcessor
    from ...preprocessing.rainfall_generator import RainfallGenerator
    from ...preprocessing.terrain_features import TerrainFeatureExtractor
    from ...preprocessing.dem.hydrological_conditioning import HydrologicalConditioner
except ImportError:
    warnings.warn(
        "Some preprocessing modules not available. Using simplified implementations."
    )

logger = logging.getLogger(__name__)


class RealDataPreprocessor:
    """Enhanced preprocessor for real-world flood prediction data."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        target_crs: str = "EPSG:3857",
        target_resolution: float = 10.0,
        enable_caching: bool = True,
    ):
        """Initialize real data preprocessor.

        Args:
            config_path: Path to configuration file
            cache_dir: Directory for caching processed data
            target_crs: Target coordinate reference system
            target_resolution: Target resolution in meters
            enable_caching: Enable data caching
        """
        self.target_crs = CRS.from_string(target_crs)
        self.target_resolution = target_resolution
        self.enable_caching = enable_caching

        # Set up caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".floodrisk_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize processors
        self._init_processors()

        logger.info(
            f"Initialized RealDataPreprocessor with target CRS: {target_crs}, "
            f"resolution: {target_resolution}m"
        )

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load processing configuration."""
        default_config = {
            "dem_processing": {
                "fill_sinks": True,
                "sink_fill_method": "planchon_darboux",
                "flow_method": "d8",
                "min_drainage_area": 1000.0,
                "stream_threshold": 1000.0,
                "smoothing_sigma": 0.5,
            },
            "spatial_processing": {
                "resampling_method": "bilinear",
                "max_memory_gb": 4.0,
                "chunk_size": 2048,
                "overlap_pixels": 128,
            },
            "rainfall_processing": {
                "time_step_minutes": 15,
                "distribution_types": ["scs_type_ii", "uniform"],
                "return_periods": [10, 25, 50, 100, 500],
                "durations": [6, 12, 24],
            },
            "feature_extraction": {
                "compute_curvature": True,
                "compute_hand": True,
                "compute_twi": True,
                "hand_max_distance": 1000.0,
                "roughness_window": 3,
            },
            "validation": {
                "check_crs_alignment": True,
                "check_extent_overlap": True,
                "check_resolution_match": True,
                "tolerance_meters": 1.0,
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)

        return default_config

    def _init_processors(self):
        """Initialize processing components."""
        # Enhanced DEM processor
        self.dem_processor = EnhancedDEMProcessor(
            target_crs=str(self.target_crs),
            resolution=self.target_resolution,
            config=self.config["dem_processing"],
        )

        # Rainfall processor
        self.rainfall_processor = EnhancedRainfallProcessor(
            config=self.config["rainfall_processing"]
        )

        # Terrain feature extractor
        self.feature_extractor = TerrainFeatureExtractor(
            cell_size=self.target_resolution
        )

        # Spatial processor for alignment and validation
        self.spatial_processor = SpatialProcessor(
            target_crs=self.target_crs,
            target_resolution=self.target_resolution,
            config=self.config["spatial_processing"],
        )

        # Data validator
        self.validator = DataValidator(config=self.config["validation"])

    def process_region(
        self,
        dem_path: Union[str, Path],
        rainfall_data: Dict[str, Any],
        region_bounds: Optional[Tuple[float, float, float, float]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, xr.DataArray]:
        """Process a complete region with DEM and rainfall data.

        Args:
            dem_path: Path to DEM file (USGS 3DEP format)
            rainfall_data: NOAA Atlas 14 rainfall data dictionary
            region_bounds: Optional bounds to clip data (minx, miny, maxx, maxy)
            output_dir: Optional directory to save processed data

        Returns:
            Dictionary of processed data arrays
        """
        logger.info(f"Processing region: DEM={dem_path}")

        # Create cache key for this processing run
        cache_key = self._create_cache_key(dem_path, rainfall_data, region_bounds)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info("Loaded processed data from cache")
            return cached_result

        # Process DEM
        logger.info("Processing DEM data...")
        dem_data = self.dem_processor.process_usgs_3dep(dem_path, region_bounds)

        # Process rainfall
        logger.info("Processing rainfall data...")
        rainfall_scenarios = self.rainfall_processor.process_noaa_atlas14(
            rainfall_data, dem_data["dem"]
        )

        # Extract terrain features
        logger.info("Extracting terrain features...")
        terrain_features = self.feature_extractor.extract_all_features(
            dem_data["dem"].values,
            flow_accumulation=dem_data.get("flow_accumulation", {}).get("values"),
            streams=dem_data.get("streams", {}).get("values"),
        )

        # Convert terrain features to xarray
        terrain_data = {}
        for name, array in terrain_features.items():
            terrain_data[name] = dem_data["dem"].copy(data=array)

        # Validate spatial alignment
        logger.info("Validating spatial alignment...")
        validation_results = self.validator.validate_spatial_alignment(
            dem_data, rainfall_scenarios, terrain_data
        )

        if not all(validation_results.values()):
            logger.warning(f"Spatial validation issues: {validation_results}")

        # Combine results
        processed_data = {
            **dem_data,
            **terrain_data,
            "rainfall_scenarios": rainfall_scenarios,
            "validation": validation_results,
        }

        # Cache results
        if self.enable_caching:
            self._save_to_cache(cache_key, processed_data)

        # Save to output directory if specified
        if output_dir:
            self._save_processed_data(processed_data, output_dir)

        logger.info("Region processing completed")
        return processed_data

    def create_training_tiles(
        self,
        processed_data: Dict[str, xr.DataArray],
        flood_labels: Optional[xr.DataArray] = None,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> List[Dict]:
        """Create training tiles from processed data.

        Args:
            processed_data: Dictionary of processed data arrays
            flood_labels: Optional flood extent labels
            tile_size: Size of tiles in pixels
            overlap: Overlap between tiles in pixels

        Returns:
            List of training tiles
        """
        logger.info(
            f"Creating training tiles: {tile_size}x{tile_size} with {overlap}px overlap"
        )

        # Select input features for tiling
        feature_arrays = []
        feature_names = []

        # Core features
        core_features = ["dem", "slope_degrees", "flow_accumulation", "hand"]
        for name in core_features:
            if name in processed_data:
                feature_arrays.append(processed_data[name])
                feature_names.append(name)

        # Add rainfall scenario (use first scenario for now)
        if "rainfall_scenarios" in processed_data:
            scenarios = processed_data["rainfall_scenarios"]
            if scenarios:
                first_scenario = list(scenarios.values())[0]
                feature_arrays.append(first_scenario)
                feature_names.append("rainfall")

        # Create tile generator
        tile_generator = EnhancedTileGenerator(
            tile_size=tile_size, overlap=overlap, config=self.config
        )

        tiles = tile_generator.generate_tiles(
            feature_arrays, feature_names, flood_labels
        )

        logger.info(f"Created {len(tiles)} training tiles")
        return tiles

    def _create_cache_key(
        self,
        dem_path: Union[str, Path],
        rainfall_data: Dict[str, Any],
        region_bounds: Optional[Tuple],
    ) -> str:
        """Create cache key for processing parameters."""
        key_data = {
            "dem_path": str(dem_path),
            "dem_mtime": (
                Path(dem_path).stat().st_mtime if Path(dem_path).exists() else 0
            ),
            "rainfall_data": str(sorted(rainfall_data.items())),
            "region_bounds": region_bounds,
            "target_crs": str(self.target_crs),
            "target_resolution": self.target_resolution,
            "config_hash": hashlib.md5(
                str(sorted(self.config.items())).encode()
            ).hexdigest(),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load processed data from cache."""
        if not self.enable_caching:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save processed data to cache."""
        if not self.enable_caching:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _save_processed_data(self, data: Dict, output_dir: Union[str, Path]):
        """Save processed data to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, array in data.items():
            if isinstance(array, xr.DataArray):
                output_file = output_dir / f"{name}.tif"
                array.rio.to_raster(output_file)
            elif name == "rainfall_scenarios":
                # Save rainfall scenarios as separate files
                scenarios_dir = output_dir / "rainfall_scenarios"
                scenarios_dir.mkdir(exist_ok=True)
                for scenario_name, scenario_array in array.items():
                    scenario_file = scenarios_dir / f"{scenario_name}.tif"
                    scenario_array.rio.to_raster(scenario_file)


class EnhancedDEMProcessor(
    BaseDEMProcessor if "BaseDEMProcessor" in locals() else object
):
    """Enhanced DEM processor with USGS 3DEP support and advanced flow routing."""

    def __init__(
        self,
        target_crs: str = "EPSG:3857",
        resolution: float = 10.0,
        config: Optional[Dict] = None,
    ):
        super().__init__(target_crs, resolution)
        self.config = config or {}
        self.has_richdem = HAS_RICHDEM

    def process_usgs_3dep(
        self,
        dem_path: Union[str, Path],
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, xr.DataArray]:
        """Process USGS 3DEP DEM with enhanced flow routing.

        Args:
            dem_path: Path to USGS 3DEP DEM file
            bounds: Optional bounds to clip (minx, miny, maxx, maxy)

        Returns:
            Dictionary of processed DEM products
        """
        logger.info(f"Processing USGS 3DEP DEM: {dem_path}")

        # Load and preprocess DEM
        dem_da = self.load_and_reproject(dem_path)

        # Clip to bounds if specified
        if bounds:
            minx, miny, maxx, maxy = bounds
            dem_da = dem_da.sel(
                x=slice(minx, maxx), y=slice(maxy, miny)  # y is typically decreasing
            )

        # Apply hydrological conditioning
        if self.config.get("fill_sinks", True):
            filled_dem = self._advanced_sink_filling(dem_da)
        else:
            filled_dem = dem_da

        # Compute flow direction and accumulation
        flow_products = self._compute_advanced_flow(filled_dem)

        # Compute derived features
        derived_features = self.compute_derived_features(filled_dem)

        # Combine results
        results = {
            "dem": dem_da,
            "filled_dem": filled_dem,
            **flow_products,
            **derived_features,
        }

        return results

    def _advanced_sink_filling(self, dem: xr.DataArray) -> xr.DataArray:
        """Advanced sink filling using RichDEM if available."""
        if not self.has_richdem:
            logger.info("Using scipy-based sink filling")
            return self._scipy_sink_filling(dem)

        logger.info("Using RichDEM for advanced sink filling")

        # Convert to RichDEM format
        rd_dem = rd.rdarray(dem.values, no_data=dem.attrs.get("_FillValue", -9999))

        method = self.config.get("sink_fill_method", "planchon_darboux")
        if method == "planchon_darboux":
            filled = rd.FillDepressions(rd_dem, epsilon=0.01, in_place=False)
        else:
            filled = rd.FillDepressions(rd_dem, in_place=False)

        # Convert back to xarray
        filled_da = dem.copy(data=np.array(filled))
        return filled_da

    def _scipy_sink_filling(self, dem: xr.DataArray) -> xr.DataArray:
        """Fallback sink filling using scipy."""
        logger.info("Applying scipy-based depression filling")

        dem_array = dem.values
        filled = dem_array.copy()

        # Simple iterative filling
        kernel = np.ones((3, 3))
        for _ in range(100):  # Max iterations
            dilated = ndimage.grey_dilation(filled, footprint=kernel)
            new_filled = np.maximum(filled, np.minimum(dilated, dem_array + 0.01))

            if np.allclose(filled, new_filled, rtol=1e-6):
                break
            filled = new_filled

        return dem.copy(data=filled)

    def _compute_advanced_flow(self, dem: xr.DataArray) -> Dict[str, xr.DataArray]:
        """Compute flow direction and accumulation with advanced algorithms."""
        results = {}

        if self.has_richdem:
            logger.info("Computing flow using RichDEM")
            rd_dem = rd.rdarray(dem.values, no_data=dem.attrs.get("_FillValue", -9999))

            # Flow direction
            flow_method = self.config.get("flow_method", "d8")
            if flow_method.upper() == "D8":
                flow_dir = rd.FlowDir(rd_dem, method="D8")
            elif flow_method.upper() == "DINF":
                flow_dir = rd.FlowDir(rd_dem, method="Dinf")
            else:
                flow_dir = rd.FlowDir(rd_dem, method="D8")

            results["flow_direction"] = dem.copy(data=np.array(flow_dir))

            # Flow accumulation
            flow_acc = rd.FlowAccumulation(rd.rdarray(flow_dir), method="D8")
            results["flow_accumulation"] = dem.copy(data=np.array(flow_acc))

        else:
            logger.info("Computing flow using scipy methods")
            # Use parent class methods
            derived = self.compute_derived_features(dem)
            results["flow_accumulation"] = derived.get(
                "flow_accumulation", dem.copy(data=np.ones_like(dem.values))
            )

        # Extract stream network
        if "flow_accumulation" in results:
            threshold = self.config.get("stream_threshold", 1000.0)
            cell_area = self.resolution**2
            threshold_cells = threshold / cell_area

            streams = results["flow_accumulation"].values >= threshold_cells
            results["streams"] = dem.copy(data=streams.astype(np.uint8))

        return results


class EnhancedRainfallProcessor(
    RainfallGenerator if "RainfallGenerator" in locals() else object
):
    """Enhanced rainfall processor for NOAA Atlas 14 data."""

    def __init__(self, config: Optional[Dict] = None):
        if hasattr(super(), "__init__"):
            super().__init__(
                time_step_minutes=config.get("time_step_minutes", 15) if config else 15
            )
        self.config = config or {}

    def process_noaa_atlas14(
        self, rainfall_data: Dict[str, Any], template_raster: xr.DataArray
    ) -> Dict[str, xr.DataArray]:
        """Process NOAA Atlas 14 rainfall data into scenarios.

        Args:
            rainfall_data: NOAA Atlas 14 precipitation data
            template_raster: Template for spatial extent and resolution

        Returns:
            Dictionary of rainfall scenario rasters
        """
        logger.info("Processing NOAA Atlas 14 rainfall data")

        scenarios = {}

        # Extract return periods and durations from data
        return_periods = self.config.get("return_periods", [10, 25, 50, 100, 500])
        durations = self.config.get("durations", [6, 12, 24])
        distributions = self.config.get(
            "distribution_types", ["scs_type_ii", "uniform"]
        )

        for rp in return_periods:
            if str(rp) not in rainfall_data:
                logger.warning(f"Return period {rp} not found in rainfall data")
                continue

            for duration in durations:
                if str(duration) not in rainfall_data[str(rp)]:
                    logger.warning(
                        f"Duration {duration}h not found for {rp}-year return period"
                    )
                    continue

                total_rainfall_mm = rainfall_data[str(rp)][str(duration)]

                for dist_type in distributions:
                    scenario_name = f"rp{rp}_d{duration}h_{dist_type}"

                    # Create uniform rainfall raster
                    rainfall_raster = self._create_rainfall_raster(
                        total_rainfall_mm, template_raster, distribution_type=dist_type
                    )

                    scenarios[scenario_name] = rainfall_raster

        logger.info(f"Created {len(scenarios)} rainfall scenarios")
        return scenarios

    def _create_rainfall_raster(
        self,
        total_rainfall_mm: float,
        template_raster: xr.DataArray,
        distribution_type: str = "uniform",
    ) -> xr.DataArray:
        """Create rainfall raster with optional spatial variability."""

        # Create base uniform field
        rainfall = np.full_like(
            template_raster.values, total_rainfall_mm, dtype=np.float32
        )

        # Add mild spatial variability for realism
        if distribution_type != "uniform":
            variability = 0.1  # 10% spatial variability
            noise = np.random.normal(1.0, variability, rainfall.shape)
            noise = ndimage.gaussian_filter(noise, sigma=2)  # Smooth spatial gradients
            rainfall = rainfall * noise
            rainfall = np.maximum(rainfall, 0)  # Ensure non-negative

        rainfall_da = template_raster.copy(data=rainfall)
        rainfall_da.attrs.update(
            {
                "units": "mm",
                "description": f"{total_rainfall_mm}mm total rainfall - {distribution_type}",
                "distribution_type": distribution_type,
            }
        )

        return rainfall_da


class SpatialProcessor:
    """Handles spatial alignment, reprojection, and validation."""

    def __init__(self, target_crs: CRS, target_resolution: float, config: Dict):
        self.target_crs = target_crs
        self.target_resolution = target_resolution
        self.config = config

    def align_rasters(self, raster_list: List[xr.DataArray]) -> List[xr.DataArray]:
        """Align multiple rasters to common grid."""
        if not raster_list:
            return []

        logger.info(f"Aligning {len(raster_list)} rasters to common grid")

        # Use first raster as reference
        reference = raster_list[0]
        aligned = [reference]

        for i, raster in enumerate(raster_list[1:], 1):
            aligned_raster = self._align_to_reference(raster, reference)
            aligned.append(aligned_raster)

        return aligned

    def _align_to_reference(
        self, raster: xr.DataArray, reference: xr.DataArray
    ) -> xr.DataArray:
        """Align raster to reference grid."""
        # Check if already aligned
        if (
            raster.x.values.shape == reference.x.values.shape
            and raster.y.values.shape == reference.y.values.shape
            and np.allclose(
                raster.x.values, reference.x.values, atol=self.target_resolution / 10
            )
            and np.allclose(
                raster.y.values, reference.y.values, atol=self.target_resolution / 10
            )
        ):
            return raster

        # Interpolate to reference grid
        aligned = raster.interp(x=reference.x, y=reference.y, method="linear")

        return aligned


class DataValidator:
    """Validates processed data quality and spatial consistency."""

    def __init__(self, config: Dict):
        self.config = config
        self.tolerance = config.get("tolerance_meters", 1.0)

    def validate_spatial_alignment(
        self,
        dem_data: Dict[str, xr.DataArray],
        rainfall_data: Dict[str, xr.DataArray],
        terrain_data: Dict[str, xr.DataArray],
    ) -> Dict[str, bool]:
        """Validate spatial alignment of all data layers."""
        results = {}

        # Get reference DEM
        reference = dem_data.get("dem")
        if reference is None:
            return {"error": False, "message": "No reference DEM found"}

        # Check DEM products alignment
        for name, array in dem_data.items():
            if name != "dem" and isinstance(array, xr.DataArray):
                results[f"dem_{name}_aligned"] = self._check_alignment(reference, array)

        # Check rainfall alignment
        for name, array in rainfall_data.items():
            if isinstance(array, xr.DataArray):
                results[f"rainfall_{name}_aligned"] = self._check_alignment(
                    reference, array
                )

        # Check terrain features alignment
        for name, array in terrain_data.items():
            if isinstance(array, xr.DataArray):
                results[f"terrain_{name}_aligned"] = self._check_alignment(
                    reference, array
                )

        # Overall alignment check
        results["overall_alignment"] = all(results.values())

        return results

    def _check_alignment(
        self, reference: xr.DataArray, test_array: xr.DataArray
    ) -> bool:
        """Check if two arrays are spatially aligned."""
        try:
            # Check shapes
            if reference.shape != test_array.shape:
                return False

            # Check coordinate alignment
            x_aligned = np.allclose(
                reference.x.values, test_array.x.values, atol=self.tolerance
            )
            y_aligned = np.allclose(
                reference.y.values, test_array.y.values, atol=self.tolerance
            )

            return x_aligned and y_aligned

        except Exception as e:
            logger.warning(f"Alignment check failed: {e}")
            return False


class EnhancedTileGenerator:
    """Enhanced tile generator with memory-efficient processing."""

    def __init__(
        self, tile_size: int = 512, overlap: int = 64, config: Optional[Dict] = None
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.config = config or {}

    def generate_tiles(
        self,
        data_arrays: List[xr.DataArray],
        feature_names: List[str],
        flood_labels: Optional[xr.DataArray] = None,
    ) -> List[Dict]:
        """Generate training tiles from multiple data arrays."""
        logger.info(
            f"Generating tiles: {self.tile_size}x{self.tile_size} with {self.overlap}px overlap"
        )

        if not data_arrays:
            return []

        # Get dimensions from first array
        reference = data_arrays[0]
        height, width = reference.shape

        tiles = []
        tile_id = 0

        for i in range(0, height - self.tile_size + 1, self.stride):
            for j in range(0, width - self.tile_size + 1, self.stride):
                tile_data = {
                    "tile_id": tile_id,
                    "row": i,
                    "col": j,
                    "bounds": {
                        "row_start": i,
                        "row_end": i + self.tile_size,
                        "col_start": j,
                        "col_end": j + self.tile_size,
                    },
                }

                # Extract data for each feature
                for k, (array, name) in enumerate(zip(data_arrays, feature_names)):
                    tile_data[name] = array.isel(
                        y=slice(i, i + self.tile_size), x=slice(j, j + self.tile_size)
                    ).values

                # Extract labels if provided
                if flood_labels is not None:
                    tile_data["labels"] = flood_labels.isel(
                        y=slice(i, i + self.tile_size), x=slice(j, j + self.tile_size)
                    ).values

                tiles.append(tile_data)
                tile_id += 1

        logger.info(f"Generated {len(tiles)} tiles")
        return tiles


def create_nashville_config() -> Dict[str, Any]:
    """Create configuration for Nashville region processing."""
    return {
        "region_name": "Nashville, TN",
        "bounds": {"minx": -87.0, "miny": 35.8, "maxx": -86.4, "maxy": 36.4},
        "dem_processing": {
            "fill_sinks": True,
            "sink_fill_method": "planchon_darboux",
            "flow_method": "d8",
            "min_drainage_area": 500.0,  # Smaller for urban area
            "stream_threshold": 500.0,
            "smoothing_sigma": 0.5,
        },
        "spatial_processing": {
            "resampling_method": "bilinear",
            "max_memory_gb": 8.0,
            "chunk_size": 1024,
            "overlap_pixels": 64,
        },
        "rainfall_processing": {
            "time_step_minutes": 15,
            "distribution_types": ["scs_type_ii", "uniform"],
            "return_periods": [10, 25, 50, 100, 500],
            "durations": [6, 12, 24],
        },
        "feature_extraction": {
            "compute_curvature": True,
            "compute_hand": True,
            "compute_twi": True,
            "hand_max_distance": 2000.0,  # Larger for floodplain
            "roughness_window": 5,  # Larger window for urban features
        },
        "validation": {
            "check_crs_alignment": True,
            "check_extent_overlap": True,
            "check_resolution_match": True,
            "tolerance_meters": 0.5,
        },
    }


# Factory function for easy instantiation
def create_preprocessor(region: str = "default", **kwargs) -> RealDataPreprocessor:
    """Create preprocessor instance for specific region."""
    if region == "nashville":
        config = create_nashville_config()
        return RealDataPreprocessor(config_path=None, **kwargs)
    else:
        return RealDataPreprocessor(**kwargs)
