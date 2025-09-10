"""
Validation Utilities and Helper Functions

Common utilities for the flood risk validation framework including:
- Configuration management
- Data loading and preprocessing
- File format conversions
- Coordinate system transformations
- Statistical utilities
- Error handling and logging setup
"""

import numpy as np
import pandas as pd
import logging
import yaml
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
import tempfile
import shutil
from datetime import datetime, timedelta
from functools import wraps
import inspect

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import pyproj

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation framework errors"""

    pass


class ConfigurationManager:
    """
    Manages configuration files and settings for the validation framework
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Configuration loaded from: {config_path or 'defaults'}")

    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
        if self.config_path and Path(self.config_path).exists():
            return self._load_yaml_config(self.config_path)
        else:
            logger.info("Using default configuration")
            return self._get_default_config()

    def _load_yaml_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Merge with defaults to ensure all required keys exist
            default_config = self._get_default_config()
            merged_config = self._merge_configs(default_config, config)

            return merged_config

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise ValidationError(f"Configuration loading failed: {e}")

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user configuration with defaults"""
        merged = default.copy()

        for key, value in user.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "metrics": {
                "calculate_iou": True,
                "calculate_csi": True,
                "calculate_regression": True,
                "calculate_classification": True,
                "depth_threshold": 0.01,
                "binary_threshold": 0.5,
            },
            "data_processing": {
                "preprocess": {
                    "clip_negative_depths": True,
                    "apply_min_threshold": True,
                    "apply_max_threshold": True,
                    "max_depth_clip": 50.0,
                },
                "spatial": {"tolerance": 1e-6, "resampling_method": "bilinear"},
            },
            "visualization": {
                "enabled": True,
                "maps": {
                    "generate_flood_maps": True,
                    "generate_comparison_maps": True,
                    "interactive_maps": True,
                },
                "plots": {"generate_scatter_plots": True, "generate_histograms": True},
            },
            "reporting": {
                "enabled": True,
                "formats": {
                    "generate_html": True,
                    "generate_pdf": False,
                    "generate_json_summary": True,
                },
                "content": {
                    "include_executive_summary": True,
                    "include_detailed_results": True,
                    "include_visualizations": True,
                    "include_recommendations": True,
                },
            },
            "logging": {"level": "INFO", "console_logging": {"enabled": True}},
            "paths": {
                "base_output_dir": "./validation_output",
                "reports_dir": "./validation_output/reports",
                "plots_dir": "./validation_output/plots",
            },
        }

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration key (e.g., 'metrics.depth_threshold')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to file

        Args:
            output_path: Path to save configuration
        """
        try:
            with open(output_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise ValidationError(f"Configuration save failed: {e}")


class DataLoader:
    """
    Handles loading and preprocessing of various data formats
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize data loader

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigurationManager()
        logger.info("DataLoader initialized")

    def load_flood_data(
        self, file_path: Union[str, Path], file_format: Optional[str] = None
    ) -> Dict:
        """
        Load flood data from various formats

        Args:
            file_path: Path to data file
            file_format: Format override (auto-detect if None)

        Returns:
            Dictionary with data and metadata
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Detect format from extension if not specified
            if file_format is None:
                file_format = self._detect_file_format(file_path)

            logger.info(f"Loading flood data: {file_path} (format: {file_format})")

            # Load based on format
            if file_format == "tiff" or file_format == "geotiff":
                return self._load_geotiff(file_path)
            elif file_format == "asc":
                return self._load_asc_grid(file_path)
            elif file_format == "csv":
                return self._load_csv_data(file_path)
            elif file_format == "netcdf":
                return self._load_netcdf(file_path)
            elif file_format == "numpy":
                return self._load_numpy_array(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        except Exception as e:
            logger.error(f"Error loading flood data: {e}")
            raise ValidationError(f"Data loading failed: {e}")

    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension"""
        extension = file_path.suffix.lower()

        format_map = {
            ".tif": "tiff",
            ".tiff": "tiff",
            ".asc": "asc",
            ".csv": "csv",
            ".nc": "netcdf",
            ".npy": "numpy",
            ".npz": "numpy",
        }

        return format_map.get(extension, "unknown")

    def _load_geotiff(self, file_path: Path) -> Dict:
        """Load GeoTIFF raster data"""
        if not HAS_RASTERIO:
            raise ValidationError("rasterio required for GeoTIFF support")

        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            # Handle nodata values
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

        return {
            "data": data,
            "transform": transform,
            "crs": crs,
            "nodata": nodata,
            "format": "geotiff",
            "metadata": {
                "file_path": str(file_path),
                "shape": data.shape,
                "data_type": str(data.dtype),
            },
        }

    def _load_asc_grid(self, file_path: Path) -> Dict:
        """Load ASCII grid file"""
        try:
            with open(file_path, "r") as f:
                # Read header
                header = {}
                for i in range(6):  # Standard ASC header
                    line = f.readline().strip().split()
                    if len(line) >= 2:
                        key = line[0].lower()
                        value = float(line[1]) if "." in line[1] else int(line[1])
                        header[key] = value

                # Read data
                data = []
                for line in f:
                    row = [float(x) for x in line.strip().split()]
                    data.append(row)

            data = np.array(data)

            # Handle nodata
            nodata_value = header.get("nodata_value", -9999)
            data = np.where(data == nodata_value, np.nan, data)

            # Create transform
            if HAS_RASTERIO:
                transform = from_bounds(
                    header["xllcorner"],
                    header["yllcorner"],
                    header["xllcorner"] + header["ncols"] * header["cellsize"],
                    header["yllcorner"] + header["nrows"] * header["cellsize"],
                    header["ncols"],
                    header["nrows"],
                )
            else:
                transform = None

            return {
                "data": data,
                "transform": transform,
                "crs": None,
                "nodata": nodata_value,
                "format": "asc",
                "header": header,
                "metadata": {
                    "file_path": str(file_path),
                    "shape": data.shape,
                    "data_type": str(data.dtype),
                },
            }

        except Exception as e:
            raise ValidationError(f"Error loading ASC file: {e}")

    def _load_csv_data(self, file_path: Path) -> Dict:
        """Load CSV point data"""
        try:
            df = pd.read_csv(file_path)

            return {
                "data": df,
                "format": "csv",
                "metadata": {
                    "file_path": str(file_path),
                    "shape": df.shape,
                    "columns": list(df.columns),
                },
            }

        except Exception as e:
            raise ValidationError(f"Error loading CSV file: {e}")

    def _load_netcdf(self, file_path: Path) -> Dict:
        """Load NetCDF data"""
        try:
            import xarray as xr

            ds = xr.open_dataset(file_path)

            return {
                "data": ds,
                "format": "netcdf",
                "metadata": {
                    "file_path": str(file_path),
                    "variables": list(ds.variables.keys()),
                    "dimensions": dict(ds.dims),
                },
            }

        except ImportError:
            raise ValidationError("xarray required for NetCDF support")
        except Exception as e:
            raise ValidationError(f"Error loading NetCDF file: {e}")

    def _load_numpy_array(self, file_path: Path) -> Dict:
        """Load NumPy array data"""
        try:
            if file_path.suffix.lower() == ".npz":
                data = np.load(file_path)
                # Assume first array is the main data
                main_key = list(data.keys())[0]
                array_data = data[main_key]
            else:
                array_data = np.load(file_path)

            return {
                "data": array_data,
                "format": "numpy",
                "metadata": {
                    "file_path": str(file_path),
                    "shape": array_data.shape,
                    "data_type": str(array_data.dtype),
                },
            }

        except Exception as e:
            raise ValidationError(f"Error loading NumPy file: {e}")


class SpatialUtils:
    """
    Spatial processing utilities for coordinate transformations and alignment
    """

    @staticmethod
    def transform_coordinates(
        coords: np.ndarray, src_crs: str, dst_crs: str
    ) -> np.ndarray:
        """
        Transform coordinates between coordinate reference systems

        Args:
            coords: Array of coordinates [[x1, y1], [x2, y2], ...]
            src_crs: Source CRS (e.g., 'EPSG:4326')
            dst_crs: Destination CRS (e.g., 'EPSG:3857')

        Returns:
            Transformed coordinates
        """
        if not HAS_PYPROJ:
            logger.warning("pyproj not available - returning original coordinates")
            return coords

        try:
            transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            x_coords, y_coords = coords[:, 0], coords[:, 1]
            x_transformed, y_transformed = transformer.transform(x_coords, y_coords)

            return np.column_stack([x_transformed, y_transformed])

        except Exception as e:
            logger.error(f"Coordinate transformation failed: {e}")
            raise ValidationError(f"Coordinate transformation error: {e}")

    @staticmethod
    def calculate_bounds(data_dict: Dict) -> Tuple[float, float, float, float]:
        """
        Calculate spatial bounds from data dictionary

        Args:
            data_dict: Data dictionary with spatial information

        Returns:
            Bounds tuple (min_x, min_y, max_x, max_y)
        """
        try:
            if "transform" in data_dict and data_dict["transform"] is not None:
                transform = data_dict["transform"]
                data = data_dict["data"]

                # Calculate bounds from transform and data shape
                min_x = transform.c
                max_y = transform.f
                max_x = min_x + (data.shape[1] * transform.a)
                min_y = max_y + (data.shape[0] * transform.e)  # e is negative

                return min_x, min_y, max_x, max_y

            elif "header" in data_dict:
                # ASCII grid bounds
                header = data_dict["header"]
                min_x = header["xllcorner"]
                min_y = header["yllcorner"]
                max_x = min_x + (header["ncols"] * header["cellsize"])
                max_y = min_y + (header["nrows"] * header["cellsize"])

                return min_x, min_y, max_x, max_y

            else:
                logger.warning("No spatial information found - using default bounds")
                return -180.0, -90.0, 180.0, 90.0

        except Exception as e:
            logger.error(f"Error calculating bounds: {e}")
            return -180.0, -90.0, 180.0, 90.0

    @staticmethod
    def align_grids(
        grid1: np.ndarray, grid2: np.ndarray, method: str = "crop"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two grids by shape

        Args:
            grid1: First grid
            grid2: Second grid
            method: Alignment method ('crop', 'pad', 'resample')

        Returns:
            Aligned grids
        """
        if grid1.shape == grid2.shape:
            return grid1, grid2

        if method == "crop":
            # Crop to minimum dimensions
            min_rows = min(grid1.shape[0], grid2.shape[0])
            min_cols = min(grid1.shape[1], grid2.shape[1])

            aligned_grid1 = grid1[:min_rows, :min_cols]
            aligned_grid2 = grid2[:min_rows, :min_cols]

        elif method == "pad":
            # Pad to maximum dimensions
            max_rows = max(grid1.shape[0], grid2.shape[0])
            max_cols = max(grid1.shape[1], grid2.shape[1])

            aligned_grid1 = np.pad(
                grid1,
                ((0, max_rows - grid1.shape[0]), (0, max_cols - grid1.shape[1])),
                constant_values=np.nan,
            )
            aligned_grid2 = np.pad(
                grid2,
                ((0, max_rows - grid2.shape[0]), (0, max_cols - grid2.shape[1])),
                constant_values=np.nan,
            )

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        logger.info(f"Grids aligned using {method} method: {aligned_grid1.shape}")
        return aligned_grid1, aligned_grid2


class StatisticalUtils:
    """
    Statistical utility functions for validation analysis
    """

    @staticmethod
    def detect_outliers(
        data: np.ndarray, method: str = "iqr", threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in data

        Args:
            data: Input data array
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold value

        Returns:
            Boolean mask of outliers
        """
        data_clean = data[~np.isnan(data)]

        if len(data_clean) == 0:
            return np.zeros_like(data, dtype=bool)

        if method == "iqr":
            Q1 = np.percentile(data_clean, 25)
            Q3 = np.percentile(data_clean, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == "zscore":
            z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
            outliers = z_scores > threshold

        elif method == "modified_zscore":
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers

    @staticmethod
    def bootstrap_confidence_interval(
        data1: np.ndarray,
        data2: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict:
        """
        Calculate bootstrap confidence interval for a metric

        Args:
            data1: First dataset
            data2: Second dataset
            metric_func: Function to calculate metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            Dictionary with metric value and confidence interval
        """
        # Original metric
        original_metric = metric_func(data1, data2)

        # Bootstrap samples
        n_samples = len(data1)
        bootstrap_metrics = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample1 = data1[indices]
            sample2 = data2[indices]

            # Calculate metric for bootstrap sample
            bootstrap_metric = metric_func(sample1, sample2)
            bootstrap_metrics.append(bootstrap_metric)

        bootstrap_metrics = np.array(bootstrap_metrics)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)

        return {
            "metric_value": original_metric,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "bootstrap_std": np.std(bootstrap_metrics),
            "bootstrap_mean": np.mean(bootstrap_metrics),
        }


class LoggingSetup:
    """
    Configures logging for the validation framework
    """

    @staticmethod
    def setup_logging(config_manager: ConfigurationManager) -> None:
        """
        Setup logging configuration

        Args:
            config_manager: Configuration manager instance
        """
        log_config = config_manager.get("logging", {})

        # Get log level
        log_level = getattr(logging, log_config.get("level", "INFO").upper())

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console logging
        if log_config.get("console_logging", {}).get("enabled", True):
            console_handler = logging.StreamHandler()
            console_level = getattr(
                logging,
                log_config.get("console_logging", {}).get("level", "INFO").upper(),
            )
            console_handler.setLevel(console_level)

            formatter = logging.Formatter(
                log_config.get(
                    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File logging
        file_config = log_config.get("file_logging", {})
        if file_config.get("enabled", False):
            log_file = file_config.get("filename", "validation.log")

            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=file_config.get("max_bytes", 10485760),  # 10MB
                backupCount=file_config.get("backup_count", 5),
            )

            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logger.info("Logging configured successfully")


# Decorator utilities
def log_execution_time(func):
    """Decorator to log function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            execution_time = datetime.now() - start_time
            logger.info(
                f"Completed {func.__name__} in {execution_time.total_seconds():.2f}s"
            )
            return result

        except Exception as e:
            execution_time = datetime.now() - start_time
            logger.error(
                f"Failed {func.__name__} after {execution_time.total_seconds():.2f}s: {e}"
            )
            raise

    return wrapper


def validate_inputs(validation_func):
    """Decorator to validate function inputs"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Apply validation
            validation_func(bound_args.arguments)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value when denominator is zero

    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def create_temp_directory() -> str:
    """
    Create temporary directory for validation processing

    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="flood_validation_")
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up temporary directory

    Args:
        temp_dir: Path to temporary directory
    """
    try:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


def generate_timestamp_string() -> str:
    """
    Generate timestamp string for file naming

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def check_memory_usage() -> Dict[str, float]:
    """
    Check current memory usage

    Returns:
        Dictionary with memory statistics
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

    except ImportError:
        logger.warning("psutil not available - cannot check memory usage")
        return {"error": "psutil not available"}
    except Exception as e:
        logger.warning(f"Error checking memory usage: {e}")
        return {"error": str(e)}


# Context managers
class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time

        if exc_type is None:
            logger.info(
                f"Completed {self.operation_name} in {duration.total_seconds():.2f}s"
            )
        else:
            logger.error(
                f"Failed {self.operation_name} after {duration.total_seconds():.2f}s"
            )

        return False  # Don't suppress exceptions

    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class TemporaryDirectory:
    """Context manager for temporary directories"""

    def __init__(self, prefix: str = "flood_validation_"):
        self.prefix = prefix
        self.temp_dir = None

    def __enter__(self) -> str:
        self.temp_dir = create_temp_directory()
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            cleanup_temp_directory(self.temp_dir)
        return False
