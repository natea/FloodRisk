"""USGS 3DEP DEM data acquisition module.

This module provides functionality for downloading 10m resolution DEM data
from the USGS 3D Elevation Program using both the TNM API and optimized
cloud services.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin
import requests
from pyproj import CRS, Transformer

from .base import BaseDataSource, DataSourceError
from ..config import BoundingBox, DataConfig


class USGS3DEPDownloader(BaseDataSource):
    """USGS 3D Elevation Program DEM downloader.

    Supports downloading 10m resolution DEM data for specified geographic areas
    using multiple data sources including The National Map API and cloud-optimized
    services.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize USGS 3DEP downloader.

        Args:
            config: Data configuration object
        """
        super().__init__(config)
        self.api_url = self.config.usgs_api_base_url
        self.static_url = self.config.usgs_static_service_url

        # Supported resolutions and their service preferences
        self.resolution_services = {
            10: ["static", "api"],  # Prefer static service for 10m
            30: ["static", "api"],
            60: ["static", "api"],
            1: ["api"],  # 1m only available through API
        }

    def get_available_data(
        self, bbox: Union[BoundingBox, Dict[str, float]], resolution: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get list of available DEM products for bounding box.

        Args:
            bbox: Geographic bounding box
            resolution: DEM resolution in meters (default: 10)
            **kwargs: Additional parameters

        Returns:
            List of available DEM products with metadata

        Raises:
            DataSourceError: If API request fails
        """
        if isinstance(bbox, dict):
            bbox = BoundingBox(**bbox)

        cache_key = self._generate_cache_key(
            bbox=bbox.to_dict(), resolution=resolution, action="available_data"
        )
        cache_path = self._get_cache_path(cache_key)

        # Check cache first
        if cached_data := self._load_from_cache(cache_path):
            return cached_data

        self.logger.info(f"Querying available DEM data for resolution {resolution}m")

        try:
            # Transform bbox to geographic coordinates if needed
            if bbox.crs != 4326:
                transformer = Transformer.from_crs(bbox.crs, 4326, always_xy=True)
                west, south = transformer.transform(bbox.west, bbox.south)
                east, north = transformer.transform(bbox.east, bbox.north)
                geo_bbox = BoundingBox(west, south, east, north, 4326)
            else:
                geo_bbox = bbox

            # Query TNM API
            params = {
                "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
                "bbox": f"{geo_bbox.west},{geo_bbox.south},{geo_bbox.east},{geo_bbox.north}",
                "format": "JSON",
                "max": 50,
            }

            if resolution == 10:
                params["datasets"] = "National Elevation Dataset (NED) 1/3 arc-second"
            elif resolution == 30:
                params["datasets"] = "National Elevation Dataset (NED) 1 arc-second"
            elif resolution == 1:
                params["datasets"] = "Lidar Point Cloud (LPC)"

            response = self.session.get(self.api_url, params=params)
            response.raise_for_status()

            data = response.json()
            products = []

            for item in data.get("items", []):
                product_info = {
                    "id": item.get("sourceId"),
                    "title": item.get("title", ""),
                    "download_url": item.get("downloadURL"),
                    "format": item.get("format", "GeoTIFF"),
                    "size_bytes": item.get("sizeInBytes", 0),
                    "bbox": {
                        "west": float(item.get("boundingBox", {}).get("minX", 0)),
                        "south": float(item.get("boundingBox", {}).get("minY", 0)),
                        "east": float(item.get("boundingBox", {}).get("maxX", 0)),
                        "north": float(item.get("boundingBox", {}).get("maxY", 0)),
                    },
                    "resolution_meters": resolution,
                    "date_created": item.get("dateCreated"),
                    "last_updated": item.get("lastUpdated"),
                    "metadata_url": item.get("metaUrl"),
                }
                products.append(product_info)

            # Cache results
            self._save_to_cache(cache_path, products)

            self.logger.info(f"Found {len(products)} DEM products")
            return products

        except requests.RequestException as e:
            raise DataSourceError(f"Failed to query USGS API: {e}")
        except Exception as e:
            raise DataSourceError(f"Error processing USGS API response: {e}")

    def download_data(
        self,
        bbox: Union[BoundingBox, Dict[str, float]],
        resolution: int = 10,
        output_dir: Optional[Path] = None,
        region_name: Optional[str] = None,
        **kwargs,
    ) -> List[Path]:
        """Download DEM data for specified bounding box.

        Args:
            bbox: Geographic bounding box
            resolution: DEM resolution in meters
            output_dir: Directory to save downloaded files
            region_name: Name for the region (used in filenames)
            **kwargs: Additional parameters

        Returns:
            List of paths to downloaded DEM files

        Raises:
            DataSourceError: If download fails
        """
        if isinstance(bbox, dict):
            bbox = BoundingBox(**bbox)

        if output_dir is None:
            output_dir = self.config.data_dir / "dem" / f"{resolution}m"

        self.logger.info(f"Starting DEM download for {resolution}m resolution")

        # Get available products
        products = self.get_available_data(bbox, resolution, **kwargs)

        if not products:
            raise DataSourceError(
                f"No DEM products found for specified area and resolution"
            )

        downloaded_files = []

        for i, product in enumerate(products):
            try:
                # Generate filename
                if region_name:
                    filename = f"{region_name}_dem_{resolution}m_{i+1}.tif"
                else:
                    filename = f"dem_{resolution}m_{product['id']}.tif"

                output_path = output_dir / filename

                # Skip if file already exists and is valid
                if output_path.exists() and self._is_cache_valid(output_path):
                    self.logger.info(f"DEM file already exists: {output_path}")
                    downloaded_files.append(output_path)
                    continue

                # Download the file
                download_url = product["download_url"]
                if not download_url:
                    self.logger.warning(f"No download URL for product {product['id']}")
                    continue

                self.logger.info(
                    f"Downloading DEM {i+1}/{len(products)}: {product['title']}"
                )

                downloaded_path = self._download_file(
                    download_url, output_path, resume=True, validate=True
                )

                downloaded_files.append(downloaded_path)

            except Exception as e:
                self.logger.error(
                    f"Failed to download product {product.get('id', 'unknown')}: {e}"
                )
                continue

        if not downloaded_files:
            raise DataSourceError("No DEM files were successfully downloaded")

        self.logger.info(f"Successfully downloaded {len(downloaded_files)} DEM files")
        return downloaded_files

    def download_region(
        self,
        region_name: str,
        resolution: int = 10,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> List[Path]:
        """Download DEM data for a predefined region.

        Args:
            region_name: Name of predefined region (e.g., 'nashville', 'tennessee')
            resolution: DEM resolution in meters
            output_dir: Directory to save downloaded files
            **kwargs: Additional parameters

        Returns:
            List of paths to downloaded DEM files

        Raises:
            DataSourceError: If region not found or download fails
        """
        bbox = self.config.get_region_bbox(region_name)
        if bbox is None:
            available_regions = list(self.config.regions.keys())
            raise DataSourceError(
                f"Region '{region_name}' not found. "
                f"Available regions: {available_regions}"
            )

        if output_dir is None:
            output_dir = (
                self.config.data_dir
                / "regions"
                / region_name
                / "dem"
                / f"{resolution}m"
            )

        self.logger.info(f"Downloading DEM data for region: {region_name}")

        return self.download_data(
            bbox=bbox,
            resolution=resolution,
            output_dir=output_dir,
            region_name=region_name,
            **kwargs,
        )

    def get_seamless_dem(
        self,
        bbox: Union[BoundingBox, Dict[str, float]],
        resolution: int = 10,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Download seamless DEM using cloud-optimized service.

        This method uses the seamless 3DEP service for faster downloads
        when available.

        Args:
            bbox: Geographic bounding box
            resolution: DEM resolution in meters
            output_path: Path for output file

        Returns:
            Path to downloaded seamless DEM file

        Raises:
            DataSourceError: If seamless service unavailable or download fails
        """
        if isinstance(bbox, dict):
            bbox = BoundingBox(**bbox)

        # Check if resolution is supported by static service
        if resolution not in [10, 30, 60]:
            raise DataSourceError(
                f"Seamless service does not support {resolution}m resolution"
            )

        # Transform to geographic coordinates if needed
        if bbox.crs != 4326:
            transformer = Transformer.from_crs(bbox.crs, 4326, always_xy=True)
            west, south = transformer.transform(bbox.west, bbox.south)
            east, north = transformer.transform(bbox.east, bbox.north)
        else:
            west, south, east, north = bbox.bounds_tuple

        if output_path is None:
            output_path = (
                self.config.data_dir
                / "dem"
                / "seamless"
                / f"seamless_dem_{resolution}m_{west}_{south}_{east}_{north}.tif"
            )

        # Construct seamless service URL
        service_url = (
            f"{self.static_url}/SRTMGL{resolution}/SRTMGL{resolution}_srtm.tif"
        )

        # Parameters for COG access with bbox
        params = {"bbox": f"{west},{south},{east},{north}", "format": "GTiff"}

        try:
            self.logger.info("Downloading seamless DEM from cloud-optimized service")

            response = self.session.get(service_url, params=params, stream=True)
            response.raise_for_status()

            # Save the seamless DEM
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                    if chunk:
                        f.write(chunk)

            self.logger.info(f"Seamless DEM downloaded: {output_path}")
            return output_path

        except requests.RequestException as e:
            self.logger.warning(
                f"Seamless service failed, falling back to standard API: {e}"
            )
            # Fall back to standard download method
            files = self.download_data(bbox, resolution, output_path.parent)
            return files[0] if files else None
