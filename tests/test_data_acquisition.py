"""Tests for data acquisition system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests

from src.data.config import BoundingBox, DataConfig
from src.data.sources.base import BaseDataSource, DataSourceError
from src.data.sources.usgs_3dep import USGS3DEPDownloader
from src.data.sources.noaa_atlas14 import NOAAAtlas14Fetcher
from src.data.manager import DataManager


class TestDataConfig:
    """Test data configuration management."""

    def test_bounding_box_creation(self):
        """Test bounding box creation and validation."""
        bbox = BoundingBox(west=-87.0, south=36.0, east=-86.0, north=37.0)

        assert bbox.west == -87.0
        assert bbox.south == 36.0
        assert bbox.east == -86.0
        assert bbox.north == 37.0
        assert bbox.crs == 4326

        assert bbox.bounds_tuple == (-87.0, 36.0, -86.0, 37.0)

    def test_bounding_box_validation(self):
        """Test bounding box coordinate validation."""
        with pytest.raises(ValueError):
            BoundingBox(west=-86.0, south=36.0, east=-87.0, north=37.0)  # west > east

        with pytest.raises(ValueError):
            BoundingBox(west=-87.0, south=37.0, east=-86.0, north=36.0)  # south > north

    def test_data_config_initialization(self):
        """Test data configuration initialization."""
        config = DataConfig()

        assert config.preferred_dem_resolution == 10
        assert config.enable_caching is True
        assert config.validate_downloads is True
        assert "nashville" in config.regions

        # Test Nashville region configuration
        nashville = config.get_region_bbox("nashville")
        assert nashville is not None
        assert nashville.west < nashville.east
        assert nashville.south < nashville.north

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict("os.environ", {"FLOODRISK_TARGET_CRS": "3857"}):
            config = DataConfig.from_env()
            assert config.target_crs == 3857


class TestBaseDataSource:
    """Test base data source functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation."""

        class MockDataSource(BaseDataSource):
            def get_available_data(self, **kwargs):
                return []

            def download_data(self, **kwargs):
                return []

        source = MockDataSource()

        key1 = source._generate_cache_key(param1="value1", param2=123)
        key2 = source._generate_cache_key(
            param2=123, param1="value1"
        )  # Same params, different order
        key3 = source._generate_cache_key(
            param1="value2", param2=123
        )  # Different value

        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different key
        assert len(key1) == 64  # SHA256 hex string length

    def test_session_creation(self):
        """Test HTTP session creation with retry strategy."""

        class MockDataSource(BaseDataSource):
            def get_available_data(self, **kwargs):
                return []

            def download_data(self, **kwargs):
                return []

        source = MockDataSource()
        session = source.session

        assert session.timeout == source.config.timeout_seconds
        assert "User-Agent" in session.headers


class TestUSGS3DEPDownloader:
    """Test USGS 3DEP downloader."""

    def test_initialization(self):
        """Test downloader initialization."""
        downloader = USGS3DEPDownloader()

        assert downloader.config is not None
        assert downloader.api_url == downloader.config.usgs_api_base_url
        assert 10 in downloader.resolution_services

    @patch("requests.Session.get")
    def test_get_available_data(self, mock_get):
        """Test querying available DEM data."""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "items": [
                {
                    "sourceId": "test_dem_001",
                    "title": "Test DEM Tile",
                    "downloadURL": "https://example.com/test.tif",
                    "format": "GeoTIFF",
                    "sizeInBytes": 1000000,
                    "boundingBox": {
                        "minX": -87.0,
                        "minY": 36.0,
                        "maxX": -86.0,
                        "maxY": 37.0,
                    },
                    "dateCreated": "2024-01-01",
                    "lastUpdated": "2024-01-01",
                }
            ]
        }
        mock_get.return_value = mock_response

        downloader = USGS3DEPDownloader()
        bbox = BoundingBox(west=-87.0, south=36.0, east=-86.0, north=37.0)

        products = downloader.get_available_data(bbox, resolution=10)

        assert len(products) == 1
        assert products[0]["id"] == "test_dem_001"
        assert products[0]["resolution_meters"] == 10
        assert products[0]["download_url"] == "https://example.com/test.tif"

    def test_resolution_service_preferences(self):
        """Test resolution service preferences."""
        downloader = USGS3DEPDownloader()

        # 10m should prefer static service
        assert downloader.resolution_services[10] == ["static", "api"]

        # 1m should only use API
        assert downloader.resolution_services[1] == ["api"]


class TestNOAAAtlas14Fetcher:
    """Test NOAA Atlas 14 fetcher."""

    def test_initialization(self):
        """Test fetcher initialization."""
        fetcher = NOAAAtlas14Fetcher()

        assert fetcher.config is not None
        assert fetcher.pfds_url == fetcher.config.noaa_pfds_base_url
        assert 1440 in fetcher.duration_minutes  # 24 hours in minutes
        assert fetcher.duration_minutes[1440] == "24-hr"

    @patch("requests.Session.get")
    def test_get_available_data(self, mock_get):
        """Test querying available precipitation data."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"status": "success"}
        mock_get.return_value = mock_response

        fetcher = NOAAAtlas14Fetcher()
        location = (-86.5, 36.0)  # Nashville coordinates

        available_data = fetcher.get_available_data(location)

        assert available_data["location"]["longitude"] == -86.5
        assert available_data["location"]["latitude"] == 36.0
        assert available_data["return_periods"] == fetcher.config.return_periods_years
        assert available_data["source"] == "NOAA Atlas 14"

    def test_duration_conversion(self):
        """Test duration conversion from hours to minutes."""
        fetcher = NOAAAtlas14Fetcher()

        # Config durations in hours should be converted to minutes
        assert fetcher.config_durations_min == [
            int(h * 60) for h in fetcher.config.durations_hours
        ]

        # Standard duration labels
        assert fetcher.duration_minutes[60] == "60-min"
        assert fetcher.duration_minutes[1440] == "24-hr"


class TestDataManager:
    """Test data manager coordination."""

    def test_initialization(self):
        """Test data manager initialization."""
        manager = DataManager()

        assert manager.config is not None
        assert isinstance(manager.usgs_downloader, USGS3DEPDownloader)
        assert isinstance(manager.noaa_fetcher, NOAAAtlas14Fetcher)

    def test_validate_data_integrity(self):
        """Test data integrity validation."""
        manager = DataManager()

        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid file
            valid_file = temp_path / "valid.tif"
            valid_file.write_text("x" * 2000)  # Above minimum size

            # Create invalid file (too small)
            invalid_file = temp_path / "invalid.tif"
            invalid_file.write_text("x")  # Below minimum size

            # Create missing file reference
            missing_file = temp_path / "missing.tif"

            results = manager.validate_data_integrity(
                [valid_file, invalid_file, missing_file], data_type="test"
            )

            assert valid_file in results["valid"]
            assert invalid_file in results["invalid"]
            assert missing_file in results["invalid"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_data_source_error(self):
        """Test DataSourceError exception."""
        with pytest.raises(DataSourceError):
            raise DataSourceError("Test error message")

    @patch("requests.Session.get")
    def test_network_error_handling(self, mock_get):
        """Test handling of network errors."""
        # Mock network error
        mock_get.side_effect = requests.ConnectionError("Network error")

        downloader = USGS3DEPDownloader()
        bbox = BoundingBox(west=-87.0, south=36.0, east=-86.0, north=37.0)

        with pytest.raises(DataSourceError):
            downloader.get_available_data(bbox)

    def test_invalid_region_handling(self):
        """Test handling of invalid region names."""
        manager = DataManager()

        with pytest.raises(DataSourceError) as exc_info:
            manager.download_all_region_data("nonexistent_region")

        assert "not found" in str(exc_info.value).lower()
        assert "available regions" in str(exc_info.value).lower()


class TestCaching:
    """Test caching functionality."""

    def test_cache_path_generation(self):
        """Test cache path generation."""

        class MockDataSource(BaseDataSource):
            def get_available_data(self, **kwargs):
                return []

            def download_data(self, **kwargs):
                return []

        source = MockDataSource()
        cache_key = "test_cache_key_123"

        cache_path = source._get_cache_path(cache_key, ".json")

        assert cache_path.name == "test_cache_key_123.json"
        assert "mockdatasource" in str(cache_path).lower()

    def test_cache_save_and_load(self):
        """Test cache save and load operations."""

        class MockDataSource(BaseDataSource):
            def get_available_data(self, **kwargs):
                return []

            def download_data(self, **kwargs):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(cache_dir=Path(temp_dir))
            source = MockDataSource(config)

            # Test data
            test_data = {"key": "value", "number": 123}
            cache_key = "test_key"
            cache_path = source._get_cache_path(cache_key)

            # Save to cache
            source._save_to_cache(cache_path, test_data)
            assert cache_path.exists()

            # Load from cache
            loaded_data = source._load_from_cache(cache_path)
            assert loaded_data == test_data


# Integration test (requires network access - mark as slow)
@pytest.mark.slow
class TestIntegration:
    """Integration tests requiring network access."""

    def test_usgs_api_connectivity(self):
        """Test actual USGS API connectivity."""
        downloader = USGS3DEPDownloader()

        # Small test area in Nashville
        bbox = BoundingBox(west=-86.8, south=36.1, east=-86.7, north=36.2)

        try:
            products = downloader.get_available_data(bbox, resolution=10)
            # Should find at least some products for Nashville area
            assert isinstance(products, list)
        except DataSourceError as e:
            pytest.skip(f"USGS API not accessible: {e}")

    def test_noaa_api_connectivity(self):
        """Test actual NOAA API connectivity."""
        fetcher = NOAAAtlas14Fetcher()

        # Nashville coordinates
        location = (-86.7816, 36.1627)

        try:
            available_data = fetcher.get_available_data(location)
            assert "location" in available_data
            assert "return_periods" in available_data
        except DataSourceError as e:
            pytest.skip(f"NOAA API not accessible: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
