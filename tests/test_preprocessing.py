"""Integration tests for DEM preprocessing and feature extraction.

This module tests the complete preprocessing pipeline including:
- DEM loading and validation
- Hydrological conditioning
- Terrain feature extraction
- Real data processing workflows
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from src.preprocessing.dem.hydrological_conditioning import HydrologicalConditioner, validate_dem_quality
from src.preprocessing.terrain.feature_extraction import TerrainFeatureExtractor, HANDCalculator


class TestDEMPreprocessing:
    """Test DEM preprocessing components with real data scenarios."""

    @pytest.fixture
    def sample_dem_data(self):
        """Create realistic synthetic DEM data for testing."""
        # Create a 100x100 DEM with realistic elevation patterns
        height, width = 100, 100
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Base terrain with realistic elevation values
        dem = 100 + 20 * np.sin(X/2) + 15 * np.cos(Y/3) + 5 * np.random.normal(0, 1, (height, width))
        
        # Add some drainage features (lower elevations)
        stream_mask = (X > 4.5) & (X < 5.5) & (Y > 3) & (Y < 7)
        dem[stream_mask] -= 10
        
        # Add some hills
        hill_mask = ((X - 7)**2 + (Y - 7)**2) < 4
        dem[hill_mask] += 30 * np.exp(-((X[hill_mask] - 7)**2 + (Y[hill_mask] - 7)**2))
        
        # Add some artificial sinks for testing
        sink_mask = ((X - 2)**2 + (Y - 2)**2) < 1
        dem[sink_mask] -= 5
        
        return dem.astype(np.float32)

    @pytest.fixture
    def sample_dem_file(self, sample_dem_data, tmp_path):
        """Create a temporary DEM file for testing file I/O."""
        dem_path = tmp_path / "test_dem.tif"
        
        # Define geospatial properties
        transform = from_bounds(
            west=0, south=0, east=1000, north=1000,
            width=sample_dem_data.shape[1], height=sample_dem_data.shape[0]
        )
        
        with rasterio.open(
            dem_path, 'w',
            driver='GTiff',
            height=sample_dem_data.shape[0],
            width=sample_dem_data.shape[1],
            count=1,
            dtype=sample_dem_data.dtype,
            crs=CRS.from_epsg(3857),
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(sample_dem_data, 1)
        
        return str(dem_path)

    @pytest.fixture
    def hydrological_conditioner(self, sample_dem_file):
        """Create a HydrologicalConditioner instance."""
        return HydrologicalConditioner(sample_dem_file)

    def test_dem_loading(self, hydrological_conditioner, sample_dem_data):
        """Test DEM loading functionality."""
        assert hydrological_conditioner.dem_array is not None
        assert hydrological_conditioner.dem_array.shape == sample_dem_data.shape
        assert hydrological_conditioner.dem_profile is not None
        assert 'crs' in hydrological_conditioner.dem_profile
        assert 'transform' in hydrological_conditioner.dem_profile

    def test_sink_filling(self, hydrological_conditioner):
        """Test sink filling algorithms."""
        original_dem = hydrological_conditioner.dem_array.copy()
        
        # Test Planchon-Darboux method
        filled_dem = hydrological_conditioner.fill_sinks(method='planchon_darboux')
        
        assert filled_dem.shape == original_dem.shape
        assert np.all(filled_dem >= original_dem), "Filled DEM should have higher or equal elevations"
        assert not np.array_equal(filled_dem, original_dem), "Sink filling should modify the DEM"
        
        # Test Wei method
        filled_dem_wei = hydrological_conditioner.fill_sinks(method='wei')
        assert filled_dem_wei.shape == original_dem.shape
        
    def test_flow_direction_calculation(self, hydrological_conditioner):
        """Test flow direction calculation."""
        filled_dem = hydrological_conditioner.fill_sinks()
        flow_dir = hydrological_conditioner.calculate_flow_direction(filled_dem)
        
        assert flow_dir.shape == filled_dem.shape
        assert flow_dir.dtype in [np.uint8, np.int32, np.float32, np.float64]
        
        # Flow direction should have valid D8 values (0-7 or equivalent)
        unique_values = np.unique(flow_dir[~np.isnan(flow_dir)])
        assert len(unique_values) <= 9, "D8 flow direction should have at most 9 unique values"

    def test_flow_accumulation_calculation(self, hydrological_conditioner):
        """Test flow accumulation calculation."""
        filled_dem = hydrological_conditioner.fill_sinks()
        flow_dir = hydrological_conditioner.calculate_flow_direction(filled_dem)
        flow_acc = hydrological_conditioner.calculate_flow_accumulation(flow_dir)
        
        assert flow_acc.shape == filled_dem.shape
        assert np.all(flow_acc >= 0), "Flow accumulation should be non-negative"
        assert np.max(flow_acc) > np.min(flow_acc), "Flow accumulation should vary across the DEM"

    def test_complete_conditioning_workflow(self, hydrological_conditioner):
        """Test the complete hydrological conditioning workflow."""
        results = hydrological_conditioner.condition_dem(
            stream_threshold=50, buffer_distance=50
        )
        
        required_keys = {
            'conditioned_dem', 'filled_dem', 'flow_direction', 
            'flow_accumulation', 'streams'
        }
        assert set(results.keys()) == required_keys
        
        # Validate each result
        original_shape = hydrological_conditioner.dem_array.shape
        for key, array in results.items():
            assert array.shape == original_shape, f"{key} has incorrect shape"
            assert not np.all(np.isnan(array)), f"{key} should not be all NaN"
        
        # Streams should be binary
        streams = results['streams']
        unique_stream_values = np.unique(streams)
        assert set(unique_stream_values).issubset({0, 1}), "Streams should be binary (0 or 1)"

    def test_dem_quality_validation(self, hydrological_conditioner):
        """Test DEM quality validation metrics."""
        results = hydrological_conditioner.condition_dem()
        metrics = validate_dem_quality(results['conditioned_dem'], results['flow_accumulation'])
        
        expected_metrics = {
            'sink_count', 'sink_percentage', 'stream_connectivity',
            'elevation_range', 'elevation_mean', 'elevation_std'
        }
        assert set(metrics.keys()) == expected_metrics
        
        # Validate metric ranges
        assert metrics['sink_count'] >= 0
        assert 0 <= metrics['sink_percentage'] <= 100
        assert 0 <= metrics['stream_connectivity'] <= 1
        assert metrics['elevation_range'] > 0
        assert not np.isnan(metrics['elevation_mean'])
        assert metrics['elevation_std'] > 0

    @patch('rasterio.open')
    def test_dem_loading_with_io_error(self, mock_rasterio_open):
        """Test DEM loading with I/O errors."""
        mock_rasterio_open.side_effect = IOError("File not found")
        
        with pytest.raises(IOError):
            HydrologicalConditioner("nonexistent_file.tif")

    def test_save_results(self, hydrological_conditioner, tmp_path):
        """Test saving conditioning results to files."""
        results = hydrological_conditioner.condition_dem()
        output_dir = tmp_path / "output"
        
        hydrological_conditioner.save_results(results, str(output_dir))
        
        # Check that files were created
        for key in results.keys():
            output_file = output_dir / f"{key}.tif"
            assert output_file.exists(), f"Output file {key}.tif was not created"
            
            # Verify file can be read
            with rasterio.open(output_file) as src:
                data = src.read(1)
                assert data.shape == results[key].shape


class TestTerrainFeatureExtraction:
    """Test terrain feature extraction with realistic scenarios."""

    @pytest.fixture
    def sample_terrain_data(self):
        """Create realistic terrain data for feature extraction."""
        height, width = 50, 50
        x = np.linspace(0, 5, width)
        y = np.linspace(0, 5, height)
        X, Y = np.meshgrid(x, y)
        
        # Create varied terrain with valleys, hills, and ridges
        dem = (
            100 +
            30 * np.sin(X) * np.cos(Y) +  # Undulating terrain
            20 * np.exp(-((X - 2.5)**2 + (Y - 2.5)**2)) +  # Central hill
            -15 * np.exp(-((X - 1)**2 + (Y - 4)**2) / 0.5)  # Valley
        )
        
        return dem.astype(np.float32)

    @pytest.fixture
    def feature_extractor(self, sample_terrain_data):
        """Create a TerrainFeatureExtractor instance."""
        pixel_size = 10.0  # 10 meters
        return TerrainFeatureExtractor(sample_terrain_data, pixel_size)

    def test_slope_calculation(self, feature_extractor):
        """Test slope calculation with different units."""
        # Test degrees
        slope_deg = feature_extractor.calculate_slope(units='degrees')
        assert slope_deg.shape == feature_extractor.dem.shape
        assert np.all((slope_deg >= 0) | np.isnan(slope_deg))
        assert np.all((slope_deg <= 90) | np.isnan(slope_deg))
        
        # Test radians
        slope_rad = feature_extractor.calculate_slope(units='radians')
        assert np.all((slope_rad >= 0) | np.isnan(slope_rad))
        assert np.all((slope_rad <= np.pi/2) | np.isnan(slope_rad))
        
        # Test percent
        slope_pct = feature_extractor.calculate_slope(units='percent')
        assert np.all((slope_pct >= 0) | np.isnan(slope_pct))
        
        # Verify unit conversions
        # slope_deg ≈ slope_rad * 180 / π
        valid_mask = ~np.isnan(slope_deg) & ~np.isnan(slope_rad)
        np.testing.assert_allclose(
            slope_deg[valid_mask], 
            slope_rad[valid_mask] * 180 / np.pi,
            rtol=1e-5
        )

    def test_aspect_calculation(self, feature_extractor):
        """Test aspect calculation."""
        aspect = feature_extractor.calculate_aspect()
        
        assert aspect.shape == feature_extractor.dem.shape
        valid_aspect = aspect[~np.isnan(aspect)]
        assert np.all(valid_aspect >= 0)
        assert np.all(valid_aspect <= 360)

    def test_curvature_calculation(self, feature_extractor):
        """Test different types of curvature calculation."""
        curvature_types = ['total', 'plan', 'profile']
        
        for curv_type in curvature_types:
            curvature = feature_extractor.calculate_curvature(curv_type)
            assert curvature.shape == feature_extractor.dem.shape
            # Curvature can be positive, negative, or zero
            valid_curv = curvature[~np.isnan(curvature)]
            assert len(valid_curv) > 0, f"{curv_type} curvature should have valid values"

    def test_roughness_calculation(self, feature_extractor):
        """Test terrain roughness calculation."""
        roughness = feature_extractor.calculate_roughness(window_size=3)
        
        assert roughness.shape == feature_extractor.dem.shape
        valid_roughness = roughness[~np.isnan(roughness)]
        assert np.all(valid_roughness >= 0), "Roughness should be non-negative"
        assert len(valid_roughness) > 0

    def test_tpi_calculation(self, feature_extractor):
        """Test Topographic Position Index calculation."""
        tpi = feature_extractor.calculate_tpi(window_size=3)
        
        assert tpi.shape == feature_extractor.dem.shape
        valid_tpi = tpi[~np.isnan(tpi)]
        # TPI can be positive (ridges), negative (valleys), or zero (flat)
        assert len(valid_tpi) > 0
        assert np.std(valid_tpi) > 0, "TPI should show variation across terrain"

    def test_tri_calculation(self, feature_extractor):
        """Test Terrain Ruggedness Index calculation."""
        tri = feature_extractor.calculate_tri(window_size=3)
        
        assert tri.shape == feature_extractor.dem.shape
        valid_tri = tri[~np.isnan(tri)]
        assert np.all(valid_tri >= 0), "TRI should be non-negative"
        assert len(valid_tri) > 0

    def test_flow_accumulation(self, feature_extractor):
        """Test flow accumulation calculation."""
        flow_acc = feature_extractor.calculate_flow_accumulation()
        
        assert flow_acc.shape == feature_extractor.dem.shape
        valid_flow_acc = flow_acc[~np.isnan(flow_acc)]
        assert np.all(valid_flow_acc >= 0), "Flow accumulation should be non-negative"
        assert len(valid_flow_acc) > 0

    def test_wetness_index_calculation(self, feature_extractor):
        """Test Topographic Wetness Index calculation."""
        flow_acc = feature_extractor.calculate_flow_accumulation()
        slope_rad = feature_extractor.calculate_slope(units='radians')
        twi = feature_extractor.calculate_wetness_index(flow_acc, slope_rad)
        
        assert twi.shape == feature_extractor.dem.shape
        valid_twi = twi[~np.isnan(twi)]
        # TWI typically ranges from ~2 to ~20
        assert len(valid_twi) > 0
        assert np.min(valid_twi) > -10, "TWI should not be extremely negative"
        assert np.max(valid_twi) < 30, "TWI should not be extremely positive"

    def test_extract_all_features(self, feature_extractor):
        """Test extraction of all terrain features."""
        features = feature_extractor.extract_all_features()
        
        expected_features = {
            'slope_degrees', 'slope_radians', 'aspect',
            'curvature_total', 'curvature_plan', 'curvature_profile',
            'roughness', 'tpi', 'tri',
            'flow_accumulation', 'wetness_index'
        }
        
        assert set(features.keys()) == expected_features
        
        # Validate all features have correct shape
        for name, feature_array in features.items():
            assert feature_array.shape == feature_extractor.dem.shape, f"{name} has incorrect shape"
            assert not np.all(np.isnan(feature_array)), f"{name} should not be all NaN"


class TestHANDCalculation:
    """Test Height Above Nearest Drainage (HAND) calculation."""

    @pytest.fixture
    def hand_test_data(self):
        """Create test data for HAND calculation."""
        height, width = 30, 30
        x = np.linspace(0, 3, width)
        y = np.linspace(0, 3, height)
        X, Y = np.meshgrid(x, y)
        
        # Create terrain with a clear drainage pattern
        dem = 100 + 10 * X + 5 * Y + 2 * np.sin(X * 2) * np.cos(Y * 2)
        
        # Create a clear stream through the middle
        stream_mask = (Y > 1.4) & (Y < 1.6)
        dem[stream_mask] -= 15
        
        # Create flow accumulation that defines the stream
        flow_acc = np.ones_like(dem)
        flow_acc[stream_mask] = 2000  # Above threshold
        
        return dem.astype(np.float32), flow_acc.astype(np.float32)

    @pytest.fixture
    def hand_calculator(self, hand_test_data):
        """Create a HANDCalculator instance."""
        dem, flow_acc = hand_test_data
        return HANDCalculator(dem, flow_acc, pixel_size=10.0, stream_threshold=1000)

    def test_stream_identification(self, hand_calculator):
        """Test stream identification from flow accumulation."""
        streams = hand_calculator.streams
        
        assert streams.shape == hand_calculator.dem.shape
        assert streams.dtype == bool
        assert np.any(streams), "Should identify at least some stream pixels"
        
        # Stream pixels should correspond to high flow accumulation
        stream_flow_acc = hand_calculator.flow_acc[streams]
        assert np.all(stream_flow_acc >= hand_calculator.stream_threshold)

    def test_hand_calculation(self, hand_calculator):
        """Test HAND calculation."""
        hand = hand_calculator.calculate_hand(max_distance=100)
        
        assert hand.shape == hand_calculator.dem.shape
        valid_hand = hand[~np.isnan(hand)]
        
        # HAND should be non-negative
        assert np.all(valid_hand >= 0), "HAND should be non-negative"
        
        # Stream pixels should have HAND close to zero
        stream_hand = hand[hand_calculator.streams]
        stream_hand = stream_hand[~np.isnan(stream_hand)]
        if len(stream_hand) > 0:
            assert np.mean(stream_hand) < 5, "Stream pixels should have low HAND values"
        
        # Non-stream pixels should generally have higher HAND values
        non_stream_mask = ~hand_calculator.streams
        non_stream_hand = hand[non_stream_mask]
        non_stream_hand = non_stream_hand[~np.isnan(non_stream_hand)]
        if len(non_stream_hand) > 0 and len(stream_hand) > 0:
            assert np.mean(non_stream_hand) > np.mean(stream_hand)

    def test_find_nearest_stream(self, hand_calculator):
        """Test nearest stream pixel finding algorithm."""
        # Test finding nearest stream from a known location
        test_row, test_col = 10, 10
        
        nearest_row, nearest_col = hand_calculator._find_nearest_stream(
            test_row, test_col, max_pixels=20
        )
        
        if nearest_row is not None and nearest_col is not None:
            # Verify the found pixel is actually a stream
            assert hand_calculator.streams[nearest_row, nearest_col]
            
            # Verify it's within the search radius
            distance = np.sqrt((nearest_row - test_row)**2 + (nearest_col - test_col)**2)
            assert distance <= 20


class TestIntegrationWorkflows:
    """Test complete preprocessing workflows with realistic scenarios."""

    @pytest.fixture
    def complex_terrain_file(self, tmp_path):
        """Create a complex terrain file for integration testing."""
        height, width = 200, 200
        x = np.linspace(0, 20, width)
        y = np.linspace(0, 20, height)
        X, Y = np.meshgrid(x, y)
        
        # Create complex terrain with multiple features
        dem = (
            500 +  # Base elevation
            100 * np.sin(X/5) * np.cos(Y/5) +  # Large-scale topography
            50 * np.sin(X/2) * np.cos(Y/3) +   # Medium-scale features
            20 * np.random.normal(0, 1, (height, width)) +  # Noise
            200 * np.exp(-((X - 10)**2 + (Y - 15)**2) / 20) +  # Mountain
            -100 * np.exp(-((X - 15)**2 + (Y - 5)**2) / 10)    # Valley
        )
        
        # Add stream network
        stream1 = (Y > 9) & (Y < 11) & (X > 5) & (X < 18)
        stream2 = (X > 9) & (X < 11) & (Y > 2) & (Y < 15)
        dem[stream1 | stream2] -= 30
        
        # Add some artificial artifacts for testing robustness
        dem[50:55, 50:55] = np.nan  # No-data region
        dem[100:105, 100:105] -= 50  # Artificial sink
        
        dem_path = tmp_path / "complex_terrain.tif"
        
        transform = from_bounds(
            west=0, south=0, east=2000, north=2000,
            width=width, height=height
        )
        
        with rasterio.open(
            dem_path, 'w',
            driver='GTiff',
            height=height, width=width, count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(32633),  # UTM Zone 33N
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(dem.astype(np.float32), 1)
        
        return str(dem_path)

    def test_complete_preprocessing_pipeline(self, complex_terrain_file, tmp_path):
        """Test the complete preprocessing pipeline from DEM to features."""
        # Step 1: Hydrological conditioning
        conditioner = HydrologicalConditioner(complex_terrain_file)
        conditioning_results = conditioner.condition_dem(
            stream_threshold=100, buffer_distance=30
        )
        
        # Step 2: Terrain feature extraction
        conditioned_dem = conditioning_results['conditioned_dem']
        pixel_size = abs(conditioner.dem_profile['transform'][0])
        
        feature_extractor = TerrainFeatureExtractor(
            conditioned_dem, pixel_size, 
            nodata_value=conditioner.dem_profile.get('nodata')
        )
        
        features = feature_extractor.extract_all_features(
            flow_direction=conditioning_results['flow_direction']
        )
        
        # Step 3: HAND calculation
        hand_calc = HANDCalculator(
            conditioned_dem, 
            conditioning_results['flow_accumulation'],
            pixel_size, 
            stream_threshold=100
        )
        hand = hand_calc.calculate_hand(max_distance=500)
        
        # Validation
        assert len(features) == 11  # All expected features
        assert hand.shape == conditioned_dem.shape
        
        # Check that features have reasonable ranges
        assert np.nanmax(features['slope_degrees']) <= 90
        assert np.nanmin(features['flow_accumulation']) >= 0
        assert np.nanmin(hand) >= 0
        
        # Save results for inspection
        output_dir = tmp_path / "pipeline_output"
        conditioner.save_results(conditioning_results, str(output_dir))
        
        # Verify output files exist and are readable
        for filename in conditioning_results.keys():
            output_file = output_dir / f"{filename}.tif"
            assert output_file.exists()
            
            with rasterio.open(output_file) as src:
                data = src.read(1)
                assert data.shape == conditioned_dem.shape

    def test_pipeline_with_missing_data(self, complex_terrain_file):
        """Test pipeline robustness with missing/invalid data."""
        conditioner = HydrologicalConditioner(complex_terrain_file)
        
        # Introduce more no-data values
        original_dem = conditioner.dem_array.copy()
        conditioner.dem_array[::10, ::10] = conditioner.dem_profile.get('nodata', -9999)
        
        try:
            results = conditioner.condition_dem()
            
            # Pipeline should complete without errors
            assert 'conditioned_dem' in results
            assert not np.all(np.isnan(results['conditioned_dem']))
            
        except Exception as e:
            pytest.fail(f"Pipeline failed with missing data: {e}")
        
        # Restore original data
        conditioner.dem_array = original_dem

    def test_performance_with_large_data(self, tmp_path):
        """Test pipeline performance with larger datasets."""
        # Create a moderately large DEM (500x500)
        height, width = 500, 500
        large_dem = np.random.normal(1000, 100, (height, width)).astype(np.float32)
        
        # Add some structure
        x = np.linspace(0, 50, width)
        y = np.linspace(0, 50, height)
        X, Y = np.meshgrid(x, y)
        large_dem += 200 * np.sin(X/10) * np.cos(Y/10)
        
        dem_path = tmp_path / "large_dem.tif"
        
        transform = from_bounds(
            west=0, south=0, east=5000, north=5000,
            width=width, height=height
        )
        
        with rasterio.open(
            dem_path, 'w',
            driver='GTiff',
            height=height, width=width, count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(32633),
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(large_dem, 1)
        
        # Time the preprocessing pipeline
        import time
        start_time = time.time()
        
        conditioner = HydrologicalConditioner(str(dem_path))
        results = conditioner.condition_dem(stream_threshold=1000)
        
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert processing_time < 60, f"Processing took too long: {processing_time} seconds"
        
        # Results should be valid
        assert results['conditioned_dem'].shape == (height, width)
        assert not np.all(np.isnan(results['conditioned_dem']))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
