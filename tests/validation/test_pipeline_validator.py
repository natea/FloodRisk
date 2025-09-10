"""
Comprehensive Test Suite for Pipeline Validation Framework

Tests for:
- DEM validation functionality
- Rainfall data validation
- Spatial consistency validation
- Simulation results validation
- Tile quality validation
- Pipeline orchestration
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import xarray as xr
from datetime import datetime
import json
from unittest.mock import Mock, patch, MagicMock

from src.validation.pipeline_validator import (
    DEMValidator, RainfallValidator, SpatialConsistencyValidator,
    SimulationValidator, TileQualityValidator, PipelineValidator,
    ValidationResult
)


class TestDEMValidator:
    """Test suite for DEM quality validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DEMValidator({
            'elevation_bounds': (-100, 3000),
            'void_threshold': 0.1,
            'smoothness_threshold': 50
        })
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def create_test_dem(self, height=100, width=100, elevation_range=(0, 1000), 
                       void_fraction=0.0, add_noise=False):
        """Create test DEM file"""
        # Generate elevation data
        np.random.seed(42)
        elevations = np.random.uniform(
            elevation_range[0], elevation_range[1], (height, width)
        ).astype(np.float32)
        
        # Add spatial correlation (smoothness)
        from scipy.ndimage import gaussian_filter
        elevations = gaussian_filter(elevations, sigma=2)
        
        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, 100, elevations.shape)
            elevations += noise
        
        # Add voids
        if void_fraction > 0:
            n_voids = int(height * width * void_fraction)
            void_indices = np.random.choice(height * width, n_voids, replace=False)
            elevations.flat[void_indices] = np.nan
        
        # Create raster file
        dem_path = self.temp_dir / 'test_dem.tif'
        
        with rasterio.open(
            dem_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=1, dtype=rasterio.float32,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-120, 30, -110, 40, width, height)
        ) as dst:
            dst.write(elevations, 1)
        
        return dem_path
    
    def test_valid_dem(self):
        """Test validation of a good quality DEM"""
        dem_path = self.create_test_dem(elevation_range=(10, 500))
        
        result = self.validator.validate(dem_path)
        
        assert result.component == 'DEM_Quality'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert 'elevation_stats' in result.details
        assert 'void_analysis' in result.details
        assert 'spatial_analysis' in result.details
        assert len(result.issues) == 0
    
    def test_dem_with_voids(self):
        """Test DEM with excessive voids"""
        dem_path = self.create_test_dem(void_fraction=0.15)  # 15% voids
        
        result = self.validator.validate(dem_path)
        
        assert result.status in ['WARN', 'FAIL']
        assert result.score < 0.8
        assert any('void' in issue.lower() for issue in result.issues)
    
    def test_dem_extreme_elevations(self):
        """Test DEM with out-of-bounds elevations"""
        dem_path = self.create_test_dem(elevation_range=(-200, 4000))  # Outside bounds
        
        result = self.validator.validate(dem_path)
        
        assert result.status in ['WARN', 'FAIL']
        assert result.score < 1.0
        assert any('elevation' in issue.lower() for issue in result.issues)
    
    def test_noisy_dem(self):
        """Test DEM with high noise/gradients"""
        dem_path = self.create_test_dem(add_noise=True)
        
        result = self.validator.validate(dem_path)
        
        assert 'spatial_analysis' in result.details
        # May or may not fail depending on noise level, but should have gradient info
        assert 'max_gradient' in result.details['spatial_analysis']
    
    def test_missing_file(self):
        """Test handling of missing DEM file"""
        missing_path = self.temp_dir / 'nonexistent.tif'
        
        result = self.validator.validate(missing_path)
        
        assert result.status == 'FAIL'
        assert result.score == 0.0
        assert 'error' in result.details
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestRainfallValidator:
    """Test suite for rainfall data validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = RainfallValidator({
            'max_intensity': 200,
            'min_coverage': 0.9,
            'missing_data_threshold': 0.15
        })
    
    def create_test_rainfall_array(self, shape=(50, 50), intensity_range=(0, 50),
                                 missing_fraction=0.0, add_negatives=False,
                                 add_temporal_dim=False):
        """Create test rainfall data"""
        np.random.seed(42)
        
        if add_temporal_dim:
            shape = (10,) + shape  # Add time dimension
        
        rainfall = np.random.uniform(
            intensity_range[0], intensity_range[1], shape
        ).astype(np.float32)
        
        # Add missing values
        if missing_fraction > 0:
            n_missing = int(np.prod(shape) * missing_fraction)
            missing_indices = np.random.choice(np.prod(shape), n_missing, replace=False)
            rainfall.flat[missing_indices] = np.nan
        
        # Add negative values if requested
        if add_negatives:
            negative_indices = np.random.choice(np.prod(shape), size=10, replace=False)
            rainfall.flat[negative_indices] = -np.abs(rainfall.flat[negative_indices])
        
        return rainfall
    
    def test_valid_rainfall_array(self):
        """Test validation of good quality rainfall data"""
        rainfall = self.create_test_rainfall_array()
        
        result = self.validator.validate(rainfall)
        
        assert result.component == 'Rainfall_Quality'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert 'value_stats' in result.details
        assert 'coverage_analysis' in result.details
    
    def test_rainfall_with_negatives(self):
        """Test rainfall with negative values"""
        rainfall = self.create_test_rainfall_array(add_negatives=True)
        
        result = self.validator.validate(rainfall)
        
        assert result.score < 1.0
        assert any('negative' in issue.lower() for issue in result.issues)
    
    def test_rainfall_extreme_values(self):
        """Test rainfall with extreme intensities"""
        rainfall = self.create_test_rainfall_array(intensity_range=(0, 300))  # Exceeds threshold
        
        result = self.validator.validate(rainfall)
        
        assert result.score < 1.0
        assert any('extreme' in issue.lower() or 'intensity' in issue.lower() for issue in result.issues)
    
    def test_rainfall_high_missing_data(self):
        """Test rainfall with excessive missing data"""
        rainfall = self.create_test_rainfall_array(missing_fraction=0.2)  # 20% missing
        
        result = self.validator.validate(rainfall)
        
        assert result.score < 1.0
        assert any('missing' in issue.lower() for issue in result.issues)
    
    def test_rainfall_temporal_data(self):
        """Test rainfall with temporal dimension"""
        rainfall = self.create_test_rainfall_array(add_temporal_dim=True)
        
        result = self.validator.validate(rainfall)
        
        assert 'temporal_analysis' in result.details
        assert result.details['has_temporal_dimension'] is True
    
    def test_xarray_rainfall(self):
        """Test rainfall data as xarray DataArray"""
        rainfall_array = self.create_test_rainfall_array(add_temporal_dim=True)
        
        # Create xarray DataArray
        rainfall_da = xr.DataArray(
            rainfall_array,
            dims=['time', 'y', 'x'],
            coords={
                'time': pd.date_range('2023-01-01', periods=10, freq='H'),
                'y': np.arange(50),
                'x': np.arange(50)
            }
        )
        
        result = self.validator.validate(rainfall_da)
        
        assert result.component == 'Rainfall_Quality'
        assert 'dimensions' in result.details
        assert result.details['has_temporal_dimension'] is True


class TestSpatialConsistencyValidator:
    """Test suite for spatial consistency validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SpatialConsistencyValidator({
            'spatial_tolerance': 0.1,
            'min_overlap': 0.9
        })
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def create_test_raster(self, filename, bounds=(-120, 30, -110, 40), 
                          shape=(100, 100), crs=CRS.from_epsg(4326)):
        """Create test raster file"""
        data = np.random.rand(*shape).astype(np.float32)
        raster_path = self.temp_dir / filename
        
        with rasterio.open(
            raster_path, 'w',
            driver='GTiff',
            height=shape[0], width=shape[1],
            count=1, dtype=rasterio.float32,
            crs=crs,
            transform=from_bounds(*bounds, shape[1], shape[0])
        ) as dst:
            dst.write(data, 1)
        
        return raster_path
    
    def test_consistent_datasets(self):
        """Test datasets with consistent spatial properties"""
        # Create datasets with same CRS, resolution, and overlapping extents
        raster1 = self.create_test_raster('raster1.tif')
        raster2 = self.create_test_raster('raster2.tif')
        
        datasets = [
            {'path': raster1, 'name': 'DEM', 'type': 'elevation'},
            {'path': raster2, 'name': 'Rainfall', 'type': 'precipitation'}
        ]
        
        result = self.validator.validate(datasets)
        
        assert result.component == 'Spatial_Consistency'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert result.details['crs_consistent'] is True
    
    def test_inconsistent_crs(self):
        """Test datasets with different CRS"""
        raster1 = self.create_test_raster('raster1.tif', crs=CRS.from_epsg(4326))
        raster2 = self.create_test_raster('raster2.tif', crs=CRS.from_epsg(3857))
        
        datasets = [
            {'path': raster1, 'name': 'DEM', 'type': 'elevation'},
            {'path': raster2, 'name': 'Rainfall', 'type': 'precipitation'}
        ]
        
        result = self.validator.validate(datasets)
        
        assert result.score < 1.0
        assert result.details['crs_consistent'] is False
        assert any('CRS' in issue or 'crs' in issue.lower() for issue in result.issues)
    
    def test_non_overlapping_extents(self):
        """Test datasets with non-overlapping extents"""
        raster1 = self.create_test_raster('raster1.tif', bounds=(-120, 30, -110, 40))
        raster2 = self.create_test_raster('raster2.tif', bounds=(-100, 50, -90, 60))  # No overlap
        
        datasets = [
            {'path': raster1, 'name': 'DEM', 'type': 'elevation'},
            {'path': raster2, 'name': 'Rainfall', 'type': 'precipitation'}
        ]
        
        result = self.validator.validate(datasets)
        
        assert result.score < 1.0
        assert any('overlap' in issue.lower() for issue in result.issues)
    
    def test_different_resolutions(self):
        """Test datasets with different resolutions"""
        raster1 = self.create_test_raster('raster1.tif', shape=(100, 100))  # Fine resolution
        raster2 = self.create_test_raster('raster2.tif', shape=(50, 50))    # Coarse resolution
        
        datasets = [
            {'path': raster1, 'name': 'DEM', 'type': 'elevation'},
            {'path': raster2, 'name': 'Rainfall', 'type': 'precipitation'}
        ]
        
        result = self.validator.validate(datasets)
        
        # Should detect resolution inconsistency
        assert not result.details['resolution_analysis']['consistent']
    
    def test_insufficient_datasets(self):
        """Test with insufficient number of datasets"""
        raster1 = self.create_test_raster('raster1.tif')
        datasets = [{'path': raster1, 'name': 'DEM', 'type': 'elevation'}]
        
        result = self.validator.validate(datasets)
        
        assert result.status == 'WARN'
        assert 'Insufficient datasets' in result.issues[0]
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestSimulationValidator:
    """Test suite for simulation results validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SimulationValidator({
            'max_depth': 20.0,
            'mass_conservation_tolerance': 0.05,
            'convergence_threshold': 1e-6
        })
    
    def create_test_simulation_results(self, add_negatives=False, extreme_depths=False,
                                     add_convergence=True, add_mass_balance=True):
        """Create test simulation results"""
        np.random.seed(42)
        
        # Generate depth field
        depths = np.random.exponential(0.5, (50, 50)).astype(np.float32)
        
        if add_negatives:
            depths[0:5, 0:5] = -1.0  # Add some negative depths
        
        if extreme_depths:
            depths[10:15, 10:15] = 100.0  # Add extreme depths
        
        # Generate velocity field
        velocities = np.random.normal(0, 2, (2, 50, 50)).astype(np.float32)
        
        results = {
            'depths': depths,
            'velocities': velocities
        }
        
        if add_convergence:
            results['convergence'] = {
                'final_residual': 1e-7,
                'iterations': 50,
                'residuals': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
            }
        
        if add_mass_balance:
            total_volume = np.sum(depths[depths > 0])
            results['inflow'] = total_volume * 1.02  # 2% mass balance error
            results['outflow'] = total_volume * 0.1
        
        return results
    
    def test_valid_simulation_results(self):
        """Test validation of good simulation results"""
        results = self.create_test_simulation_results()
        
        validation_result = self.validator.validate(results)
        
        assert validation_result.component == 'Simulation_Quality'
        assert validation_result.status in ['PASS', 'WARN']  # May warn on mass balance
        assert 'depth_analysis' in validation_result.details
        assert 'convergence_analysis' in validation_result.details
    
    def test_simulation_with_negative_depths(self):
        """Test simulation with negative depths"""
        results = self.create_test_simulation_results(add_negatives=True)
        
        validation_result = self.validator.validate(results)
        
        assert validation_result.score < 1.0
        assert any('negative' in issue.lower() for issue in validation_result.issues)
    
    def test_simulation_extreme_depths(self):
        """Test simulation with extreme depths"""
        results = self.create_test_simulation_results(extreme_depths=True)
        
        validation_result = self.validator.validate(results)
        
        assert validation_result.score < 1.0
        assert any('extreme' in issue.lower() for issue in validation_result.issues)
    
    def test_poor_convergence(self):
        """Test simulation with poor convergence"""
        results = self.create_test_simulation_results()
        results['convergence']['final_residual'] = 1e-3  # Poor convergence
        results['convergence']['iterations'] = 1000  # Too many iterations
        
        validation_result = self.validator.validate(results)
        
        assert validation_result.score < 1.0
        assert any('converge' in issue.lower() for issue in validation_result.issues)
    
    def test_mass_conservation_issues(self):
        """Test simulation with mass conservation problems"""
        results = self.create_test_simulation_results()
        # Create large mass balance error
        results['inflow'] = 1000.0
        results['outflow'] = 100.0
        
        validation_result = self.validator.validate(results)
        
        assert validation_result.score < 1.0
        assert any('mass' in issue.lower() or 'conservation' in issue.lower() 
                  for issue in validation_result.issues)


class TestTileQualityValidator:
    """Test suite for tile quality validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = TileQualityValidator({
            'target_flood_ratio': (0.1, 0.9),
            'min_tiles': 50,
            'edge_threshold': 3
        })
    
    def create_test_tiles(self, num_tiles=100, tile_size=(64, 64), 
                         flood_ratios=None, inconsistent_sizes=False):
        """Create test tiles data"""
        np.random.seed(42)
        
        tiles = []
        
        for i in range(num_tiles):
            if inconsistent_sizes and i % 10 == 0:
                size = (32, 32)  # Different size for some tiles
            else:
                size = tile_size
            
            # Generate tile data
            if flood_ratios:
                flood_ratio = flood_ratios[min(i, len(flood_ratios)-1)]
            else:
                flood_ratio = np.random.uniform(0.1, 0.9)  # Good range
            
            tile_data = np.random.uniform(0, 1, size)
            # Set flood ratio by thresholding
            threshold = np.percentile(tile_data, (1 - flood_ratio) * 100)
            tile_data = (tile_data > threshold).astype(float) * np.random.uniform(0.1, 2.0, size)
            
            tiles.append({'data': tile_data})
        
        return {
            'tiles': tiles,
            'metadata': {
                'tile_size': tile_size,
                'overlap': 0.1,
                'projection': 'EPSG:4326',
                'bounds': [-120, 30, -110, 40]
            }
        }
    
    def test_valid_tiles(self):
        """Test validation of good quality tiles"""
        tiles_info = self.create_test_tiles()
        
        result = self.validator.validate(tiles_info)
        
        assert result.component == 'Tile_Quality'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert 'flood_balance' in result.details
        assert 'tile_count' in result.details
    
    def test_insufficient_tiles(self):
        """Test with insufficient number of tiles"""
        tiles_info = self.create_test_tiles(num_tiles=20)  # Below threshold
        
        result = self.validator.validate(tiles_info)
        
        assert result.score < 1.0
        assert any('insufficient' in issue.lower() for issue in result.issues)
    
    def test_poor_flood_balance(self):
        """Test tiles with poor flood/dry balance"""
        # Create tiles with poor balance (all very low flood ratios)
        poor_ratios = [0.01] * 50 + [0.02] * 50  # All very low flood ratios
        tiles_info = self.create_test_tiles(flood_ratios=poor_ratios)
        
        result = self.validator.validate(tiles_info)
        
        assert result.score < 1.0
        assert any('balance' in issue.lower() for issue in result.issues)
    
    def test_extreme_tiles(self):
        """Test with too many extreme (all-flood or all-dry) tiles"""
        # Create many extreme tiles
        extreme_ratios = [0.0] * 40 + [1.0] * 40 + [0.5] * 20
        tiles_info = self.create_test_tiles(flood_ratios=extreme_ratios)
        
        result = self.validator.validate(tiles_info)
        
        assert result.score < 1.0
        assert any('extreme' in issue.lower() for issue in result.issues)
    
    def test_inconsistent_tile_sizes(self):
        """Test tiles with inconsistent sizes"""
        tiles_info = self.create_test_tiles(inconsistent_sizes=True)
        
        result = self.validator.validate(tiles_info)
        
        assert result.score < 1.0
        assert not result.details['tile_sizes']['size_consistent']
        assert any('inconsistent' in issue.lower() for issue in result.issues)
    
    def test_empty_tiles_data(self):
        """Test with no tiles"""
        tiles_info = {'tiles': [], 'metadata': {}}
        
        result = self.validator.validate(tiles_info)
        
        assert result.status == 'FAIL'
        assert result.score == 0.0


class TestPipelineValidator:
    """Test suite for overall pipeline validation orchestration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = PipelineValidator()
    
    @patch('src.validation.pipeline_validator.rasterio.open')
    def test_full_pipeline_validation(self, mock_rasterio_open):
        """Test complete pipeline validation"""
        # Mock rasterio file reading
        mock_dataset = MagicMock()
        mock_dataset.read.return_value = np.random.rand(100, 100)
        mock_dataset.transform = from_bounds(-120, 30, -110, 40, 100, 100)
        mock_dataset.crs = CRS.from_epsg(4326)
        mock_dataset.bounds = (-120, 30, -110, 40)
        mock_dataset.shape = (100, 100)
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        pipeline_data = {
            'dem_path': '/fake/path/dem.tif',
            'rainfall_data': np.random.uniform(0, 50, (50, 50)),
            'spatial_datasets': [
                {'path': '/fake/path/dem.tif', 'name': 'DEM', 'type': 'elevation'},
                {'path': '/fake/path/rainfall.tif', 'name': 'Rainfall', 'type': 'precipitation'}
            ],
            'simulation_results': {
                'depths': np.random.exponential(0.5, (50, 50)),
                'convergence': {'final_residual': 1e-7, 'iterations': 50}
            },
            'tiles_info': {
                'tiles': [{'data': np.random.rand(64, 64)} for _ in range(100)],
                'metadata': {'tile_size': (64, 64)}
            }
        }
        
        results = self.validator.validate_full_pipeline(pipeline_data)
        
        assert len(results) > 0
        assert all(isinstance(result, ValidationResult) for result in results.values())
        assert self.validator.get_pipeline_status() in ['PASS', 'WARN', 'FAIL']
    
    def test_partial_pipeline_validation(self):
        """Test pipeline validation with only some components"""
        pipeline_data = {
            'rainfall_data': np.random.uniform(0, 50, (50, 50))
        }
        
        results = self.validator.validate_full_pipeline(pipeline_data)
        
        assert 'rainfall' in results
        assert len(results) == 1
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        # Add some mock results
        self.validator.results = [
            ValidationResult(
                component='Test_Component',
                status='PASS',
                score=0.9,
                details={'test': 'data'},
                issues=[],
                timestamp=datetime.now()
            )
        ]
        
        report = self.validator.generate_validation_report()
        
        assert 'validation_summary' in report
        assert 'component_results' in report
        assert 'recommendations' in report
        assert report['validation_summary']['overall_score'] == 0.9
    
    def test_empty_pipeline_validation(self):
        """Test pipeline validation with no data"""
        results = self.validator.validate_full_pipeline({})
        
        assert len(results) == 0
        assert self.validator.get_pipeline_status() == 'UNKNOWN'


class TestValidationResult:
    """Test suite for ValidationResult data structure"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and serialization"""
        result = ValidationResult(
            component='Test_Component',
            status='PASS',
            score=0.95,
            details={'metric1': 0.8, 'metric2': 'good'},
            issues=['Minor issue'],
            timestamp=datetime.now()
        )
        
        assert result.component == 'Test_Component'
        assert result.status == 'PASS'
        assert result.score == 0.95
        assert len(result.issues) == 1
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['component'] == 'Test_Component'
        assert isinstance(result_dict['timestamp'], str)  # ISO format


class TestIntegration:
    """Integration tests for the complete validation system"""
    
    def test_real_world_workflow_simulation(self):
        """Simulate a complete real-world validation workflow"""
        # This test would simulate the entire pipeline with realistic data
        # In practice, this would use actual DEM files, rainfall data, etc.
        
        # Create pipeline validator
        validator = PipelineValidator()
        
        # Simulate pipeline data (in real use, this would be actual file paths and data)
        with patch('src.validation.pipeline_validator.rasterio.open') as mock_rasterio:
            # Mock DEM file
            mock_dataset = MagicMock()
            mock_dataset.read.return_value = np.random.uniform(10, 1000, (100, 100))
            mock_dataset.transform = from_bounds(-120, 30, -110, 40, 100, 100)
            mock_dataset.crs = CRS.from_epsg(4326)
            mock_dataset.bounds = (-120, 30, -110, 40)
            mock_dataset.shape = (100, 100)
            mock_rasterio.return_value.__enter__.return_value = mock_dataset
            
            pipeline_data = {
                'dem_path': '/mock/dem.tif',
                'rainfall_data': np.random.uniform(0, 30, (100, 100)),
                'spatial_datasets': [
                    {'path': '/mock/dem.tif', 'name': 'DEM', 'type': 'elevation'}
                ],
                'simulation_results': {
                    'depths': np.random.exponential(0.3, (100, 100)),
                    'velocities': np.random.normal(0, 1, (2, 100, 100)),
                    'convergence': {
                        'final_residual': 1e-8,
                        'iterations': 45,
                        'residuals': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
                    },
                    'inflow': 1000.0,
                    'outflow': 950.0
                },
                'tiles_info': {
                    'tiles': [{
                        'data': np.random.uniform(0, 1, (64, 64)) > 0.7
                    } for _ in range(150)],
                    'metadata': {
                        'tile_size': (64, 64),
                        'overlap': 0.1,
                        'projection': 'EPSG:4326'
                    }
                }
            }
            
            # Run complete validation
            results = validator.validate_full_pipeline(pipeline_data)
            
            # Verify results
            assert len(results) >= 4  # At least DEM, rainfall, simulation, tiles
            
            # Generate report
            report = validator.generate_validation_report()
            assert report is not None
            assert 'validation_summary' in report
            
            # Check overall pipeline status
            status = validator.get_pipeline_status()
            assert status in ['PASS', 'WARN', 'FAIL']


# Fixtures and utilities for testing
@pytest.fixture
def sample_validation_results():
    """Fixture providing sample validation results for testing"""
    return [
        ValidationResult(
            component='DEM_Quality',
            status='PASS',
            score=0.95,
            details={'elevation_stats': {'min': 10, 'max': 1000}},
            issues=[],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='Rainfall_Quality', 
            status='WARN',
            score=0.75,
            details={'value_stats': {'min': 0, 'max': 150}},
            issues=['Some extreme values detected'],
            timestamp=datetime.now()
        ),
        ValidationResult(
            component='Spatial_Consistency',
            status='PASS',
            score=0.90,
            details={'crs_consistent': True},
            issues=[],
            timestamp=datetime.now()
        )
    ]


def test_with_sample_results(sample_validation_results):
    """Test using the sample results fixture"""
    assert len(sample_validation_results) == 3
    assert sample_validation_results[0].status == 'PASS'
    assert sample_validation_results[1].status == 'WARN'


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
