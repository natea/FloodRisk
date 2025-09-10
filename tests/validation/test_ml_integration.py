"""
Test Suite for ML Integration Validation

Tests for:
- ML data format validation
- Model performance validation
- Label quality validation
- ML pipeline integration
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.validation.ml_integration_validator import (
    MLDataValidator, ModelPerformanceValidator, LabelQualityValidator,
    MLIntegrationValidator
)
from src.validation.pipeline_validator import ValidationResult


class MockFloodDataset(Dataset):
    """Mock dataset for testing ML validation"""
    
    def __init__(self, size=1000, input_shape=(3, 256, 256), output_shape=(1, 256, 256),
                 inconsistent_format=False, memory_intensive=False):
        self.size = size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inconsistent_format = inconsistent_format
        self.memory_intensive = memory_intensive
        
        # Generate consistent data
        np.random.seed(42)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        np.random.seed(idx)  # For reproducible "random" data
        
        if self.inconsistent_format and idx % 10 == 0:
            # Return inconsistent format for some items
            return torch.randn(2, 128, 128), torch.randn(2, 128, 128)
        
        if self.memory_intensive:
            # Create very large tensors
            input_data = torch.randn(self.input_shape[0], 1024, 1024)
            output_data = torch.randn(self.output_shape[0], 1024, 1024)
        else:
            input_data = torch.randn(*self.input_shape)
            output_data = torch.randn(*self.output_shape)
            
        return input_data, output_data


class TestMLDataValidator:
    """Test suite for ML data format validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = MLDataValidator({
            'expected_input_shape': (3, 256, 256),
            'expected_output_shape': (1, 256, 256),
            'batch_size': 16,
            'max_memory_gb': 8
        })
    
    def test_valid_dataset(self):
        """Test validation of properly formatted dataset"""
        dataset = MockFloodDataset(size=100)
        
        result = self.validator.validate(dataset)
        
        assert result.component == 'ML_Data_Compatibility'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert result.details['dataset_size'] == 100
        assert result.details['sample_validation']['format_consistent']
        assert result.details['sample_validation']['shape_valid']
    
    def test_inconsistent_format(self):
        """Test dataset with inconsistent data format"""
        dataset = MockFloodDataset(size=100, inconsistent_format=True)
        
        result = self.validator.validate(dataset)
        
        assert result.score < 1.0
        assert not result.details['sample_validation']['format_consistent']
        assert any('inconsistent' in issue.lower() for issue in result.issues)
    
    def test_wrong_shape(self):
        """Test dataset with wrong tensor shapes"""
        dataset = MockFloodDataset(
            size=50,
            input_shape=(5, 128, 128),  # Wrong number of channels
            output_shape=(2, 128, 128)  # Wrong number of output channels
        )
        
        result = self.validator.validate(dataset)
        
        assert result.score < 1.0
        assert not result.details['sample_validation']['shape_valid']
        assert any('shape' in issue.lower() for issue in result.issues)
    
    def test_memory_intensive_dataset(self):
        """Test dataset with high memory requirements"""
        dataset = MockFloodDataset(size=1000, memory_intensive=True)
        
        result = self.validator.validate(dataset)
        
        # Should warn about high memory usage
        if 'memory_analysis' in result.details and 'estimated_memory_gb' in result.details['memory_analysis']:
            memory_gb = result.details['memory_analysis']['estimated_memory_gb']
            if memory_gb > 8:  # Exceeds our threshold
                assert result.score < 1.0
                assert any('memory' in issue.lower() for issue in result.issues)
    
    def test_dataloader_compatibility(self):
        """Test DataLoader compatibility"""
        dataset = MockFloodDataset(size=50)
        
        result = self.validator.validate(dataset)
        
        assert 'dataloader_validation' in result.details
        assert result.details['dataloader_validation']['batch_loading_success']
    
    def test_empty_dataset(self):
        """Test empty dataset"""
        dataset = MockFloodDataset(size=0)
        
        result = self.validator.validate(dataset)
        
        assert result.status == 'FAIL'
        assert result.score == 0.0
        assert 'empty' in result.issues[0].lower()
    
    def test_data_type_validation(self):
        """Test data type validation"""
        dataset = MockFloodDataset(size=10)
        
        result = self.validator.validate(dataset)
        
        assert 'dtype_validation' in result.details
        assert result.details['dtype_validation']['dtypes_valid']


class TestModelPerformanceValidator:
    """Test suite for model performance validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ModelPerformanceValidator({
            'min_accuracy': 0.7,
            'max_overfitting_gap': 0.1,
            'min_convergence_epochs': 10,
            'inference_speed_threshold': 100  # ms
        })
    
    def create_training_history(self, converged=True, overfitting=False, epochs=20):
        """Create mock training history"""
        np.random.seed(42)
        
        if converged:
            # Decreasing loss that stabilizes
            train_loss = [1.0 - (i * 0.8) / epochs + np.random.normal(0, 0.01) for i in range(epochs)]
            train_loss = np.maximum(train_loss, 0.1)  # Floor at 0.1
        else:
            # Oscillating loss that doesn't converge
            train_loss = [0.5 + 0.3 * np.sin(i * 0.5) + np.random.normal(0, 0.05) for i in range(epochs)]
        
        if overfitting:
            # Validation loss increases while training decreases
            val_loss = train_loss[:10] + [train_loss[9] + (i-9) * 0.05 for i in range(10, epochs)]
        else:
            # Good validation loss
            val_loss = [l + np.random.normal(0, 0.02) for l in train_loss]
        
        # Generate accuracies (inverse relationship with loss)
        train_accuracy = [1 - l + np.random.normal(0, 0.01) for l in train_loss]
        val_accuracy = [1 - l + np.random.normal(0, 0.01) for l in val_loss]
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }
    
    def test_good_model_performance(self):
        """Test validation of well-performing model"""
        model_results = {
            'training_history': self.create_training_history(converged=True, overfitting=False),
            'test_metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'iou': 0.78
            },
            'inference_metrics': {
                'avg_time_ms': 50.0,
                'max_time_ms': 80.0,
                'memory_mb': 512.0,
                'throughput': 20.0
            },
            'model_info': {
                'parameters': 15_000_000,  # 15M parameters
                'size_mb': 60.0,
                'architecture': 'U-Net',
                'input_shape': (3, 256, 256),
                'output_shape': (1, 256, 256)
            }
        }
        
        result = self.validator.validate(model_results)
        
        assert result.component == 'Model_Performance'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert result.details['training_analysis']['converged']
        assert not result.details['training_analysis']['overfitting_detected']
        assert result.details['performance_analysis']['meets_accuracy_threshold']
    
    def test_non_convergent_model(self):
        """Test model that didn't converge"""
        model_results = {
            'training_history': self.create_training_history(converged=False, overfitting=False),
            'test_metrics': {'accuracy': 0.65}  # Below threshold
        }
        
        result = self.validator.validate(model_results)
        
        assert result.score < 1.0
        assert not result.details['training_analysis']['converged']
        assert any('converge' in issue.lower() for issue in result.issues)
    
    def test_overfitting_model(self):
        """Test model with overfitting"""
        model_results = {
            'training_history': self.create_training_history(converged=True, overfitting=True),
            'test_metrics': {'accuracy': 0.75}
        }
        
        result = self.validator.validate(model_results)
        
        assert result.score < 1.0
        assert result.details['training_analysis']['overfitting_detected']
        assert any('overfitting' in issue.lower() for issue in result.issues)
    
    def test_low_accuracy_model(self):
        """Test model with poor test accuracy"""
        model_results = {
            'test_metrics': {
                'accuracy': 0.60,  # Below threshold
                'precision': 0.58,
                'recall': 0.62,
                'f1_score': 0.60
            }
        }
        
        result = self.validator.validate(model_results)
        
        assert result.score < 1.0
        assert not result.details['performance_analysis']['meets_accuracy_threshold']
        assert any('accuracy' in issue.lower() for issue in result.issues)
    
    def test_slow_inference(self):
        """Test model with slow inference"""
        model_results = {
            'test_metrics': {'accuracy': 0.85},
            'inference_metrics': {
                'avg_time_ms': 150.0,  # Above threshold
                'max_time_ms': 200.0,
                'throughput': 5.0
            }
        }
        
        result = self.validator.validate(model_results)
        
        assert result.score < 1.0
        assert not result.details['inference_analysis']['meets_speed_threshold']
        assert any('slow' in issue.lower() or 'speed' in issue.lower() for issue in result.issues)
    
    def test_real_vs_dummy_comparison(self):
        """Test real vs dummy data performance comparison"""
        model_results = {
            'test_metrics': {'accuracy': 0.80},
            'real_data_metrics': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84
            },
            'dummy_data_metrics': {
                'accuracy': 0.75,  # Lower than real (good)
                'precision': 0.73,
                'recall': 0.77
            }
        }
        
        result = self.validator.validate(model_results)
        
        comparison = result.details['real_vs_dummy_comparison']
        assert comparison['real_significantly_better']  # Real should be better
        assert not comparison['dummy_better']
    
    def test_dummy_better_than_real(self):
        """Test concerning case where dummy data performs better"""
        model_results = {
            'test_metrics': {'accuracy': 0.75},
            'real_data_metrics': {
                'accuracy': 0.70  # Real data worse
            },
            'dummy_data_metrics': {
                'accuracy': 0.78  # Dummy better (concerning)
            }
        }
        
        result = self.validator.validate(model_results)
        
        assert result.score < 1.0
        comparison = result.details['real_vs_dummy_comparison']
        assert comparison['dummy_better']
        assert any('dummy' in issue.lower() for issue in result.issues)


class TestLabelQualityValidator:
    """Test suite for label quality validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = LabelQualityValidator({
            'min_class_ratio': 0.01,
            'max_class_ratio': 0.99,
            'spatial_coherence_threshold': 0.8
        })
    
    def create_binary_labels(self, shape=(256, 256), flood_ratio=0.3, 
                           add_noise=False, spatially_coherent=True):
        """Create binary flood labels"""
        np.random.seed(42)
        
        if spatially_coherent:
            # Create spatially coherent labels (flood regions)
            from scipy.ndimage import gaussian_filter
            
            # Start with random noise
            base = np.random.rand(*shape)
            # Smooth it to create coherent regions
            smoothed = gaussian_filter(base, sigma=10)
            # Threshold to get desired flood ratio
            threshold = np.percentile(smoothed, (1 - flood_ratio) * 100)
            labels = (smoothed > threshold).astype(int)
        else:
            # Create spatially incoherent labels (salt and pepper)
            labels = (np.random.rand(*shape) < flood_ratio).astype(int)
        
        if add_noise:
            # Flip random pixels
            noise_mask = np.random.rand(*shape) < 0.05  # 5% noise
            labels[noise_mask] = 1 - labels[noise_mask]
        
        return labels
    
    def test_good_quality_labels(self):
        """Test well-balanced, spatially coherent labels"""
        labels = self.create_binary_labels(flood_ratio=0.4, spatially_coherent=True)
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert result.component == 'Label_Quality'
        assert result.status == 'PASS'
        assert result.score >= 0.8
        assert not result.details['class_balance']['imbalance_severe']
        assert result.details['spatial_analysis']['meets_threshold']
    
    def test_severe_class_imbalance(self):
        """Test labels with severe class imbalance"""
        labels = self.create_binary_labels(flood_ratio=0.005)  # Only 0.5% flood
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert result.score < 1.0
        assert result.details['class_balance']['imbalance_severe']
        assert any('imbalance' in issue.lower() for issue in result.issues)
    
    def test_spatially_incoherent_labels(self):
        """Test labels with poor spatial coherence"""
        labels = self.create_binary_labels(spatially_coherent=False)
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert result.score < 1.0
        assert not result.details['spatial_analysis']['meets_threshold']
        assert any('coherence' in issue.lower() for issue in result.issues)
    
    def test_noisy_labels(self):
        """Test labels with significant noise"""
        labels = self.create_binary_labels(add_noise=True)
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        # Check noise detection
        if result.details['noise_analysis']['estimated_noise_ratio'] > 0.1:
            assert result.score < 1.0
            assert any('noise' in issue.lower() for issue in result.issues)
    
    def test_multiclass_labels(self):
        """Test multi-class labels"""
        np.random.seed(42)
        # Create 3-class labels: background, shallow flood, deep flood
        labels = np.random.choice([0, 1, 2], size=(100, 100), p=[0.5, 0.3, 0.2])
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert result.details['class_balance']['unique_classes'] == 3
    
    def test_continuous_labels(self):
        """Test continuous depth labels"""
        np.random.seed(42)
        # Create continuous flood depth labels
        labels = np.random.exponential(0.5, size=(100, 100))
        labels[labels < 0.1] = 0  # Set shallow areas to dry
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert 'value_analysis' in result.details
        assert result.details['value_analysis']['range_info'] == 'regression_positive'
    
    def test_invalid_label_values(self):
        """Test labels with invalid value ranges"""
        # Create labels with negative values (invalid for flood depths)
        labels = np.random.normal(0, 1, size=(50, 50))
        labels_data = {'labels': labels}
        
        result = self.validator.validate(labels_data)
        
        assert not result.details['value_analysis']['valid_range']
        assert result.details['value_analysis']['has_negative_values']
    
    def test_empty_labels(self):
        """Test empty labels data"""
        labels_data = {}
        
        result = self.validator.validate(labels_data)
        
        assert result.status == 'FAIL'
        assert result.score == 0.0
        assert 'No labels' in result.issues[0]


class TestMLIntegrationValidator:
    """Test suite for ML integration validation orchestration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = MLIntegrationValidator()
    
    def test_complete_ml_pipeline_validation(self):
        """Test complete ML pipeline validation"""
        # Create test data
        dataset = MockFloodDataset(size=100)
        labels = np.random.choice([0, 1], size=(100, 256, 256), p=[0.7, 0.3])
        
        ml_data = {
            'dataset': dataset,
            'labels_data': {'labels': labels},
            'model_results': {
                'training_history': {
                    'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.19, 0.18],
                    'val_loss': [1.1, 0.85, 0.65, 0.45, 0.35, 0.30, 0.28, 0.26, 0.25, 0.24],
                    'train_accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87],
                    'val_accuracy': [0.48, 0.58, 0.68, 0.73, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84]
                },
                'test_metrics': {
                    'accuracy': 0.83,
                    'precision': 0.81,
                    'recall': 0.85,
                    'f1_score': 0.83
                }
            }
        }
        
        results = self.validator.validate_ml_pipeline(ml_data)
        
        assert len(results) == 3  # Data, performance, labels
        assert 'ml_data' in results
        assert 'model_performance' in results
        assert 'label_quality' in results
        
        # Check overall status
        status = self.validator.get_ml_pipeline_status()
        assert status in ['PASS', 'WARN', 'FAIL']
    
    def test_partial_ml_validation(self):
        """Test ML validation with only some components"""
        ml_data = {
            'dataset': MockFloodDataset(size=50)
        }
        
        results = self.validator.validate_ml_pipeline(ml_data)
        
        assert len(results) == 1
        assert 'ml_data' in results
    
    def test_ml_report_generation(self):
        """Test ML validation report generation"""
        # Add mock results
        self.validator.results = [
            ValidationResult(
                component='ML_Data_Compatibility',
                status='PASS',
                score=0.9,
                details={'dataset_size': 1000},
                issues=[],
                timestamp=datetime.now()
            ),
            ValidationResult(
                component='Model_Performance',
                status='WARN',
                score=0.75,
                details={'performance_analysis': {'accuracy': 0.75}},
                issues=['Low accuracy'],
                timestamp=datetime.now()
            )
        ]
        
        report = self.validator.generate_ml_validation_report()
        
        assert 'ml_validation_summary' in report
        assert 'ml_component_results' in report
        assert 'ml_recommendations' in report
        assert report['ml_validation_summary']['overall_ml_score'] == 0.825  # Average of 0.9 and 0.75
    
    def test_failed_ml_pipeline(self):
        """Test ML pipeline with failures"""
        # Dataset with incompatible format
        dataset = MockFloodDataset(size=50, inconsistent_format=True)
        
        # Poor labels
        labels = np.random.choice([0, 1], size=(50, 256, 256), p=[0.999, 0.001])  # Severe imbalance
        
        ml_data = {
            'dataset': dataset,
            'labels_data': {'labels': labels},
            'model_results': {
                'test_metrics': {'accuracy': 0.5}  # Poor performance
            }
        }
        
        results = self.validator.validate_ml_pipeline(ml_data)
        
        # Should have issues in multiple components
        failed_components = [r for r in results.values() if r.status == 'FAIL']
        warning_components = [r for r in results.values() if r.status == 'WARN']
        
        assert len(failed_components) > 0 or len(warning_components) > 0
        
        # Overall status should reflect issues
        status = self.validator.get_ml_pipeline_status()
        assert status in ['WARN', 'FAIL']


class TestIntegrationMLValidation:
    """Integration tests for ML validation system"""
    
    def test_realistic_ml_workflow(self):
        """Test realistic ML validation workflow"""
        # Create realistic dataset
        class RealisticFloodDataset(Dataset):
            def __init__(self, size=500):
                self.size = size
                np.random.seed(42)
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Simulate multi-channel input (DEM, rainfall, land use)
                dem_channel = torch.randn(1, 256, 256) * 500 + 200  # Elevation
                rainfall_channel = torch.exponential(torch.ones(1, 256, 256) * 0.1) * 50  # Rainfall
                landuse_channel = torch.randint(0, 5, (1, 256, 256)).float()  # Land use categories
                
                input_tensor = torch.cat([dem_channel, rainfall_channel, landuse_channel], dim=0)
                
                # Generate realistic flood depth target
                # Higher rainfall + lower elevation = more flood
                flood_potential = rainfall_channel[0] - (dem_channel[0] - 200) / 1000
                flood_depth = torch.clamp(flood_potential, 0, 5)  # Max 5m depth
                
                return input_tensor, flood_depth.unsqueeze(0)
        
        # Create realistic training history
        realistic_history = {
            'train_loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30],
            'val_loss': [2.7, 2.0, 1.4, 1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.40],
            'train_accuracy': [0.45, 0.55, 0.65, 0.72, 0.78, 0.83, 0.86, 0.88, 0.89, 0.90],
            'val_accuracy': [0.43, 0.53, 0.63, 0.70, 0.76, 0.80, 0.83, 0.85, 0.86, 0.87]
        }
        
        # Realistic labels (flood depths)
        np.random.seed(42)
        realistic_labels = np.random.exponential(0.3, size=(100, 256, 256))
        realistic_labels[realistic_labels < 0.05] = 0  # Set shallow to dry
        realistic_labels = np.clip(realistic_labels, 0, 10)  # Max 10m depth
        
        # Complete ML validation
        validator = MLIntegrationValidator()
        
        ml_data = {
            'dataset': RealisticFloodDataset(),
            'labels_data': {'labels': realistic_labels},
            'model_results': {
                'training_history': realistic_history,
                'test_metrics': {
                    'accuracy': 0.86,
                    'precision': 0.83,
                    'recall': 0.89,
                    'f1_score': 0.86,
                    'iou': 0.76,
                    'mae': 0.15,  # Mean absolute error for depth
                    'rmse': 0.32   # Root mean square error
                },
                'real_data_metrics': {
                    'accuracy': 0.86,
                    'iou': 0.76
                },
                'dummy_data_metrics': {
                    'accuracy': 0.72,  # Dummy performs worse (good)
                    'iou': 0.62
                },
                'inference_metrics': {
                    'avg_time_ms': 75.0,
                    'max_time_ms': 120.0,
                    'memory_mb': 1024.0,
                    'throughput': 13.3
                },
                'model_info': {
                    'parameters': 25_000_000,
                    'size_mb': 95.0,
                    'architecture': 'DeepLabV3+',
                    'input_shape': (3, 256, 256),
                    'output_shape': (1, 256, 256)
                }
            }
        }
        
        # Run validation
        results = validator.validate_ml_pipeline(ml_data)
        
        # Verify comprehensive validation
        assert len(results) == 3
        
        # Generate report
        report = validator.generate_ml_validation_report()
        assert 'ml_validation_summary' in report
        
        # Should pass with realistic good data
        status = validator.get_ml_pipeline_status()
        assert status == 'PASS'


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
