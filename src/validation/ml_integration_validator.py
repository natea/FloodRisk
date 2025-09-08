"""
ML Pipeline Integration Validator

Validates integration between flood simulation pipeline and ML training infrastructure:
- Data format compatibility
- Training pipeline integration
- Model performance validation
- Label quality validation  
- Inference pipeline validation
- Real vs dummy data comparison
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .pipeline_validator import BaseValidator, ValidationResult
from .metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLDataValidator(BaseValidator):
    """
    ML Data Format and Compatibility Validator
    
    Validates:
    - Data format compatibility with ML pipeline
    - Feature tensor shapes and types
    - Label tensor shapes and types
    - Data loader compatibility
    - Memory requirements
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.expected_input_shape = config.get('expected_input_shape', (256, 256, 3))  # H, W, C
        self.expected_output_shape = config.get('expected_output_shape', (256, 256, 1))
        self.batch_size = config.get('batch_size', 32)
        self.max_memory_gb = config.get('max_memory_gb', 16)
        
    def validate(self, dataset: Any, **kwargs) -> ValidationResult:
        """Validate ML dataset compatibility"""
        try:
            issues = []
            details = {}
            score = 1.0
            
            # 1. Dataset structure validation
            if hasattr(dataset, '__len__'):
                dataset_size = len(dataset)
                details['dataset_size'] = dataset_size
                
                if dataset_size == 0:
                    issues.append("Dataset is empty")
                    score = 0.0
                    return self._create_result('ML_Data_Compatibility', 'FAIL', score, details, issues)
                
                # Sample a few items to check format
                sample_indices = np.random.choice(min(10, dataset_size), size=min(5, dataset_size), replace=False)
                sample_validation = self._validate_samples(dataset, sample_indices)
                details['sample_validation'] = sample_validation
                
                if not sample_validation['format_consistent']:
                    issues.append("Inconsistent data format across samples")
                    score -= 0.4
                
                if not sample_validation['shape_valid']:
                    issues.append(f"Invalid tensor shapes: expected input {self.expected_input_shape}, output {self.expected_output_shape}")
                    score -= 0.3
            
            # 2. Memory requirements validation
            if hasattr(dataset, '__getitem__'):
                memory_analysis = self._analyze_memory_requirements(dataset)
                details['memory_analysis'] = memory_analysis
                
                if memory_analysis['estimated_memory_gb'] > self.max_memory_gb:
                    issues.append(f"High memory requirement ({memory_analysis['estimated_memory_gb']:.2f}GB) exceeds limit")
                    score -= 0.2
            
            # 3. Data loader compatibility
            try:
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                batch_validation = self._validate_dataloader(dataloader)
                details['dataloader_validation'] = batch_validation
                
                if not batch_validation['batch_loading_success']:
                    issues.append("DataLoader batch loading failed")
                    score -= 0.3
            except Exception as e:
                issues.append(f"DataLoader creation failed: {e}")
                score -= 0.3
            
            # 4. Data type validation
            dtype_validation = self._validate_data_types(dataset)
            details['dtype_validation'] = dtype_validation
            
            if not dtype_validation['dtypes_valid']:
                issues.append("Invalid data types for ML training")
                score -= 0.2
            
            # Determine status
            if score >= 0.8:
                status = 'PASS'
            elif score >= 0.6:
                status = 'WARN'
            else:
                status = 'FAIL'
            
            self.logger.info(f"ML data validation completed: {status} (score: {score:.3f})")
            return self._create_result('ML_Data_Compatibility', status, max(0.0, score), details, issues)
            
        except Exception as e:
            self.logger.error(f"ML data validation failed: {e}")
            return self._create_result(
                'ML_Data_Compatibility', 'FAIL', 0.0,
                {'error': str(e)}, [f"Validation failed: {e}"]
            )
    
    def _validate_samples(self, dataset: Any, indices: List[int]) -> Dict[str, Any]:
        """Validate sample format consistency"""
        sample_shapes = []
        sample_types = []
        format_consistent = True
        shape_valid = True
        
        try:
            for idx in indices:
                sample = dataset[idx]
                
                if isinstance(sample, (tuple, list)) and len(sample) == 2:
                    input_data, target_data = sample
                    
                    # Check input shape
                    if hasattr(input_data, 'shape'):
                        input_shape = input_data.shape
                        sample_shapes.append(('input', input_shape))
                        
                        # Validate against expected shape (allowing batch dimension flexibility)
                        expected_shape = self.expected_input_shape
                        if len(input_shape) == len(expected_shape):
                            if input_shape[-len(expected_shape):] != expected_shape:
                                shape_valid = False
                        elif len(input_shape) == len(expected_shape) + 1:  # with batch dim
                            if input_shape[-len(expected_shape):] != expected_shape:
                                shape_valid = False
                        else:
                            shape_valid = False
                    
                    # Check target shape
                    if hasattr(target_data, 'shape'):
                        target_shape = target_data.shape
                        sample_shapes.append(('target', target_shape))
                        
                        # Similar validation for target shape
                        expected_shape = self.expected_output_shape
                        if len(target_shape) == len(expected_shape):
                            if target_shape[-len(expected_shape):] != expected_shape:
                                shape_valid = False
                        elif len(target_shape) == len(expected_shape) + 1:  # with batch dim
                            if target_shape[-len(expected_shape):] != expected_shape:
                                shape_valid = False
                        else:
                            shape_valid = False
                    
                    # Check data types
                    sample_types.append({
                        'input_type': str(type(input_data)),
                        'target_type': str(type(target_data))
                    })
                else:
                    format_consistent = False
                    
        except Exception as e:
            format_consistent = False
            sample_shapes.append(('error', str(e)))
        
        return {
            'format_consistent': format_consistent,
            'shape_valid': shape_valid,
            'sample_shapes': sample_shapes,
            'sample_types': sample_types
        }
    
    def _analyze_memory_requirements(self, dataset: Any) -> Dict[str, Any]:
        """Analyze memory requirements for dataset"""
        try:
            # Sample a few items to estimate memory usage
            sample_item = dataset[0]
            
            if isinstance(sample_item, (tuple, list)) and len(sample_item) == 2:
                input_data, target_data = sample_item
                
                # Calculate memory per item
                input_memory = 0
                target_memory = 0
                
                if hasattr(input_data, 'nbytes'):
                    input_memory = input_data.nbytes
                elif hasattr(input_data, 'element_size'):
                    input_memory = input_data.numel() * input_data.element_size()
                
                if hasattr(target_data, 'nbytes'):
                    target_memory = target_data.nbytes
                elif hasattr(target_data, 'element_size'):
                    target_memory = target_data.numel() * target_data.element_size()
                
                bytes_per_item = input_memory + target_memory
                dataset_size = len(dataset)
                total_bytes = bytes_per_item * dataset_size
                
                # Account for batch loading
                batch_memory = bytes_per_item * self.batch_size
                
                return {
                    'bytes_per_item': bytes_per_item,
                    'estimated_total_bytes': total_bytes,
                    'estimated_memory_gb': total_bytes / (1024**3),
                    'batch_memory_mb': batch_memory / (1024**2)
                }
        except Exception as e:
            return {'error': str(e)}
        
        return {'error': 'Could not analyze memory requirements'}
    
    def _validate_dataloader(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Validate DataLoader functionality"""
        try:
            # Try to load a batch
            batch = next(iter(dataloader))
            
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
                
                return {
                    'batch_loading_success': True,
                    'batch_input_shape': list(inputs.shape) if hasattr(inputs, 'shape') else 'unknown',
                    'batch_target_shape': list(targets.shape) if hasattr(targets, 'shape') else 'unknown',
                    'batch_size': len(inputs) if hasattr(inputs, '__len__') else 'unknown'
                }
            else:
                return {
                    'batch_loading_success': False,
                    'error': 'Batch format not as expected (input, target) tuple'
                }
                
        except Exception as e:
            return {
                'batch_loading_success': False,
                'error': str(e)
            }
    
    def _validate_data_types(self, dataset: Any) -> Dict[str, Any]:
        """Validate data types for ML compatibility"""
        try:
            sample = dataset[0]
            
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                input_data, target_data = sample
                
                # Check if data is tensor-like
                input_valid = hasattr(input_data, 'dtype') and hasattr(input_data, 'shape')
                target_valid = hasattr(target_data, 'dtype') and hasattr(target_data, 'shape')
                
                # Check for float types (typically required for ML)
                input_float = False
                target_numeric = False
                
                if input_valid:
                    input_float = 'float' in str(input_data.dtype)
                
                if target_valid:
                    target_numeric = any(t in str(target_data.dtype) for t in ['float', 'int', 'long'])
                
                return {
                    'dtypes_valid': input_valid and target_valid and input_float and target_numeric,
                    'input_dtype': str(input_data.dtype) if input_valid else 'unknown',
                    'target_dtype': str(target_data.dtype) if target_valid else 'unknown',
                    'input_tensor_like': input_valid,
                    'target_tensor_like': target_valid
                }
            
            return {'dtypes_valid': False, 'error': 'Sample format not as expected'}
            
        except Exception as e:
            return {'dtypes_valid': False, 'error': str(e)}


class ModelPerformanceValidator(BaseValidator):
    """
    ML Model Performance Validator
    
    Validates:
    - Training convergence
    - Model performance metrics
    - Overfitting detection
    - Real vs dummy data performance comparison
    - Inference speed and accuracy
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_accuracy = config.get('min_accuracy', 0.7)
        self.max_overfitting_gap = config.get('max_overfitting_gap', 0.1)  # 10% gap
        self.min_convergence_epochs = config.get('min_convergence_epochs', 10)
        self.inference_speed_threshold = config.get('inference_speed_threshold', 100)  # ms per sample
        
    def validate(self, model_results: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate model performance"""
        try:
            issues = []
            details = {}
            score = 1.0
            
            # 1. Training metrics validation
            if 'training_history' in model_results:
                training_analysis = self._analyze_training_history(model_results['training_history'])
                details['training_analysis'] = training_analysis
                
                if not training_analysis['converged']:
                    issues.append("Model did not converge during training")
                    score -= 0.4
                
                if training_analysis['overfitting_detected']:
                    issues.append(f"Overfitting detected (gap: {training_analysis['train_val_gap']:.3f})")
                    score -= 0.2
            
            # 2. Final performance validation
            if 'test_metrics' in model_results:
                performance_analysis = self._analyze_test_performance(model_results['test_metrics'])
                details['performance_analysis'] = performance_analysis
                
                if performance_analysis['accuracy'] < self.min_accuracy:
                    issues.append(f"Low test accuracy ({performance_analysis['accuracy']:.3f}) below threshold")
                    score -= 0.3
            
            # 3. Real vs dummy data comparison
            if 'real_data_metrics' in model_results and 'dummy_data_metrics' in model_results:
                comparison_analysis = self._compare_real_vs_dummy(
                    model_results['real_data_metrics'],
                    model_results['dummy_data_metrics']
                )
                details['real_vs_dummy_comparison'] = comparison_analysis
                
                if comparison_analysis['real_significantly_better']:
                    # This is actually good - real data should perform better
                    pass
                elif comparison_analysis['dummy_better']:
                    issues.append("Dummy data performs better than real data - possible data quality issues")
                    score -= 0.3
                else:
                    issues.append("No significant performance difference between real and dummy data")
                    score -= 0.1
            
            # 4. Inference speed validation
            if 'inference_metrics' in model_results:
                inference_analysis = self._analyze_inference_performance(model_results['inference_metrics'])
                details['inference_analysis'] = inference_analysis
                
                if inference_analysis['avg_inference_time_ms'] > self.inference_speed_threshold:
                    issues.append(f"Slow inference speed ({inference_analysis['avg_inference_time_ms']:.1f}ms)")
                    score -= 0.1
            
            # 5. Model architecture validation
            if 'model_info' in model_results:
                architecture_analysis = self._validate_model_architecture(model_results['model_info'])
                details['architecture_analysis'] = architecture_analysis
                
                if architecture_analysis['parameter_count'] > 100_000_000:  # 100M parameters
                    issues.append("Very large model size may cause deployment issues")
                    score -= 0.05
            
            # Determine status
            if score >= 0.8:
                status = 'PASS'
            elif score >= 0.6:
                status = 'WARN'
            else:
                status = 'FAIL'
            
            self.logger.info(f"Model performance validation completed: {status} (score: {score:.3f})")
            return self._create_result('Model_Performance', status, max(0.0, score), details, issues)
            
        except Exception as e:
            self.logger.error(f"Model performance validation failed: {e}")
            return self._create_result(
                'Model_Performance', 'FAIL', 0.0,
                {'error': str(e)}, [f"Validation failed: {e}"]
            )
    
    def _analyze_training_history(self, training_history: Dict[str, List]) -> Dict[str, Any]:
        """Analyze training convergence and overfitting"""
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        train_accuracies = training_history.get('train_accuracy', [])
        val_accuracies = training_history.get('val_accuracy', [])
        
        analysis = {
            'total_epochs': len(train_losses),
            'converged': False,
            'overfitting_detected': False,
            'train_val_gap': 0.0
        }
        
        if len(train_losses) >= self.min_convergence_epochs:
            # Check convergence (loss stabilization)
            recent_losses = train_losses[-5:] if len(train_losses) >= 5 else train_losses
            loss_std = np.std(recent_losses)
            analysis['converged'] = loss_std < 0.01  # Loss variation < 0.01
            analysis['final_train_loss'] = train_losses[-1]
            
            if val_losses:
                analysis['final_val_loss'] = val_losses[-1]
                
                # Check overfitting (val loss increasing while train loss decreasing)
                if len(val_losses) >= 10:
                    recent_val_trend = np.polyfit(range(5), val_losses[-5:], 1)[0]  # slope
                    recent_train_trend = np.polyfit(range(5), train_losses[-5:], 1)[0]
                    
                    analysis['overfitting_detected'] = recent_val_trend > 0 and recent_train_trend < 0
        
        # Calculate train-validation gap
        if train_accuracies and val_accuracies and len(train_accuracies) == len(val_accuracies):
            final_train_acc = train_accuracies[-1]
            final_val_acc = val_accuracies[-1]
            analysis['train_val_gap'] = final_train_acc - final_val_acc
            analysis['final_train_accuracy'] = final_train_acc
            analysis['final_val_accuracy'] = final_val_acc
            
            if analysis['train_val_gap'] > self.max_overfitting_gap:
                analysis['overfitting_detected'] = True
        
        return analysis
    
    def _analyze_test_performance(self, test_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze final test performance"""
        return {
            'accuracy': test_metrics.get('accuracy', 0.0),
            'precision': test_metrics.get('precision', 0.0),
            'recall': test_metrics.get('recall', 0.0),
            'f1_score': test_metrics.get('f1_score', 0.0),
            'iou': test_metrics.get('iou', 0.0),
            'meets_accuracy_threshold': test_metrics.get('accuracy', 0.0) >= self.min_accuracy
        }
    
    def _compare_real_vs_dummy(self, real_metrics: Dict[str, float], 
                             dummy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare performance on real vs dummy data"""
        real_acc = real_metrics.get('accuracy', 0.0)
        dummy_acc = dummy_metrics.get('accuracy', 0.0)
        
        accuracy_diff = real_acc - dummy_acc
        
        return {
            'real_accuracy': real_acc,
            'dummy_accuracy': dummy_acc,
            'accuracy_difference': accuracy_diff,
            'real_significantly_better': accuracy_diff > 0.05,  # 5% better
            'dummy_better': accuracy_diff < -0.02,  # Dummy 2% better (concerning)
            'real_metrics': real_metrics,
            'dummy_metrics': dummy_metrics
        }
    
    def _analyze_inference_performance(self, inference_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze inference speed and memory usage"""
        return {
            'avg_inference_time_ms': inference_metrics.get('avg_time_ms', 0.0),
            'max_inference_time_ms': inference_metrics.get('max_time_ms', 0.0),
            'memory_usage_mb': inference_metrics.get('memory_mb', 0.0),
            'throughput_samples_per_sec': inference_metrics.get('throughput', 0.0),
            'meets_speed_threshold': inference_metrics.get('avg_time_ms', float('inf')) <= self.inference_speed_threshold
        }
    
    def _validate_model_architecture(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model architecture and size"""
        return {
            'parameter_count': model_info.get('parameters', 0),
            'model_size_mb': model_info.get('size_mb', 0.0),
            'architecture': model_info.get('architecture', 'unknown'),
            'input_shape': model_info.get('input_shape', 'unknown'),
            'output_shape': model_info.get('output_shape', 'unknown')
        }


class LabelQualityValidator(BaseValidator):
    """
    Training Label Quality Validator
    
    Validates:
    - Label consistency and accuracy
    - Class balance in labels
    - Spatial coherence of flood labels
    - Label noise detection
    - Ground truth alignment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_class_ratio = config.get('min_class_ratio', 0.01)  # At least 1% of minority class
        self.max_class_ratio = config.get('max_class_ratio', 0.99)  # At most 99% of majority class
        self.spatial_coherence_threshold = config.get('spatial_coherence_threshold', 0.8)
        
    def validate(self, labels_data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate label quality"""
        try:
            issues = []
            details = {}
            score = 1.0
            
            labels = labels_data.get('labels')
            if labels is None:
                return self._create_result(
                    'Label_Quality', 'FAIL', 0.0,
                    {'error': 'No labels provided'}, ['No labels found']
                )
            
            # Convert to numpy if needed
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            
            # 1. Class balance validation
            class_analysis = self._analyze_class_balance(labels)
            details['class_balance'] = class_analysis
            
            if class_analysis['imbalance_severe']:
                issues.append(f"Severe class imbalance detected (minority class: {class_analysis['minority_ratio']:.3f})")
                score -= 0.3
            elif class_analysis['imbalance_moderate']:
                issues.append(f"Moderate class imbalance (minority class: {class_analysis['minority_ratio']:.3f})")
                score -= 0.1
            
            # 2. Label consistency validation
            consistency_analysis = self._check_label_consistency(labels)
            details['consistency_analysis'] = consistency_analysis
            
            if consistency_analysis['inconsistent_labels'] > 0.05:  # 5% inconsistent
                issues.append(f"High label inconsistency ({consistency_analysis['inconsistent_labels']*100:.1f}%)")
                score -= 0.2
            
            # 3. Spatial coherence validation (for image labels)
            if labels.ndim >= 2:
                spatial_analysis = self._analyze_spatial_coherence(labels)
                details['spatial_analysis'] = spatial_analysis
                
                if spatial_analysis['coherence_score'] < self.spatial_coherence_threshold:
                    issues.append(f"Low spatial coherence in labels ({spatial_analysis['coherence_score']:.3f})")
                    score -= 0.2
            
            # 4. Noise detection
            noise_analysis = self._detect_label_noise(labels)
            details['noise_analysis'] = noise_analysis
            
            if noise_analysis['estimated_noise_ratio'] > 0.1:  # 10% noise
                issues.append(f"High estimated label noise ({noise_analysis['estimated_noise_ratio']*100:.1f}%)")
                score -= 0.2
            
            # 5. Value range validation
            value_analysis = self._validate_label_values(labels)
            details['value_analysis'] = value_analysis
            
            if not value_analysis['valid_range']:
                issues.append("Labels contain values outside expected range")
                score -= 0.1
            
            # Determine status
            if score >= 0.8:
                status = 'PASS'
            elif score >= 0.6:
                status = 'WARN'
            else:
                status = 'FAIL'
            
            self.logger.info(f"Label quality validation completed: {status} (score: {score:.3f})")
            return self._create_result('Label_Quality', status, max(0.0, score), details, issues)
            
        except Exception as e:
            self.logger.error(f"Label quality validation failed: {e}")
            return self._create_result(
                'Label_Quality', 'FAIL', 0.0,
                {'error': str(e)}, [f"Validation failed: {e}"]
            )
    
    def _analyze_class_balance(self, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze class distribution and balance"""
        unique_values, counts = np.unique(labels.flatten(), return_counts=True)
        total_samples = len(labels.flatten())
        
        class_ratios = counts / total_samples
        minority_ratio = np.min(class_ratios)
        majority_ratio = np.max(class_ratios)
        
        return {
            'unique_classes': len(unique_values),
            'class_values': unique_values.tolist(),
            'class_counts': counts.tolist(),
            'class_ratios': class_ratios.tolist(),
            'minority_ratio': float(minority_ratio),
            'majority_ratio': float(majority_ratio),
            'imbalance_severe': minority_ratio < self.min_class_ratio,
            'imbalance_moderate': minority_ratio < 0.1,  # Less than 10%
            'balance_score': 1.0 - abs(0.5 - minority_ratio) if len(unique_values) == 2 else 1.0 - np.std(class_ratios)
        }
    
    def _check_label_consistency(self, labels: np.ndarray) -> Dict[str, Any]:
        """Check for label consistency issues"""
        # This is a simplified check - in practice, you'd have more sophisticated methods
        flat_labels = labels.flatten()
        
        # Check for invalid values (assuming binary or multi-class labels should be integers)
        valid_values = np.all(np.isfinite(flat_labels))
        
        # Count potential inconsistencies (e.g., isolated pixels in binary segmentation)
        inconsistent_count = 0
        if labels.ndim == 2:  # For 2D labels
            # Simple consistency check: count pixels that differ from all neighbors
            h, w = labels.shape
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = labels[i, j]
                    neighbors = labels[i-1:i+2, j-1:j+2]
                    if np.sum(neighbors == center) == 1:  # Only center pixel has this value
                        inconsistent_count += 1
        
        inconsistent_ratio = inconsistent_count / len(flat_labels) if len(flat_labels) > 0 else 0
        
        return {
            'valid_values': valid_values,
            'inconsistent_pixels': inconsistent_count,
            'inconsistent_labels': inconsistent_ratio,
            'total_pixels': len(flat_labels)
        }
    
    def _analyze_spatial_coherence(self, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial coherence of labels"""
        if labels.ndim not in [2, 3]:
            return {'error': 'Spatial analysis requires 2D or 3D labels'}
        
        # For each label patch, calculate coherence with neighbors
        from scipy.ndimage import generic_filter
        
        def coherence_filter(neighborhood):
            center = neighborhood[len(neighborhood)//2]
            return np.sum(neighborhood == center) / len(neighborhood)
        
        if labels.ndim == 2:
            coherence_map = generic_filter(labels, coherence_filter, size=3)
            coherence_score = np.mean(coherence_map)
        else:
            # For 3D (batch of 2D), compute average coherence
            coherence_scores = []
            for i in range(labels.shape[0]):
                coherence_map = generic_filter(labels[i], coherence_filter, size=3)
                coherence_scores.append(np.mean(coherence_map))
            coherence_score = np.mean(coherence_scores)
        
        return {
            'coherence_score': float(coherence_score),
            'meets_threshold': coherence_score >= self.spatial_coherence_threshold
        }
    
    def _detect_label_noise(self, labels: np.ndarray) -> Dict[str, Any]:
        """Detect potential label noise"""
        # Simplified noise detection - in practice, you'd use more sophisticated methods
        flat_labels = labels.flatten()
        
        # Statistical outlier detection for continuous labels
        if len(np.unique(flat_labels)) > 10:  # Treat as continuous
            q1, q3 = np.percentile(flat_labels, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((flat_labels < lower_bound) | (flat_labels > upper_bound))
            noise_ratio = outliers / len(flat_labels)
        else:
            # For discrete labels, check for spatial inconsistency as noise proxy
            if labels.ndim == 2:
                from scipy.ndimage import median_filter
                smoothed = median_filter(labels, size=3)
                noise_pixels = np.sum(labels != smoothed)
                noise_ratio = noise_pixels / labels.size
            else:
                noise_ratio = 0.0  # Default for other cases
        
        return {
            'estimated_noise_ratio': float(noise_ratio),
            'noise_pixels': int(noise_ratio * len(flat_labels)),
            'total_pixels': len(flat_labels)
        }
    
    def _validate_label_values(self, labels: np.ndarray) -> Dict[str, Any]:
        """Validate label value ranges"""
        flat_labels = labels.flatten()
        
        min_val = float(np.min(flat_labels))
        max_val = float(np.max(flat_labels))
        
        # Check if values are in expected range (adjust based on your label format)
        # For binary segmentation: [0, 1]
        # For multi-class: [0, num_classes-1]
        # For regression: positive values for depths
        
        valid_range = True
        range_info = "unknown"
        
        unique_values = np.unique(flat_labels)
        num_unique = len(unique_values)
        
        if num_unique == 2 and set(unique_values).issubset({0, 1}):
            # Binary classification
            valid_range = True
            range_info = "binary_classification"
        elif num_unique <= 20 and all(v >= 0 and v == int(v) for v in unique_values):
            # Multi-class classification
            valid_range = True
            range_info = "multi_class_classification"
        elif min_val >= 0:
            # Regression (depths should be positive)
            valid_range = True
            range_info = "regression_positive"
        else:
            valid_range = False
            range_info = "invalid_range"
        
        return {
            'valid_range': valid_range,
            'range_info': range_info,
            'min_value': min_val,
            'max_value': max_val,
            'unique_values': num_unique,
            'has_negative_values': min_val < 0,
            'has_non_integer_values': not all(v == int(v) for v in unique_values[:10])  # Check first 10
        }


class MLIntegrationValidator:
    """
    Main ML Integration Validator orchestrating all ML validation components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize validators
        self.data_validator = MLDataValidator(self.config.get('data', {}))
        self.performance_validator = ModelPerformanceValidator(self.config.get('performance', {}))
        self.label_validator = LabelQualityValidator(self.config.get('labels', {}))
        
        self.results = []
    
    def validate_ml_pipeline(self, ml_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Run complete ML pipeline validation
        
        Args:
            ml_data: Dictionary containing ML pipeline components
                - dataset: Training dataset
                - model_results: Model training and evaluation results
                - labels_data: Training labels information
        
        Returns:
            Dictionary of validation results by component
        """
        self.logger.info("Starting ML pipeline validation")
        validation_results = {}
        
        # 1. Data Format Validation
        if 'dataset' in ml_data:
            try:
                data_result = self.data_validator.validate(ml_data['dataset'])
                validation_results['ml_data'] = data_result
                self.results.append(data_result)
                self.logger.info(f"ML data validation: {data_result.status}")
            except Exception as e:
                self.logger.error(f"ML data validation failed: {e}")
        
        # 2. Model Performance Validation
        if 'model_results' in ml_data:
            try:
                perf_result = self.performance_validator.validate(ml_data['model_results'])
                validation_results['model_performance'] = perf_result
                self.results.append(perf_result)
                self.logger.info(f"Model performance validation: {perf_result.status}")
            except Exception as e:
                self.logger.error(f"Model performance validation failed: {e}")
        
        # 3. Label Quality Validation
        if 'labels_data' in ml_data:
            try:
                label_result = self.label_validator.validate(ml_data['labels_data'])
                validation_results['label_quality'] = label_result
                self.results.append(label_result)
                self.logger.info(f"Label quality validation: {label_result.status}")
            except Exception as e:
                self.logger.error(f"Label quality validation failed: {e}")
        
        self.logger.info("ML pipeline validation completed")
        return validation_results
    
    def get_ml_pipeline_status(self) -> str:
        """Get overall ML pipeline validation status"""
        if not self.results:
            return 'UNKNOWN'
        
        statuses = [r.status for r in self.results]
        
        if any(s == 'FAIL' for s in statuses):
            return 'FAIL'
        elif any(s == 'WARN' for s in statuses):
            return 'WARN'
        else:
            return 'PASS'
    
    def generate_ml_validation_report(self, output_path: Union[str, Path] = None) -> Dict[str, Any]:
        """Generate ML validation report"""
        if not self.results:
            return {}
        
        # Calculate ML-specific metrics
        ml_scores = [r.score for r in self.results if r.score is not None]
        ml_score = np.mean(ml_scores) if ml_scores else 0.0
        
        report = {
            'ml_validation_summary': {
                'overall_ml_score': ml_score,
                'ml_pipeline_status': self.get_ml_pipeline_status(),
                'components_validated': len(self.results),
                'timestamp': datetime.now().isoformat()
            },
            'ml_component_results': [result.to_dict() for result in self.results],
            'ml_recommendations': self._generate_ml_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"ML validation report saved to {output_path}")
        
        return report
    
    def _generate_ml_recommendations(self) -> List[str]:
        """Generate ML-specific recommendations"""
        recommendations = []
        
        for result in self.results:
            if result.component == 'ML_Data_Compatibility' and result.score < 0.8:
                recommendations.append("Review dataset format and compatibility with ML training pipeline")
            
            elif result.component == 'Model_Performance' and result.score < 0.8:
                recommendations.append("Improve model training: check convergence, overfitting, hyperparameters")
            
            elif result.component == 'Label_Quality' and result.score < 0.8:
                recommendations.append("Review label quality: check class balance, consistency, noise")
        
        if not recommendations:
            recommendations.append("ML pipeline validation passed - ready for production deployment")
        
        return recommendations
    
    def clear_results(self):
        """Clear ML validation results"""
        self.results.clear()
        self.logger.info("ML validation results cleared")
