"""
Flood Risk Validation Framework

A comprehensive validation system for flood risk models including:
- Quantitative metrics (IoU, MAE, RMSE, CSI, F1)
- Model comparison with LISFLOOD-FP
- Insurance claims validation with NFIP data
- Interactive visualizations
- Automated report generation

Usage:
    from src.validation import ValidationFramework
    
    validator = ValidationFramework(config_path='config/validation.yaml')
    results = validator.validate_model(predictions, observations)
"""

from .metrics import (
    MetricsCalculator,
    IoUCalculator,
    RegressionMetrics,
    ClassificationMetrics,
    CriticalSuccessIndex
)

from .lisflood_validator import LISFLOODValidator
from .nfip_validator import NFIPValidator
from .visualization import FloodVisualization
from .report_generator import ValidationReportGenerator

__version__ = "1.0.0"
__author__ = "Flood Risk Team"

__all__ = [
    'MetricsCalculator',
    'IoUCalculator', 
    'RegressionMetrics',
    'ClassificationMetrics',
    'CriticalSuccessIndex',
    'LISFLOODValidator',
    'NFIPValidator',
    'FloodVisualization',
    'ValidationReportGenerator'
]


class ValidationFramework:
    """
    Main validation framework orchestrator
    """
    
    def __init__(self, config_path=None):
        """Initialize validation framework with configuration"""
        self.config = self._load_config(config_path)
        self.metrics = MetricsCalculator()
        self.lisflood = LISFLOODValidator()
        self.nfip = NFIPValidator()
        self.viz = FloodVisualization()
        self.reporter = ValidationReportGenerator()
    
    def _load_config(self, config_path):
        """Load validation configuration"""
        if config_path is None:
            return self._default_config()
        
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self):
        """Default validation configuration"""
        return {
            'metrics': {
                'calculate_iou': True,
                'calculate_csi': True,
                'calculate_regression': True,
                'calculate_classification': True
            },
            'validation': {
                'lisflood_comparison': True,
                'nfip_validation': True
            },
            'visualization': {
                'generate_maps': True,
                'interactive_plots': True
            },
            'reporting': {
                'generate_html': True,
                'generate_pdf': False,
                'include_visualizations': True
            }
        }
    
    def validate_model(self, predictions, observations, metadata=None):
        """
        Complete model validation pipeline
        
        Args:
            predictions: Model predictions (numpy array or path to file)
            observations: Ground truth observations (numpy array or path to file)
            metadata: Additional metadata for validation
        
        Returns:
            ValidationResults object with all metrics and visualizations
        """
        results = {}
        
        # Calculate metrics
        if self.config['metrics']['calculate_iou']:
            results['iou'] = self.metrics.calculate_iou(predictions, observations)
        
        if self.config['metrics']['calculate_csi']:
            results['csi'] = self.metrics.calculate_csi(predictions, observations)
        
        if self.config['metrics']['calculate_regression']:
            results['regression'] = self.metrics.calculate_regression_metrics(predictions, observations)
        
        if self.config['metrics']['calculate_classification']:
            results['classification'] = self.metrics.calculate_classification_metrics(predictions, observations)
        
        # Generate visualizations
        if self.config['visualization']['generate_maps']:
            results['visualizations'] = self.viz.create_comparison_map(predictions, observations, metadata)
        
        # Generate report
        if self.config['reporting']['generate_html']:
            results['report'] = self.reporter.generate_html_report(results, metadata)
        
        return ValidationResults(results)


class ValidationResults:
    """Container for validation results"""
    
    def __init__(self, results):
        self.results = results
        self._metrics = results.get('metrics', {})
        self._visualizations = results.get('visualizations', {})
        self._report = results.get('report', None)
    
    def get_metric(self, metric_name):
        """Get specific metric value"""
        return self._metrics.get(metric_name)
    
    def get_summary(self):
        """Get summary of all metrics"""
        summary = {}
        for key, value in self._metrics.items():
            if isinstance(value, dict):
                summary.update(value)
            else:
                summary[key] = value
        return summary
    
    def save_report(self, output_path):
        """Save validation report to file"""
        if self._report:
            with open(output_path, 'w') as f:
                f.write(self._report)
    
    def __repr__(self):
        return f"ValidationResults(metrics={len(self._metrics)}, visualizations={len(self._visualizations)})"