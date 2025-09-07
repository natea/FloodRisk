"""
Flood Risk Visualization Module

Creates comprehensive visualizations for flood risk model validation including:
- Interactive flood maps with Folium
- Statistical plots with Plotly
- Comparison maps (model vs observations)
- Performance dashboards
- Spatial analysis plots
- Time series visualizations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
import json
from datetime import datetime
import base64
import io

try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    warnings.warn("folium not available - no interactive maps")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available - limited plotting functionality")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - fallback plotting only")

try:
    import rasterio
    from rasterio.plot import show
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodVisualization:
    """
    Main flood risk visualization class
    """
    
    def __init__(self, style: str = 'plotly_white'):
        """
        Initialize flood visualization system
        
        Args:
            style: Default plotting style
        """
        self.style = style
        if HAS_PLOTLY:
            pio.templates.default = style
        
        logger.info(f"FloodVisualization initialized with style: {style}")
    
    def create_comparison_map(self, predictions: np.ndarray, observations: np.ndarray,
                            metadata: Optional[Dict] = None, 
                            output_path: Optional[str] = None) -> Dict:
        """
        Create interactive comparison map of predictions vs observations
        
        Args:
            predictions: Model predictions
            observations: Ground truth observations
            metadata: Spatial metadata (bounds, CRS, etc.)
            output_path: Path to save HTML map
            
        Returns:
            Dictionary with map data and statistics
        """
        try:
            logger.info("Creating flood comparison map")
            
            if not HAS_FOLIUM:
                logger.warning("Folium not available - creating static comparison")
                return self._create_static_comparison(predictions, observations, metadata)
            
            # Get bounds from metadata or use defaults
            bounds = metadata.get('bounds', [-95, 25, -85, 35]) if metadata else [-95, 25, -85, 35]
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
            folium.TileLayer('CartoDB positron', name='CartoDB Light').add_to(m)
            
            # Create flood extent overlays
            pred_overlay = self._create_flood_overlay(predictions, bounds, 'Predictions', 'blue')
            obs_overlay = self._create_flood_overlay(observations, bounds, 'Observations', 'red')
            
            pred_overlay.add_to(m)
            obs_overlay.add_to(m)
            
            # Add difference overlay
            difference = predictions - observations
            diff_overlay = self._create_difference_overlay(difference, bounds)
            diff_overlay.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add metrics popup
            metrics_html = self._create_metrics_popup(predictions, observations)
            folium.Marker(
                [center_lat, center_lon],
                popup=folium.Popup(metrics_html, max_width=300),
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
            
            # Save map if output path provided
            if output_path:
                m.save(output_path)
                logger.info(f"Interactive map saved to: {output_path}")
            
            # Return map data
            return {
                'map_object': m,
                'html_string': m._repr_html_(),
                'bounds': bounds,
                'center': [center_lat, center_lon],
                'statistics': self._calculate_map_statistics(predictions, observations)
            }
            
        except Exception as e:
            logger.error(f"Error creating comparison map: {e}")
            raise RuntimeError(f"Map creation failed: {e}")
    
    def _create_flood_overlay(self, data: np.ndarray, bounds: List[float], 
                             name: str, color: str) -> folium.FeatureGroup:
        """
        Create flood extent overlay for folium map
        """
        # Create binary flood extent
        flood_extent = data > 0.01  # 1cm threshold
        
        # Convert to GeoJSON-like format (simplified)
        # In practice, this would use proper raster to vector conversion
        overlay = folium.FeatureGroup(name=name)
        
        # Add flood extent as rectangles (simplified representation)
        rows, cols = data.shape
        lat_step = (bounds[3] - bounds[1]) / rows
        lon_step = (bounds[2] - bounds[0]) / cols
        
        # Sample every nth cell to avoid too many features
        sample_rate = max(1, min(rows, cols) // 50)
        
        for i in range(0, rows, sample_rate):
            for j in range(0, cols, sample_rate):
                if flood_extent[i, j]:
                    lat = bounds[1] + i * lat_step
                    lon = bounds[0] + j * lon_step
                    
                    folium.Rectangle(
                        bounds=[[lat, lon], [lat + lat_step, lon + lon_step]],
                        color=color,
                        fill=True,
                        fillOpacity=0.3,
                        weight=0.5,
                        popup=f"{name}: {data[i, j]:.2f}m"
                    ).add_to(overlay)
        
        return overlay
    
    def _create_difference_overlay(self, difference: np.ndarray, bounds: List[float]) -> folium.FeatureGroup:
        """
        Create difference overlay showing prediction errors
        """
        overlay = folium.FeatureGroup(name='Prediction Errors')
        
        rows, cols = difference.shape
        lat_step = (bounds[3] - bounds[1]) / rows
        lon_step = (bounds[2] - bounds[0]) / cols
        
        # Sample for visualization
        sample_rate = max(1, min(rows, cols) // 50)
        
        for i in range(0, rows, sample_rate):
            for j in range(0, cols, sample_rate):
                diff_val = difference[i, j]
                if abs(diff_val) > 0.1:  # Only show significant differences
                    lat = bounds[1] + i * lat_step
                    lon = bounds[0] + j * lon_step
                    
                    # Color based on over/under prediction
                    color = 'red' if diff_val > 0 else 'blue'
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=min(abs(diff_val) * 5, 20),
                        popup=f"Error: {diff_val:.2f}m",
                        color=color,
                        fillOpacity=0.6
                    ).add_to(overlay)
        
        return overlay
    
    def _create_metrics_popup(self, predictions: np.ndarray, observations: np.ndarray) -> str:
        """
        Create HTML popup with validation metrics
        """
        from .metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(predictions, observations)
        
        html = """
        <div style="font-family: Arial, sans-serif; width: 250px;">
            <h4 style="color: #2E8B57;">Validation Metrics</h4>
        """
        
        # IoU metrics
        if 'iou' in metrics:
            iou_score = metrics['iou'].get('iou', 0)
            html += f"<p><strong>IoU Score:</strong> {iou_score:.3f}</p>"
        
        # Classification metrics
        if 'classification' in metrics and 'error' not in metrics['classification']:
            f1 = metrics['classification'].get('f1_score', 0)
            accuracy = metrics['classification'].get('accuracy', 0)
            html += f"<p><strong>F1 Score:</strong> {f1:.3f}</p>"
            html += f"<p><strong>Accuracy:</strong> {accuracy:.3f}</p>"
        
        # Regression metrics
        if 'regression' in metrics and 'error' not in metrics['regression']:
            rmse = metrics['regression'].get('rmse', 0)
            mae = metrics['regression'].get('mae', 0)
            html += f"<p><strong>RMSE:</strong> {rmse:.3f}m</p>"
            html += f"<p><strong>MAE:</strong> {mae:.3f}m</p>"
        
        html += "</div>"
        return html
    
    def _calculate_map_statistics(self, predictions: np.ndarray, observations: np.ndarray) -> Dict:
        """
        Calculate statistics for map display
        """
        return {
            'prediction_area': float(np.sum(predictions > 0.01)),
            'observation_area': float(np.sum(observations > 0.01)),
            'overlap_area': float(np.sum((predictions > 0.01) & (observations > 0.01))),
            'max_prediction': float(np.max(predictions)),
            'max_observation': float(np.max(observations))
        }
    
    def create_performance_dashboard(self, validation_results: Dict, 
                                   output_path: Optional[str] = None) -> Dict:
        """
        Create comprehensive performance dashboard with Plotly
        
        Args:
            validation_results: Results from validation analysis
            output_path: Path to save HTML dashboard
            
        Returns:
            Dictionary with dashboard components
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available - creating simple dashboard")
            return self._create_simple_dashboard(validation_results)
        
        try:
            logger.info("Creating performance dashboard")
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'IoU Score', 'Classification Metrics',
                    'Regression Metrics', 'ROC Curve',
                    'Confusion Matrix', 'Error Distribution'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "histogram"}]
                ]
            )
            
            # Extract metrics
            standard_metrics = validation_results.get('standard_metrics', {})
            
            # IoU Indicator
            iou_score = 0
            if 'iou' in standard_metrics:
                iou_score = standard_metrics['iou'].get('iou', 0)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=iou_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "IoU Score"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Classification metrics bar chart
            if 'classification' in standard_metrics and 'error' not in standard_metrics['classification']:
                class_metrics = standard_metrics['classification']
                metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
                metrics_values = [
                    class_metrics.get('precision', 0),
                    class_metrics.get('recall', 0),
                    class_metrics.get('f1_score', 0),
                    class_metrics.get('accuracy', 0)
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=metrics_names,
                        y=metrics_values,
                        marker_color='skyblue',
                        name='Classification'
                    ),
                    row=1, col=2
                )
            
            # Regression metrics
            if 'regression' in standard_metrics and 'error' not in standard_metrics['regression']:
                reg_metrics = standard_metrics['regression']
                reg_names = ['MAE', 'RMSE', 'R²', 'NSE']
                reg_values = [
                    reg_metrics.get('mae', 0),
                    reg_metrics.get('rmse', 0),
                    reg_metrics.get('r_squared', 0),
                    reg_metrics.get('nse', 0)
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=reg_names,
                        y=reg_values,
                        marker_color='lightcoral',
                        name='Regression'
                    ),
                    row=2, col=1
                )
            
            # ROC curve (if available)
            if 'roc_analysis' in validation_results:
                roc_data = validation_results['roc_analysis']
                fig.add_trace(
                    go.Scatter(
                        x=roc_data.get('fpr', []),
                        y=roc_data.get('tpr', []),
                        mode='lines',
                        name=f"ROC (AUC = {roc_data.get('roc_auc', 0):.3f})",
                        line=dict(color='darkorange', width=2)
                    ),
                    row=2, col=2
                )
                
                # Add diagonal reference line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        line=dict(dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Confusion matrix heatmap
            if 'classification' in standard_metrics and 'error' not in standard_metrics['classification']:
                class_metrics = standard_metrics['classification']
                confusion_matrix = [
                    [class_metrics.get('true_negatives', 0), class_metrics.get('false_positives', 0)],
                    [class_metrics.get('false_negatives', 0), class_metrics.get('true_positives', 0)]
                ]
                
                fig.add_trace(
                    go.Heatmap(
                        z=confusion_matrix,
                        x=['No Flood', 'Flood'],
                        y=['No Flood', 'Flood'],
                        colorscale='Blues',
                        showscale=False
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text="Flood Risk Model Performance Dashboard",
                title_font_size=20,
                showlegend=False
            )
            
            # Save dashboard if output path provided
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Dashboard saved to: {output_path}")
            
            return {
                'figure': fig,
                'html_string': fig.to_html(),
                'components': self._extract_dashboard_components(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise RuntimeError(f"Dashboard creation failed: {e}")
    
    def create_spatial_analysis_plots(self, predictions: np.ndarray, observations: np.ndarray,
                                    metadata: Optional[Dict] = None) -> Dict:
        """
        Create spatial analysis visualizations
        
        Args:
            predictions: Model predictions
            observations: Ground truth observations
            metadata: Spatial metadata
            
        Returns:
            Dictionary with spatial analysis plots
        """
        try:
            logger.info("Creating spatial analysis plots")
            
            plots = {}
            
            # Difference map
            if HAS_MATPLOTLIB:
                difference = predictions - observations
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Spatial Analysis of Flood Predictions', fontsize=16)
                
                # Predictions
                im1 = axes[0, 0].imshow(predictions, cmap='Blues', aspect='auto')
                axes[0, 0].set_title('Model Predictions')
                plt.colorbar(im1, ax=axes[0, 0], label='Depth (m)')
                
                # Observations
                im2 = axes[0, 1].imshow(observations, cmap='Reds', aspect='auto')
                axes[0, 1].set_title('Observations')
                plt.colorbar(im2, ax=axes[0, 1], label='Depth (m)')
                
                # Difference
                im3 = axes[1, 0].imshow(difference, cmap='RdBu_r', aspect='auto')
                axes[1, 0].set_title('Prediction Error (Pred - Obs)')
                plt.colorbar(im3, ax=axes[1, 0], label='Error (m)')
                
                # Absolute difference
                im4 = axes[1, 1].imshow(np.abs(difference), cmap='hot', aspect='auto')
                axes[1, 1].set_title('Absolute Error')
                plt.colorbar(im4, ax=axes[1, 1], label='|Error| (m)')
                
                plt.tight_layout()
                
                # Convert to base64 for web display
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                spatial_plot_b64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                plots['spatial_analysis'] = {
                    'image_b64': spatial_plot_b64,
                    'format': 'png'
                }
            
            # Scatter plot with Plotly
            if HAS_PLOTLY:
                # Sample points for scatter plot
                sample_size = min(5000, predictions.size)
                indices = np.random.choice(predictions.size, sample_size, replace=False)
                pred_sample = predictions.flat[indices]
                obs_sample = observations.flat[indices]
                
                scatter_fig = go.Figure()
                
                scatter_fig.add_trace(
                    go.Scatter(
                        x=obs_sample,
                        y=pred_sample,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=np.abs(pred_sample - obs_sample),
                            colorscale='Viridis',
                            colorbar=dict(title="Absolute Error"),
                            line=dict(width=0.5, color='white')
                        ),
                        name='Predictions vs Observations'
                    )
                )
                
                # Add perfect prediction line
                max_val = max(np.max(pred_sample), np.max(obs_sample))
                scatter_fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        line=dict(dash='dash', color='red'),
                        name='Perfect Prediction'
                    )
                )
                
                scatter_fig.update_layout(
                    title='Predictions vs Observations Scatter Plot',
                    xaxis_title='Observed Depth (m)',
                    yaxis_title='Predicted Depth (m)',
                    width=600,
                    height=500
                )
                
                plots['scatter_plot'] = {
                    'figure': scatter_fig,
                    'html': scatter_fig.to_html()
                }
            
            logger.info("Spatial analysis plots created successfully")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating spatial analysis plots: {e}")
            raise RuntimeError(f"Spatial plotting failed: {e}")
    
    def create_time_series_validation(self, temporal_data: Dict,
                                    output_path: Optional[str] = None) -> Dict:
        """
        Create time series validation plots for temporal flood models
        
        Args:
            temporal_data: Dictionary with time series data
            output_path: Path to save plots
            
        Returns:
            Dictionary with time series plots
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for time series plots")
            return {}
        
        try:
            logger.info("Creating time series validation plots")
            
            # Extract time series data
            timestamps = temporal_data.get('timestamps', [])
            predicted_series = temporal_data.get('predictions', [])
            observed_series = temporal_data.get('observations', [])
            
            # Create time series plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=predicted_series,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=observed_series,
                    mode='lines+markers',
                    name='Observations',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                )
            )
            
            # Add error bands if available
            if 'prediction_std' in temporal_data:
                pred_upper = np.array(predicted_series) + np.array(temporal_data['prediction_std'])
                pred_lower = np.array(predicted_series) - np.array(temporal_data['prediction_std'])
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps + timestamps[::-1],
                        y=list(pred_upper) + list(pred_lower[::-1]),
                        fill='toself',
                        fillcolor='rgba(0, 100, 80, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Prediction Uncertainty'
                    )
                )
            
            fig.update_layout(
                title='Time Series Validation',
                xaxis_title='Time',
                yaxis_title='Water Depth (m)',
                hovermode='x unified',
                width=800,
                height=500
            )
            
            # Save if output path provided
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Time series plot saved to: {output_path}")
            
            return {
                'figure': fig,
                'html': fig.to_html(),
                'statistics': self._calculate_time_series_stats(predicted_series, observed_series)
            }
            
        except Exception as e:
            logger.error(f"Error creating time series plots: {e}")
            raise RuntimeError(f"Time series plotting failed: {e}")
    
    def _create_static_comparison(self, predictions: np.ndarray, observations: np.ndarray,
                                metadata: Optional[Dict] = None) -> Dict:
        """
        Create static comparison plots when Folium is not available
        """
        if not HAS_MATPLOTLIB:
            return {'error': 'No plotting libraries available'}
        
        logger.info("Creating static comparison plots")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Flood Extent Comparison', fontsize=16)
        
        # Predictions
        im1 = axes[0].imshow(predictions, cmap='Blues', aspect='auto')
        axes[0].set_title('Model Predictions')
        plt.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Observations
        im2 = axes[1].imshow(observations, cmap='Reds', aspect='auto')
        axes[1].set_title('Observations')
        plt.colorbar(im2, ax=axes[1], label='Depth (m)')
        
        # Difference
        difference = predictions - observations
        im3 = axes[2].imshow(difference, cmap='RdBu_r', aspect='auto')
        axes[2].set_title('Difference (Pred - Obs)')
        plt.colorbar(im3, ax=axes[2], label='Error (m)')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        static_plot_b64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'static_comparison': {
                'image_b64': static_plot_b64,
                'format': 'png'
            }
        }
    
    def _create_simple_dashboard(self, validation_results: Dict) -> Dict:
        """
        Create simple text-based dashboard when Plotly is not available
        """
        logger.info("Creating simple text dashboard")
        
        # Extract key metrics
        standard_metrics = validation_results.get('standard_metrics', {})
        
        dashboard_text = "=== FLOOD RISK MODEL PERFORMANCE DASHBOARD ===\n\n"
        
        # IoU metrics
        if 'iou' in standard_metrics:
            iou_score = standard_metrics['iou'].get('iou', 0)
            dashboard_text += f"Intersection over Union (IoU): {iou_score:.4f}\n"
        
        # Classification metrics
        if 'classification' in standard_metrics and 'error' not in standard_metrics['classification']:
            class_metrics = standard_metrics['classification']
            dashboard_text += "\n--- Classification Metrics ---\n"
            dashboard_text += f"Accuracy: {class_metrics.get('accuracy', 0):.4f}\n"
            dashboard_text += f"Precision: {class_metrics.get('precision', 0):.4f}\n"
            dashboard_text += f"Recall: {class_metrics.get('recall', 0):.4f}\n"
            dashboard_text += f"F1-Score: {class_metrics.get('f1_score', 0):.4f}\n"
        
        # Regression metrics
        if 'regression' in standard_metrics and 'error' not in standard_metrics['regression']:
            reg_metrics = standard_metrics['regression']
            dashboard_text += "\n--- Regression Metrics ---\n"
            dashboard_text += f"MAE: {reg_metrics.get('mae', 0):.4f} m\n"
            dashboard_text += f"RMSE: {reg_metrics.get('rmse', 0):.4f} m\n"
            dashboard_text += f"R²: {reg_metrics.get('r_squared', 0):.4f}\n"
            dashboard_text += f"Nash-Sutcliffe: {reg_metrics.get('nse', 0):.4f}\n"
        
        return {
            'text_dashboard': dashboard_text,
            'format': 'text'
        }
    
    def _extract_dashboard_components(self, validation_results: Dict) -> Dict:
        """
        Extract dashboard components for programmatic access
        """
        components = {}
        
        standard_metrics = validation_results.get('standard_metrics', {})
        
        # Key performance indicators
        components['kpis'] = {}
        if 'iou' in standard_metrics:
            components['kpis']['iou'] = standard_metrics['iou'].get('iou', 0)
        if 'classification' in standard_metrics and 'error' not in standard_metrics['classification']:
            components['kpis']['accuracy'] = standard_metrics['classification'].get('accuracy', 0)
            components['kpis']['f1_score'] = standard_metrics['classification'].get('f1_score', 0)
        if 'regression' in standard_metrics and 'error' not in standard_metrics['regression']:
            components['kpis']['rmse'] = standard_metrics['regression'].get('rmse', 0)
            components['kpis']['mae'] = standard_metrics['regression'].get('mae', 0)
        
        return components
    
    def _calculate_time_series_stats(self, predictions: List[float], observations: List[float]) -> Dict:
        """
        Calculate time series validation statistics
        """
        if not predictions or not observations:
            return {}
        
        pred_array = np.array(predictions)
        obs_array = np.array(observations)
        
        # Basic statistics
        correlation = np.corrcoef(pred_array, obs_array)[0, 1] if len(pred_array) > 1 else 0
        rmse = np.sqrt(np.mean((pred_array - obs_array) ** 2))
        mae = np.mean(np.abs(pred_array - obs_array))
        
        return {
            'correlation': float(correlation),
            'rmse': float(rmse),
            'mae': float(mae),
            'mean_prediction': float(np.mean(pred_array)),
            'mean_observation': float(np.mean(obs_array))
        }


# Utility functions
def create_flood_colormap(n_colors: int = 256) -> Any:
    """
    Create custom colormap for flood depth visualization
    
    Args:
        n_colors: Number of colors in the colormap
        
    Returns:
        Colormap object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    # Define color transitions: white -> light blue -> blue -> dark blue
    colors = ['white', 'lightcyan', 'lightblue', 'blue', 'darkblue']
    n_bins = len(colors) - 1
    
    cmap = mcolors.LinearSegmentedColormap.from_list('flood_depth', colors, N=n_colors)
    
    return cmap


def save_validation_plots(plots_dict: Dict, output_dir: str) -> List[str]:
    """
    Save all validation plots to specified directory
    
    Args:
        plots_dict: Dictionary with plot data
        output_dir: Output directory path
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for plot_name, plot_data in plots_dict.items():
        try:
            if 'figure' in plot_data and HAS_PLOTLY:
                # Save Plotly figure
                file_path = output_path / f"{plot_name}.html"
                plot_data['figure'].write_html(str(file_path))
                saved_files.append(str(file_path))
            
            elif 'image_b64' in plot_data:
                # Save base64 encoded image
                file_path = output_path / f"{plot_name}.png"
                image_data = base64.b64decode(plot_data['image_b64'])
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                saved_files.append(str(file_path))
            
            elif 'html_string' in plot_data:
                # Save HTML string
                file_path = output_path / f"{plot_name}.html"
                with open(file_path, 'w') as f:
                    f.write(plot_data['html_string'])
                saved_files.append(str(file_path))
        
        except Exception as e:
            logger.error(f"Failed to save plot {plot_name}: {e}")
    
    logger.info(f"Saved {len(saved_files)} plot files to {output_dir}")
    return saved_files


def generate_plot_gallery(plots_dict: Dict, title: str = "Validation Results") -> str:
    """
    Generate HTML gallery of all plots
    
    Args:
        plots_dict: Dictionary with plot data
        title: Gallery title
        
    Returns:
        HTML string with plot gallery
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot-container {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
            .plot-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """
    
    for plot_name, plot_data in plots_dict.items():
        html += f'<div class="plot-container">'
        html += f'<div class="plot-title">{plot_name.replace("_", " ").title()}</div>'
        
        if 'html_string' in plot_data:
            html += plot_data['html_string']
        elif 'image_b64' in plot_data:
            html += f'<img src="data:image/png;base64,{plot_data["image_b64"]}" alt="{plot_name}">'
        elif 'text_dashboard' in plot_data:
            html += f'<pre>{plot_data["text_dashboard"]}</pre>'
        
        html += '</div>'
    
    html += """
    </body>
    </html>
    """
    
    return html