"""
Validation Report Generator

Generates comprehensive validation reports for flood risk models including:
- HTML reports with interactive visualizations
- PDF reports for formal documentation
- Executive summaries for stakeholders
- Technical detailed reports for researchers
- Comparison reports across multiple models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import base64
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings
import io
import tempfile

try:
    from jinja2 import Template, Environment, FileSystemLoader

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    warnings.warn("jinja2 not available - limited template functionality")

try:
    import weasyprint

    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    warnings.warn("weasyprint not available - no PDF generation")

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .visualization import FloodVisualization
from .metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    title: str = "Flood Risk Model Validation Report"
    subtitle: str = ""
    author: str = "Validation System"
    organization: str = "Flood Risk Assessment Team"
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_detailed_results: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_appendices: bool = False
    template_style: str = "professional"
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None
    output_format: str = "html"  # 'html', 'pdf', 'both'


class HTMLReportGenerator:
    """
    Generates HTML validation reports with interactive content
    """

    def __init__(self, config: ReportConfig):
        """
        Initialize HTML report generator

        Args:
            config: Report configuration
        """
        self.config = config
        self.visualizer = FloodVisualization()
        logger.info("HTML Report Generator initialized")

    def generate_html_report(
        self,
        validation_results: Dict,
        output_path: str,
        additional_data: Optional[Dict] = None,
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            validation_results: Results from validation analysis
            output_path: Path for output HTML file
            additional_data: Additional data to include in report

        Returns:
            Path to generated HTML report
        """
        try:
            logger.info(f"Generating HTML report: {output_path}")

            # Prepare report data
            report_data = self._prepare_report_data(validation_results, additional_data)

            # Generate report sections
            sections = self._generate_report_sections(report_data)

            # Create HTML content
            html_content = self._create_html_content(sections)

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise RuntimeError(f"HTML report generation failed: {e}")

    def _prepare_report_data(
        self, validation_results: Dict, additional_data: Optional[Dict] = None
    ) -> Dict:
        """
        Prepare and structure data for report generation
        """
        logger.info("Preparing report data")

        # Base report data
        report_data = {
            "config": asdict(self.config),
            "generation_timestamp": datetime.now().isoformat(),
            "validation_results": validation_results,
            "metadata": validation_results.get("metadata", {}),
            "additional_data": additional_data or {},
        }

        # Extract key metrics for summary
        report_data["key_metrics"] = self._extract_key_metrics(validation_results)

        # Calculate performance grades
        report_data["performance_grades"] = self._calculate_performance_grades(
            validation_results
        )

        # Generate recommendations
        report_data["recommendations"] = self._generate_recommendations(
            validation_results
        )

        return report_data

    def _extract_key_metrics(self, validation_results: Dict) -> Dict:
        """
        Extract key metrics for executive summary
        """
        key_metrics = {}

        standard_metrics = validation_results.get("standard_metrics", {})

        # IoU Score
        if "iou" in standard_metrics:
            key_metrics["iou_score"] = {
                "value": standard_metrics["iou"].get("iou", 0),
                "label": "Intersection over Union",
                "description": "Overlap accuracy of flood extents",
                "range": [0, 1],
                "higher_better": True,
            }

        # F1 Score
        if (
            "classification" in standard_metrics
            and "error" not in standard_metrics["classification"]
        ):
            key_metrics["f1_score"] = {
                "value": standard_metrics["classification"].get("f1_score", 0),
                "label": "F1 Score",
                "description": "Balance of precision and recall",
                "range": [0, 1],
                "higher_better": True,
            }

        # RMSE
        if (
            "regression" in standard_metrics
            and "error" not in standard_metrics["regression"]
        ):
            key_metrics["rmse"] = {
                "value": standard_metrics["regression"].get("rmse", 0),
                "label": "Root Mean Square Error",
                "description": "Average prediction error magnitude",
                "range": [0, float("inf")],
                "higher_better": False,
                "unit": "m",
            }

        # Nash-Sutcliffe Efficiency
        if "regression" in standard_metrics and "nse" in standard_metrics["regression"]:
            key_metrics["nse"] = {
                "value": standard_metrics["regression"]["nse"],
                "label": "Nash-Sutcliffe Efficiency",
                "description": "Model efficiency compared to mean",
                "range": [-float("inf"), 1],
                "higher_better": True,
            }

        # Critical Success Index
        if "csi" in standard_metrics:
            key_metrics["csi"] = {
                "value": standard_metrics["csi"].get("csi", 0),
                "label": "Critical Success Index",
                "description": "Threat score for categorical forecasts",
                "range": [0, 1],
                "higher_better": True,
            }

        return key_metrics

    def _calculate_performance_grades(self, validation_results: Dict) -> Dict:
        """
        Calculate letter grades for different aspects of performance
        """
        grades = {}

        standard_metrics = validation_results.get("standard_metrics", {})

        # Flood extent accuracy grade (based on IoU)
        if "iou" in standard_metrics:
            iou_score = standard_metrics["iou"].get("iou", 0)
            grades["flood_extent"] = self._score_to_grade(iou_score)

        # Classification performance grade (based on F1)
        if (
            "classification" in standard_metrics
            and "error" not in standard_metrics["classification"]
        ):
            f1_score = standard_metrics["classification"].get("f1_score", 0)
            grades["classification"] = self._score_to_grade(f1_score)

        # Depth accuracy grade (based on normalized RMSE)
        if (
            "regression" in standard_metrics
            and "error" not in standard_metrics["regression"]
        ):
            rmse = standard_metrics["regression"].get("rmse", float("inf"))
            # Normalize RMSE (assuming max reasonable depth is 10m)
            normalized_rmse = min(rmse / 10.0, 1.0)
            rmse_score = 1.0 - normalized_rmse
            grades["depth_accuracy"] = self._score_to_grade(rmse_score)

        # Overall grade (average of available grades)
        if grades:
            grade_values = [self._grade_to_numeric(g) for g in grades.values()]
            overall_numeric = np.mean(grade_values)
            grades["overall"] = self._numeric_to_grade(overall_numeric)

        return grades

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _grade_to_numeric(self, grade: str) -> float:
        """Convert letter grade to numeric value"""
        grade_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
        return grade_map.get(grade, 0.0)

    def _numeric_to_grade(self, numeric: float) -> str:
        """Convert numeric value to letter grade"""
        if numeric >= 3.5:
            return "A"
        elif numeric >= 2.5:
            return "B"
        elif numeric >= 1.5:
            return "C"
        elif numeric >= 0.5:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self, validation_results: Dict) -> List[Dict]:
        """
        Generate actionable recommendations based on validation results
        """
        recommendations = []

        standard_metrics = validation_results.get("standard_metrics", {})

        # IoU-based recommendations
        if "iou" in standard_metrics:
            iou_score = standard_metrics["iou"].get("iou", 0)
            if iou_score < 0.5:
                recommendations.append(
                    {
                        "priority": "High",
                        "category": "Model Accuracy",
                        "issue": "Poor flood extent prediction",
                        "recommendation": "Review model parameters, topographic data quality, and boundary conditions. Consider recalibration with local flood events.",
                        "metric": f"IoU Score: {iou_score:.3f}",
                    }
                )
            elif iou_score > 0.85:
                recommendations.append(
                    {
                        "priority": "Low",
                        "category": "Model Performance",
                        "issue": "Excellent flood extent prediction",
                        "recommendation": "Model performs well for flood extent prediction. Consider operational deployment with regular monitoring.",
                        "metric": f"IoU Score: {iou_score:.3f}",
                    }
                )

        # RMSE-based recommendations
        if (
            "regression" in standard_metrics
            and "error" not in standard_metrics["regression"]
        ):
            rmse = standard_metrics["regression"].get("rmse", 0)
            if rmse > 1.0:
                recommendations.append(
                    {
                        "priority": "High",
                        "category": "Depth Accuracy",
                        "issue": "High depth prediction errors",
                        "recommendation": "Investigate bathymetric data accuracy, friction parameters, and model resolution. Consider ensemble modeling approaches.",
                        "metric": f"RMSE: {rmse:.3f}m",
                    }
                )

        # False alarm recommendations
        if (
            "classification" in standard_metrics
            and "error" not in standard_metrics["classification"]
        ):
            far = standard_metrics["classification"].get("false_alarm_rate", 0)
            if far > 0.3:
                recommendations.append(
                    {
                        "priority": "Medium",
                        "category": "False Alarms",
                        "issue": "High false alarm rate",
                        "recommendation": "Review threshold settings and consider post-processing filters to reduce false positives. Validate against local knowledge.",
                        "metric": f"False Alarm Rate: {far:.3f}",
                    }
                )

        # Nash-Sutcliffe recommendations
        if "regression" in standard_metrics and "nse" in standard_metrics["regression"]:
            nse = standard_metrics["regression"]["nse"]
            if nse < 0:
                recommendations.append(
                    {
                        "priority": "Critical",
                        "category": "Model Performance",
                        "issue": "Model performs worse than mean",
                        "recommendation": "Model shows poor performance. Complete model review recommended including data quality, physical processes, and parameterization.",
                        "metric": f"Nash-Sutcliffe Efficiency: {nse:.3f}",
                    }
                )

        return recommendations

    def _generate_report_sections(self, report_data: Dict) -> Dict:
        """
        Generate content for each report section
        """
        logger.info("Generating report sections")

        sections = {}

        # Executive Summary
        if self.config.include_executive_summary:
            sections["executive_summary"] = self._generate_executive_summary(
                report_data
            )

        # Methodology
        if self.config.include_methodology:
            sections["methodology"] = self._generate_methodology_section()

        # Results
        if self.config.include_detailed_results:
            sections["detailed_results"] = self._generate_results_section(report_data)

        # Visualizations
        if self.config.include_visualizations:
            sections["visualizations"] = self._generate_visualizations_section(
                report_data
            )

        # Recommendations
        if self.config.include_recommendations:
            sections["recommendations"] = self._generate_recommendations_section(
                report_data
            )

        return sections

    def _generate_executive_summary(self, report_data: Dict) -> str:
        """Generate executive summary section"""

        key_metrics = report_data.get("key_metrics", {})
        performance_grades = report_data.get("performance_grades", {})

        summary = f"""
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="summary-highlights">
                <div class="overall-grade">
                    <h3>Overall Performance: Grade {performance_grades.get('overall', 'N/A')}</h3>
                </div>
                
                <div class="key-metrics-grid">
        """

        # Add key metrics cards
        for metric_key, metric_data in key_metrics.items():
            value = metric_data["value"]
            if isinstance(value, float):
                value_str = f"{value:.3f}"
                if metric_data.get("unit"):
                    value_str += f" {metric_data['unit']}"
            else:
                value_str = str(value)

            summary += f"""
                    <div class="metric-card">
                        <div class="metric-value">{value_str}</div>
                        <div class="metric-label">{metric_data['label']}</div>
                        <div class="metric-description">{metric_data['description']}</div>
                    </div>
            """

        summary += """
                </div>
            </div>
            
            <div class="summary-text">
                <p>This report presents the validation results of the flood risk model against 
                ground truth observations and reference datasets. The model performance is 
                evaluated across multiple dimensions including spatial accuracy, depth prediction 
                quality, and categorical classification performance.</p>
            </div>
        </div>
        """

        return summary

    def _generate_methodology_section(self) -> str:
        """Generate methodology section"""

        return """
        <div class="methodology">
            <h2>Validation Methodology</h2>
            
            <h3>Metrics Used</h3>
            <ul>
                <li><strong>Intersection over Union (IoU):</strong> Measures spatial overlap between predicted and observed flood extents</li>
                <li><strong>Root Mean Square Error (RMSE):</strong> Quantifies average depth prediction errors</li>
                <li><strong>Nash-Sutcliffe Efficiency (NSE):</strong> Compares model performance to mean observations</li>
                <li><strong>F1 Score:</strong> Balances precision and recall for flood/no-flood classification</li>
                <li><strong>Critical Success Index (CSI):</strong> Evaluates categorical forecast skill</li>
            </ul>
            
            <h3>Validation Datasets</h3>
            <ul>
                <li><strong>Ground Truth Observations:</strong> High-resolution flood depth measurements</li>
                <li><strong>LISFLOOD-FP Reference:</strong> Benchmark 2D hydraulic model simulations</li>
                <li><strong>NFIP Claims Data:</strong> Historical flood insurance claims for real-world validation</li>
            </ul>
            
            <h3>Quality Assurance</h3>
            <p>All validation data underwent rigorous quality control including spatial alignment verification, 
            temporal consistency checks, and outlier detection. Statistical significance testing was applied 
            where appropriate.</p>
        </div>
        """

    def _generate_results_section(self, report_data: Dict) -> str:
        """Generate detailed results section"""

        validation_results = report_data["validation_results"]
        standard_metrics = validation_results.get("standard_metrics", {})

        results_html = """
        <div class="detailed-results">
            <h2>Detailed Results</h2>
        """

        # IoU Results
        if "iou" in standard_metrics:
            iou_data = standard_metrics["iou"]
            results_html += f"""
            <h3>Spatial Overlap Analysis</h3>
            <div class="results-table">
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>IoU Score</td><td>{iou_data.get('iou', 0):.4f}</td><td>Intersection over Union</td></tr>
                    <tr><td>Intersection Area</td><td>{iou_data.get('intersection_area', 0):,}</td><td>Cells correctly predicted as flooded</td></tr>
                    <tr><td>Union Area</td><td>{iou_data.get('union_area', 0):,}</td><td>Total flooded area (predicted + observed)</td></tr>
                    <tr><td>Predicted Area</td><td>{iou_data.get('predicted_area', 0):,}</td><td>Total predicted flood extent</td></tr>
                    <tr><td>Observed Area</td><td>{iou_data.get('observed_area', 0):,}</td><td>Total observed flood extent</td></tr>
                </table>
            </div>
            """

        # Classification Results
        if (
            "classification" in standard_metrics
            and "error" not in standard_metrics["classification"]
        ):
            class_data = standard_metrics["classification"]
            results_html += f"""
            <h3>Classification Performance</h3>
            <div class="results-table">
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Accuracy</td><td>{class_data.get('accuracy', 0):.4f}</td><td>Overall correct classifications</td></tr>
                    <tr><td>Precision</td><td>{class_data.get('precision', 0):.4f}</td><td>Correct flood predictions / Total flood predictions</td></tr>
                    <tr><td>Recall</td><td>{class_data.get('recall', 0):.4f}</td><td>Correct flood predictions / Total actual floods</td></tr>
                    <tr><td>F1-Score</td><td>{class_data.get('f1_score', 0):.4f}</td><td>Harmonic mean of precision and recall</td></tr>
                    <tr><td>Specificity</td><td>{class_data.get('specificity', 0):.4f}</td><td>Correct no-flood predictions / Total actual no-floods</td></tr>
                </table>
            </div>
            """

        # Regression Results
        if (
            "regression" in standard_metrics
            and "error" not in standard_metrics["regression"]
        ):
            reg_data = standard_metrics["regression"]
            results_html += f"""
            <h3>Depth Prediction Analysis</h3>
            <div class="results-table">
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>RMSE</td><td>{reg_data.get('rmse', 0):.4f} m</td><td>Root Mean Square Error</td></tr>
                    <tr><td>MAE</td><td>{reg_data.get('mae', 0):.4f} m</td><td>Mean Absolute Error</td></tr>
                    <tr><td>Bias</td><td>{reg_data.get('bias', 0):.4f} m</td><td>Mean prediction bias</td></tr>
                    <tr><td>R²</td><td>{reg_data.get('r_squared', 0):.4f}</td><td>Coefficient of determination</td></tr>
                    <tr><td>NSE</td><td>{reg_data.get('nse', 0):.4f}</td><td>Nash-Sutcliffe Efficiency</td></tr>
                </table>
            </div>
            """

        results_html += "</div>"
        return results_html

    def _generate_visualizations_section(self, report_data: Dict) -> str:
        """Generate visualizations section"""

        return """
        <div class="visualizations">
            <h2>Visualizations</h2>
            <div id="visualization-container">
                <div class="viz-placeholder">
                    <p>Interactive visualizations would be embedded here including:</p>
                    <ul>
                        <li>Flood extent comparison maps</li>
                        <li>Performance dashboard</li>
                        <li>Spatial analysis plots</li>
                        <li>Statistical distribution charts</li>
                    </ul>
                </div>
            </div>
        </div>
        """

    def _generate_recommendations_section(self, report_data: Dict) -> str:
        """Generate recommendations section"""

        recommendations = report_data.get("recommendations", [])

        rec_html = """
        <div class="recommendations">
            <h2>Recommendations</h2>
        """

        if not recommendations:
            rec_html += "<p>No specific recommendations generated.</p>"
        else:
            # Group by priority
            high_priority = [r for r in recommendations if r["priority"] == "High"]
            medium_priority = [r for r in recommendations if r["priority"] == "Medium"]
            low_priority = [r for r in recommendations if r["priority"] == "Low"]

            for priority_group, title in [
                (high_priority, "High Priority"),
                (medium_priority, "Medium Priority"),
                (low_priority, "Low Priority"),
            ]:
                if priority_group:
                    rec_html += f"<h3>{title}</h3><div class='recommendation-list'>"

                    for rec in priority_group:
                        rec_html += f"""
                        <div class="recommendation-item {rec['priority'].lower()}-priority">
                            <div class="rec-category">{rec['category']}</div>
                            <div class="rec-issue"><strong>Issue:</strong> {rec['issue']}</div>
                            <div class="rec-action"><strong>Recommendation:</strong> {rec['recommendation']}</div>
                            <div class="rec-metric"><strong>Supporting Metric:</strong> {rec['metric']}</div>
                        </div>
                        """

                    rec_html += "</div>"

        rec_html += "</div>"
        return rec_html

    def _create_html_content(self, sections: Dict) -> str:
        """
        Create complete HTML content with styling
        """
        css_styles = self._get_css_styles()

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.config.title}</title>
            <style>
                {css_styles}
            </style>
        </head>
        <body>
            <div class="report-container">
                <header class="report-header">
                    <h1>{self.config.title}</h1>
                    {f'<h2 class="subtitle">{self.config.subtitle}</h2>' if self.config.subtitle else ''}
                    <div class="report-meta">
                        <p><strong>Author:</strong> {self.config.author}</p>
                        <p><strong>Organization:</strong> {self.config.organization}</p>
                        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </header>
                
                <main class="report-content">
        """

        # Add sections
        for section_name, section_content in sections.items():
            html_content += f'<section class="report-section {section_name}">\n{section_content}\n</section>\n'

        html_content += """
                </main>
                
                <footer class="report-footer">
                    <p>Generated by Flood Risk Validation Framework</p>
                </footer>
            </div>
        </body>
        </html>
        """

        return html_content

    def _get_css_styles(self) -> str:
        """
        Get CSS styles for the report
        """
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .report-header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5rem;
        }
        
        .subtitle {
            margin: 0 0 20px 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .report-meta p {
            margin: 5px 0;
            font-size: 0.9rem;
        }
        
        .report-content {
            padding: 30px;
        }
        
        .report-section {
            margin-bottom: 40px;
        }
        
        h2 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        .executive-summary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .overall-grade {
            text-align: center;
            margin-bottom: 25px;
        }
        
        .overall-grade h3 {
            font-size: 2rem;
            margin: 0;
            color: white;
        }
        
        .key-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 3px;
        }
        
        .metric-description {
            font-size: 0.75rem;
            opacity: 0.9;
        }
        
        .results-table table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        .results-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .recommendation-list {
            margin-bottom: 25px;
        }
        
        .recommendation-item {
            border-left: 4px solid #ccc;
            padding: 15px;
            margin-bottom: 15px;
            background: #f8f9fa;
            border-radius: 0 5px 5px 0;
        }
        
        .high-priority {
            border-left-color: #dc3545;
            background: #fff5f5;
        }
        
        .medium-priority {
            border-left-color: #ffc107;
            background: #fffbf0;
        }
        
        .low-priority {
            border-left-color: #28a745;
            background: #f0fff4;
        }
        
        .rec-category {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .rec-issue,
        .rec-action,
        .rec-metric {
            margin-bottom: 8px;
        }
        
        .viz-placeholder {
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-radius: 5px;
            border: 2px dashed #dee2e6;
        }
        
        .report-footer {
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
        }
        
        @media print {
            body {
                background-color: white;
            }
            
            .report-container {
                box-shadow: none;
            }
        }
        """


class ValidationReportGenerator:
    """
    Main report generator class that coordinates different report types
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize validation report generator

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.html_generator = HTMLReportGenerator(self.config)
        self.metrics_calc = MetricsCalculator()

        logger.info("Validation Report Generator initialized")

    def generate_comprehensive_report(
        self,
        validation_results: Dict,
        output_path: str,
        additional_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate comprehensive validation report

        Args:
            validation_results: Results from validation analysis
            output_path: Base path for output files (extension will be added)
            additional_data: Additional data to include

        Returns:
            Dictionary with paths to generated reports
        """
        try:
            logger.info(f"Generating comprehensive validation report: {output_path}")

            generated_files = {}

            # Generate HTML report
            if self.config.output_format in ["html", "both"]:
                html_path = f"{output_path}.html"
                self.html_generator.generate_html_report(
                    validation_results, html_path, additional_data
                )
                generated_files["html"] = html_path

            # Generate PDF report (if requested and available)
            if self.config.output_format in ["pdf", "both"] and HAS_WEASYPRINT:
                pdf_path = f"{output_path}.pdf"
                self._generate_pdf_report(validation_results, pdf_path, additional_data)
                generated_files["pdf"] = pdf_path
            elif self.config.output_format in ["pdf", "both"]:
                logger.warning("PDF generation requested but weasyprint not available")

            # Generate summary JSON
            summary_path = f"{output_path}_summary.json"
            self._generate_summary_json(validation_results, summary_path)
            generated_files["summary"] = summary_path

            logger.info(
                f"Report generation completed. Files: {list(generated_files.keys())}"
            )
            return generated_files

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise RuntimeError(f"Report generation failed: {e}")

    def _generate_pdf_report(
        self,
        validation_results: Dict,
        output_path: str,
        additional_data: Optional[Dict] = None,
    ) -> None:
        """
        Generate PDF report from HTML content
        """
        try:
            logger.info(f"Generating PDF report: {output_path}")

            # Generate HTML content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            ) as temp_file:
                temp_html_path = temp_file.name
                self.html_generator.generate_html_report(
                    validation_results, temp_html_path, additional_data
                )

            # Convert to PDF
            weasyprint.HTML(temp_html_path).write_pdf(output_path)

            # Clean up
            Path(temp_html_path).unlink()

            logger.info(f"PDF report generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise RuntimeError(f"PDF generation failed: {e}")

    def _generate_summary_json(
        self, validation_results: Dict, output_path: str
    ) -> None:
        """
        Generate JSON summary of validation results
        """
        try:
            logger.info(f"Generating summary JSON: {output_path}")

            # Extract key information for JSON summary
            standard_metrics = validation_results.get("standard_metrics", {})

            summary = {
                "report_metadata": {
                    "title": self.config.title,
                    "generated_at": datetime.now().isoformat(),
                    "author": self.config.author,
                    "organization": self.config.organization,
                },
                "performance_summary": {},
                "key_metrics": {},
                "data_summary": validation_results.get("data_summary", {}),
                "recommendations_count": 0,
            }

            # Extract key metrics
            if "iou" in standard_metrics:
                summary["key_metrics"]["iou_score"] = standard_metrics["iou"].get(
                    "iou", 0
                )

            if (
                "classification" in standard_metrics
                and "error" not in standard_metrics["classification"]
            ):
                cls_metrics = standard_metrics["classification"]
                summary["key_metrics"].update(
                    {
                        "accuracy": cls_metrics.get("accuracy", 0),
                        "f1_score": cls_metrics.get("f1_score", 0),
                        "precision": cls_metrics.get("precision", 0),
                        "recall": cls_metrics.get("recall", 0),
                    }
                )

            if (
                "regression" in standard_metrics
                and "error" not in standard_metrics["regression"]
            ):
                reg_metrics = standard_metrics["regression"]
                summary["key_metrics"].update(
                    {
                        "rmse": reg_metrics.get("rmse", 0),
                        "mae": reg_metrics.get("mae", 0),
                        "r_squared": reg_metrics.get("r_squared", 0),
                        "nash_sutcliffe": reg_metrics.get("nse", 0),
                    }
                )

            # Performance grades
            html_data = self.html_generator._prepare_report_data(validation_results)
            summary["performance_grades"] = html_data.get("performance_grades", {})
            summary["recommendations_count"] = len(html_data.get("recommendations", []))

            # Save JSON
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Summary JSON generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating summary JSON: {e}")
            raise RuntimeError(f"JSON summary generation failed: {e}")

    def generate_comparison_report(
        self, model_results: Dict[str, Dict], output_path: str
    ) -> str:
        """
        Generate comparison report for multiple models

        Args:
            model_results: Dictionary with model names as keys and validation results as values
            output_path: Path for output comparison report

        Returns:
            Path to generated comparison report
        """
        try:
            logger.info(f"Generating model comparison report: {output_path}")

            # Create comparison HTML
            html_content = self._create_comparison_html(model_results)

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Comparison report generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise RuntimeError(f"Comparison report generation failed: {e}")

    def _create_comparison_html(self, model_results: Dict[str, Dict]) -> str:
        """
        Create HTML content for model comparison report
        """
        # Extract metrics for all models
        comparison_data = {}
        for model_name, results in model_results.items():
            standard_metrics = results.get("standard_metrics", {})

            model_summary = {
                "iou": standard_metrics.get("iou", {}).get("iou", 0),
                "f1_score": 0,
                "accuracy": 0,
                "rmse": 0,
                "mae": 0,
                "r_squared": 0,
            }

            if (
                "classification" in standard_metrics
                and "error" not in standard_metrics["classification"]
            ):
                cls = standard_metrics["classification"]
                model_summary.update(
                    {
                        "f1_score": cls.get("f1_score", 0),
                        "accuracy": cls.get("accuracy", 0),
                    }
                )

            if (
                "regression" in standard_metrics
                and "error" not in standard_metrics["regression"]
            ):
                reg = standard_metrics["regression"]
                model_summary.update(
                    {
                        "rmse": reg.get("rmse", 0),
                        "mae": reg.get("mae", 0),
                        "r_squared": reg.get("r_squared", 0),
                    }
                )

            comparison_data[model_name] = model_summary

        # Create comparison table
        metrics_to_compare = ["iou", "f1_score", "accuracy", "rmse", "mae", "r_squared"]
        metric_labels = {
            "iou": "IoU Score",
            "f1_score": "F1 Score",
            "accuracy": "Accuracy",
            "rmse": "RMSE (m)",
            "mae": "MAE (m)",
            "r_squared": "R²",
        }

        table_html = """
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
        """

        for model_name in comparison_data.keys():
            table_html += f"<th>{model_name}</th>"

        table_html += """
                    <th>Best Model</th>
                </tr>
            </thead>
            <tbody>
        """

        for metric in metrics_to_compare:
            table_html += f"""
                <tr>
                    <td><strong>{metric_labels[metric]}</strong></td>
            """

            # Find best model for this metric
            metric_values = {
                name: data[metric] for name, data in comparison_data.items()
            }

            # For RMSE and MAE, lower is better
            if metric in ["rmse", "mae"]:
                best_model = min(metric_values.keys(), key=lambda x: metric_values[x])
            else:
                best_model = max(metric_values.keys(), key=lambda x: metric_values[x])

            for model_name, model_data in comparison_data.items():
                value = model_data[metric]
                css_class = "best-value" if model_name == best_model else ""

                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                table_html += f'<td class="{css_class}">{value_str}</td>'

            table_html += f"<td><strong>{best_model}</strong></td>"
            table_html += "</tr>"

        table_html += """
            </tbody>
        </table>
        """

        # Complete HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Comparison Report</title>
            <style>
                {self.html_generator._get_css_styles()}
                
                .comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                
                .comparison-table th,
                .comparison-table td {{
                    padding: 12px;
                    text-align: center;
                    border: 1px solid #ddd;
                }}
                
                .comparison-table th {{
                    background-color: #667eea;
                    color: white;
                    font-weight: 600;
                }}
                
                .best-value {{
                    background-color: #d4edda;
                    font-weight: bold;
                    color: #155724;
                }}
                
                .comparison-table tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="report-container">
                <header class="report-header">
                    <h1>Model Comparison Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                
                <main class="report-content">
                    <section class="comparison-section">
                        <h2>Performance Comparison</h2>
                        <p>The following table compares key performance metrics across all validated models. 
                        The best performing model for each metric is highlighted.</p>
                        
                        {table_html}
                        
                        <h3>Summary</h3>
                        <p>This comparison helps identify the most suitable model for different aspects of flood prediction:</p>
                        <ul>
                            <li><strong>Spatial Accuracy:</strong> Best IoU score indicates superior flood extent prediction</li>
                            <li><strong>Classification Performance:</strong> Highest F1 score shows best balance of precision and recall</li>
                            <li><strong>Depth Accuracy:</strong> Lowest RMSE indicates most accurate depth predictions</li>
                            <li><strong>Overall Fit:</strong> Highest R² shows best correlation with observations</li>
                        </ul>
                    </section>
                </main>
                
                <footer class="report-footer">
                    <p>Generated by Flood Risk Validation Framework</p>
                </footer>
            </div>
        </body>
        </html>
        """

        return html_content


# Utility functions
def create_report_config(title: str, **kwargs) -> ReportConfig:
    """
    Create report configuration with specified parameters

    Args:
        title: Report title
        **kwargs: Additional configuration parameters

    Returns:
        ReportConfig object
    """
    return ReportConfig(title=title, **kwargs)


def batch_generate_reports(
    validation_results_list: List[Dict],
    output_dir: str,
    config: Optional[ReportConfig] = None,
) -> List[str]:
    """
    Generate reports for multiple validation runs

    Args:
        validation_results_list: List of validation result dictionaries
        output_dir: Output directory for reports
        config: Report configuration

    Returns:
        List of generated report file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = ValidationReportGenerator(config)
    generated_files = []

    for i, results in enumerate(validation_results_list):
        try:
            report_path = output_path / f"validation_report_{i+1:03d}"
            files = generator.generate_comprehensive_report(results, str(report_path))
            generated_files.extend(files.values())

        except Exception as e:
            logger.error(f"Failed to generate report {i+1}: {e}")

    logger.info(f"Generated {len(generated_files)} report files in {output_dir}")
    return generated_files
