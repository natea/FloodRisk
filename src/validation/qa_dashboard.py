"""
Automated QA Dashboard and Reporting System

Provides:
- Interactive validation dashboards
- Automated QA report generation
- Real-time validation monitoring
- Performance trend analysis
- Alert system for quality issues
- Comparative analysis tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import asdict
import base64
import io

from .pipeline_validator import ValidationResult, PipelineValidator
from .ml_integration_validator import MLIntegrationValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend("Agg")


class ValidationDatabase:
    """
    SQLite database for storing validation results and history
    """

    def __init__(self, db_path: Union[str, Path] = "validation_history.db"):
        self.db_path = Path(db_path)
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pipeline_type TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    component_count INTEGER NOT NULL,
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    details TEXT NOT NULL,
                    issues TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES validation_runs (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    run_id INTEGER,
                    FOREIGN KEY (run_id) REFERENCES validation_runs (id)
                )
            """
            )

    def store_validation_run(
        self,
        pipeline_type: str,
        overall_status: str,
        overall_score: float,
        results: List[ValidationResult],
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Store validation run and results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO validation_runs 
                (timestamp, pipeline_type, overall_status, overall_score, component_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    pipeline_type,
                    overall_status,
                    overall_score,
                    len(results),
                    json.dumps(metadata or {}),
                ),
            )

            run_id = cursor.lastrowid

            # Store individual results
            for result in results:
                conn.execute(
                    """
                    INSERT INTO validation_results 
                    (run_id, component, status, score, details, issues, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        result.component,
                        result.status,
                        result.score,
                        json.dumps(result.details),
                        json.dumps(result.issues),
                        result.timestamp.isoformat(),
                    ),
                )

            return run_id

    def get_validation_history(self, days: int = 30) -> pd.DataFrame:
        """Get validation history"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM validation_runs 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """,
                conn,
                params=(cutoff_date,),
            )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_component_trends(self, component: str, days: int = 30) -> pd.DataFrame:
        """Get trends for specific component"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT vr.*, vrun.timestamp as run_timestamp
                FROM validation_results vr
                JOIN validation_runs vrun ON vr.run_id = vrun.id
                WHERE vr.component = ? AND vrun.timestamp >= ?
                ORDER BY vrun.timestamp DESC
            """,
                conn,
                params=(component, cutoff_date),
            )

        if not df.empty:
            df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])

        return df

    def store_alert(
        self, severity: str, component: str, message: str, run_id: Optional[int] = None
    ):
        """Store validation alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alerts (timestamp, severity, component, message, run_id)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), severity, component, message, run_id),
            )

    def get_active_alerts(self) -> pd.DataFrame:
        """Get active (unresolved) alerts"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM alerts WHERE resolved = FALSE
                ORDER BY timestamp DESC
            """,
                conn,
            )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df


class QAVisualizer:
    """
    Quality Assurance Visualization Generator
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        plt.style.use(style)
        self.color_palette = sns.color_palette("husl", 8)

    def create_validation_summary_plot(
        self, results: List[ValidationResult], save_path: Optional[Path] = None
    ) -> str:
        """Create validation summary visualization"""
        # Extract data
        components = [r.component for r in results]
        scores = [r.score for r in results]
        statuses = [r.status for r in results]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Score bar chart
        colors = [
            "green" if s == "PASS" else "orange" if s == "WARN" else "red"
            for s in statuses
        ]
        bars = ax1.bar(range(len(components)), scores, color=colors, alpha=0.7)
        ax1.set_xlabel("Components")
        ax1.set_ylabel("Validation Score")
        ax1.set_title("Validation Scores by Component")
        ax1.set_xticks(range(len(components)))
        ax1.set_xticklabels(
            [c.replace("_", "\n") for c in components], rotation=45, ha="right"
        )
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # Status pie chart
        status_counts = pd.Series(statuses).value_counts()
        colors_pie = ["green", "orange", "red"]
        wedges, texts, autotexts = ax2.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie[: len(status_counts)],
        )
        ax2.set_title("Validation Status Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            # Return base64 encoded image
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"

    def create_trend_analysis_plot(
        self, history_df: pd.DataFrame, save_path: Optional[Path] = None
    ) -> str:
        """Create validation trend analysis"""
        if history_df.empty:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(
                0.5,
                0.5,
                "No historical data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax.set_title("Validation Score Trends")
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Score trend over time
            ax1.plot(
                history_df["timestamp"],
                history_df["overall_score"],
                marker="o",
                linewidth=2,
                markersize=6,
            )
            ax1.set_ylabel("Overall Score")
            ax1.set_title("Validation Score Trend Over Time")
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # Status distribution over time (stacked area)
            status_pivot = (
                history_df.groupby(["timestamp", "overall_status"])
                .size()
                .unstack(fill_value=0)
            )
            if not status_pivot.empty:
                ax2.stackplot(
                    status_pivot.index,
                    status_pivot.get("PASS", 0),
                    status_pivot.get("WARN", 0),
                    status_pivot.get("FAIL", 0),
                    labels=["PASS", "WARN", "FAIL"],
                    colors=["green", "orange", "red"],
                    alpha=0.7,
                )
                ax2.set_ylabel("Count")
                ax2.set_xlabel("Date")
                ax2.set_title("Validation Status Distribution Over Time")
                ax2.legend(loc="upper right")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"

    def create_component_heatmap(
        self, component_scores: Dict[str, List[float]], save_path: Optional[Path] = None
    ) -> str:
        """Create component performance heatmap"""
        # Convert to DataFrame for heatmap
        df_data = []
        for component, scores in component_scores.items():
            for i, score in enumerate(scores):
                df_data.append(
                    {"Component": component, "Run": f"Run {i+1}", "Score": score}
                )

        if not df_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No component data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=16,
            )
            ax.set_title("Component Performance Heatmap")
        else:
            df = pd.DataFrame(df_data)
            pivot_df = df.pivot(index="Component", columns="Run", values="Score")

            fig, ax = plt.subplots(
                figsize=(max(8, len(pivot_df.columns)), max(6, len(pivot_df.index)))
            )
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                ax=ax,
                fmt=".3f",
                cbar_kws={"label": "Validation Score"},
            )
            ax.set_title("Component Performance Heatmap")
            ax.set_xlabel("Validation Runs")
            ax.set_ylabel("Components")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"

    def create_interactive_dashboard(
        self, results: List[ValidationResult], history_df: pd.DataFrame
    ) -> str:
        """Create interactive Plotly dashboard"""
        # Create subplot structure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Validation Scores",
                "Score Trends",
                "Status Distribution",
                "Component Details",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}],
            ],
        )

        # 1. Validation Scores Bar Chart
        if results:
            components = [r.component.replace("_", "<br>") for r in results]
            scores = [r.score for r in results]
            colors = [
                (
                    "green"
                    if r.status == "PASS"
                    else "orange" if r.status == "WARN" else "red"
                )
                for r in results
            ]

            fig.add_trace(
                go.Bar(
                    x=components,
                    y=scores,
                    marker_color=colors,
                    name="Scores",
                    text=[f"{s:.3f}" for s in scores],
                    textposition="outside",
                ),
                row=1,
                col=1,
            )

        # 2. Score Trends
        if not history_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"],
                    y=history_df["overall_score"],
                    mode="lines+markers",
                    name="Score Trend",
                    line=dict(color="blue"),
                ),
                row=1,
                col=2,
            )

        # 3. Status Distribution
        if results:
            statuses = [r.status for r in results]
            status_counts = pd.Series(statuses).value_counts()
            colors_pie = {"PASS": "green", "WARN": "orange", "FAIL": "red"}

            fig.add_trace(
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    marker=dict(
                        colors=[colors_pie.get(s, "gray") for s in status_counts.index]
                    ),
                ),
                row=2,
                col=1,
            )

        # 4. Component Issue Counts
        if results:
            issue_counts = [len(r.issues) for r in results]
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=issue_counts,
                    marker_color="red",
                    name="Issues",
                    opacity=0.6,
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=800, showlegend=False, title_text="Validation Dashboard", title_x=0.5
        )

        # Update y-axes
        fig.update_yaxis(range=[0, 1], title_text="Score", row=1, col=1)
        fig.update_yaxis(range=[0, 1], title_text="Score", row=1, col=2)
        fig.update_yaxis(title_text="Issue Count", row=2, col=2)

        # Convert to HTML
        return pyo.plot(fig, output_type="div", include_plotlyjs=True)


class QAReportGenerator:
    """
    Automated QA Report Generator
    """

    def __init__(self, db: ValidationDatabase, visualizer: QAVisualizer):
        self.db = db
        self.visualizer = visualizer
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_html_report(
        self,
        results: List[ValidationResult],
        pipeline_type: str = "FloodRisk ML Pipeline",
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate comprehensive HTML report"""
        # Calculate summary statistics
        overall_score = np.mean([r.score for r in results]) if results else 0.0
        overall_status = self._determine_overall_status(results)

        # Get historical data
        history_df = self.db.get_validation_history()

        # Generate visualizations
        summary_plot = self.visualizer.create_validation_summary_plot(results)
        trend_plot = self.visualizer.create_trend_analysis_plot(history_df)
        interactive_dashboard = self.visualizer.create_interactive_dashboard(
            results, history_df
        )

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FloodRisk QA Validation Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 8px 16px;
                    border-radius: 20px;
                    color: white;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .status-pass {{ background-color: #4CAF50; }}
                .status-warn {{ background-color: #FF9800; }}
                .status-fail {{ background-color: #F44336; }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 10px;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                    display: inline-block;
                    min-width: 200px;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    color: #6c757d;
                    margin-top: 5px;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .section h2 {{
                    color: #333;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                .component-result {{
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    margin: 10px 0;
                    padding: 15px;
                }}
                .component-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .component-name {{
                    font-weight: bold;
                    font-size: 1.1em;
                }}
                .score {{
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .issues-list {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                }}
                .issue-item {{
                    margin: 5px 0;
                    padding-left: 20px;
                    position: relative;
                }}
                .issue-item:before {{
                    content: "⚠";
                    position: absolute;
                    left: 0;
                    color: #f39c12;
                }}
                .visualization {{
                    text-align: center;
                    margin: 20px 0;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .recommendations {{
                    background: #e8f4f8;
                    border-left: 4px solid #17a2b8;
                    padding: 20px;
                    border-radius: 4px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #e0e0e0;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌊 FloodRisk ML Pipeline Validation Report</h1>
                    <h2>{pipeline_type}</h2>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Overall Status:</strong> 
                        <span class="status-badge status-{overall_status.lower()}">{overall_status}</span>
                    </p>
                </div>
                
                <div class="section">
                    <div class="metric-card">
                        <div class="metric-value">{overall_score:.3f}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(results)}</div>
                        <div class="metric-label">Components Validated</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len([r for r in results if r.status == 'PASS'])}</div>
                        <div class="metric-label">Passed Components</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{sum(len(r.issues) for r in results)}</div>
                        <div class="metric-label">Total Issues</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Validation Summary</h2>
                    <div class="visualization">
                        <img src="{summary_plot}" alt="Validation Summary" style="max-width: 100%; height: auto;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Performance Trends</h2>
                    <div class="visualization">
                        <img src="{trend_plot}" alt="Validation Trends" style="max-width: 100%; height: auto;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Component Results</h2>
                    {self._generate_component_results_html(results)}
                </div>
                
                <div class="section">
                    <h2>📋 Recommendations</h2>
                    <div class="recommendations">
                        {self._generate_recommendations_html(results)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Interactive Dashboard</h2>
                    {interactive_dashboard}
                </div>
                
                <div class="footer">
                    <p>Report generated by FloodRisk QA Validation System</p>
                    <p>For questions or issues, contact the FloodRisk development team</p>
                </div>
            </div>
        </body>
        </html>
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"HTML report saved to {output_path}")

        return html_content

    def _determine_overall_status(self, results: List[ValidationResult]) -> str:
        """Determine overall validation status"""
        if not results:
            return "UNKNOWN"

        statuses = [r.status for r in results]

        if any(s == "FAIL" for s in statuses):
            return "FAIL"
        elif any(s == "WARN" for s in statuses):
            return "WARN"
        else:
            return "PASS"

    def _generate_component_results_html(self, results: List[ValidationResult]) -> str:
        """Generate HTML for component results"""
        html_parts = []

        for result in results:
            status_class = f"status-{result.status.lower()}"

            issues_html = ""
            if result.issues:
                issues_html = '<div class="issues-list">'
                issues_html += "<strong>Issues:</strong>"
                for issue in result.issues:
                    issues_html += f'<div class="issue-item">{issue}</div>'
                issues_html += "</div>"

            html_parts.append(
                f"""
                <div class="component-result">
                    <div class="component-header">
                        <div class="component-name">{result.component.replace('_', ' ').title()}</div>
                        <div>
                            <span class="score">{result.score:.3f}</span>
                            <span class="status-badge {status_class}">{result.status}</span>
                        </div>
                    </div>
                    {issues_html}
                    <div style="margin-top: 10px; font-size: 0.9em; color: #6c757d;">
                        <strong>Timestamp:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                </div>
            """
            )

        return "".join(html_parts)

    def _generate_recommendations_html(self, results: List[ValidationResult]) -> str:
        """Generate HTML for recommendations"""
        recommendations = []

        # Generate recommendations based on results
        for result in results:
            if result.status == "FAIL":
                recommendations.append(
                    f"🚨 CRITICAL: Address issues in {result.component.replace('_', ' ')} before proceeding"
                )
            elif result.status == "WARN":
                recommendations.append(
                    f"⚠️ WARNING: Review and potentially fix issues in {result.component.replace('_', ' ')}"
                )

        if not recommendations:
            recommendations.append(
                "✅ All validation checks passed successfully - pipeline ready for production"
            )

        html_parts = ["<ul>"]
        for rec in recommendations:
            html_parts.append(f"<li>{rec}</li>")
        html_parts.append("</ul>")

        return "".join(html_parts)

    def generate_json_report(
        self, results: List[ValidationResult], output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate machine-readable JSON report"""
        overall_score = np.mean([r.score for r in results]) if results else 0.0
        overall_status = self._determine_overall_status(results)

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "validation_framework_version": "1.0.0",
                "report_type": "qa_validation",
            },
            "summary": {
                "overall_score": overall_score,
                "overall_status": overall_status,
                "total_components": len(results),
                "passed_components": len([r for r in results if r.status == "PASS"]),
                "warning_components": len([r for r in results if r.status == "WARN"]),
                "failed_components": len([r for r in results if r.status == "FAIL"]),
                "total_issues": sum(len(r.issues) for r in results),
            },
            "component_results": [asdict(result) for result in results],
            "recommendations": self._generate_recommendations_list(results),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"JSON report saved to {output_path}")

        return report

    def _generate_recommendations_list(
        self, results: List[ValidationResult]
    ) -> List[str]:
        """Generate recommendations as list"""
        recommendations = []

        for result in results:
            if result.status == "FAIL":
                recommendations.append(
                    f"CRITICAL: Address issues in {result.component}"
                )
            elif result.status == "WARN":
                recommendations.append(f"WARNING: Review issues in {result.component}")

        if not recommendations:
            recommendations.append("All validation checks passed successfully")

        return recommendations


class QAAlertSystem:
    """
    Automated Alert System for Quality Issues
    """

    def __init__(self, db: ValidationDatabase):
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)

        # Alert thresholds
        self.critical_score_threshold = 0.6
        self.warning_score_threshold = 0.8
        self.trend_degradation_threshold = 0.1  # 10% degradation

    def check_and_generate_alerts(
        self, results: List[ValidationResult], run_id: Optional[int] = None
    ):
        """Check validation results and generate alerts"""
        for result in results:
            # Critical score alerts
            if result.score < self.critical_score_threshold:
                self.db.store_alert(
                    "CRITICAL",
                    result.component,
                    f"Validation score ({result.score:.3f}) below critical threshold",
                    run_id,
                )
                self.logger.critical(
                    f"CRITICAL ALERT: {result.component} score {result.score:.3f}"
                )

            # Warning score alerts
            elif result.score < self.warning_score_threshold:
                self.db.store_alert(
                    "WARNING",
                    result.component,
                    f"Validation score ({result.score:.3f}) below warning threshold",
                    run_id,
                )
                self.logger.warning(
                    f"WARNING ALERT: {result.component} score {result.score:.3f}"
                )

            # Issue count alerts
            if len(result.issues) > 5:
                self.db.store_alert(
                    "WARNING",
                    result.component,
                    f"High number of issues detected ({len(result.issues)})",
                    run_id,
                )

        # Check for trend degradation
        self._check_trend_alerts(run_id)

    def _check_trend_alerts(self, run_id: Optional[int] = None):
        """Check for performance trend degradation"""
        history = self.db.get_validation_history(days=7)

        if len(history) >= 3:
            # Check if scores are consistently declining
            recent_scores = history.head(3)["overall_score"].tolist()

            if len(recent_scores) >= 3:
                # Check if each score is lower than the previous
                declining = all(
                    recent_scores[i] < recent_scores[i - 1]
                    for i in range(1, len(recent_scores))
                )

                if declining:
                    total_decline = recent_scores[-1] - recent_scores[0]
                    if abs(total_decline) > self.trend_degradation_threshold:
                        self.db.store_alert(
                            "WARNING",
                            "TREND_ANALYSIS",
                            f"Consistent score degradation detected ({total_decline:.3f} over recent runs)",
                            run_id,
                        )
                        self.logger.warning(
                            "TREND ALERT: Validation scores consistently declining"
                        )

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of active alerts"""
        alerts_df = self.db.get_active_alerts()

        if alerts_df.empty:
            return {"total_alerts": 0, "by_severity": {}, "by_component": {}}

        return {
            "total_alerts": len(alerts_df),
            "by_severity": alerts_df["severity"].value_counts().to_dict(),
            "by_component": alerts_df["component"].value_counts().to_dict(),
            "latest_alert": (
                alerts_df.iloc[0].to_dict() if not alerts_df.empty else None
            ),
        }


class QADashboard:
    """
    Main QA Dashboard Class orchestrating all QA components
    """

    def __init__(self, db_path: Union[str, Path] = "validation_history.db"):
        self.db = ValidationDatabase(db_path)
        self.visualizer = QAVisualizer()
        self.report_generator = QAReportGenerator(self.db, self.visualizer)
        self.alert_system = QAAlertSystem(self.db)
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_validation_results(
        self,
        results: List[ValidationResult],
        pipeline_type: str = "FloodRisk ML Pipeline",
        generate_report: bool = True,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Process validation results through complete QA pipeline"""
        self.logger.info(f"Processing validation results for {pipeline_type}")

        # Calculate overall metrics
        overall_score = np.mean([r.score for r in results]) if results else 0.0
        overall_status = self.report_generator._determine_overall_status(results)

        # Store in database
        run_id = self.db.store_validation_run(
            pipeline_type, overall_status, overall_score, results
        )

        # Generate and check alerts
        self.alert_system.check_and_generate_alerts(results, run_id)

        # Generate reports if requested
        reports = {}
        if generate_report:
            if report_path:
                html_report = self.report_generator.generate_html_report(
                    results, pipeline_type, report_path
                )
                json_report = self.report_generator.generate_json_report(
                    results, report_path.with_suffix(".json")
                )
            else:
                html_report = self.report_generator.generate_html_report(
                    results, pipeline_type
                )
                json_report = self.report_generator.generate_json_report(results)

            reports = {"html_report": html_report, "json_report": json_report}

        # Get alert summary
        alert_summary = self.alert_system.get_alert_summary()

        return {
            "run_id": run_id,
            "overall_score": overall_score,
            "overall_status": overall_status,
            "alert_summary": alert_summary,
            "reports": reports,
            "processed_components": len(results),
        }

    def get_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Get dashboard data for visualization"""
        history = self.db.get_validation_history(days)
        alerts = self.db.get_active_alerts()

        return {
            "validation_history": (
                history.to_dict("records") if not history.empty else []
            ),
            "active_alerts": alerts.to_dict("records") if not alerts.empty else [],
            "summary_stats": {
                "total_runs": len(history),
                "avg_score": (
                    history["overall_score"].mean() if not history.empty else 0
                ),
                "success_rate": (
                    len(history[history["overall_status"] == "PASS"])
                    / len(history)
                    * 100
                    if not history.empty
                    else 0
                ),
                "active_alerts_count": len(alerts),
            },
        }

    def cleanup_old_data(self, days: int = 90):
        """Clean up old validation data"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db.db_path) as conn:
            # Delete old validation results
            cursor = conn.execute(
                """
                DELETE FROM validation_results 
                WHERE run_id IN (
                    SELECT id FROM validation_runs WHERE timestamp < ?
                )
            """,
                (cutoff_date,),
            )

            # Delete old validation runs
            cursor = conn.execute(
                """
                DELETE FROM validation_runs WHERE timestamp < ?
            """,
                (cutoff_date,),
            )

            # Delete old resolved alerts
            cursor = conn.execute(
                """
                DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE
            """,
                (cutoff_date,),
            )

        self.logger.info(f"Cleaned up data older than {days} days")
