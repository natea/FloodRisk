"""Progress Tracking System for FloodRisk Pipeline.

This module provides comprehensive progress tracking and monitoring
capabilities for the flood risk modeling pipeline.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading


@dataclass
class StageProgress:
    """Progress information for a pipeline stage."""
    stage_name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    completion_percentage: float = 0.0
    current_operation: str = ""
    error_message: str = ""
    substages: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ProgressMetrics:
    """Overall pipeline progress metrics."""
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    overall_progress: float = 0.0
    estimated_remaining_minutes: Optional[float] = None
    current_stage: str = ""
    pipeline_start_time: Optional[datetime] = None
    pipeline_end_time: Optional[datetime] = None
    total_runtime_seconds: float = 0.0


class ProgressTracker:
    """Real-time progress tracking for pipeline execution."""
    
    def __init__(self, log_file: Optional[Path] = None, auto_save_interval: float = 10.0):
        """Initialize progress tracker.
        
        Args:
            log_file: Path to save progress log
            auto_save_interval: Interval in seconds for automatic saving
        """
        self.log_file = log_file
        self.auto_save_interval = auto_save_interval
        
        # Progress state
        self.stages: Dict[str, StageProgress] = {}
        self.metrics = ProgressMetrics()
        self.stage_order: List[str] = []
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        
        # Auto-save timer
        self._save_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        
        # Load existing progress if log file exists
        if self.log_file and self.log_file.exists():
            self._load_progress()
    
    def start_pipeline(self, stage_names: List[str]):
        """Start tracking a new pipeline execution.
        
        Args:
            stage_names: List of stage names in execution order
        """
        with self._lock:
            self.pipeline_start_time = datetime.now()
            self.metrics.pipeline_start_time = self.pipeline_start_time
            self.stage_order = stage_names
            self.metrics.total_stages = len(stage_names)
            
            # Initialize stages
            for stage_name in stage_names:
                self.stages[stage_name] = StageProgress(stage_name=stage_name)
            
            self._start_auto_save()
            self._save_progress()
    
    def start_stage(self, stage_name: str):
        """Start tracking a pipeline stage.
        
        Args:
            stage_name: Name of the stage
        """
        with self._lock:
            if stage_name not in self.stages:
                self.stages[stage_name] = StageProgress(stage_name=stage_name)
            
            stage = self.stages[stage_name]
            stage.status = "running"
            stage.start_time = datetime.now()
            stage.completion_percentage = 0.0
            
            self.metrics.current_stage = stage_name
            self._update_metrics()
    
    def update_stage_progress(self, stage_name: str, completion: float, operation: str = ""):
        """Update progress for a specific stage.
        
        Args:
            stage_name: Name of the stage
            completion: Completion percentage (0.0 to 1.0)
            operation: Current operation description
        """
        with self._lock:
            if stage_name not in self.stages:
                return
            
            stage = self.stages[stage_name]
            stage.completion_percentage = min(1.0, max(0.0, completion))
            stage.current_operation = operation
            
            self._update_metrics()
    
    def complete_stage(self, stage_name: str, duration_seconds: float = 0.0, 
                      success: bool = True, error: str = ""):
        """Mark a stage as completed.
        
        Args:
            stage_name: Name of the stage
            duration_seconds: Stage duration in seconds
            success: Whether the stage completed successfully
            error: Error message if stage failed
        """
        with self._lock:
            if stage_name not in self.stages:
                return
            
            stage = self.stages[stage_name]
            stage.end_time = datetime.now()
            stage.duration_seconds = duration_seconds
            stage.completion_percentage = 1.0
            
            if success:
                stage.status = "completed"
                self.metrics.completed_stages += 1
            else:
                stage.status = "failed"
                stage.error_message = error
                self.metrics.failed_stages += 1
            
            self._update_metrics()
    
    def add_stage_metric(self, stage_name: str, metric_name: str, value: Any):
        """Add a metric to a specific stage.
        
        Args:
            stage_name: Name of the stage
            metric_name: Name of the metric
            value: Metric value
        """
        with self._lock:
            if stage_name not in self.stages:
                return
            
            self.stages[stage_name].metrics[metric_name] = value
    
    def add_substage(self, stage_name: str, substage_name: str, progress: float = 0.0, 
                    status: str = "running"):
        """Add or update a substage within a main stage.
        
        Args:
            stage_name: Name of the main stage
            substage_name: Name of the substage
            progress: Substage progress (0.0 to 1.0)
            status: Substage status
        """
        with self._lock:
            if stage_name not in self.stages:
                return
            
            self.stages[stage_name].substages[substage_name] = {
                "progress": progress,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
    
    def end_pipeline(self, success: bool = True):
        """Mark the pipeline as completed.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        with self._lock:
            self.pipeline_end_time = datetime.now()
            self.metrics.pipeline_end_time = self.pipeline_end_time
            
            if self.pipeline_start_time:
                self.metrics.total_runtime_seconds = (
                    self.pipeline_end_time - self.pipeline_start_time
                ).total_seconds()
            
            self._stop_auto_save()
            self._save_progress()
    
    def _update_metrics(self):
        """Update overall pipeline metrics."""
        if not self.stages:
            return
        
        # Calculate overall progress
        total_stages = len(self.stage_order) if self.stage_order else len(self.stages)
        completed_stages = sum(1 for stage in self.stages.values() 
                             if stage.status == "completed")
        running_stages = [stage for stage in self.stages.values() 
                         if stage.status == "running"]
        
        # Base progress from completed stages
        base_progress = completed_stages / total_stages if total_stages > 0 else 0.0
        
        # Add partial progress from running stages
        running_progress = 0.0
        if running_stages:
            running_progress = sum(stage.completion_percentage for stage in running_stages)
            running_progress /= total_stages
        
        self.metrics.overall_progress = min(1.0, base_progress + running_progress)
        self.metrics.completed_stages = completed_stages
        
        # Estimate remaining time
        if self.pipeline_start_time and self.metrics.overall_progress > 0:
            elapsed_minutes = (datetime.now() - self.pipeline_start_time).total_seconds() / 60
            if self.metrics.overall_progress > 0.1:  # Only estimate after 10% completion
                estimated_total_minutes = elapsed_minutes / self.metrics.overall_progress
                self.metrics.estimated_remaining_minutes = max(
                    0, estimated_total_minutes - elapsed_minutes
                )
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary.
        
        Returns:
            Dictionary containing progress information
        """
        with self._lock:
            return {
                "metrics": asdict(self.metrics),
                "stages": {name: asdict(stage) for name, stage in self.stages.items()},
                "stage_order": self.stage_order,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_active_operations(self) -> List[Dict[str, str]]:
        """Get list of currently active operations.
        
        Returns:
            List of active operation descriptions
        """
        with self._lock:
            active = []
            for stage in self.stages.values():
                if stage.status == "running" and stage.current_operation:
                    active.append({
                        "stage": stage.stage_name,
                        "operation": stage.current_operation,
                        "progress": f"{stage.completion_percentage:.1%}"
                    })
            return active
    
    def get_stage_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of stage execution.
        
        Returns:
            List of stage execution timeline events
        """
        with self._lock:
            timeline = []
            for stage_name in self.stage_order:
                if stage_name in self.stages:
                    stage = self.stages[stage_name]
                    timeline.append({
                        "stage": stage_name,
                        "status": stage.status,
                        "start_time": stage.start_time.isoformat() if stage.start_time else None,
                        "end_time": stage.end_time.isoformat() if stage.end_time else None,
                        "duration_seconds": stage.duration_seconds,
                        "error": stage.error_message if stage.error_message else None
                    })
            return timeline
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all stages.
        
        Returns:
            Dictionary containing performance analysis
        """
        with self._lock:
            completed_stages = [s for s in self.stages.values() if s.status == "completed"]
            
            if not completed_stages:
                return {"status": "no_completed_stages"}
            
            durations = [s.duration_seconds for s in completed_stages]
            
            return {
                "total_completed_stages": len(completed_stages),
                "average_stage_duration_seconds": sum(durations) / len(durations),
                "min_stage_duration_seconds": min(durations),
                "max_stage_duration_seconds": max(durations),
                "total_processing_time_seconds": sum(durations),
                "stages_by_duration": [
                    {
                        "stage": s.stage_name,
                        "duration_seconds": s.duration_seconds,
                        "duration_minutes": s.duration_seconds / 60
                    }
                    for s in sorted(completed_stages, key=lambda x: x.duration_seconds, reverse=True)
                ]
            }
    
    def _save_progress(self):
        """Save progress to file."""
        if not self.log_file:
            return
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            summary = self.get_progress_summary()
            
            # Handle datetime serialization
            def datetime_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(self.log_file, 'w') as f:
                json.dump(summary, f, indent=2, default=datetime_serializer)
                
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")
    
    def _load_progress(self):
        """Load progress from file."""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Restore metrics
            if "metrics" in data:
                metrics_data = data["metrics"]
                self.metrics = ProgressMetrics(**metrics_data)
                
                # Convert datetime strings back to datetime objects
                if self.metrics.pipeline_start_time:
                    self.metrics.pipeline_start_time = datetime.fromisoformat(
                        self.metrics.pipeline_start_time
                    )
                if self.metrics.pipeline_end_time:
                    self.metrics.pipeline_end_time = datetime.fromisoformat(
                        self.metrics.pipeline_end_time
                    )
            
            # Restore stages
            if "stages" in data:
                for stage_name, stage_data in data["stages"].items():
                    stage = StageProgress(**stage_data)
                    
                    # Convert datetime strings back to datetime objects
                    if stage.start_time:
                        stage.start_time = datetime.fromisoformat(stage.start_time)
                    if stage.end_time:
                        stage.end_time = datetime.fromisoformat(stage.end_time)
                    
                    self.stages[stage_name] = stage
            
            # Restore stage order
            if "stage_order" in data:
                self.stage_order = data["stage_order"]
                
        except Exception as e:
            print(f"Warning: Failed to load progress: {e}")
    
    def _start_auto_save(self):
        """Start automatic progress saving."""
        if self.auto_save_interval > 0:
            self._save_progress()
            self._save_timer = threading.Timer(self.auto_save_interval, self._start_auto_save)
            self._save_timer.daemon = True
            self._save_timer.start()
    
    def _stop_auto_save(self):
        """Stop automatic progress saving."""
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None
    
    def print_progress_summary(self):
        """Print a formatted progress summary to console."""
        with self._lock:
            print("\n" + "="*60)
            print(f"PIPELINE PROGRESS SUMMARY")
            print("="*60)
            print(f"Overall Progress: {self.metrics.overall_progress:.1%}")
            print(f"Completed Stages: {self.metrics.completed_stages}/{self.metrics.total_stages}")
            print(f"Current Stage: {self.metrics.current_stage}")
            
            if self.metrics.estimated_remaining_minutes is not None:
                print(f"Estimated Remaining: {self.metrics.estimated_remaining_minutes:.1f} minutes")
            
            if self.metrics.total_runtime_seconds > 0:
                print(f"Total Runtime: {self.metrics.total_runtime_seconds/60:.1f} minutes")
            
            print("\nStage Status:")
            for stage_name in self.stage_order:
                if stage_name in self.stages:
                    stage = self.stages[stage_name]
                    status_symbol = {
                        "pending": "⏳",
                        "running": "🔄", 
                        "completed": "✅",
                        "failed": "❌"
                    }.get(stage.status, "❓")
                    
                    duration_str = ""
                    if stage.duration_seconds > 0:
                        duration_str = f" ({stage.duration_seconds/60:.1f}m)"
                    
                    progress_str = ""
                    if stage.status == "running":
                        progress_str = f" [{stage.completion_percentage:.1%}]"
                    
                    print(f"  {status_symbol} {stage_name}{progress_str}{duration_str}")
                    
                    if stage.current_operation:
                        print(f"      {stage.current_operation}")
            
            active_ops = self.get_active_operations()
            if active_ops:
                print("\nActive Operations:")
                for op in active_ops:
                    print(f"  🔄 {op['stage']}: {op['operation']} ({op['progress']})")
            
            print("="*60)
    
    def export_progress_report(self, output_file: Path):
        """Export detailed progress report.
        
        Args:
            output_file: Path to save the report
        """
        with self._lock:
            report = {
                "pipeline_summary": self.get_progress_summary(),
                "stage_timeline": self.get_stage_timeline(),
                "performance_metrics": self.get_performance_metrics(),
                "generated_at": datetime.now().isoformat()
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)