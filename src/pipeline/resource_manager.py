"""Resource Management System for FloodRisk Pipeline.

This module provides monitoring and management of system resources
including memory, disk space, and CPU usage during pipeline execution.
"""

import os
import psutil
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
import json


logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    max_memory_gb: Optional[float] = None
    max_disk_space_gb: Optional[float] = None
    memory_warning_threshold: float = 0.85  # 85% of max
    disk_warning_threshold: float = 0.90    # 90% of max
    monitoring_interval_seconds: float = 30.0
    cleanup_intermediate: bool = True
    enable_memory_profiling: bool = True
    enable_disk_monitoring: bool = True


@dataclass 
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: datetime
    memory_used_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_available_gb: float
    disk_percent: float
    cpu_percent: float
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'memory_used_gb': self.memory_used_gb,
            'memory_percent': self.memory_percent,
            'disk_used_gb': self.disk_used_gb,
            'disk_available_gb': self.disk_available_gb,
            'disk_percent': self.disk_percent,
            'cpu_percent': self.cpu_percent,
            'process_count': self.process_count
        }


class ResourceManager:
    """Monitors and manages system resources during pipeline execution."""
    
    def __init__(self, config: ResourceConfig):
        """Initialize resource manager.
        
        Args:
            config: Resource management configuration
        """
        self.config = config
        self.monitoring = False
        self.snapshots: List[ResourceSnapshot] = []
        self.cleanup_callbacks: List[Callable] = []
        
        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Current process
        self.process = psutil.Process()
        
        # Resource limits
        self.memory_limit_bytes = None
        self.disk_limit_bytes = None
        
        if config.max_memory_gb:
            self.memory_limit_bytes = config.max_memory_gb * 1024**3
        if config.max_disk_space_gb:
            self.disk_limit_bytes = config.max_disk_space_gb * 1024**3
        
        # Warnings and alerts
        self.warnings_issued: List[str] = []
        self.last_warning_time = {}
        
        logger.info(f"ResourceManager initialized with limits: "
                   f"Memory={config.max_memory_gb}GB, Disk={config.max_disk_space_gb}GB")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            logger.warning("Resource monitoring already started")
            return
        
        self.monitoring = True
        self._stop_monitoring.clear()
        
        if self.config.enable_memory_profiling or self.config.enable_disk_monitoring:
            self._monitor_thread = threading.Thread(target=self._monitoring_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._stop_monitoring.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                
                # Check limits and issue warnings
                self._check_resource_limits(snapshot)
                
                # Trigger cleanup if needed
                if self._should_trigger_cleanup(snapshot):
                    self._trigger_cleanup()
                
                # Limit snapshot history (keep last 1000 snapshots)
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            self._stop_monitoring.wait(self.config.monitoring_interval_seconds)
    
    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage snapshot."""
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_used_gb = memory_info.used / 1024**3
        memory_percent = memory_info.percent
        
        # Disk usage (for current working directory)
        disk_info = psutil.disk_usage('.')
        disk_used_gb = disk_info.used / 1024**3
        disk_available_gb = disk_info.free / 1024**3
        disk_percent = (disk_info.used / disk_info.total) * 100
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Process count
        process_count = len(psutil.pids())
        
        return ResourceSnapshot(
            timestamp=datetime.now(),
            memory_used_gb=memory_used_gb,
            memory_percent=memory_percent,
            disk_used_gb=disk_used_gb,
            disk_available_gb=disk_available_gb,
            disk_percent=disk_percent,
            cpu_percent=cpu_percent,
            process_count=process_count
        )
    
    def _check_resource_limits(self, snapshot: ResourceSnapshot):
        """Check if resource limits are being approached or exceeded."""
        current_time = datetime.now()
        
        # Memory checks
        if self.memory_limit_bytes and self.config.enable_memory_profiling:
            memory_used_bytes = snapshot.memory_used_gb * 1024**3
            memory_usage_ratio = memory_used_bytes / self.memory_limit_bytes
            
            if memory_usage_ratio > self.config.memory_warning_threshold:
                self._issue_warning(
                    "memory_high",
                    f"Memory usage is high: {snapshot.memory_used_gb:.1f}GB "
                    f"({memory_usage_ratio:.1%} of limit)",
                    current_time
                )
            
            if memory_usage_ratio > 1.0:
                self._issue_warning(
                    "memory_exceeded",
                    f"Memory limit exceeded: {snapshot.memory_used_gb:.1f}GB > {self.memory_limit_bytes/1024**3:.1f}GB",
                    current_time
                )
        
        # Disk checks  
        if self.disk_limit_bytes and self.config.enable_disk_monitoring:
            disk_usage_ratio = snapshot.disk_percent / 100.0
            
            if disk_usage_ratio > self.config.disk_warning_threshold:
                self._issue_warning(
                    "disk_high",
                    f"Disk usage is high: {snapshot.disk_used_gb:.1f}GB "
                    f"({disk_usage_ratio:.1%})",
                    current_time
                )
    
    def _issue_warning(self, warning_type: str, message: str, timestamp: datetime):
        """Issue a resource warning."""
        # Avoid spamming warnings (only issue once per 5 minutes per type)
        if warning_type in self.last_warning_time:
            if timestamp - self.last_warning_time[warning_type] < timedelta(minutes=5):
                return
        
        self.last_warning_time[warning_type] = timestamp
        self.warnings_issued.append(f"{timestamp.isoformat()}: {message}")
        
        logger.warning(f"Resource Warning: {message}")
    
    def _should_trigger_cleanup(self, snapshot: ResourceSnapshot) -> bool:
        """Determine if automatic cleanup should be triggered."""
        if not self.config.cleanup_intermediate:
            return False
        
        # Trigger cleanup if memory or disk usage is very high
        memory_critical = False
        disk_critical = False
        
        if self.memory_limit_bytes:
            memory_usage_ratio = (snapshot.memory_used_gb * 1024**3) / self.memory_limit_bytes
            memory_critical = memory_usage_ratio > 0.95
        
        if self.disk_limit_bytes:
            disk_usage_ratio = snapshot.disk_percent / 100.0
            disk_critical = disk_usage_ratio > 0.95
        
        return memory_critical or disk_critical
    
    def _trigger_cleanup(self):
        """Trigger automatic cleanup of resources."""
        logger.info("Triggering automatic resource cleanup")
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function.
        
        Args:
            callback: Function to call when cleanup is triggered
        """
        self.cleanup_callbacks.append(callback)
    
    def get_current_usage(self) -> ResourceSnapshot:
        """Get current resource usage snapshot.
        
        Returns:
            Current resource usage snapshot
        """
        return self._capture_snapshot()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary.
        
        Returns:
            Dictionary containing resource usage statistics
        """
        if not self.snapshots:
            current = self._capture_snapshot()
            return {
                "monitoring_active": self.monitoring,
                "current": current.to_dict(),
                "history_available": False
            }
        
        current = self.snapshots[-1]
        
        # Calculate statistics over monitoring period
        memory_values = [s.memory_used_gb for s in self.snapshots]
        disk_values = [s.disk_used_gb for s in self.snapshots]
        cpu_values = [s.cpu_percent for s in self.snapshots]
        
        return {
            "monitoring_active": self.monitoring,
            "monitoring_duration_minutes": len(self.snapshots) * self.config.monitoring_interval_seconds / 60,
            "snapshots_collected": len(self.snapshots),
            "current": current.to_dict(),
            "statistics": {
                "memory_gb": {
                    "current": memory_values[-1],
                    "average": sum(memory_values) / len(memory_values),
                    "peak": max(memory_values),
                    "minimum": min(memory_values)
                },
                "disk_gb": {
                    "current": disk_values[-1],
                    "average": sum(disk_values) / len(disk_values),
                    "peak": max(disk_values),
                    "minimum": min(disk_values)
                },
                "cpu_percent": {
                    "current": cpu_values[-1],
                    "average": sum(cpu_values) / len(cpu_values),
                    "peak": max(cpu_values),
                    "minimum": min(cpu_values)
                }
            },
            "limits": {
                "memory_limit_gb": self.config.max_memory_gb,
                "disk_limit_gb": self.config.max_disk_space_gb
            },
            "warnings_issued": len(self.warnings_issued),
            "warnings": self.warnings_issued[-10:]  # Last 10 warnings
        }
    
    def get_resource_trend(self, minutes: int = 60) -> Dict[str, List[Dict]]:
        """Get resource usage trend over specified time period.
        
        Args:
            minutes: Time period in minutes
            
        Returns:
            Dictionary containing trend data
        """
        if not self.snapshots:
            return {"memory": [], "disk": [], "cpu": []}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_snapshots = [
            s for s in self.snapshots
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            recent_snapshots = self.snapshots[-1:]
        
        return {
            "memory": [
                {"timestamp": s.timestamp.isoformat(), "value": s.memory_used_gb}
                for s in recent_snapshots
            ],
            "disk": [
                {"timestamp": s.timestamp.isoformat(), "value": s.disk_used_gb}
                for s in recent_snapshots
            ],
            "cpu": [
                {"timestamp": s.timestamp.isoformat(), "value": s.cpu_percent}
                for s in recent_snapshots
            ]
        }
    
    def export_resource_report(self, output_file: Path):
        """Export detailed resource usage report.
        
        Args:
            output_file: Path to save the report
        """
        report = {
            "resource_manager_config": self.config.__dict__,
            "usage_summary": self.get_usage_summary(),
            "trend_data": self.get_resource_trend(minutes=120),  # 2 hours
            "all_snapshots": [s.to_dict() for s in self.snapshots],
            "generated_at": datetime.now().isoformat()
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resource report exported to: {output_file}")
    
    def check_available_resources(self) -> Dict[str, bool]:
        """Check if sufficient resources are available for pipeline execution.
        
        Returns:
            Dictionary indicating resource availability status
        """
        snapshot = self._capture_snapshot()
        
        status = {
            "memory_available": True,
            "disk_available": True,
            "overall_status": True
        }
        
        # Check memory
        if self.memory_limit_bytes:
            memory_used_bytes = snapshot.memory_used_gb * 1024**3
            memory_ratio = memory_used_bytes / self.memory_limit_bytes
            
            if memory_ratio > 0.9:  # 90% threshold
                status["memory_available"] = False
                status["memory_message"] = f"Memory usage too high: {memory_ratio:.1%}"
        
        # Check disk space
        if snapshot.disk_available_gb < 1.0:  # Less than 1GB available
            status["disk_available"] = False
            status["disk_message"] = f"Low disk space: {snapshot.disk_available_gb:.1f}GB available"
        
        status["overall_status"] = status["memory_available"] and status["disk_available"]
        
        return status
    
    def estimate_pipeline_resource_needs(self, 
                                       num_simulations: int,
                                       dem_size_mb: float,
                                       enable_ml_training: bool = True) -> Dict[str, float]:
        """Estimate resource needs for pipeline execution.
        
        Args:
            num_simulations: Number of flood simulations to run
            dem_size_mb: Size of DEM file in MB
            enable_ml_training: Whether ML training is enabled
            
        Returns:
            Dictionary with estimated resource requirements
        """
        # Base memory requirements
        base_memory_gb = 2.0  # Python runtime, libraries, etc.
        
        # DEM processing memory (typically 3-4x the DEM size)
        dem_processing_memory_gb = (dem_size_mb / 1024) * 3.5
        
        # Simulation memory (per parallel simulation)
        sim_memory_per_job_gb = 0.5  # LISFLOOD-FP memory usage
        parallel_sims = min(num_simulations, 4)  # Assume max 4 parallel
        simulation_memory_gb = sim_memory_per_job_gb * parallel_sims
        
        # ML training memory (if enabled)
        ml_memory_gb = 4.0 if enable_ml_training else 0.0
        
        # Total memory estimate
        total_memory_gb = (base_memory_gb + dem_processing_memory_gb + 
                          simulation_memory_gb + ml_memory_gb)
        
        # Disk space requirements
        # DEM and derived products
        dem_products_gb = (dem_size_mb / 1024) * 5  # DEM + derived features
        
        # Simulation outputs (depth + extent rasters per simulation)
        sim_output_per_scenario_mb = dem_size_mb * 2  # Depth + extent
        total_sim_outputs_gb = (sim_output_per_scenario_mb * num_simulations) / 1024
        
        # Intermediate processing files
        intermediate_gb = total_sim_outputs_gb * 0.5
        
        # ML training data and models
        ml_disk_gb = total_sim_outputs_gb * 0.3 if enable_ml_training else 0.0
        
        # Total disk estimate (with 20% buffer)
        total_disk_gb = (dem_products_gb + total_sim_outputs_gb + 
                        intermediate_gb + ml_disk_gb) * 1.2
        
        return {
            "estimated_memory_gb": total_memory_gb,
            "estimated_disk_gb": total_disk_gb,
            "breakdown": {
                "base_memory_gb": base_memory_gb,
                "dem_processing_memory_gb": dem_processing_memory_gb,
                "simulation_memory_gb": simulation_memory_gb,
                "ml_memory_gb": ml_memory_gb,
                "dem_products_disk_gb": dem_products_gb,
                "simulation_outputs_disk_gb": total_sim_outputs_gb,
                "intermediate_files_disk_gb": intermediate_gb,
                "ml_training_disk_gb": ml_disk_gb
            }
        }
    
    def cleanup_temporary_files(self, directories: List[Path]):
        """Clean up temporary files in specified directories.
        
        Args:
            directories: List of directories to clean
        """
        cleaned_files = 0
        cleaned_size_mb = 0
        
        for directory in directories:
            if not directory.exists():
                continue
            
            try:
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        # Only clean temporary and intermediate files
                        if any(pattern in file_path.name.lower() 
                              for pattern in ['temp', 'tmp', 'intermediate', 'cache']):
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleaned_files += 1
                            cleaned_size_mb += file_size / 1024**2
                            
            except Exception as e:
                logger.warning(f"Error cleaning directory {directory}: {e}")
        
        if cleaned_files > 0:
            logger.info(f"Cleaned up {cleaned_files} temporary files, "
                       f"freed {cleaned_size_mb:.1f} MB")
        
        return {"files_cleaned": cleaned_files, "size_freed_mb": cleaned_size_mb}