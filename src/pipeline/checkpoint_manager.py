"""Checkpoint and Recovery System for FloodRisk Pipeline.

This module provides checkpoint/resume functionality for long-running
pipeline executions, enabling recovery from failures and interruptions.
"""

import os
import json
import pickle
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    DATA_ACQUISITION = "data_acquisition" 
    DATA_PREPROCESSING = "data_preprocessing"
    SIMULATION_SETUP = "simulation_setup"
    SIMULATION_EXECUTION = "simulation_execution"
    SIMULATION_VALIDATION = "simulation_validation"
    ML_TRAINING_SETUP = "ml_training_setup"
    ML_TRAINING_EXECUTION = "ml_training_execution"
    FINAL_VALIDATION = "final_validation"
    COMPLETION = "completion"


@dataclass
class PipelineState:
    """Complete pipeline state for checkpointing."""
    pipeline_id: str
    stage: PipelineStage
    timestamp: datetime
    results: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    progress_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        state_dict = asdict(self)
        state_dict['stage'] = self.stage.value
        state_dict['timestamp'] = self.timestamp.isoformat()
        return state_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create from dictionary."""
        # Convert stage enum
        data['stage'] = PipelineStage(data['stage'])
        # Convert timestamp
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    pipeline_id: str
    stage: PipelineStage
    timestamp: datetime
    checkpoint_size_bytes: int
    file_path: str
    valid: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'pipeline_id': self.pipeline_id,
            'stage': self.stage.value,
            'timestamp': self.timestamp.isoformat(),
            'checkpoint_size_bytes': self.checkpoint_size_bytes,
            'file_path': self.file_path,
            'valid': self.valid,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        data['stage'] = PipelineStage(data['stage'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CheckpointManager:
    """Manages pipeline checkpoints and recovery."""
    
    def __init__(self, checkpoint_dir: Path, auto_save_interval: float = 30.0):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            auto_save_interval: Minutes between auto-saves (0 to disable)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # State tracking
        self.current_state: Optional[PipelineState] = None
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.auto_save_enabled = auto_save_interval > 0
        
        # Auto-save thread
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()
        
        # Load existing checkpoints
        self._load_checkpoint_metadata()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")
    
    def save_checkpoint(self, state: PipelineState, description: str = "") -> str:
        """Save a pipeline checkpoint.
        
        Args:
            state: Current pipeline state
            description: Optional checkpoint description
            
        Returns:
            Checkpoint ID
        """
        try:
            checkpoint_id = self._generate_checkpoint_id(state)
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            # Save state to file
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Create metadata
            file_size = checkpoint_file.stat().st_size
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                pipeline_id=state.pipeline_id,
                stage=state.stage,
                timestamp=state.timestamp,
                checkpoint_size_bytes=file_size,
                file_path=str(checkpoint_file),
                description=description
            )
            
            # Store metadata
            self.checkpoints[checkpoint_id] = metadata
            self.current_state = state
            
            # Save metadata index
            self._save_checkpoint_metadata()
            
            logger.info(f"Checkpoint saved: {checkpoint_id} ({file_size/1024:.1f} KB)")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[PipelineState]:
        """Load a pipeline checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Pipeline state or None if not found
        """
        if checkpoint_id not in self.checkpoints:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None
        
        metadata = self.checkpoints[checkpoint_id]
        checkpoint_file = Path(metadata.file_path)
        
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint file missing: {checkpoint_file}")
            metadata.valid = False
            self._save_checkpoint_metadata()
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            self.current_state = state
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_latest_checkpoint(self, pipeline_id: Optional[str] = None) -> Optional[PipelineState]:
        """Get the most recent checkpoint.
        
        Args:
            pipeline_id: Optional pipeline ID filter
            
        Returns:
            Latest pipeline state or None
        """
        valid_checkpoints = [
            metadata for metadata in self.checkpoints.values()
            if metadata.valid and (pipeline_id is None or metadata.pipeline_id == pipeline_id)
        ]
        
        if not valid_checkpoints:
            return None
        
        # Sort by timestamp and get latest
        latest = sorted(valid_checkpoints, key=lambda x: x.timestamp)[-1]
        return self.load_checkpoint(latest.checkpoint_id)
    
    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List available checkpoints.
        
        Args:
            pipeline_id: Optional pipeline ID filter
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = list(self.checkpoints.values())
        
        if pipeline_id:
            checkpoints = [cp for cp in checkpoints if cp.pipeline_id == pipeline_id]
        
        # Sort by timestamp (newest first)
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        if checkpoint_id not in self.checkpoints:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        metadata = self.checkpoints[checkpoint_id]
        checkpoint_file = Path(metadata.file_path)
        
        try:
            # Delete checkpoint file
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            # Remove from metadata
            del self.checkpoints[checkpoint_id]
            self._save_checkpoint_metadata()
            
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_count: int = 10, 
                               max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old checkpoints.
        
        Args:
            keep_count: Number of recent checkpoints to keep per pipeline
            max_age_days: Maximum age in days
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0
        freed_bytes = 0
        
        # Group by pipeline
        pipeline_checkpoints = {}
        for cp_id, metadata in self.checkpoints.items():
            if metadata.pipeline_id not in pipeline_checkpoints:
                pipeline_checkpoints[metadata.pipeline_id] = []
            pipeline_checkpoints[metadata.pipeline_id].append((cp_id, metadata))
        
        # Clean up each pipeline's checkpoints
        for pipeline_id, checkpoints in pipeline_checkpoints.items():
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
            
            # Keep recent checkpoints, delete old ones
            for i, (cp_id, metadata) in enumerate(checkpoints):
                should_delete = False
                
                # Delete if too old
                if metadata.timestamp < cutoff_date:
                    should_delete = True
                    
                # Delete if beyond keep count (but always keep at least 2)
                elif i >= keep_count and i >= 2:
                    should_delete = True
                
                if should_delete:
                    freed_bytes += metadata.checkpoint_size_bytes
                    if self.delete_checkpoint(cp_id):
                        deleted_count += 1
        
        logger.info(f"Cleanup completed: deleted {deleted_count} checkpoints, "
                   f"freed {freed_bytes/1024/1024:.1f} MB")
        
        return {
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes,
            "remaining_checkpoints": len(self.checkpoints)
        }
    
    def start_auto_save(self, state: PipelineState):
        """Start automatic checkpoint saving.
        
        Args:
            state: Initial pipeline state
        """
        if not self.auto_save_enabled:
            return
        
        self.current_state = state
        
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self.stop_auto_save()
        
        self._stop_auto_save.clear()
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop)
        self._auto_save_thread.daemon = True
        self._auto_save_thread.start()
        
        logger.info(f"Auto-save started (interval: {self.auto_save_interval} minutes)")
    
    def stop_auto_save(self):
        """Stop automatic checkpoint saving."""
        if self._auto_save_thread:
            self._stop_auto_save.set()
            self._auto_save_thread.join(timeout=5.0)
            self._auto_save_thread = None
        
        logger.info("Auto-save stopped")
    
    def _auto_save_loop(self):
        """Auto-save loop."""
        interval_seconds = self.auto_save_interval * 60
        
        while not self._stop_auto_save.is_set():
            self._stop_auto_save.wait(interval_seconds)
            
            if not self._stop_auto_save.is_set() and self.current_state:
                try:
                    # Update timestamp for auto-save
                    self.current_state.timestamp = datetime.now()
                    self.save_checkpoint(self.current_state, "Auto-save")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
    
    def update_current_state(self, **updates):
        """Update current pipeline state.
        
        Args:
            **updates: State updates to apply
        """
        if self.current_state:
            for key, value in updates.items():
                if hasattr(self.current_state, key):
                    setattr(self.current_state, key, value)
            
            # Update timestamp
            self.current_state.timestamp = datetime.now()
    
    def save_error_state(self, stage: PipelineStage, error: str, pipeline_id: str):
        """Save error state for recovery analysis.
        
        Args:
            stage: Stage where error occurred
            error: Error message
            pipeline_id: Pipeline ID
        """
        error_state = PipelineState(
            pipeline_id=pipeline_id,
            stage=stage,
            timestamp=datetime.now(),
            results={"error": error, "failed_at": stage.value}
        )
        
        try:
            checkpoint_id = self.save_checkpoint(
                error_state, 
                f"Error state - {error[:50]}..."
            )
            logger.info(f"Error state saved: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to save error state: {e}")
    
    def get_recovery_options(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """Get available recovery options for a failed pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            List of recovery options
        """
        pipeline_checkpoints = [
            cp for cp in self.checkpoints.values()
            if cp.pipeline_id == pipeline_id and cp.valid
        ]
        
        if not pipeline_checkpoints:
            return []
        
        # Sort by stage order for recovery recommendations
        stage_order = list(PipelineStage)
        pipeline_checkpoints.sort(
            key=lambda x: (stage_order.index(x.stage), x.timestamp),
            reverse=True
        )
        
        recovery_options = []
        for cp in pipeline_checkpoints:
            # Calculate what stages would need to be re-run
            current_stage_idx = stage_order.index(cp.stage)
            remaining_stages = [s.value for s in stage_order[current_stage_idx:]]
            
            option = {
                "checkpoint_id": cp.checkpoint_id,
                "stage": cp.stage.value,
                "timestamp": cp.timestamp.isoformat(),
                "description": cp.description,
                "remaining_stages": remaining_stages,
                "estimated_time_saved_percent": (current_stage_idx / len(stage_order)) * 100
            }
            recovery_options.append(option)
        
        return recovery_options
    
    def create_recovery_plan(self, checkpoint_id: str) -> Dict[str, Any]:
        """Create a recovery plan from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to recover from
            
        Returns:
            Recovery plan details
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        metadata = self.checkpoints[checkpoint_id]
        state = self.load_checkpoint(checkpoint_id)
        
        if not state:
            raise ValueError(f"Could not load checkpoint: {checkpoint_id}")
        
        # Determine remaining stages
        stage_order = list(PipelineStage)
        current_stage_idx = stage_order.index(state.stage)
        remaining_stages = stage_order[current_stage_idx:]
        
        recovery_plan = {
            "checkpoint_id": checkpoint_id,
            "pipeline_id": state.pipeline_id,
            "recovery_stage": state.stage.value,
            "checkpoint_timestamp": metadata.timestamp.isoformat(),
            "remaining_stages": [s.value for s in remaining_stages],
            "completed_results": list(state.results.keys()),
            "configuration": state.configuration,
            "estimated_completion_percentage": (current_stage_idx / len(stage_order)) * 100,
            "recovery_instructions": [
                f"1. Load checkpoint: {checkpoint_id}",
                f"2. Resume from stage: {state.stage.value}",
                f"3. Execute remaining {len(remaining_stages)} stages",
                "4. Monitor for any configuration changes needed"
            ]
        }
        
        return recovery_plan
    
    def _generate_checkpoint_id(self, state: PipelineState) -> str:
        """Generate unique checkpoint ID."""
        # Use pipeline ID, stage, and timestamp hash
        data = f"{state.pipeline_id}_{state.stage.value}_{state.timestamp.isoformat()}"
        hash_digest = hashlib.md5(data.encode()).hexdigest()[:8]
        return f"cp_{state.pipeline_id}_{state.stage.value}_{hash_digest}"
    
    def _load_checkpoint_metadata(self):
        """Load checkpoint metadata from disk."""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for cp_id, cp_data in data.items():
                metadata = CheckpointMetadata.from_dict(cp_data)
                self.checkpoints[cp_id] = metadata
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata: {e}")
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            metadata_dict = {
                cp_id: metadata.to_dict()
                for cp_id, metadata in self.checkpoints.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint system summary.
        
        Returns:
            Dictionary with checkpoint system status
        """
        total_size = sum(cp.checkpoint_size_bytes for cp in self.checkpoints.values())
        valid_checkpoints = [cp for cp in self.checkpoints.values() if cp.valid]
        
        # Group by pipeline
        pipeline_counts = {}
        for cp in self.checkpoints.values():
            pipeline_counts[cp.pipeline_id] = pipeline_counts.get(cp.pipeline_id, 0) + 1
        
        return {
            "checkpoint_directory": str(self.checkpoint_dir),
            "total_checkpoints": len(self.checkpoints),
            "valid_checkpoints": len(valid_checkpoints),
            "total_size_mb": total_size / 1024 / 1024,
            "auto_save_enabled": self.auto_save_enabled,
            "auto_save_interval_minutes": self.auto_save_interval,
            "pipeline_counts": pipeline_counts,
            "oldest_checkpoint": min(
                (cp.timestamp for cp in self.checkpoints.values()),
                default=None
            ),
            "newest_checkpoint": max(
                (cp.timestamp for cp in self.checkpoints.values()),
                default=None
            )
        }