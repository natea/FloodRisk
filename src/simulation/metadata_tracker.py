"""Metadata tracking and provenance system for simulation pipeline.

This module provides comprehensive tracking of simulation metadata,
provenance, and lineage for reproducibility and quality assurance.
"""

import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for input/output files."""

    file_path: str
    file_size_bytes: int
    checksum_md5: str
    created_at: str
    modified_at: str
    file_type: str  # 'dem', 'rainfall', 'manning', 'infiltration', 'depth', 'extent'

    # Spatial metadata (if applicable)
    spatial_bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    crs: Optional[str] = None
    pixel_size_m: Optional[float] = None
    grid_shape: Optional[List[int]] = None  # [height, width]


@dataclass
class SimulationProvenance:
    """Complete provenance record for a simulation."""

    # Unique identifiers
    simulation_id: str
    provenance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Input provenance
    input_files: Dict[str, FileMetadata] = field(default_factory=dict)
    parameter_config: Dict = field(default_factory=dict)

    # Processing provenance
    software_version: Dict = field(default_factory=dict)  # LISFLOOD-FP, Python versions
    execution_environment: Dict = field(default_factory=dict)  # OS, hardware
    processing_parameters: Dict = field(default_factory=dict)

    # Output provenance
    output_files: Dict[str, FileMetadata] = field(default_factory=dict)
    processing_statistics: Dict = field(default_factory=dict)

    # Quality assurance
    validation_results: Dict = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)

    # Temporal tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_start: Optional[str] = None
    processing_end: Optional[str] = None

    # Chain of processing
    parent_simulations: List[str] = field(default_factory=list)  # For derived products
    child_simulations: List[str] = field(default_factory=list)  # For downstream usage


@dataclass
class BatchProvenance:
    """Provenance for batch simulation execution."""

    batch_id: str
    provenance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Batch configuration
    batch_config: Dict = field(default_factory=dict)
    scenario_definitions: List[Dict] = field(default_factory=list)

    # Execution tracking
    individual_simulations: List[str] = field(default_factory=list)  # simulation_ids
    execution_summary: Dict = field(default_factory=dict)

    # Aggregate quality metrics
    batch_validation: Dict = field(default_factory=dict)

    # Temporal tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    batch_start: Optional[str] = None
    batch_end: Optional[str] = None


class SimulationMetadata:
    """Comprehensive metadata tracking system for flood simulations."""

    def __init__(
        self,
        metadata_dir: str = "metadata",
        enable_file_checksums: bool = True,
        track_environment: bool = True,
    ):
        """Initialize metadata tracker.

        Args:
            metadata_dir: Directory to store metadata files
            enable_file_checksums: Whether to calculate file checksums
            track_environment: Whether to track execution environment
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.enable_checksums = enable_file_checksums
        self.track_environment = track_environment

        # Initialize tracking database (JSON-based for simplicity)
        self.simulation_db = self.metadata_dir / "simulations.json"
        self.batch_db = self.metadata_dir / "batches.json"

        self._init_databases()

        logger.info(f"SimulationMetadata initialized with metadata dir: {metadata_dir}")

    def _init_databases(self):
        """Initialize metadata databases."""
        if not self.simulation_db.exists():
            self._save_json({}, self.simulation_db)

        if not self.batch_db.exists():
            self._save_json({}, self.batch_db)

    def create_simulation_provenance(
        self, simulation_id: str, input_files: Dict[str, str], parameter_config: Dict
    ) -> SimulationProvenance:
        """Create provenance record for a simulation.

        Args:
            simulation_id: Unique simulation identifier
            input_files: Dictionary mapping file types to paths
            parameter_config: Simulation configuration parameters

        Returns:
            SimulationProvenance object
        """
        provenance = SimulationProvenance(
            simulation_id=simulation_id, parameter_config=parameter_config.copy()
        )

        # Process input files
        for file_type, file_path in input_files.items():
            if Path(file_path).exists():
                file_metadata = self._create_file_metadata(file_path, file_type)
                provenance.input_files[file_type] = file_metadata
            else:
                logger.warning(f"Input file not found: {file_path}")

        # Capture software and environment info
        if self.track_environment:
            provenance.software_version = self._get_software_versions()
            provenance.execution_environment = self._get_execution_environment()

        provenance.processing_start = datetime.now().isoformat()

        logger.info(f"Created simulation provenance: {simulation_id}")
        return provenance

    def update_simulation_outputs(
        self,
        provenance: SimulationProvenance,
        output_files: Dict[str, str],
        processing_stats: Dict,
        validation_results: Optional[Dict] = None,
    ):
        """Update provenance with simulation outputs.

        Args:
            provenance: SimulationProvenance object to update
            output_files: Dictionary mapping file types to output paths
            processing_stats: Processing statistics
            validation_results: Validation results
        """
        # Process output files
        for file_type, file_path in output_files.items():
            if Path(file_path).exists():
                file_metadata = self._create_file_metadata(file_path, file_type)
                provenance.output_files[file_type] = file_metadata

        # Update processing information
        provenance.processing_statistics = processing_stats.copy()
        provenance.processing_end = datetime.now().isoformat()

        if validation_results:
            provenance.validation_results = validation_results.copy()

            # Extract quality flags
            if validation_results.get("status") == "failed":
                provenance.quality_flags.append("validation_failed")
            elif validation_results.get("status") == "warning":
                provenance.quality_flags.append("validation_warnings")

            # Add specific quality flags based on validation issues
            for error in validation_results.get("errors", []):
                provenance.quality_flags.append(f"error_{error.split(':')[0].lower()}")

        logger.debug(f"Updated simulation outputs for: {provenance.simulation_id}")

    def save_simulation_provenance(self, provenance: SimulationProvenance):
        """Save simulation provenance to database.

        Args:
            provenance: SimulationProvenance object to save
        """
        # Load existing database
        db = self._load_json(self.simulation_db)

        # Add or update simulation
        db[provenance.simulation_id] = asdict(provenance)

        # Save database
        self._save_json(db, self.simulation_db)

        # Also save individual provenance file
        individual_file = self.metadata_dir / f"sim_{provenance.simulation_id}.json"
        self._save_json(asdict(provenance), individual_file)

        logger.info(f"Saved simulation provenance: {provenance.simulation_id}")

    def create_batch_provenance(
        self, batch_id: str, batch_config: Dict, scenario_definitions: List[Dict]
    ) -> BatchProvenance:
        """Create provenance record for batch execution.

        Args:
            batch_id: Unique batch identifier
            batch_config: Batch configuration
            scenario_definitions: List of scenario configurations

        Returns:
            BatchProvenance object
        """
        provenance = BatchProvenance(
            batch_id=batch_id,
            batch_config=batch_config.copy(),
            scenario_definitions=[s.copy() for s in scenario_definitions],
        )

        provenance.batch_start = datetime.now().isoformat()

        logger.info(f"Created batch provenance: {batch_id}")
        return provenance

    def update_batch_execution(
        self,
        provenance: BatchProvenance,
        individual_results: List[Dict],
        batch_summary: Dict,
        validation_summary: Optional[Dict] = None,
    ):
        """Update batch provenance with execution results.

        Args:
            provenance: BatchProvenance object to update
            individual_results: List of individual simulation results
            batch_summary: Batch execution summary
            validation_summary: Batch validation summary
        """
        # Extract simulation IDs
        provenance.individual_simulations = [
            result.get("simulation_id")
            for result in individual_results
            if result.get("simulation_id")
        ]

        provenance.execution_summary = batch_summary.copy()
        provenance.batch_end = datetime.now().isoformat()

        if validation_summary:
            provenance.batch_validation = validation_summary.copy()

        logger.debug(f"Updated batch execution for: {provenance.batch_id}")

    def save_batch_provenance(self, provenance: BatchProvenance):
        """Save batch provenance to database.

        Args:
            provenance: BatchProvenance object to save
        """
        # Load existing database
        db = self._load_json(self.batch_db)

        # Add or update batch
        db[provenance.batch_id] = asdict(provenance)

        # Save database
        self._save_json(db, self.batch_db)

        # Also save individual batch file
        individual_file = self.metadata_dir / f"batch_{provenance.batch_id}.json"
        self._save_json(asdict(provenance), individual_file)

        logger.info(f"Saved batch provenance: {provenance.batch_id}")

    def get_simulation_provenance(
        self, simulation_id: str
    ) -> Optional[SimulationProvenance]:
        """Retrieve simulation provenance by ID.

        Args:
            simulation_id: Simulation identifier

        Returns:
            SimulationProvenance object or None if not found
        """
        db = self._load_json(self.simulation_db)

        if simulation_id in db:
            return SimulationProvenance(**db[simulation_id])

        return None

    def get_batch_provenance(self, batch_id: str) -> Optional[BatchProvenance]:
        """Retrieve batch provenance by ID.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchProvenance object or None if not found
        """
        db = self._load_json(self.batch_db)

        if batch_id in db:
            return BatchProvenance(**db[batch_id])

        return None

    def query_simulations(
        self,
        quality_flags: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        return_periods: Optional[List[int]] = None,
    ) -> List[str]:
        """Query simulations based on criteria.

        Args:
            quality_flags: Required quality flags
            date_range: Date range tuple (start, end) in ISO format
            return_periods: Required return periods

        Returns:
            List of simulation IDs matching criteria
        """
        db = self._load_json(self.simulation_db)
        matches = []

        for sim_id, sim_data in db.items():
            # Check quality flags
            if quality_flags:
                sim_flags = sim_data.get("quality_flags", [])
                if not all(flag in sim_flags for flag in quality_flags):
                    continue

            # Check date range
            if date_range:
                created_at = sim_data.get("created_at")
                if created_at:
                    if not (date_range[0] <= created_at <= date_range[1]):
                        continue

            # Check return periods (from parameter config)
            if return_periods:
                param_config = sim_data.get("parameter_config", {})
                # This would need to be adapted based on actual parameter structure
                # For now, assuming return period is stored in config
                sim_rp = param_config.get("return_period")
                if sim_rp not in return_periods:
                    continue

            matches.append(sim_id)

        logger.info(f"Query found {len(matches)} simulations")
        return matches

    def create_lineage_report(self, simulation_id: str) -> Dict:
        """Create complete lineage report for a simulation.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Comprehensive lineage report
        """
        provenance = self.get_simulation_provenance(simulation_id)

        if not provenance:
            raise ValueError(f"Simulation not found: {simulation_id}")

        # Calculate processing duration
        processing_duration = None
        if provenance.processing_start and provenance.processing_end:
            start = datetime.fromisoformat(
                provenance.processing_start.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                provenance.processing_end.replace("Z", "+00:00")
            )
            processing_duration = (end - start).total_seconds()

        report = {
            "simulation_id": simulation_id,
            "provenance_id": provenance.provenance_id,
            "created_at": provenance.created_at,
            "processing_duration_seconds": processing_duration,
            "input_summary": {
                "file_count": len(provenance.input_files),
                "file_types": list(provenance.input_files.keys()),
                "total_input_size_mb": sum(
                    f.file_size_bytes for f in provenance.input_files.values()
                )
                / 1024
                / 1024,
            },
            "output_summary": {
                "file_count": len(provenance.output_files),
                "file_types": list(provenance.output_files.keys()),
                "total_output_size_mb": sum(
                    f.file_size_bytes for f in provenance.output_files.values()
                )
                / 1024
                / 1024,
            },
            "quality_assessment": {
                "quality_flags": provenance.quality_flags,
                "validation_status": provenance.validation_results.get(
                    "status", "unknown"
                ),
                "has_warnings": len(provenance.validation_results.get("warnings", []))
                > 0,
                "has_errors": len(provenance.validation_results.get("errors", [])) > 0,
            },
            "parameter_hash": self._hash_dict(provenance.parameter_config),
            "input_file_hashes": {
                file_type: metadata.checksum_md5
                for file_type, metadata in provenance.input_files.items()
            },
            "reproducibility_info": {
                "software_versions": provenance.software_version,
                "execution_environment": provenance.execution_environment,
                "parameter_config": provenance.parameter_config,
            },
        }

        return report

    def export_metadata(self, output_file: str, format: str = "json"):
        """Export all metadata to file.

        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == "json":
            self._export_json(output_file)
        elif format == "csv":
            self._export_csv(output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _create_file_metadata(self, file_path: str, file_type: str) -> FileMetadata:
        """Create metadata for a file."""
        path_obj = Path(file_path)
        stat = path_obj.stat()

        # Calculate checksum if enabled
        checksum = ""
        if self.enable_checksums:
            checksum = self._calculate_md5(file_path)

        metadata = FileMetadata(
            file_path=str(path_obj.absolute()),
            file_size_bytes=stat.st_size,
            checksum_md5=checksum,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            file_type=file_type,
        )

        # Add spatial metadata for raster files if possible
        if file_type in ["dem", "depth", "extent"] and self._has_rasterio():
            try:
                import rasterio

                with rasterio.open(file_path) as src:
                    metadata.spatial_bounds = list(src.bounds)
                    metadata.crs = str(src.crs)
                    metadata.pixel_size_m = abs(
                        src.transform[0]
                    )  # Assuming square pixels
                    metadata.grid_shape = [src.height, src.width]
            except:
                pass  # Ignore spatial metadata errors

        return metadata

    def _calculate_md5(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""

    def _hash_dict(self, d: Dict) -> str:
        """Calculate hash of dictionary for reproducibility."""
        json_str = json.dumps(d, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _get_software_versions(self) -> Dict:
        """Get software version information."""
        import sys
        import platform

        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "floodrisk_version": "1.0.0",  # Would come from package info
        }

        # Try to get LISFLOOD-FP version (placeholder)
        try:
            # This would need to be implemented based on LISFLOOD-FP version detection
            versions["lisflood_fp"] = "unknown"
        except:
            pass

        return versions

    def _get_execution_environment(self) -> Dict:
        """Get execution environment information."""
        import platform
        import os

        env = {
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "working_directory": str(Path.cwd()),
            "user": os.environ.get("USER", "unknown"),
        }

        # Add memory info if available
        try:
            import psutil

            env["memory_gb"] = psutil.virtual_memory().total / 1024**3
        except ImportError:
            pass

        return env

    def _has_rasterio(self) -> bool:
        """Check if rasterio is available."""
        try:
            import rasterio

            return True
        except ImportError:
            return False

    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_json(self, data: Dict, file_path: Path):
        """Save data as JSON."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_json(self, output_file: str):
        """Export metadata as JSON."""
        simulations = self._load_json(self.simulation_db)
        batches = self._load_json(self.batch_db)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "simulations": simulations,
            "batches": batches,
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metadata to {output_file}")

    def _export_csv(self, output_file: str):
        """Export metadata as CSV (simplified)."""
        import csv

        simulations = self._load_json(self.simulation_db)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "simulation_id",
                    "created_at",
                    "processing_start",
                    "processing_end",
                    "input_file_count",
                    "output_file_count",
                    "quality_flags",
                    "validation_status",
                    "max_depth_m",
                    "flood_fraction",
                ]
            )

            # Data rows
            for sim_id, sim_data in simulations.items():
                stats = sim_data.get("processing_statistics", {})
                validation = sim_data.get("validation_results", {})

                writer.writerow(
                    [
                        sim_id,
                        sim_data.get("created_at", ""),
                        sim_data.get("processing_start", ""),
                        sim_data.get("processing_end", ""),
                        len(sim_data.get("input_files", {})),
                        len(sim_data.get("output_files", {})),
                        ";".join(sim_data.get("quality_flags", [])),
                        validation.get("status", ""),
                        stats.get("max_depth_m", ""),
                        stats.get("flood_fraction", ""),
                    ]
                )

        logger.info(f"Exported metadata CSV to {output_file}")
