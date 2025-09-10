"""Parameter file generation for LISFLOOD-FP simulations.

This module generates LISFLOOD-FP parameter files (.par) from DEM and rainfall inputs,
supporting different return periods and hyetograph patterns for comprehensive
training dataset generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HyetographConfig:
    """Configuration for rainfall temporal distribution."""

    pattern_type: str  # 'uniform', 'front_loaded', 'center_loaded', 'back_loaded'
    total_depth_mm: float  # 24-hour total rainfall depth
    duration_hours: float = 24.0
    timestep_hours: float = 1.0

    # Pattern-specific parameters
    peak_hour: Optional[float] = None  # Hour of maximum intensity (for center_loaded)
    front_factor: float = 2.0  # Intensity multiplier for front-loaded pattern
    center_factor: float = 3.0  # Peak intensity multiplier for center-loaded

    def __post_init__(self):
        """Set default peak hour for center-loaded pattern."""
        if self.pattern_type == "center_loaded" and self.peak_hour is None:
            self.peak_hour = 12.0  # Peak at noon


@dataclass
class ReturnPeriodConfig:
    """Configuration for different return period scenarios."""

    return_period_years: int
    rainfall_depth_24h_mm: float
    description: str = ""

    # Sub-design scenarios for negative examples
    is_sub_design: bool = False

    def __post_init__(self):
        """Generate description if not provided."""
        if not self.description:
            self.description = f"{self.return_period_years}-year return period"


class ParameterFileGenerator:
    """Generates LISFLOOD-FP parameter files for simulation scenarios."""

    def __init__(
        self,
        base_config_dir: str = "LISFLOOD-FP/Nashville",
        template_dir: Optional[str] = None,
    ):
        """Initialize parameter file generator.

        Args:
            base_config_dir: Directory containing reference configuration files
            template_dir: Directory for parameter file templates
        """
        self.base_config_dir = Path(base_config_dir)
        self.template_dir = Path(template_dir) if template_dir else self.base_config_dir

        # Default file mapping for standard inputs
        self.default_files = {
            "dem_file": "final_dem.asc",
            "manning_file": "manning.asc",
            "infiltration_file": "infiltration.asc",
        }

        logger.info(
            f"Parameter generator initialized with base config: {self.base_config_dir}"
        )

    def generate_scenario_parameters(
        self,
        dem_file: str,
        return_periods: List[ReturnPeriodConfig],
        hyetograph_patterns: List[HyetographConfig],
        output_dir: str,
        base_simulation_config: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate parameter files for multiple scenarios.

        Args:
            dem_file: Path to DEM file
            return_periods: List of return period configurations
            hyetograph_patterns: List of rainfall pattern configurations
            output_dir: Directory to save parameter files
            base_simulation_config: Base configuration overrides

        Returns:
            List of generated scenario configurations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        scenarios = []

        for rp_config in return_periods:
            for hyet_config in hyetograph_patterns:
                # Create scenario ID
                scenario_id = self._create_scenario_id(rp_config, hyet_config)

                # Generate rainfall file
                rainfall_file = self._generate_rainfall_file(
                    hyet_config,
                    rp_config.rainfall_depth_24h_mm,
                    output_path / f"rainfall_{scenario_id}.rain",
                )

                # Generate parameter file
                par_config = self._create_parameter_config(
                    dem_file=dem_file,
                    rainfall_file=str(rainfall_file),
                    scenario_id=scenario_id,
                    base_config=base_simulation_config,
                )

                par_file = self._generate_parameter_file(
                    par_config, output_path / f"simulation_{scenario_id}.par"
                )

                # Create scenario metadata
                scenario = {
                    "scenario_id": scenario_id,
                    "return_period": asdict(rp_config),
                    "hyetograph": asdict(hyet_config),
                    "parameter_file": str(par_file),
                    "rainfall_file": str(rainfall_file),
                    "config": par_config,
                    # Store original paths for file copying
                    "original_dem_file": dem_file,
                    "created_at": datetime.now().isoformat(),
                }

                scenarios.append(scenario)

                logger.info(f"Generated scenario: {scenario_id}")

        # Save scenario manifest
        self._save_scenario_manifest(scenarios, output_path / "scenarios.json")

        logger.info(f"Generated {len(scenarios)} simulation scenarios in {output_path}")
        return scenarios

    def _create_scenario_id(
        self, rp_config: ReturnPeriodConfig, hyet_config: HyetographConfig
    ) -> str:
        """Create unique scenario identifier."""
        rp_part = f"{rp_config.return_period_years}yr"
        if rp_config.is_sub_design:
            rp_part += "_sub"

        pattern_part = hyet_config.pattern_type
        if hyet_config.pattern_type == "center_loaded":
            pattern_part += f"_h{int(hyet_config.peak_hour)}"

        return f"{rp_part}_{pattern_part}_{int(hyet_config.total_depth_mm)}mm"

    def _generate_rainfall_file(
        self, hyet_config: HyetographConfig, total_depth_mm: float, output_file: Path
    ) -> str:
        """Generate LISFLOOD-FP rainfall file with specified hyetograph.

        Args:
            hyet_config: Hyetograph configuration
            total_depth_mm: Total 24-hour rainfall depth
            output_file: Output file path

        Returns:
            Path to generated rainfall file
        """
        # Generate hourly intensities based on pattern type
        hourly_intensities = self._create_hyetograph_pattern(
            hyet_config, total_depth_mm
        )

        # Write LISFLOOD-FP rainfall file format
        lines = ["#rainfall file", f"{int(hyet_config.duration_hours)} hours"]

        for hour, intensity in enumerate(hourly_intensities):
            lines.append(f"{intensity:.6f}\t{hour}")

        output_file.write_text("\n".join(lines) + "\n")

        logger.debug(f"Generated rainfall file: {output_file}")
        logger.debug(
            f"Total depth: {np.sum(hourly_intensities):.2f}mm, "
            f"Peak intensity: {np.max(hourly_intensities):.2f}mm/h"
        )

        return str(output_file)

    def _create_hyetograph_pattern(
        self, config: HyetographConfig, total_depth_mm: float
    ) -> np.ndarray:
        """Create hourly intensity pattern based on hyetograph type."""
        n_hours = int(config.duration_hours / config.timestep_hours)
        hours = np.arange(n_hours) * config.timestep_hours

        if config.pattern_type == "uniform":
            intensities = np.full(n_hours, total_depth_mm / config.duration_hours)

        elif config.pattern_type == "front_loaded":
            # Higher intensities in first 6 hours
            intensities = np.ones(n_hours)
            intensities[:6] *= config.front_factor
            intensities = intensities * total_depth_mm / np.sum(intensities)

        elif config.pattern_type == "back_loaded":
            # Higher intensities in last 6 hours
            intensities = np.ones(n_hours)
            intensities[-6:] *= config.front_factor
            intensities = intensities * total_depth_mm / np.sum(intensities)

        elif config.pattern_type == "center_loaded":
            # Bell curve centered on peak_hour
            peak_hour = config.peak_hour or 12.0
            std_hours = 4.0  # Standard deviation for bell curve

            intensities = np.exp(-0.5 * ((hours - peak_hour) / std_hours) ** 2)
            intensities *= config.center_factor / np.max(intensities)

            # Add base level
            base_level = 0.2
            intensities = base_level + (1 - base_level) * intensities

            # Normalize to total depth
            intensities = intensities * total_depth_mm / np.sum(intensities)

        else:
            raise ValueError(f"Unknown hyetograph pattern: {config.pattern_type}")

        return intensities

    def _create_parameter_config(
        self,
        dem_file: str,
        rainfall_file: str,
        scenario_id: str,
        base_config: Optional[Dict] = None,
    ) -> Dict:
        """Create parameter configuration dictionary."""
        # Start with default configuration
        # Use just the filename since files are copied to the sim directory
        config = {
            "dem_file": os.path.basename(dem_file),
            "rainfall_file": os.path.basename(rainfall_file),
            "manning_file": self.default_files["manning_file"],
            "infiltration_file": self.default_files["infiltration_file"],
            "output_prefix": f"res_{scenario_id}",
            "output_directory": f"results_{scenario_id}",
            "sim_time": 86400.0,  # 24 hours
            "initial_timestep": 10.0,
            "acceleration": True,
            "depthoff": True,
            "elevoff": True,
        }

        # Apply any base configuration overrides
        if base_config:
            config.update(base_config)

        return config

    def _generate_parameter_file(self, config: Dict, output_file: Path) -> str:
        """Generate LISFLOOD-FP parameter file."""
        lines = []
        lines.append("# LISFLOOD-FP parameter file - Auto-generated")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Scenario: {config.get('output_prefix', 'unknown')}")
        lines.append("")

        # Essential parameters
        lines.append(f"DEMfile\t\t\t{config['dem_file']}")
        lines.append(f"resroot\t\t\t{config['output_prefix']}")
        lines.append(f"dirroot\t\t\t{config['output_directory']}")
        lines.append(f"sim_time\t\t{config['sim_time']}")
        lines.append(f"initial_tstep\t\t{config['initial_timestep']}")

        # Input files
        lines.append(f"rainfall\t\t{config['rainfall_file']}")
        lines.append(f"manningfile\t\t{config['manning_file']}")
        lines.append(f"infiltration\t\t{config['infiltration_file']}")

        # Optional files
        if "boundary_file" in config:
            lines.append(f"bcifile\t\t\t{config['boundary_file']}")

        if "floodplain_friction" in config:
            lines.append(f"fpfric\t\t\t{config['floodplain_friction']}")

        # Physics flags
        if config.get("depthoff", False):
            lines.append("depthoff")

        if config.get("elevoff", False):
            lines.append("elevoff")

        if config.get("acceleration", False):
            lines.append("acceleration")

        # Write file
        output_file.write_text("\n".join(lines) + "\n")

        logger.debug(f"Generated parameter file: {output_file}")
        return str(output_file)

    def _save_scenario_manifest(self, scenarios: List[Dict], output_file: Path):
        """Save scenario manifest as JSON."""
        import json

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "total_scenarios": len(scenarios),
            "scenarios": scenarios,
        }

        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Saved scenario manifest: {output_file}")

    @staticmethod
    def create_standard_return_periods() -> List[ReturnPeriodConfig]:
        """Create standard return period configurations for Nashville."""
        return [
            # Main design events for positive examples
            ReturnPeriodConfig(
                return_period_years=100,
                rainfall_depth_24h_mm=177.8,  # 7.01 inches
                description="100-year design storm - primary training target",
            ),
            ReturnPeriodConfig(
                return_period_years=500,
                rainfall_depth_24h_mm=222.25,  # 8.75 inches
                description="500-year extreme event - secondary training target",
            ),
            # Sub-design events for negative examples
            ReturnPeriodConfig(
                return_period_years=10,
                rainfall_depth_24h_mm=111.76,  # 4.4 inches
                description="10-year sub-design event - negative example",
                is_sub_design=True,
            ),
            ReturnPeriodConfig(
                return_period_years=25,
                rainfall_depth_24h_mm=142.24,  # 5.6 inches
                description="25-year sub-design event - negative example",
                is_sub_design=True,
            ),
        ]

    @staticmethod
    def create_standard_hyetographs() -> List[HyetographConfig]:
        """Create standard hyetograph patterns for diverse training."""
        return [
            HyetographConfig(
                pattern_type="uniform",
                total_depth_mm=0,  # Will be set by return period
                description="Uniform intensity distribution",
            ),
            HyetographConfig(
                pattern_type="front_loaded",
                total_depth_mm=0,
                front_factor=2.5,
                description="Front-loaded storm pattern",
            ),
            HyetographConfig(
                pattern_type="center_loaded",
                total_depth_mm=0,
                peak_hour=12.0,
                center_factor=3.0,
                description="Center-loaded storm pattern",
            ),
            HyetographConfig(
                pattern_type="back_loaded",
                total_depth_mm=0,
                front_factor=2.5,  # Reused for back loading
                description="Back-loaded storm pattern",
            ),
        ]
