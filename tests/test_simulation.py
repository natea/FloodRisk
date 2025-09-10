"""Integration tests for the simulation pipeline.

This module tests the complete LISFLOOD-FP simulation integration,
including parameter generation, execution, and result processing.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from src.simulation import (
    LisfloodSimulator,
    ParameterFileGenerator,
    SimulationBatch,
    ResultProcessor,
    SimulationValidator,
    SimulationMetadata,
)
from src.simulation.lisflood_simulator import SimulationConfig
from src.simulation.parameter_generator import ReturnPeriodConfig, HyetographConfig
from src.simulation.validation import ValidationThresholds
from src.simulation.result_processor import ProcessingConfig


class TestLisfloodSimulator:
    """Test LISFLOOD-FP simulator integration."""

    @pytest.fixture
    def mock_executable(self, tmp_path):
        """Create mock LISFLOOD-FP executable."""
        exe_path = tmp_path / "lisflood"
        exe_path.write_text("#!/bin/bash\necho 'Mock LISFLOOD-FP'")
        exe_path.chmod(0o755)
        return str(exe_path)

    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary input files."""
        files = {}

        # Create mock DEM file
        dem_file = tmp_path / "dem.asc"
        dem_file.write_text("ncols 100\nnrows 100\n")
        files["dem"] = str(dem_file)

        # Create mock rainfall file
        rain_file = tmp_path / "rainfall.rain"
        rain_content = "#rainfall file\n24 hours\n"
        for h in range(24):
            rain_content += f"5.0\t{h}\n"
        rain_file.write_text(rain_content)
        files["rainfall"] = str(rain_file)

        # Create mock manning file
        manning_file = tmp_path / "manning.asc"
        manning_file.write_text("ncols 100\nnrows 100\n")
        files["manning"] = str(manning_file)

        # Create mock infiltration file
        infil_file = tmp_path / "infiltration.asc"
        infil_file.write_text("ncols 100\nnrows 100\n")
        files["infiltration"] = str(infil_file)

        return files

    def test_simulator_initialization(self, mock_executable):
        """Test simulator initialization."""
        simulator = LisfloodSimulator(
            lisflood_executable=mock_executable, working_directory="test_runs"
        )

        assert simulator.lisflood_exe == mock_executable
        assert simulator.working_dir.name == "test_runs"

    def test_parameter_file_generation(self, temp_files):
        """Test parameter file generation."""
        config = SimulationConfig(
            dem_file=temp_files["dem"],
            rainfall_file=temp_files["rainfall"],
            manning_file=temp_files["manning"],
            infiltration_file=temp_files["infiltration"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            simulator = LisfloodSimulator(working_directory=temp_dir)
            sim_dir = Path(temp_dir) / "test_sim"
            sim_dir.mkdir()

            par_file = simulator._generate_parameter_file(config, sim_dir)

            assert Path(par_file).exists()

            # Check parameter file content
            content = Path(par_file).read_text()
            assert "DEMfile" in content
            assert "rainfall" in content
            assert "manningfile" in content
            assert "acceleration" in content

    @patch("subprocess.run")
    def test_simulation_execution_success(self, mock_run, mock_executable, temp_files):
        """Test successful simulation execution."""
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Simulation completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        config = SimulationConfig(
            dem_file=temp_files["dem"],
            rainfall_file=temp_files["rainfall"],
            manning_file=temp_files["manning"],
            infiltration_file=temp_files["infiltration"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            simulator = LisfloodSimulator(
                lisflood_executable=mock_executable, working_directory=temp_dir
            )

            # Create mock output file
            sim_dir = Path(temp_dir) / "test_sim"
            results_dir = sim_dir / config.output_directory
            results_dir.mkdir(parents=True)

            # Create mock depth output (.max file)
            depth_data = np.random.random((100, 100)) * 2.0  # Random depths 0-2m
            max_file = results_dir / f"{config.output_prefix}.max"
            depth_data.astype(np.float32).tofile(max_file)

            with patch.object(
                simulator, "_load_lisflood_output", return_value=depth_data
            ):
                result = simulator.run_simulation(config, "test_sim")

            assert result["status"] == "success"
            assert result["simulation_id"] == "test_sim"
            assert "outputs" in result
            assert "statistics" in result["outputs"]

    @patch("subprocess.run")
    def test_simulation_execution_failure(self, mock_run, mock_executable, temp_files):
        """Test simulation execution failure handling."""
        # Mock failed subprocess run
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Simulation failed"
        mock_run.return_value = mock_result

        config = SimulationConfig(
            dem_file=temp_files["dem"],
            rainfall_file=temp_files["rainfall"],
            manning_file=temp_files["manning"],
            infiltration_file=temp_files["infiltration"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            simulator = LisfloodSimulator(
                lisflood_executable=mock_executable, working_directory=temp_dir
            )

            result = simulator.run_simulation(config, "test_sim_fail")

            assert result["status"] == "failed"
            assert "error" in result


class TestParameterFileGenerator:
    """Test parameter file generation."""

    def test_standard_return_periods(self):
        """Test standard return period generation."""
        return_periods = ParameterFileGenerator.create_standard_return_periods()

        assert len(return_periods) == 4

        # Check 100-year return period
        rp_100 = next(rp for rp in return_periods if rp.return_period_years == 100)
        assert rp_100.rainfall_depth_24h_mm == 177.8
        assert not rp_100.is_sub_design

        # Check sub-design event
        rp_10 = next(rp for rp in return_periods if rp.return_period_years == 10)
        assert rp_10.is_sub_design

    def test_hyetograph_generation(self):
        """Test hyetograph pattern generation."""
        hyetographs = ParameterFileGenerator.create_standard_hyetographs()

        assert (
            len(hyetographs) >= 3
        )  # uniform, front_loaded, center_loaded, back_loaded

        patterns = [h.pattern_type for h in hyetographs]
        assert "uniform" in patterns
        assert "center_loaded" in patterns

    def test_scenario_generation(self, tmp_path):
        """Test complete scenario generation."""
        generator = ParameterFileGenerator()

        return_periods = [
            ReturnPeriodConfig(100, 177.8, "100-year test"),
            ReturnPeriodConfig(10, 111.8, "10-year test", is_sub_design=True),
        ]

        hyetographs = [
            HyetographConfig("uniform", 0),
            HyetographConfig("center_loaded", 0, peak_hour=12.0),
        ]

        scenarios = generator.generate_scenario_parameters(
            dem_file="test_dem.asc",
            return_periods=return_periods,
            hyetograph_patterns=hyetographs,
            output_dir=str(tmp_path),
        )

        assert len(scenarios) == 4  # 2 return periods × 2 patterns

        # Check scenario files were created
        for scenario in scenarios:
            assert Path(scenario["parameter_file"]).exists()
            assert Path(scenario["rainfall_file"]).exists()

        # Check scenario manifest
        manifest_file = tmp_path / "scenarios.json"
        assert manifest_file.exists()

        with open(manifest_file) as f:
            manifest = json.load(f)
            assert len(manifest["scenarios"]) == 4


class TestResultProcessor:
    """Test simulation result processing."""

    @pytest.fixture
    def mock_depth_data(self):
        """Create mock depth data."""
        # Create realistic depth pattern
        data = np.zeros((50, 50))

        # Add flooded areas
        data[20:30, 15:25] = np.random.uniform(0.1, 2.0, (10, 10))  # Main flood area
        data[35:40, 30:35] = np.random.uniform(0.05, 0.5, (5, 5))  # Smaller flood area

        # Add some noise
        data += np.random.normal(0, 0.01, data.shape)
        data = np.maximum(data, 0)  # Ensure no negative values

        return data

    def test_flood_extent_creation(self, mock_depth_data):
        """Test flood extent creation from depth data."""
        processor = ResultProcessor()

        extent = processor._create_flood_extent(mock_depth_data)

        assert extent.dtype == np.uint8
        assert np.any(extent)  # Should have some flooded areas

        # Check threshold application
        expected_flooded = mock_depth_data >= processor.config.flood_depth_threshold_m
        # Note: actual extent may differ due to morphological processing
        assert np.sum(extent) <= np.sum(expected_flooded)

    def test_statistics_calculation(self, mock_depth_data):
        """Test flood statistics calculation."""
        processor = ResultProcessor()
        flood_extent = processor._create_flood_extent(mock_depth_data)

        stats = processor._calculate_statistics(mock_depth_data, flood_extent)

        # Check required statistics
        required_keys = [
            "total_pixels",
            "flooded_pixels",
            "flood_fraction",
            "max_depth_m",
            "mean_depth_all_m",
            "mean_depth_flooded_m",
        ]

        for key in required_keys:
            assert key in stats

        assert stats["total_pixels"] == mock_depth_data.size
        assert 0 <= stats["flood_fraction"] <= 1
        assert stats["max_depth_m"] >= 0

    def test_quality_control_checks(self, mock_depth_data):
        """Test quality control validation."""
        processor = ResultProcessor()
        flood_extent = processor._create_flood_extent(mock_depth_data)
        stats = processor._calculate_statistics(mock_depth_data, flood_extent)

        qc = processor._quality_control_checks(mock_depth_data, flood_extent, stats)

        assert "status" in qc
        assert qc["status"] in ["passed", "passed_with_warnings", "failed"]
        assert "warnings" in qc
        assert "errors" in qc

    def test_batch_processing(self, mock_depth_data, tmp_path):
        """Test batch result processing."""
        processor = ResultProcessor()

        # Create mock simulation results
        depth_file = tmp_path / "test_depth.max"
        mock_depth_data.astype(np.float32).tofile(depth_file)

        simulation_results = [
            {
                "status": "success",
                "simulation_id": "test_sim_1",
                "outputs": {"depth_file": str(depth_file)},
            }
        ]

        batch_summary = processor.process_batch_outputs(
            simulation_results, str(tmp_path)
        )

        assert batch_summary["successful_processing"] == 1
        assert "batch_statistics" in batch_summary


class TestSimulationValidator:
    """Test simulation validation."""

    def test_validation_thresholds(self):
        """Test validation threshold configuration."""
        thresholds = ValidationThresholds()

        assert thresholds.min_reasonable_depth_m > 0
        assert thresholds.max_reasonable_depth_m > thresholds.min_reasonable_depth_m
        assert 0 < thresholds.max_flood_fraction <= 1

    def test_single_simulation_validation(self):
        """Test single simulation validation."""
        validator = SimulationValidator()

        # Mock successful simulation result
        sim_result = {
            "status": "success",
            "simulation_id": "test_sim",
            "runtime_seconds": 1800,
            "outputs": {
                "statistics": {
                    "total_pixels": 10000,
                    "flooded_pixels": 500,
                    "flood_fraction": 0.05,
                    "max_depth_m": 1.5,
                    "mean_depth_flooded_m": 0.3,
                }
            },
        }

        validation = validator.validate_single_simulation(sim_result)

        assert validation.simulation_id == "test_sim"
        assert validation.status in ["passed", "warning", "failed"]
        assert 0 <= validation.overall_score <= 100

    def test_batch_validation(self):
        """Test batch simulation validation."""
        validator = SimulationValidator()

        # Mock batch of simulation results
        sim_results = [
            {
                "status": "success",
                "simulation_id": f"test_sim_{i}",
                "runtime_seconds": 1800 + i * 100,
                "outputs": {
                    "statistics": {
                        "total_pixels": 10000,
                        "flooded_pixels": 500 + i * 50,
                        "flood_fraction": (500 + i * 50) / 10000,
                        "max_depth_m": 1.5 + i * 0.1,
                        "mean_depth_flooded_m": 0.3 + i * 0.05,
                    }
                },
            }
            for i in range(3)
        ]

        batch_summary = validator.validate_batch_results(
            sim_results, detailed_validation=False
        )

        assert batch_summary["summary"]["total_simulations"] == 3
        assert batch_summary["summary"]["success_rate"] >= 0
        assert "individual_results" in batch_summary


class TestSimulationMetadata:
    """Test metadata tracking system."""

    def test_metadata_initialization(self, tmp_path):
        """Test metadata system initialization."""
        metadata = SimulationMetadata(metadata_dir=str(tmp_path))

        assert metadata.simulation_db.exists()
        assert metadata.batch_db.exists()

    def test_simulation_provenance_creation(self, tmp_path):
        """Test simulation provenance creation."""
        metadata = SimulationMetadata(metadata_dir=str(tmp_path))

        # Create temporary input files
        dem_file = tmp_path / "test_dem.asc"
        dem_file.write_text("test dem data")

        rain_file = tmp_path / "test_rain.rain"
        rain_file.write_text("test rainfall data")

        input_files = {"dem": str(dem_file), "rainfall": str(rain_file)}

        parameter_config = {"sim_time": 86400.0, "return_period": 100}

        provenance = metadata.create_simulation_provenance(
            "test_sim", input_files, parameter_config
        )

        assert provenance.simulation_id == "test_sim"
        assert len(provenance.input_files) == 2
        assert provenance.parameter_config == parameter_config

    def test_batch_provenance_creation(self, tmp_path):
        """Test batch provenance creation."""
        metadata = SimulationMetadata(metadata_dir=str(tmp_path))

        batch_config = {"max_parallel_jobs": 4}
        scenarios = [{"scenario_id": "test_1"}, {"scenario_id": "test_2"}]

        provenance = metadata.create_batch_provenance(
            "test_batch", batch_config, scenarios
        )

        assert provenance.batch_id == "test_batch"
        assert provenance.batch_config == batch_config
        assert len(provenance.scenario_definitions) == 2


class TestSimulationBatch:
    """Test batch simulation orchestration."""

    @pytest.fixture
    def mock_simulator(self):
        """Create mock simulator."""
        simulator = Mock(spec=LisfloodSimulator)
        simulator.working_dir = Path("test_runs")
        return simulator

    @pytest.fixture
    def mock_param_generator(self):
        """Create mock parameter generator."""
        generator = Mock(spec=ParameterFileGenerator)
        return generator

    def test_batch_initialization(self, mock_simulator, mock_param_generator):
        """Test batch orchestrator initialization."""
        batch = SimulationBatch(mock_simulator, mock_param_generator)

        assert batch.simulator == mock_simulator
        assert batch.param_gen == mock_param_generator
        assert batch.scenarios == []

    def test_batch_creation_from_config(
        self, mock_simulator, mock_param_generator, tmp_path
    ):
        """Test batch creation from configuration."""
        # Mock parameter generation
        mock_scenarios = [
            {
                "scenario_id": "test_100yr_uniform",
                "parameter_file": str(tmp_path / "test.par"),
                "rainfall_file": str(tmp_path / "test.rain"),
            }
        ]
        mock_param_generator.generate_scenario_parameters.return_value = mock_scenarios

        batch = SimulationBatch(mock_simulator, mock_param_generator)

        return_periods = [ReturnPeriodConfig(100, 177.8)]
        hyetographs = [HyetographConfig("uniform", 177.8)]

        batch_id = batch.create_batch_from_config(
            dem_file="test.asc",
            return_periods=return_periods,
            hyetograph_patterns=hyetographs,
            output_dir=str(tmp_path),
        )

        assert batch_id.startswith("batch_")
        assert len(batch.scenarios) == 1


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete simulation pipeline."""

    def test_end_to_end_pipeline(self, tmp_path):
        """Test complete end-to-end pipeline with mocked LISFLOOD-FP."""
        # This would test the full pipeline but requires significant mocking
        # For now, we'll test component integration

        # Create mock input files
        dem_file = tmp_path / "dem.asc"
        dem_file.write_text("ncols 50\nnrows 50\n")

        # Initialize components
        param_gen = ParameterFileGenerator()
        result_processor = ResultProcessor()
        validator = SimulationValidator()
        metadata_tracker = SimulationMetadata(metadata_dir=str(tmp_path / "metadata"))

        # Test component interactions
        return_periods = ParameterFileGenerator.create_standard_return_periods()[
            :2
        ]  # Just 2 for testing
        hyetographs = ParameterFileGenerator.create_standard_hyetographs()[:2]

        scenarios = param_gen.generate_scenario_parameters(
            dem_file=str(dem_file),
            return_periods=return_periods,
            hyetograph_patterns=hyetographs,
            output_dir=str(tmp_path / "scenarios"),
        )

        assert len(scenarios) == 4  # 2 return periods × 2 patterns

        # Test metadata creation for scenarios
        for scenario in scenarios[:1]:  # Test one scenario
            provenance = metadata_tracker.create_simulation_provenance(
                scenario["scenario_id"],
                {"dem": str(dem_file), "rainfall": scenario["rainfall_file"]},
                scenario["config"],
            )

            assert provenance.simulation_id == scenario["scenario_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
