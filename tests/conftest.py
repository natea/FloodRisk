"""
Pytest configuration and shared fixtures for FloodRisk integration tests.

This module provides common fixtures and configurations for all test modules.
"""

import pytest
import numpy as np
import torch
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock
import warnings

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set random seeds for reproducible tests
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


@pytest.fixture(scope="session")
def test_device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        device = "cuda"
        # Clear GPU cache at start of session
        torch.cuda.empty_cache()
    else:
        device = "cpu"

    yield device

    # Cleanup at end of session
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture(scope="session")
def test_data_shapes():
    """Standard data shapes for testing."""
    return {
        "small": (32, 32),
        "medium": (64, 64),
        "large": (128, 128),
        "production": (256, 256),
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for validation."""
    return {
        "max_inference_time": 5.0,  # seconds
        "max_memory_mb": 1000,  # MB
        "min_accuracy": 0.7,  # R²
        "max_mae": 0.5,  # meters
        "max_rmse": 0.8,  # meters
    }


# Pytest hooks for test customization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "production: marks tests for production validation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark integration tests
        if item.fspath.basename.startswith("test_"):
            item.add_marker(pytest.mark.integration)

        # Mark production tests
        if "production" in item.name.lower():
            item.add_marker(pytest.mark.production)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    import gc

    gc.collect()
