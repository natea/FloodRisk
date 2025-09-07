"""
FloodRisk Integration Test Suite

This package contains comprehensive integration tests for the FloodRisk flood prediction system.

Test Categories:
- test_preprocessing.py: DEM processing and feature extraction
- test_model.py: CNN model architecture and training
- test_validation.py: Model validation and metrics
- test_api.py: REST API endpoints and web services
- test_e2e.py: End-to-end pipeline testing

Usage:
    # Run all tests
    pytest tests/

    # Run specific test file
    pytest tests/test_model.py

    # Run with coverage
    pytest tests/ --cov=src --cov-report=html

    # Run only fast tests
    pytest tests/ -m "not slow"

    # Run production validation tests
    pytest tests/ -m "production"

    # Run tests in parallel
    pytest tests/ -n auto
"""

__version__ = "1.0.0"
__author__ = "FloodRisk Development Team"