"""
FloodRisk API package.
FastAPI service for flood risk prediction and validation.
"""

from .main import app
from .config import get_settings
from .inference import get_predictor

__version__ = "1.0.0"
__author__ = "FloodRisk Team"

__all__ = ["app", "get_settings", "get_predictor"]