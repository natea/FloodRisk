"""
Utility functions for the FastAPI service.
Common helpers for data processing, validation, and formatting.
"""

import os
import time
import psutil
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
import json
import numpy as np
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import get_settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking metrics
start_time = time.time()
prediction_count = 0
response_times = []
validation_count = 0


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with timestamp and random component."""
    timestamp = int(time.time() * 1000000)  # microseconds
    return f"{prefix}_{timestamp}_{hashlib.md5(str(timestamp).encode()).hexdigest()[:8]}"


def get_current_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return 0.0


def get_uptime_seconds() -> int:
    """Get service uptime in seconds."""
    return int(time.time() - start_time)


def track_prediction():
    """Increment prediction counter."""
    global prediction_count
    prediction_count += 1


def track_response_time(response_time_ms: float):
    """Track response time for metrics."""
    global response_times
    response_times.append(response_time_ms)
    # Keep only last 1000 response times
    if len(response_times) > 1000:
        response_times = response_times[-1000:]


def get_average_response_time() -> float:
    """Get average response time in milliseconds."""
    if not response_times:
        return 0.0
    return sum(response_times) / len(response_times)


def get_predictions_last_hour() -> int:
    """Get number of predictions in the last hour."""
    # This is a simplified implementation
    # In production, you'd want to use a time-series database
    return min(prediction_count, 100)  # Placeholder


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate geographic coordinates."""
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def validate_input_data(data: Dict[str, Any]) -> List[str]:
    """Validate input data and return list of validation errors."""
    errors = []
    
    # Check required fields
    if "latitude" not in data or "longitude" not in data:
        errors.append("Latitude and longitude are required")
    
    # Validate coordinates if present
    if "latitude" in data and "longitude" in data:
        if not validate_coordinates(data["latitude"], data["longitude"]):
            errors.append("Invalid coordinates provided")
    
    # Validate numerical ranges
    if "soil_moisture" in data and data["soil_moisture"] is not None:
        if not 0 <= data["soil_moisture"] <= 100:
            errors.append("Soil moisture must be between 0 and 100")
    
    if "building_density" in data and data["building_density"] is not None:
        if not 0 <= data["building_density"] <= 100:
            errors.append("Building density must be between 0 and 100")
    
    return errors


def format_risk_level(risk_score: float) -> str:
    """Convert numerical risk score to risk level category."""
    if risk_score < 0.25:
        return "low"
    elif risk_score < 0.5:
        return "moderate"
    elif risk_score < 0.75:
        return "high"
    else:
        return "extreme"


def calculate_confidence(features: Dict[str, Any], model_metadata: Dict[str, Any] = None) -> float:
    """Calculate prediction confidence based on input data quality."""
    confidence = 1.0
    
    # Reduce confidence for missing key features
    key_features = ["current_rainfall", "elevation", "drainage_capacity"]
    missing_features = sum(1 for feature in key_features if features.get(feature) is None)
    confidence -= missing_features * 0.1
    
    # Reduce confidence for extreme values
    if features.get("current_rainfall", 0) > 100:  # Very high rainfall
        confidence -= 0.1
    
    if features.get("elevation") is not None and features["elevation"] < 0:  # Below sea level
        confidence -= 0.05
    
    return max(0.1, min(1.0, confidence))  # Keep between 0.1 and 1.0


def generate_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    """Generate risk mitigation recommendations based on risk level and factors."""
    recommendations = []
    
    if risk_level in ["high", "extreme"]:
        recommendations.extend([
            "Consider immediate evacuation if advised by authorities",
            "Move to higher ground if possible",
            "Avoid driving through flooded areas"
        ])
    
    if risk_level in ["moderate", "high", "extreme"]:
        recommendations.extend([
            "Monitor local weather alerts",
            "Prepare emergency supplies",
            "Identify safe evacuation routes"
        ])
    
    # Factor-specific recommendations
    if "poor_drainage" in risk_factors:
        recommendations.append("Report drainage issues to local authorities")
    
    if "heavy_rainfall" in risk_factors:
        recommendations.append("Avoid outdoor activities during heavy rain")
    
    return list(set(recommendations))  # Remove duplicates


def identify_risk_factors(input_data: Dict[str, Any], risk_score: float) -> List[str]:
    """Identify primary risk factors contributing to flood risk."""
    factors = []
    
    # Rainfall-related factors
    if input_data.get("current_rainfall", 0) > 20:
        factors.append("heavy_rainfall")
    
    if input_data.get("forecast_rainfall"):
        forecast_total = sum(input_data["forecast_rainfall"])
        if forecast_total > 50:
            factors.append("forecast_heavy_rainfall")
    
    # Topographical factors
    if input_data.get("elevation") is not None and input_data["elevation"] < 5:
        factors.append("low_elevation")
    
    # Infrastructure factors
    if input_data.get("drainage_capacity", float('inf')) < 30:
        factors.append("poor_drainage")
    
    if input_data.get("building_density", 0) > 70:
        factors.append("high_urban_density")
    
    # Soil conditions
    if input_data.get("soil_moisture", 0) > 80:
        factors.append("saturated_soil")
    
    return factors


def estimate_affected_area(risk_score: float, population_density: float = None) -> float:
    """Estimate affected area in square kilometers."""
    # Simple estimation based on risk score
    base_area = risk_score * 2.0  # Base area in km²
    
    # Adjust for population density (higher density = more concentrated risk)
    if population_density:
        density_factor = min(population_density / 1000, 2.0)  # Cap at 2x
        base_area *= density_factor
    
    return round(base_area, 2)


def estimate_time_to_peak(current_conditions: Dict[str, Any]) -> Optional[int]:
    """Estimate time to peak flood level in hours."""
    if not current_conditions.get("current_rainfall"):
        return None
    
    # Simple estimation based on rainfall intensity
    rainfall_intensity = current_conditions["current_rainfall"]
    
    if rainfall_intensity > 50:
        return 2  # Very heavy rain, quick response
    elif rainfall_intensity > 20:
        return 4  # Heavy rain, moderate response
    elif rainfall_intensity > 10:
        return 8  # Moderate rain, slower response
    else:
        return 12  # Light rain, slow response


def create_error_response(message: str, detail: str = None, error_code: str = None) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": message,
        "detail": detail,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat()
    }


def timing_middleware(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
        finally:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            track_response_time(execution_time)
            logger.info(f"{func.__name__} executed in {execution_time:.2f}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
        finally:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            track_response_time(execution_time)
            logger.info(f"{func.__name__} executed in {execution_time:.2f}ms")
        return result
    
    # Return appropriate wrapper based on function type
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data to prevent injection attacks."""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized[key] = value.replace(";", "").replace("--", "").strip()
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Sanitize lists
            sanitized[key] = [
                item.replace(";", "").replace("--", "").strip() if isinstance(item, str) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def validate_api_key(api_key: str) -> bool:
    """Validate API key."""
    settings = get_settings()
    if not settings.api_key:
        return True  # No API key required
    
    return api_key == settings.api_key


class APIKeyAuth(HTTPBearer):
    """Custom API key authentication."""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        settings = get_settings()
        
        # Skip authentication if no API key is configured
        if not settings.api_key:
            return None
        
        credentials = await super().__call__(request)
        
        if credentials and validate_api_key(credentials.credentials):
            return credentials
        
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def format_prediction_output(
    risk_score: float,
    input_data: Dict[str, Any],
    model_version: str,
    include_confidence: bool = True
) -> Dict[str, Any]:
    """Format prediction output with all required fields."""
    risk_level = format_risk_level(risk_score)
    confidence = calculate_confidence(input_data) if include_confidence else None
    risk_factors = identify_risk_factors(input_data, risk_score)
    
    output = {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 3),
        "model_version": model_version,
        "prediction_timestamp": datetime.utcnow().isoformat()
    }
    
    if include_confidence:
        output["confidence"] = round(confidence, 3)
    
    # Add optional fields if we can calculate them
    output["predicted_water_level"] = round(risk_score * 3.0, 2)  # Simple estimation
    output["time_to_peak"] = estimate_time_to_peak(input_data)
    output["affected_area"] = estimate_affected_area(
        risk_score, 
        input_data.get("population_density")
    )
    output["primary_factors"] = risk_factors
    output["recommendations"] = generate_recommendations(risk_level, risk_factors)
    
    return output