"""
Pydantic models for request/response validation.
Defines data structures for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import json


class RiskLevel(str, Enum):
    """Flood risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class PredictionInput(BaseModel):
    """Input data for flood risk prediction."""
    
    # Location data
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    elevation: Optional[float] = Field(None, description="Elevation in meters above sea level")
    
    # Historical data
    historical_rainfall: Optional[List[float]] = Field(None, description="Historical rainfall data in mm")
    historical_water_levels: Optional[List[float]] = Field(None, description="Historical water levels in meters")
    
    # Current conditions
    current_rainfall: Optional[float] = Field(None, ge=0, description="Current rainfall in mm")
    current_water_level: Optional[float] = Field(None, ge=0, description="Current water level in meters")
    soil_moisture: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture percentage")
    
    # Infrastructure data
    drainage_capacity: Optional[float] = Field(None, ge=0, description="Drainage capacity in cubic meters per second")
    population_density: Optional[float] = Field(None, ge=0, description="Population per square kilometer")
    building_density: Optional[float] = Field(None, ge=0, le=100, description="Building coverage percentage")
    
    # Forecast data
    forecast_rainfall: Optional[List[float]] = Field(None, description="Forecasted rainfall for next 24-72 hours")
    forecast_hours: Optional[int] = Field(24, ge=1, le=168, description="Forecast period in hours")
    
    @validator("historical_rainfall", "historical_water_levels", "forecast_rainfall")
    def validate_lists(cls, v):
        """Validate list inputs."""
        if v is not None and len(v) == 0:
            return None
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "elevation": 10.0,
                "current_rainfall": 5.2,
                "current_water_level": 1.8,
                "soil_moisture": 65.0,
                "drainage_capacity": 50.0,
                "population_density": 8500.0,
                "building_density": 75.0,
                "forecast_rainfall": [2.1, 8.5, 15.2],
                "forecast_hours": 48
            }
        }


class BatchPredictionInput(BaseModel):
    """Input for batch prediction requests."""
    
    predictions: List[PredictionInput] = Field(..., max_items=100, description="List of prediction inputs")
    include_confidence: bool = Field(True, description="Include confidence scores in response")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "latitude": 40.7128,
                        "longitude": -74.0060,
                        "current_rainfall": 5.2
                    }
                ],
                "include_confidence": True
            }
        }


class PredictionOutput(BaseModel):
    """Output from flood risk prediction."""
    
    risk_level: RiskLevel = Field(..., description="Predicted flood risk level")
    risk_score: float = Field(..., ge=0, le=1, description="Numerical risk score (0-1)")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence (0-1)")
    
    # Detailed predictions
    predicted_water_level: Optional[float] = Field(None, description="Predicted peak water level in meters")
    time_to_peak: Optional[int] = Field(None, description="Time to peak flood level in hours")
    affected_area: Optional[float] = Field(None, description="Estimated affected area in square kilometers")
    
    # Risk factors
    primary_factors: Optional[List[str]] = Field(None, description="Primary contributing risk factors")
    recommendations: Optional[List[str]] = Field(None, description="Risk mitigation recommendations")
    
    # Metadata
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_level": "moderate",
                "risk_score": 0.65,
                "confidence": 0.85,
                "predicted_water_level": 2.3,
                "time_to_peak": 6,
                "affected_area": 1.2,
                "primary_factors": ["heavy_rainfall", "poor_drainage"],
                "recommendations": ["Monitor water levels", "Prepare evacuation routes"],
                "model_version": "1.0.0",
                "prediction_timestamp": "2023-12-07T10:30:00Z"
            }
        }


class BatchPredictionOutput(BaseModel):
    """Output for batch prediction requests."""
    
    predictions: List[PredictionOutput] = Field(..., description="List of prediction outputs")
    total_processed: int = Field(..., description="Total number of predictions processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "risk_level": "moderate",
                        "risk_score": 0.65,
                        "confidence": 0.85,
                        "model_version": "1.0.0"
                    }
                ],
                "total_processed": 1,
                "processing_time_ms": 156.7
            }
        }


class ValidationRequest(BaseModel):
    """Request for model validation."""
    
    actual_flood_occurred: bool = Field(..., description="Whether flood actually occurred")
    actual_water_level: Optional[float] = Field(None, description="Actual peak water level in meters")
    actual_affected_area: Optional[float] = Field(None, description="Actual affected area in square kilometers")
    prediction_id: Optional[str] = Field(None, description="Original prediction ID for tracking")
    validation_notes: Optional[str] = Field(None, max_length=1000, description="Additional validation notes")
    
    class Config:
        schema_extra = {
            "example": {
                "actual_flood_occurred": True,
                "actual_water_level": 2.1,
                "actual_affected_area": 0.8,
                "prediction_id": "pred_123456",
                "validation_notes": "Flood occurred but was less severe than predicted"
            }
        }


class ValidationResponse(BaseModel):
    """Response for validation request."""
    
    validation_id: str = Field(..., description="Unique validation ID")
    accuracy_score: Optional[float] = Field(None, description="Prediction accuracy score")
    status: str = Field(..., description="Validation status")
    recorded_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "validation_id": "val_789012",
                "accuracy_score": 0.82,
                "status": "recorded",
                "recorded_timestamp": "2023-12-07T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model loading status")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-12-07T10:30:00Z",
                "version": "1.0.0",
                "model_status": "loaded",
                "dependencies": {
                    "database": "connected",
                    "model": "loaded"
                }
            }
        }


class MetricsResponse(BaseModel):
    """API metrics response."""
    
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_last_hour: int = Field(..., description="Predictions in the last hour")
    average_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    model_accuracy: Optional[float] = Field(None, description="Current model accuracy")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 15420,
                "predictions_last_hour": 127,
                "average_response_time_ms": 89.5,
                "model_accuracy": 0.87,
                "uptime_seconds": 86400,
                "memory_usage_mb": 512.3
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation failed",
                "detail": "Latitude must be between -90 and 90 degrees",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2023-12-07T10:30:00Z"
            }
        }