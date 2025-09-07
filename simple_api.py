#!/usr/bin/env python
"""
Simple test API for FloodRisk - minimal dependencies
Run with: python simple_api.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import random

# Create FastAPI app
app = FastAPI(
    title="FloodRisk API (Demo Mode)",
    description="Simplified demo API for flood risk prediction",
    version="1.0.0"
)

# Simple data models
class FloodPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    rainfall_mm: float
    elevation_m: Optional[float] = None
    
class FloodPredictionResponse(BaseModel):
    flood_depth_m: float
    risk_level: str
    confidence: float
    message: str

# Root endpoint
@app.get("/")
def root():
    return {
        "name": "FloodRisk API",
        "status": "running",
        "mode": "demo",
        "endpoints": {
            "docs": "http://localhost:8000/docs",
            "health": "http://localhost:8000/health",
            "predict": "http://localhost:8000/api/v1/predict"
        }
    }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "floodrisk-api",
        "version": "1.0.0",
        "mode": "demo"
    }

# Simple prediction endpoint (mock data for demo)
@app.post("/api/v1/predict", response_model=FloodPredictionResponse)
def predict_flood_depth(request: FloodPredictionRequest):
    """
    Predict flood depth based on location and rainfall.
    Note: This is a demo endpoint with simulated results.
    """
    
    # Simulate flood depth calculation
    # In reality, this would use the trained ML model
    base_depth = request.rainfall_mm / 100.0
    location_factor = abs(request.latitude % 1) + abs(request.longitude % 1)
    
    flood_depth = base_depth * (1 + location_factor * 0.5)
    flood_depth = min(flood_depth, 10.0)  # Cap at 10m
    flood_depth = round(flood_depth + random.uniform(-0.1, 0.1), 2)
    
    # Determine risk level
    if flood_depth < 0.3:
        risk_level = "Low"
    elif flood_depth < 1.0:
        risk_level = "Moderate"
    elif flood_depth < 2.0:
        risk_level = "High"
    else:
        risk_level = "Extreme"
    
    # Confidence based on data availability
    confidence = 0.85 if request.elevation_m else 0.65
    
    return FloodPredictionResponse(
        flood_depth_m=flood_depth,
        risk_level=risk_level,
        confidence=confidence,
        message=f"Demo prediction for location ({request.latitude}, {request.longitude}) with {request.rainfall_mm}mm rainfall"
    )

# Batch prediction endpoint
@app.post("/api/v1/predict/batch")
def predict_batch(requests: List[FloodPredictionRequest]):
    """
    Batch prediction for multiple locations.
    """
    results = []
    for req in requests:
        result = predict_flood_depth(req)
        results.append(result)
    return {"predictions": results, "count": len(results)}

# Get available models (mock)
@app.get("/api/v1/models")
def list_models():
    return {
        "models": [
            {
                "id": "flood-cnn-v1",
                "name": "Multi-Scale CNN",
                "version": "1.0.0",
                "status": "demo",
                "accuracy": 0.92
            }
        ]
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌊 FloodRisk API - Demo Mode")
    print("="*60)
    print("\nStarting server...")
    print("\n📍 API Documentation: http://localhost:8000/docs")
    print("📍 Health Check: http://localhost:8000/health")
    print("📍 Prediction Endpoint: http://localhost:8000/api/v1/predict")
    print("\n✨ Try the interactive docs at http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)