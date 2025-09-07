"""
FastAPI backend for Nashville Flood Risk Demo
Provides API endpoints for flood risk predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import random
from datetime import datetime
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Nashville Flood Risk API",
    description="AI-powered flood risk assessment for Nashville",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    scenario: str = "100yr"
    include_uncertainty: bool = True

class PredictionResponse(BaseModel):
    flood_probability: float
    flood_depth: float
    risk_level: str
    confidence: float
    elevation: float
    slope: float
    flow_accumulation: float
    processing_time: float
    uncertainty_bounds: Optional[Dict[str, float]] = None
    timestamp: str
    model_version: str = "v2.0"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

# Mock model class for demo
class FloodRiskModel:
    """Mock flood risk model for demonstration"""
    
    def __init__(self):
        self.model_loaded = True
        self.scenario_multipliers = {
            '10yr': 0.3,
            '25yr': 0.5,
            '50yr': 0.7,
            '100yr': 1.0,
            '500yr': 1.5
        }
        
    def predict(self, lat: float, lng: float, scenario: str) -> Dict[str, Any]:
        """Generate mock prediction based on location and scenario"""
        
        # Simulate higher risk near Cumberland River
        river_distance = self._distance_to_river(lat, lng)
        is_near_river = river_distance < 0.05
        
        # Get scenario multiplier
        multiplier = self.scenario_multipliers.get(scenario, 1.0)
        
        # Calculate base probability based on proximity to river
        if is_near_river:
            base_probability = 0.6 + (0.05 - river_distance) * 4
        else:
            base_probability = 0.1 + random.random() * 0.2
        
        # Apply scenario multiplier
        probability = min(base_probability * multiplier, 0.95)
        
        # Calculate expected flood depth
        if is_near_river:
            depth = (0.5 + random.random() * 1.5) * multiplier
        else:
            depth = random.random() * 0.3 * multiplier
        
        # Determine risk level
        if depth > 1.0:
            risk_level = 'extreme'
        elif depth > 0.3:
            risk_level = 'high'
        elif depth > 0.1:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        # Generate terrain features
        elevation = 150 + random.random() * 50 - (river_distance * 100)
        slope = max(0, 2 + random.random() * 3 - river_distance * 10)
        flow_accumulation = 100 + random.random() * 900 + (is_near_river * 1000)
        
        # Model confidence (higher near river where we have more data)
        confidence = 0.85 + (0.1 if is_near_river else 0) + random.random() * 0.05
        
        # Processing time simulation
        processing_time = 0.2 + random.random() * 0.3
        
        return {
            'flood_probability': probability,
            'flood_depth': depth,
            'risk_level': risk_level,
            'confidence': confidence,
            'elevation': elevation,
            'slope': slope,
            'flow_accumulation': flow_accumulation,
            'processing_time': processing_time,
            'uncertainty_bounds': {
                'depth_lower': max(0, depth - depth * 0.2),
                'depth_upper': depth + depth * 0.3,
                'probability_lower': max(0, probability - 0.1),
                'probability_upper': min(1, probability + 0.1)
            }
        }
    
    def _distance_to_river(self, lat: float, lng: float) -> float:
        """Calculate approximate distance to Cumberland River"""
        # Cumberland River approximate coordinates through Nashville
        river_lat = 36.1566
        river_lng = -86.7842
        
        # Simple Euclidean distance (good enough for demo)
        distance = np.sqrt((lat - river_lat)**2 + (lng - river_lng)**2)
        return distance

# Initialize model
model = FloodRiskModel()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main demo page"""
    with open("demo/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model.model_loaded,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_flood_risk(request: PredictionRequest):
    """
    Predict flood risk for a given location and scenario
    """
    try:
        # Validate coordinates (Nashville area)
        if not (35.9 < request.latitude < 36.4 and -87.1 < request.longitude < -86.5):
            raise HTTPException(
                status_code=400,
                detail="Coordinates must be within Nashville area"
            )
        
        # Get prediction from model
        result = model.predict(
            request.latitude,
            request.longitude,
            request.scenario
        )
        
        # Prepare response
        response = PredictionResponse(
            flood_probability=result['flood_probability'],
            flood_depth=result['flood_depth'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            elevation=result['elevation'],
            slope=result['slope'],
            flow_accumulation=result['flow_accumulation'],
            processing_time=result['processing_time'],
            uncertainty_bounds=result['uncertainty_bounds'] if request.include_uncertainty else None,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scenarios")
async def get_scenarios():
    """Get available rainfall scenarios"""
    return {
        "scenarios": [
            {"id": "10yr", "name": "10-Year Storm", "rainfall": "4.86 inches", "probability": 0.10},
            {"id": "25yr", "name": "25-Year Storm", "rainfall": "5.91 inches", "probability": 0.04},
            {"id": "50yr", "name": "50-Year Storm", "rainfall": "6.80 inches", "probability": 0.02},
            {"id": "100yr", "name": "100-Year Storm", "rainfall": "7.75 inches", "probability": 0.01},
            {"id": "500yr", "name": "500-Year Storm", "rainfall": "10.4 inches", "probability": 0.002}
        ]
    }

@app.get("/api/statistics")
async def get_statistics():
    """Get demo statistics"""
    return {
        "total_predictions": random.randint(1000, 5000),
        "average_processing_time": 0.35,
        "model_accuracy": {
            "iou": 0.75,
            "rmse": 0.38,
            "r2": 0.72
        },
        "coverage": {
            "area_km2": 1362,
            "properties_assessed": random.randint(50000, 100000)
        },
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/batch-predict")
async def batch_predict(locations: list[PredictionRequest]):
    """Batch prediction endpoint for multiple locations"""
    results = []
    for location in locations:
        try:
            result = model.predict(
                location.latitude,
                location.longitude,
                location.scenario
            )
            results.append({
                "latitude": location.latitude,
                "longitude": location.longitude,
                "prediction": result
            })
        except Exception as e:
            results.append({
                "latitude": location.latitude,
                "longitude": location.longitude,
                "error": str(e)
            })
    
    return {"results": results, "count": len(results)}

# Mount static files if needed
if os.path.exists("demo/static"):
    app.mount("/static", StaticFiles(directory="demo/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Nashville Flood Risk Demo Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)