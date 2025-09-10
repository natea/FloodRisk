"""
Vercel serverless function for Nashville Flood Risk Demo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import random
from datetime import datetime
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
    model_config = {"protected_namespaces": ()}
    
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
    model_config = {"protected_namespaces": ()}
    
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

# HTML template for the main page
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nashville Flood Risk Assessment - AI-Powered Demo</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        #map {
            height: 100vh;
            width: 100%;
            z-index: 1;
        }
        
        .control-panel {
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            width: 350px;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .title-section {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .title-section h1 {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .risk-level {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .risk-low { background: #d4edda; color: #155724; }
        .risk-moderate { background: #fff3cd; color: #856404; }
        .risk-high { background: #f8d7da; color: #721c24; }
        .risk-extreme { background: #f5c6cb; color: #491217; }
        
        .metric-card {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.85rem;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .btn-custom {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            margin: 5px 0;
            width: 100%;
            transition: transform 0.2s;
        }
        
        .btn-custom:hover {
            transform: translateY(-2px);
            color: white;
        }
        
        .legend {
            background: white;
            padding: 10px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner-border {
            width: 2rem;
            height: 2rem;
        }
        
        .info-alert {
            background: #e8f4fd;
            border-left: 4px solid #0066cc;
            padding: 12px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        
        .clickable-map-hint {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            z-index: 1000;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        
        .infrastructure-marker {
            background: #0066cc;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .claims-marker {
            background: #dc3545;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            border: 2px solid white;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="control-panel">
        <div class="title-section">
            <h1><i class="fas fa-water"></i> Nashville Flood Risk</h1>
            <div class="subtitle">AI-Powered Assessment System</div>
        </div>
        
        <div class="info-alert">
            <i class="fas fa-info-circle"></i> <strong>Click anywhere on the map</strong> to get instant flood risk predictions
        </div>
        
        <div class="mb-3">
            <label class="form-label"><i class="fas fa-cloud-rain"></i> Rainfall Scenario</label>
            <select id="scenario" class="form-select">
                <option value="10yr">10-Year Storm (4.86")</option>
                <option value="25yr">25-Year Storm (5.91")</option>
                <option value="50yr">50-Year Storm (6.80")</option>
                <option value="100yr" selected>100-Year Storm (7.75")</option>
                <option value="500yr">500-Year Storm (10.4")</option>
            </select>
        </div>
        
        <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="showInfrastructure">
            <label class="form-check-label" for="showInfrastructure">
                Show Critical Infrastructure
            </label>
        </div>
        
        <div class="form-check form-switch mb-3">
            <input class="form-check-input" type="checkbox" id="showClaims">
            <label class="form-check-label" for="showClaims">
                Show Historical Claims
            </label>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="mt-2">Analyzing location...</div>
        </div>
        
        <div id="results" style="display: none;">
            <h5 class="mt-3 mb-3"><i class="fas fa-chart-line"></i> Risk Assessment</h5>
            
            <div id="riskLevel" class="risk-level"></div>
            
            <div class="stats-grid">
                <div class="metric-card">
                    <div class="metric-label">Flood Probability</div>
                    <div class="metric-value" id="probability">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Expected Depth</div>
                    <div class="metric-value" id="depth">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value" id="confidence">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Processing Time</div>
                    <div class="metric-value" id="processingTime">-</div>
                </div>
            </div>
            
            <h6 class="mt-3"><i class="fas fa-mountain"></i> Terrain Features</h6>
            <div class="stats-grid">
                <div class="metric-card">
                    <div class="metric-label">Elevation</div>
                    <div class="metric-value" id="elevation">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Slope</div>
                    <div class="metric-value" id="slope">-</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Flow Accumulation</div>
                <div class="metric-value" id="flowAccumulation">-</div>
            </div>
        </div>
        
        <div class="legend">
            <h6><i class="fas fa-map-legend"></i> Risk Level Legend</h6>
            <div class="legend-item">
                <div class="legend-color" style="background: #28a745;"></div>
                <span>Low Risk (< 0.1m)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffc107;"></div>
                <span>Moderate (0.1 - 0.3m)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #fd7e14;"></div>
                <span>High (0.3 - 1.0m)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #dc3545;"></div>
                <span>Extreme (> 1.0m)</span>
            </div>
        </div>
        
        <button class="btn btn-custom mt-3" onclick="showStatistics()">
            <i class="fas fa-chart-bar"></i> View Statistics
        </button>
    </div>
    
    <div class="clickable-map-hint">
        <i class="fas fa-mouse-pointer"></i> Click on the map to assess flood risk
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // Initialize map centered on Nashville
        const map = L.map('map').setView([36.1627, -86.7816], 11);
        
        // Add base map layers
        const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        });
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri'
        });
        
        streetLayer.addTo(map);
        
        // Layer control
        const baseMaps = {
            "Street Map": streetLayer,
            "Satellite": satelliteLayer
        };
        
        L.control.layers(baseMaps).addTo(map);
        
        // Store markers
        let currentMarker = null;
        let riskCircle = null;
        let infrastructureMarkers = [];
        let claimsMarkers = [];
        
        // Critical infrastructure locations
        const infrastructure = [
            {name: "Vanderbilt Hospital", lat: 36.1408, lng: -86.8028},
            {name: "Nissan Stadium", lat: 36.1665, lng: -86.7713},
            {name: "Nashville Airport", lat: 36.1245, lng: -86.6782},
            {name: "Music City Center", lat: 36.1580, lng: -86.7761},
            {name: "Tennessee State Capitol", lat: 36.1656, lng: -86.7841}
        ];
        
        // Mock historical claims (random points near river)
        function generateClaims() {
            const claims = [];
            for (let i = 0; i < 50; i++) {
                claims.push({
                    lat: 36.1566 + (Math.random() - 0.5) * 0.1,
                    lng: -86.7842 + (Math.random() - 0.5) * 0.15
                });
            }
            return claims;
        }
        
        const historicalClaims = generateClaims();
        
        // Toggle infrastructure layer
        document.getElementById('showInfrastructure').addEventListener('change', (e) => {
            if (e.target.checked) {
                infrastructure.forEach(item => {
                    const marker = L.marker([item.lat, item.lng], {
                        icon: L.divIcon({
                            className: 'infrastructure-marker',
                            html: `<i class="fas fa-building"></i> ${item.name}`,
                            iconSize: [150, 30]
                        })
                    }).addTo(map);
                    infrastructureMarkers.push(marker);
                });
            } else {
                infrastructureMarkers.forEach(marker => map.removeLayer(marker));
                infrastructureMarkers = [];
            }
        });
        
        // Toggle claims layer
        document.getElementById('showClaims').addEventListener('change', (e) => {
            if (e.target.checked) {
                historicalClaims.forEach(claim => {
                    const marker = L.circleMarker([claim.lat, claim.lng], {
                        radius: 4,
                        fillColor: '#dc3545',
                        color: '#fff',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);
                    claimsMarkers.push(marker);
                });
            } else {
                claimsMarkers.forEach(marker => map.removeLayer(marker));
                claimsMarkers = [];
            }
        });
        
        // Map click handler
        map.on('click', async function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            
            // Hide hint after first click
            document.querySelector('.clickable-map-hint').style.display = 'none';
            
            // Remove previous marker and circle
            if (currentMarker) map.removeLayer(currentMarker);
            if (riskCircle) map.removeLayer(riskCircle);
            
            // Add new marker
            currentMarker = L.marker([lat, lng]).addTo(map);
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            // Get selected scenario
            const scenario = document.getElementById('scenario').value;
            
            try {
                // Make API call
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        latitude: lat,
                        longitude: lng,
                        scenario: scenario,
                        include_uncertainty: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const data = await response.json();
                
                // Update UI with results
                updateResults(data, lat, lng);
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error getting prediction. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function updateResults(data, lat, lng) {
            // Show results section
            document.getElementById('results').style.display = 'block';
            
            // Update risk level
            const riskElement = document.getElementById('riskLevel');
            riskElement.className = `risk-level risk-${data.risk_level}`;
            riskElement.textContent = data.risk_level.toUpperCase() + ' RISK';
            
            // Update metrics
            document.getElementById('probability').textContent = (data.flood_probability * 100).toFixed(1) + '%';
            document.getElementById('depth').textContent = data.flood_depth.toFixed(2) + ' m';
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(0) + '%';
            document.getElementById('processingTime').textContent = data.processing_time.toFixed(3) + ' s';
            
            // Update terrain features
            document.getElementById('elevation').textContent = data.elevation.toFixed(1) + ' m';
            document.getElementById('slope').textContent = data.slope.toFixed(1) + '°';
            document.getElementById('flowAccumulation').textContent = data.flow_accumulation.toFixed(0);
            
            // Add risk visualization circle
            const color = data.risk_level === 'low' ? '#28a745' :
                         data.risk_level === 'moderate' ? '#ffc107' :
                         data.risk_level === 'high' ? '#fd7e14' : '#dc3545';
            
            const radius = Math.max(100, Math.min(1000, data.flood_depth * 500));
            
            riskCircle = L.circle([lat, lng], {
                color: color,
                fillColor: color,
                fillOpacity: 0.3,
                radius: radius
            }).addTo(map);
        }
        
        async function showStatistics() {
            try {
                const response = await fetch('/api/statistics');
                const stats = await response.json();
                
                alert(`Demo Statistics:\n
Total Predictions: ${stats.total_predictions}
Average Processing Time: ${stats.average_processing_time}s
Model Accuracy (IoU): ${stats.model_accuracy.iou}
Coverage Area: ${stats.coverage.area_km2} km²
Properties Assessed: ${stats.coverage.properties_assessed}`);
            } catch (error) {
                console.error('Error fetching statistics:', error);
            }
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main demo page"""
    return HTMLResponse(content=HTML_TEMPLATE)

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

# Handler for Vercel
handler = app