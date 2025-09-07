#!/usr/bin/env python
"""
Example client for FloodRisk API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check if API is running"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def predict_single_location():
    """Make a single flood prediction"""
    data = {
        "latitude": 37.7749,  # San Francisco
        "longitude": -122.4194,
        "rainfall_mm": 150.0,
        "elevation_m": 50.0
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/predict", json=data)
    result = response.json()
    
    print("\n📍 Single Location Prediction:")
    print(f"Location: ({data['latitude']}, {data['longitude']})")
    print(f"Rainfall: {data['rainfall_mm']}mm")
    print(f"Predicted Flood Depth: {result['flood_depth_m']}m")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']*100:.0f}%")
    
    return result

def predict_multiple_locations():
    """Make batch predictions"""
    locations = [
        {"latitude": 37.7749, "longitude": -122.4194, "rainfall_mm": 150.0},  # SF
        {"latitude": 34.0522, "longitude": -118.2437, "rainfall_mm": 200.0},  # LA
        {"latitude": 40.7128, "longitude": -74.0060, "rainfall_mm": 100.0},   # NYC
        {"latitude": 51.5074, "longitude": -0.1278, "rainfall_mm": 80.0},     # London
        {"latitude": 48.8566, "longitude": 2.3522, "rainfall_mm": 120.0}      # Paris
    ]
    
    response = requests.post(f"{BASE_URL}/api/v1/predict/batch", json=locations)
    results = response.json()
    
    print("\n📊 Batch Predictions:")
    print(f"Total Locations: {results['count']}")
    print("-" * 60)
    
    for i, pred in enumerate(results['predictions']):
        loc = locations[i]
        print(f"Location {i+1}: ({loc['latitude']:.2f}, {loc['longitude']:.2f})")
        print(f"  Rainfall: {loc['rainfall_mm']}mm")
        print(f"  Flood Depth: {pred['flood_depth_m']}m")
        print(f"  Risk Level: {pred['risk_level']}")
        print()

def get_models():
    """List available models"""
    response = requests.get(f"{BASE_URL}/api/v1/models")
    models = response.json()
    
    print("\n🤖 Available Models:")
    for model in models['models']:
        print(f"- {model['name']} (v{model['version']})")
        print(f"  ID: {model['id']}")
        print(f"  Status: {model['status']}")
        print(f"  Accuracy: {model['accuracy']*100:.0f}%")

if __name__ == "__main__":
    print("=" * 60)
    print("🌊 FloodRisk API Client Example")
    print("=" * 60)
    
    # Check health
    check_health()
    
    # Single prediction
    predict_single_location()
    
    # Batch predictions
    predict_multiple_locations()
    
    # List models
    get_models()
    
    print("\n✅ All examples completed!")