#!/usr/bin/env python3
"""
Quick test script for the FloodRisk API.
Tests basic functionality of all endpoints.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install required packages: pip install fastapi uvicorn httpx pytest")
    sys.exit(1)


def test_api_endpoints():
    """Test all API endpoints."""
    client = TestClient(app)
    
    print("Testing FloodRisk API endpoints...")
    print("=" * 50)
    
    # Test root endpoint
    print("1. Testing root endpoint (/)...")
    response = client.get("/")
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    print(f"   ✓ Response: {response.json()}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint (/health)...")
    response = client.get("/health")
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    health_data = response.json()
    print(f"   ✓ Service status: {health_data['status']}")
    print(f"   ✓ Model status: {health_data['model_status']}")
    
    # Test metrics endpoint
    print("\n3. Testing metrics endpoint (/metrics)...")
    response = client.get("/metrics")
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    metrics_data = response.json()
    print(f"   ✓ Total predictions: {metrics_data['total_predictions']}")
    print(f"   ✓ Memory usage: {metrics_data['memory_usage_mb']:.2f} MB")
    
    # Test version endpoint
    print("\n4. Testing version endpoint (/version)...")
    response = client.get("/version")
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    print(f"   ✓ Version: {response.json()['version']}")
    
    # Test prediction endpoint
    print("\n5. Testing prediction endpoint (/predict)...")
    test_prediction_data = {
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
    
    response = client.post("/predict", json=test_prediction_data)
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    prediction = response.json()
    print(f"   ✓ Risk level: {prediction['risk_level']}")
    print(f"   ✓ Risk score: {prediction['risk_score']}")
    print(f"   ✓ Confidence: {prediction.get('confidence', 'N/A')}")
    
    # Test batch prediction endpoint
    print("\n6. Testing batch prediction endpoint (/predict/batch)...")
    batch_data = {
        "predictions": [
            test_prediction_data,
            {
                "latitude": 34.0522,
                "longitude": -118.2437,
                "current_rainfall": 2.1,
                "elevation": 84.0
            }
        ],
        "include_confidence": True
    }
    
    response = client.post("/predict/batch", json=batch_data)
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    batch_result = response.json()
    print(f"   ✓ Processed: {batch_result['total_processed']} predictions")
    print(f"   ✓ Processing time: {batch_result['processing_time_ms']:.2f} ms")
    
    # Test validation endpoint
    print("\n7. Testing validation endpoint (/validate)...")
    validation_data = {
        "actual_flood_occurred": True,
        "actual_water_level": 2.1,
        "actual_affected_area": 0.8,
        "prediction_id": "pred_123456",
        "validation_notes": "Test validation"
    }
    
    response = client.post("/validate", json=validation_data)
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    validation = response.json()
    print(f"   ✓ Validation ID: {validation['validation_id']}")
    print(f"   ✓ Accuracy score: {validation.get('accuracy_score', 'N/A')}")
    
    # Test model info endpoint
    print("\n8. Testing model info endpoint (/model/info)...")
    response = client.get("/model/info")
    assert response.status_code == 200
    print(f"   ✓ Status: {response.status_code}")
    model_info = response.json()
    print(f"   ✓ Model status: {model_info['status']}")
    print(f"   ✓ Model type: {model_info.get('model_type', 'N/A')}")
    print(f"   ✓ Feature count: {model_info.get('feature_count', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! FloodRisk API is working correctly.")
    
    # Print API usage examples
    print("\n📝 API Usage Examples:")
    print(f"   • Health check: GET http://localhost:8000/health")
    print(f"   • Single prediction: POST http://localhost:8000/predict")
    print(f"   • Batch prediction: POST http://localhost:8000/predict/batch")
    print(f"   • API documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    try:
        test_api_endpoints()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)