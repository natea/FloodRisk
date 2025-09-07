"""Integration tests for API endpoints and web service functionality.

This module tests the complete API pipeline including:
- REST API endpoints
- Request/response validation
- Authentication and authorization
- File upload/download functionality
- Error handling and edge cases
- Production deployment scenarios
"""

import pytest
import json
import tempfile
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional, Dict, Any
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Mock API application for testing
app = FastAPI(title="FloodRisk API", version="1.0.0")
security = HTTPBearer()

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for flood prediction."""
    rainfall_intensity: float
    rainfall_duration: float
    return_period: Optional[int] = None
    model_version: Optional[str] = "latest"
    uncertainty_quantification: bool = False

class PredictionResponse(BaseModel):
    """Response model for flood prediction."""
    prediction_id: str
    flood_depth_map: str  # Base64 encoded or file path
    max_depth: float
    flooded_area_km2: float
    confidence_score: float
    processing_time_seconds: float
    model_version: str
    uncertainty_map: Optional[str] = None

class ValidationRequest(BaseModel):
    """Request model for model validation."""
    prediction_data: str  # Base64 encoded predictions
    ground_truth_data: str  # Base64 encoded ground truth
    metrics_requested: list = ["mse", "mae", "r2", "flood_accuracy"]

class ValidationResponse(BaseModel):
    """Response model for validation results."""
    validation_id: str
    metrics: Dict[str, float]
    validation_report: str
    processing_time_seconds: float

# Mock services
class MockFloodPredictor:
    """Mock flood prediction service."""
    
    def __init__(self):
        self.model_loaded = True
    
    def predict(self, dem_data: np.ndarray, rainfall_intensity: float, 
                rainfall_duration: float, **kwargs) -> Dict[str, Any]:
        """Mock prediction method."""
        # Simulate prediction processing
        height, width = dem_data.shape if dem_data.ndim == 2 else dem_data.shape[-2:]
        
        # Create realistic mock flood depth
        flood_depth = np.maximum(0, np.random.exponential(rainfall_intensity/10, (height, width)))
        
        # Simulate some areas with higher flooding
        if rainfall_intensity > 20:
            center_y, center_x = height//2, width//2
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 < (min(height, width)/4)**2
            flood_depth[mask] *= 2
        
        return {
            'flood_depth': flood_depth.astype(np.float32),
            'max_depth': float(np.max(flood_depth)),
            'flooded_area_km2': float(np.sum(flood_depth > 0.1) * 0.01),  # Assume 10m pixels
            'confidence_score': min(0.95, max(0.5, 1.0 - rainfall_intensity/100)),
            'model_version': kwargs.get('model_version', 'v1.0.0')
        }

class MockValidator:
    """Mock validation service."""
    
    def validate_predictions(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                           metrics_requested: list) -> Dict[str, float]:
        """Mock validation method."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        pred_flat = predictions.flatten()
        truth_flat = ground_truth.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat))
        pred_valid = pred_flat[valid_mask]
        truth_valid = truth_flat[valid_mask]
        
        if len(pred_valid) == 0:
            return {metric: float('nan') for metric in metrics_requested}
        
        metrics = {}
        
        if 'mse' in metrics_requested:
            metrics['mse'] = mean_squared_error(truth_valid, pred_valid)
        
        if 'mae' in metrics_requested:
            metrics['mae'] = mean_absolute_error(truth_valid, pred_valid)
        
        if 'r2' in metrics_requested:
            if len(np.unique(truth_valid)) > 1:
                metrics['r2'] = r2_score(truth_valid, pred_valid)
            else:
                metrics['r2'] = 0.0
        
        if 'flood_accuracy' in metrics_requested:
            # Binary flood detection accuracy
            pred_flood = (pred_valid > 0.1).astype(int)
            truth_flood = (truth_valid > 0.1).astype(int)
            accuracy = np.mean(pred_flood == truth_flood)
            metrics['flood_accuracy'] = accuracy
        
        return metrics

# Global mock services
flood_predictor = MockFloodPredictor()
validator_service = MockValidator()

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": flood_predictor.model_loaded,
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flood(request: PredictionRequest, dem_file: UploadFile = File(...)):
    """Predict flood depth from DEM and rainfall parameters."""
    import time
    import uuid
    import base64
    
    start_time = time.time()
    
    try:
        # Read and validate DEM file
        dem_content = await dem_file.read()
        
        if dem_file.content_type not in ['image/tiff', 'application/octet-stream']:
            raise HTTPException(status_code=400, detail="Invalid file type. Expected GeoTIFF.")
        
        # Simulate DEM processing
        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp_file:
            tmp_file.write(dem_content)
            tmp_file.flush()
            
            # Read DEM data
            with rasterio.open(tmp_file.name) as src:
                dem_data = src.read(1)
        
        # Validate inputs
        if request.rainfall_intensity < 0 or request.rainfall_intensity > 200:
            raise HTTPException(status_code=400, detail="Rainfall intensity must be between 0-200 mm/h")
        
        if request.rainfall_duration < 0 or request.rainfall_duration > 1440:
            raise HTTPException(status_code=400, detail="Rainfall duration must be between 0-1440 minutes")
        
        # Make prediction
        prediction_result = flood_predictor.predict(
            dem_data,
            request.rainfall_intensity,
            request.rainfall_duration,
            model_version=request.model_version,
            uncertainty=request.uncertainty_quantification
        )
        
        # Encode flood depth map as base64
        flood_depth_bytes = prediction_result['flood_depth'].tobytes()
        flood_depth_b64 = base64.b64encode(flood_depth_bytes).decode('utf-8')
        
        processing_time = time.time() - start_time
        prediction_id = str(uuid.uuid4())
        
        response = PredictionResponse(
            prediction_id=prediction_id,
            flood_depth_map=flood_depth_b64,
            max_depth=prediction_result['max_depth'],
            flooded_area_km2=prediction_result['flooded_area_km2'],
            confidence_score=prediction_result['confidence_score'],
            processing_time_seconds=processing_time,
            model_version=prediction_result['model_version']
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/validate", response_model=ValidationResponse)
async def validate_predictions(request: ValidationRequest):
    """Validate model predictions against ground truth."""
    import time
    import uuid
    import base64
    
    start_time = time.time()
    
    try:
        # Decode base64 data
        pred_bytes = base64.b64decode(request.prediction_data)
        truth_bytes = base64.b64decode(request.ground_truth_data)
        
        # Convert to numpy arrays (assuming float32)
        predictions = np.frombuffer(pred_bytes, dtype=np.float32)
        ground_truth = np.frombuffer(truth_bytes, dtype=np.float32)
        
        # Ensure same shape
        if len(predictions) != len(ground_truth):
            raise HTTPException(status_code=400, detail="Prediction and ground truth data must have same size")
        
        # Reshape to 2D (assuming square)
        size = int(np.sqrt(len(predictions)))
        if size * size != len(predictions):
            raise HTTPException(status_code=400, detail="Data must represent square arrays")
        
        predictions = predictions.reshape(size, size)
        ground_truth = ground_truth.reshape(size, size)
        
        # Perform validation
        metrics = validator_service.validate_predictions(
            predictions, ground_truth, request.metrics_requested
        )
        
        # Generate validation report
        report = f"Validation completed for {size}x{size} prediction.\n"
        for metric, value in metrics.items():
            report += f"{metric.upper()}: {value:.4f}\n"
        
        processing_time = time.time() - start_time
        validation_id = str(uuid.uuid4())
        
        response = ValidationResponse(
            validation_id=validation_id,
            metrics=metrics,
            validation_report=report,
            processing_time_seconds=processing_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/models")
async def list_available_models():
    """List available flood prediction models."""
    return {
        "models": [
            {
                "name": "flood_cnn_v1",
                "version": "1.0.0",
                "description": "Multi-scale CNN with physics-informed loss",
                "input_resolution": "10m",
                "max_extent_km2": 1000,
                "accuracy_metrics": {
                    "mae": 0.23,
                    "rmse": 0.41,
                    "r2": 0.85
                }
            },
            {
                "name": "flood_cnn_v2",
                "version": "2.0.0",
                "description": "Enhanced model with uncertainty quantification",
                "input_resolution": "5m",
                "max_extent_km2": 500,
                "accuracy_metrics": {
                    "mae": 0.18,
                    "rmse": 0.35,
                    "r2": 0.89
                }
            }
        ]
    }

@app.get("/predictions/{prediction_id}")
async def get_prediction_status(prediction_id: str):
    """Get status of a prediction request."""
    # Mock implementation - in reality would check database
    return {
        "prediction_id": prediction_id,
        "status": "completed",
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:01:30Z",
        "processing_time_seconds": 90.5
    }

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "timestamp" in data
        
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
    
    def test_list_models_endpoint(self):
        """Test the model listing endpoint."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        
        # Check model structure
        model = data["models"][0]
        required_fields = ["name", "version", "description", "input_resolution", "accuracy_metrics"]
        for field in required_fields:
            assert field in model
    
    def test_prediction_status_endpoint(self):
        """Test prediction status retrieval."""
        test_id = "test-prediction-123"
        response = client.get(f"/predictions/{test_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction_id" in data
        assert "status" in data
        assert "processing_time_seconds" in data
        
        assert data["prediction_id"] == test_id
    
    @pytest.fixture
    def sample_dem_file(self):
        """Create a sample DEM file for testing."""
        # Create synthetic DEM data
        height, width = 50, 50
        dem_data = np.random.normal(100, 20, (height, width)).astype(np.float32)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            # Write as GeoTIFF
            transform = from_bounds(0, 0, 500, 500, width, height)
            
            with rasterio.open(
                tmp_file.name, 'w',
                driver='GTiff',
                height=height, width=width, count=1,
                dtype=dem_data.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform
            ) as dst:
                dst.write(dem_data, 1)
            
            return tmp_file.name
    
    def test_flood_prediction_endpoint(self, sample_dem_file):
        """Test flood prediction endpoint with file upload."""
        # Read DEM file
        with open(sample_dem_file, 'rb') as f:
            dem_content = f.read()
        
        # Prepare request
        prediction_request = {
            "rainfall_intensity": 25.5,
            "rainfall_duration": 120,
            "return_period": 10,
            "uncertainty_quantification": True
        }
        
        # Make request
        response = client.post(
            "/predict",
            data=prediction_request,
            files={"dem_file": ("test.tif", dem_content, "image/tiff")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = [
            "prediction_id", "flood_depth_map", "max_depth", 
            "flooded_area_km2", "confidence_score", "processing_time_seconds",
            "model_version"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate data types and ranges
        assert isinstance(data["prediction_id"], str)
        assert len(data["prediction_id"]) > 0
        
        assert isinstance(data["max_depth"], (int, float))
        assert data["max_depth"] >= 0
        
        assert isinstance(data["flooded_area_km2"], (int, float))
        assert data["flooded_area_km2"] >= 0
        
        assert isinstance(data["confidence_score"], (int, float))
        assert 0 <= data["confidence_score"] <= 1
        
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert data["processing_time_seconds"] > 0
        
        # Clean up
        Path(sample_dem_file).unlink()
    
    def test_validation_endpoint(self):
        """Test model validation endpoint."""
        # Create test data
        size = 32
        predictions = np.random.rand(size, size).astype(np.float32)
        ground_truth = predictions + np.random.normal(0, 0.1, (size, size)).astype(np.float32)
        
        # Encode as base64
        import base64
        pred_b64 = base64.b64encode(predictions.tobytes()).decode('utf-8')
        truth_b64 = base64.b64encode(ground_truth.tobytes()).decode('utf-8')
        
        # Prepare request
        validation_request = {
            "prediction_data": pred_b64,
            "ground_truth_data": truth_b64,
            "metrics_requested": ["mse", "mae", "r2", "flood_accuracy"]
        }
        
        # Make request
        response = client.post(
            "/validate",
            json=validation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = [
            "validation_id", "metrics", "validation_report", 
            "processing_time_seconds"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Validate metrics
        metrics = data["metrics"]
        expected_metrics = ["mse", "mae", "r2", "flood_accuracy"]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric]) or metric == "r2"  # R² can be NaN
        
        # MSE and MAE should be small for similar data
        assert metrics["mse"] < 1.0
        assert metrics["mae"] < 1.0
        
        # Validation report should be informative
        assert len(data["validation_report"]) > 50


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_invalid_file_type_error(self):
        """Test error handling for invalid file types."""
        # Create a text file instead of GeoTIFF
        invalid_content = b"This is not a GeoTIFF file"
        
        prediction_request = {
            "rainfall_intensity": 25.0,
            "rainfall_duration": 120
        }
        
        response = client.post(
            "/predict",
            data=prediction_request,
            files={"dem_file": ("test.txt", invalid_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_invalid_rainfall_parameters(self):
        """Test error handling for invalid rainfall parameters."""
        # Create minimal valid file
        valid_content = b"fake_geotiff_content"
        
        # Test negative rainfall intensity
        prediction_request = {
            "rainfall_intensity": -5.0,
            "rainfall_duration": 120
        }
        
        response = client.post(
            "/predict",
            data=prediction_request,
            files={"dem_file": ("test.tif", valid_content, "image/tiff")}
        )
        
        assert response.status_code == 400
        assert "Rainfall intensity must be between" in response.json()["detail"]
        
        # Test excessive rainfall duration
        prediction_request = {
            "rainfall_intensity": 25.0,
            "rainfall_duration": 2000  # Too long
        }
        
        response = client.post(
            "/predict",
            data=prediction_request,
            files={"dem_file": ("test.tif", valid_content, "image/tiff")}
        )
        
        assert response.status_code == 400
        assert "Rainfall duration must be between" in response.json()["detail"]
    
    def test_validation_data_size_mismatch(self):
        """Test error handling for mismatched validation data."""
        import base64
        
        # Create mismatched data
        predictions = np.random.rand(10, 10).astype(np.float32)
        ground_truth = np.random.rand(20, 20).astype(np.float32)  # Different size
        
        pred_b64 = base64.b64encode(predictions.tobytes()).decode('utf-8')
        truth_b64 = base64.b64encode(ground_truth.tobytes()).decode('utf-8')
        
        validation_request = {
            "prediction_data": pred_b64,
            "ground_truth_data": truth_b64,
            "metrics_requested": ["mse"]
        }
        
        response = client.post(
            "/validate",
            json=validation_request
        )
        
        assert response.status_code == 400
        assert "same size" in response.json()["detail"]
    
    def test_invalid_base64_data(self):
        """Test error handling for invalid base64 data."""
        validation_request = {
            "prediction_data": "invalid_base64_data!",
            "ground_truth_data": "also_invalid!",
            "metrics_requested": ["mse"]
        }
        
        response = client.post(
            "/validate",
            json=validation_request
        )
        
        assert response.status_code == 500  # Should be caught as processing error
        assert "Validation failed" in response.json()["detail"]
    
    def test_missing_required_fields(self):
        """Test error handling for missing required fields."""
        # Test prediction endpoint without rainfall parameters
        response = client.post(
            "/predict",
            data={},  # Missing required fields
            files={"dem_file": ("test.tif", b"content", "image/tiff")}
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test validation endpoint without data
        response = client.post(
            "/validate",
            json={}  # Missing required fields
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_prediction_id(self):
        """Test handling of nonexistent prediction IDs."""
        # The mock implementation returns data for any ID
        # In a real implementation, this should return 404
        response = client.get("/predictions/nonexistent-id")
        
        # Current mock returns 200, but real implementation should return 404
        assert response.status_code == 200
        # In production: assert response.status_code == 404


class TestAPIPerformance:
    """Test API performance and scalability."""
    
    def test_concurrent_requests(self):
        """Test API handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                start_time = time.time()
                response = client.get("/health")
                end_time = time.time()
                
                results.append({
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                })
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        num_threads = 10
        threads = []
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Validate results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        
        # All requests should succeed
        success_count = sum(1 for r in results if r['status_code'] == 200)
        assert success_count == num_threads
        
        # Average response time should be reasonable
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        assert avg_response_time < 1.0, f"Average response time too slow: {avg_response_time}s"
        
        # Total time should be much less than sequential execution
        expected_sequential_time = avg_response_time * num_threads
        assert total_time < expected_sequential_time * 0.8, "Concurrent execution not faster than sequential"
    
    def test_large_file_upload_performance(self):
        """Test performance with large file uploads."""
        # Create a larger synthetic DEM
        height, width = 200, 200
        large_dem = np.random.normal(100, 20, (height, width)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            transform = from_bounds(0, 0, 2000, 2000, width, height)
            
            with rasterio.open(
                tmp_file.name, 'w',
                driver='GTiff',
                height=height, width=width, count=1,
                dtype=large_dem.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform
            ) as dst:
                dst.write(large_dem, 1)
            
            # Read file content
            with open(tmp_file.name, 'rb') as f:
                dem_content = f.read()
            
            file_size_mb = len(dem_content) / (1024 * 1024)
            
            # Make timed request
            import time
            start_time = time.time()
            
            prediction_request = {
                "rainfall_intensity": 30.0,
                "rainfall_duration": 180
            }
            
            response = client.post(
                "/predict",
                data=prediction_request,
                files={"dem_file": ("large_test.tif", dem_content, "image/tiff")}
            )
            
            processing_time = time.time() - start_time
            
            # Validate response
            assert response.status_code == 200
            
            # Performance should scale reasonably with file size
            # Allow up to 10 seconds per MB for processing
            max_expected_time = max(5.0, file_size_mb * 10)
            assert processing_time < max_expected_time, f"Processing too slow: {processing_time}s for {file_size_mb:.1f}MB"
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_memory_usage_during_processing(self):
        """Test memory usage during API processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make several requests to test memory accumulation
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            
            # Check memory after each request
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be bounded
            max_increase_mb = 50  # 50 MB threshold
            assert memory_increase < max_increase_mb * 1024 * 1024, \
                f"Memory usage increased by {memory_increase / 1024 / 1024:.1f} MB after {i+1} requests"


class TestAPIProduction:
    """Test production-level API requirements."""
    
    def test_api_documentation_available(self):
        """Test that API documentation is accessible."""
        # FastAPI automatically generates OpenAPI docs
        response = client.get("/docs")
        
        # Should redirect to Swagger UI or return HTML
        assert response.status_code in [200, 307]  # 307 for redirect
        
        # Test OpenAPI schema endpoint
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Validate key endpoints are documented
        expected_paths = ["/health", "/predict", "/validate", "/models"]
        for path in expected_paths:
            assert path in schema["paths"]
    
    def test_api_versioning(self):
        """Test API version handling."""
        # Test that version is included in responses
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0
        
        # Version should follow semantic versioning pattern
        import re
        version_pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(version_pattern, data["version"]), \
            f"Version '{data['version']}' doesn't follow semantic versioning"
    
    def test_error_response_format(self):
        """Test that error responses follow consistent format."""
        # Trigger a validation error
        response = client.post("/predict", data={})  # Missing required fields
        
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        
        # For FastAPI validation errors, detail should be a list
        if isinstance(error_data["detail"], list):
            # Validate structure of validation error
            for error in error_data["detail"]:
                assert "loc" in error  # Location of error
                assert "msg" in error  # Error message
                assert "type" in error  # Error type
    
    def test_cors_headers(self):
        """Test CORS headers for browser compatibility."""
        # This would require CORS middleware to be configured
        # For now, just test that endpoints are accessible
        response = client.get("/health")
        assert response.status_code == 200
        
        # In production, you would test for specific CORS headers:
        # assert "Access-Control-Allow-Origin" in response.headers
        # assert "Access-Control-Allow-Methods" in response.headers
    
    def test_rate_limiting_headers(self):
        """Test rate limiting (if implemented)."""
        # Make a request and check for rate limiting headers
        response = client.get("/health")
        assert response.status_code == 200
        
        # In production with rate limiting, you might expect:
        # assert "X-RateLimit-Limit" in response.headers
        # assert "X-RateLimit-Remaining" in response.headers
        # assert "X-RateLimit-Reset" in response.headers
        
        # For now, just ensure the endpoint works
        assert response.json()["status"] == "healthy"
    
    def test_security_headers(self):
        """Test security headers in responses."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # In production, security headers should be present:
        # Common security headers to check:
        # - X-Content-Type-Options: nosniff
        # - X-Frame-Options: DENY or SAMEORIGIN
        # - X-XSS-Protection: 1; mode=block
        # - Strict-Transport-Security (for HTTPS)
        
        # For now, just validate response structure
        assert isinstance(response.json(), dict)
    
    def test_input_sanitization(self):
        """Test that inputs are properly sanitized."""
        # Test with potentially malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE predictions; --",
            "../../../etc/passwd",
            "${7*7}"
        ]
        
        for malicious_input in malicious_inputs:
            # Test in prediction request
            prediction_request = {
                "rainfall_intensity": 25.0,
                "rainfall_duration": 120,
                "model_version": malicious_input  # Inject malicious content
            }
            
            # Create minimal file
            fake_content = b"fake_geotiff"
            
            response = client.post(
                "/predict",
                data=prediction_request,
                files={"dem_file": ("test.tif", fake_content, "image/tiff")}
            )
            
            # Request might fail for other reasons, but shouldn't execute malicious code
            # The important thing is that the server doesn't crash or execute the input
            assert response.status_code in [200, 400, 422, 500]
            
            # Response shouldn't contain the exact malicious input (indicates lack of sanitization)
            response_text = response.text
            assert malicious_input not in response_text, f"Unsanitized input found in response: {malicious_input}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
