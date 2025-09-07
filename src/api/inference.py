"""
Model inference pipeline for flood risk prediction.
Handles model loading, preprocessing, and prediction logic.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

from .config import get_settings
from .utils import (
    format_prediction_output,
    validate_input_data,
    sanitize_input,
    timing_middleware
)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class ModelNotLoadedException(Exception):
    """Exception raised when model is not properly loaded."""
    pass


class InferenceError(Exception):
    """Exception raised during inference process."""
    pass


class FloodRiskPredictor:
    """Main predictor class for flood risk assessment."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self.model_metadata = {}
        self._is_loaded = False
        self.settings = get_settings()
        
    def load_model(self, model_path: str = None) -> bool:
        """Load the trained model and preprocessor."""
        try:
            model_path = model_path or self.settings.model_path
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}, creating mock model")
                self._create_mock_model()
                return True
            
            # Try to load pickled model
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle different model file formats
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names', self._get_default_features())
                    self.model_metadata = model_data.get('metadata', {})
                else:
                    # Assume it's just the model
                    self.model = model_data
                    self.scaler = StandardScaler()  # Default scaler
                    self.feature_names = self._get_default_features()
                    
            except (pickle.PickleError, joblib.externals.loky.process_executor.TerminatedWorkerError):
                # Try joblib format
                model_data = joblib.load(model_path)
                self.model = model_data
                self.scaler = StandardScaler()  # Default scaler
                self.feature_names = self._get_default_features()
            
            self.model_version = self.settings.model_version
            self._is_loaded = True
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Feature count: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            logger.info("Creating mock model for development/testing")
            self._create_mock_model()
            return True
    
    def _create_mock_model(self):
        """Create a mock model for development/testing purposes."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create a simple mock model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = self._get_default_features()
        
        # Create some dummy training data to fit the model
        n_samples = 100
        X_dummy = np.random.rand(n_samples, len(self.feature_names))
        y_dummy = np.random.rand(n_samples)  # Risk scores between 0 and 1
        
        # Fit the scaler and model
        X_scaled = self.scaler.fit_transform(X_dummy)
        self.model.fit(X_scaled, y_dummy)
        
        self.model_version = "mock-1.0.0"
        self.model_metadata = {
            "type": "mock",
            "created_at": datetime.utcnow().isoformat(),
            "features": len(self.feature_names)
        }
        self._is_loaded = True
        
        logger.info("Mock model created successfully")
    
    def _get_default_features(self) -> List[str]:
        """Get default feature names for the model."""
        return [
            "latitude",
            "longitude",
            "elevation",
            "current_rainfall",
            "current_water_level",
            "soil_moisture",
            "drainage_capacity",
            "population_density",
            "building_density",
            "forecast_rainfall_sum",
            "forecast_rainfall_max",
            "historical_rainfall_mean",
            "historical_water_level_mean"
        ]
    
    def _extract_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Extract and process features from input data."""
        features = {}
        
        # Basic geographic features
        features["latitude"] = input_data.get("latitude", 0.0)
        features["longitude"] = input_data.get("longitude", 0.0)
        features["elevation"] = input_data.get("elevation", 0.0)
        
        # Current conditions
        features["current_rainfall"] = input_data.get("current_rainfall", 0.0)
        features["current_water_level"] = input_data.get("current_water_level", 0.0)
        features["soil_moisture"] = input_data.get("soil_moisture", 50.0)  # Default to 50%
        
        # Infrastructure
        features["drainage_capacity"] = input_data.get("drainage_capacity", 25.0)  # Default capacity
        features["population_density"] = input_data.get("population_density", 1000.0)  # Default density
        features["building_density"] = input_data.get("building_density", 30.0)  # Default 30%
        
        # Process forecast data
        forecast_rainfall = input_data.get("forecast_rainfall", [])
        if forecast_rainfall:
            features["forecast_rainfall_sum"] = sum(forecast_rainfall)
            features["forecast_rainfall_max"] = max(forecast_rainfall)
        else:
            features["forecast_rainfall_sum"] = 0.0
            features["forecast_rainfall_max"] = 0.0
        
        # Process historical data
        historical_rainfall = input_data.get("historical_rainfall", [])
        if historical_rainfall:
            features["historical_rainfall_mean"] = np.mean(historical_rainfall)
        else:
            features["historical_rainfall_mean"] = 10.0  # Default
        
        historical_water_levels = input_data.get("historical_water_levels", [])
        if historical_water_levels:
            features["historical_water_level_mean"] = np.mean(historical_water_levels)
        else:
            features["historical_water_level_mean"] = 1.0  # Default
        
        # Convert to numpy array in the correct order
        feature_vector = np.array([
            features[feature_name] for feature_name in self.feature_names
        ]).reshape(1, -1)
        
        return feature_vector
    
    @timing_middleware
    def predict(self, input_data: Dict[str, Any], include_confidence: bool = True) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self._is_loaded:
            raise ModelNotLoadedException("Model not loaded. Call load_model() first.")
        
        try:
            # Sanitize and validate input
            input_data = sanitize_input(input_data)
            validation_errors = validate_input_data(input_data)
            
            if validation_errors:
                raise InferenceError(f"Input validation failed: {', '.join(validation_errors)}")
            
            # Extract features
            features = self._extract_features(input_data)
            
            # Scale features if scaler is available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Make prediction
            risk_score = self.model.predict(features)[0]
            
            # Ensure risk score is between 0 and 1
            risk_score = np.clip(risk_score, 0.0, 1.0)
            
            # Format output
            output = format_prediction_output(
                risk_score=risk_score,
                input_data=input_data,
                model_version=self.model_version,
                include_confidence=include_confidence
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise InferenceError(f"Prediction failed: {str(e)}")
    
    @timing_middleware
    def predict_batch(self, input_data_list: List[Dict[str, Any]], include_confidence: bool = True) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        if not self._is_loaded:
            raise ModelNotLoadedException("Model not loaded. Call load_model() first.")
        
        if len(input_data_list) > self.settings.max_prediction_batch_size:
            raise InferenceError(f"Batch size exceeds maximum of {self.settings.max_prediction_batch_size}")
        
        results = []
        
        try:
            # Process all inputs to features
            all_features = []
            valid_inputs = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    # Sanitize and validate input
                    input_data = sanitize_input(input_data)
                    validation_errors = validate_input_data(input_data)
                    
                    if validation_errors:
                        logger.warning(f"Skipping invalid input {i}: {', '.join(validation_errors)}")
                        continue
                    
                    features = self._extract_features(input_data)
                    all_features.append(features[0])  # Remove the extra dimension
                    valid_inputs.append(input_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing input {i}: {str(e)}")
                    continue
            
            if not all_features:
                raise InferenceError("No valid inputs found in batch")
            
            # Convert to numpy array and scale
            features_array = np.array(all_features)
            
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            # Make batch prediction
            risk_scores = self.model.predict(features_array)
            
            # Ensure risk scores are between 0 and 1
            risk_scores = np.clip(risk_scores, 0.0, 1.0)
            
            # Format outputs
            for i, (input_data, risk_score) in enumerate(zip(valid_inputs, risk_scores)):
                try:
                    output = format_prediction_output(
                        risk_score=float(risk_score),
                        input_data=input_data,
                        model_version=self.model_version,
                        include_confidence=include_confidence
                    )
                    results.append(output)
                except Exception as e:
                    logger.error(f"Error formatting output {i}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise InferenceError(f"Batch prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "model_version": self.model_version,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "metadata": self.model_metadata
        }
    
    def validate_prediction(
        self, 
        prediction_id: str,
        actual_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a previous prediction against actual results."""
        # This is a placeholder implementation
        # In production, you'd store predictions and calculate actual accuracy
        
        validation_id = f"val_{int(datetime.utcnow().timestamp())}"
        
        # Simple accuracy calculation (placeholder)
        accuracy_score = 0.85  # Mock accuracy
        
        return {
            "validation_id": validation_id,
            "prediction_id": prediction_id,
            "accuracy_score": accuracy_score,
            "status": "recorded",
            "recorded_timestamp": datetime.utcnow().isoformat()
        }


# Global predictor instance
_predictor = None


def get_predictor() -> FloodRiskPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = FloodRiskPredictor()
        _predictor.load_model()
    return _predictor


def reload_model() -> bool:
    """Reload the model (useful for model updates)."""
    global _predictor
    if _predictor is not None:
        return _predictor.load_model()
    return False