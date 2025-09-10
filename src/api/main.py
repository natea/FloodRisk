"""
FastAPI main application for flood risk prediction service.
Provides REST API endpoints for prediction, validation, health checks, and metrics.
"""

import time
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_settings
from .schemas import (
    PredictionInput,
    BatchPredictionInput,
    PredictionOutput,
    BatchPredictionOutput,
    ValidationRequest,
    ValidationResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)
from .inference import get_predictor, ModelNotLoadedException, InferenceError
from .utils import (
    get_current_memory_usage,
    get_uptime_seconds,
    get_average_response_time,
    get_predictions_last_hour,
    track_prediction,
    generate_id,
    create_error_response,
    timing_middleware,
    APIKeyAuth,
    prediction_count,
)
from .metrics import MetricsMiddleware, get_metrics as get_prometheus_metrics
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting FloodRisk API service")
    try:
        predictor = get_predictor()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")

    yield

    # Shutdown
    logger.info("Shutting down FloodRisk API service")


# Initialize FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI service for flood risk prediction and validation",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add Prometheus metrics middleware
app.add_middleware(MetricsMiddleware)

if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"],
    )

# Initialize authentication
auth_scheme = APIKeyAuth(auto_error=False)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    """Handle model not loaded exceptions."""
    return JSONResponse(
        status_code=503,
        content=create_error_response(
            message="Model not available",
            detail=str(exc),
            error_code="MODEL_NOT_LOADED",
        ),
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """Handle inference errors."""
    return JSONResponse(
        status_code=400,
        content=create_error_response(
            message="Prediction failed", detail=str(exc), error_code="INFERENCE_ERROR"
        ),
    )


@app.exception_handler(ValueError)
async def validation_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content=create_error_response(
            message="Validation error", detail=str(exc), error_code="VALIDATION_ERROR"
        ),
    )


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    summary="Make flood risk prediction",
    description="Predict flood risk for a specific location based on various environmental and infrastructure factors.",
)
@timing_middleware
async def predict_flood_risk(
    prediction_input: PredictionInput,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> PredictionOutput:
    """Make a single flood risk prediction."""
    try:
        predictor = get_predictor()

        # Convert Pydantic model to dict
        input_data = prediction_input.dict()

        # Make prediction
        result = predictor.predict(input_data, include_confidence=True)

        # Track metrics in background
        background_tasks.add_task(track_prediction)

        # Track Prometheus metrics
        from .metrics import (
            prediction_count as prom_prediction_count,
            model_inference_time,
        )

        prom_prediction_count.labels(
            risk_level=result.get("risk_level", "unknown")
        ).inc()

        logger.info(
            f"Prediction made for coordinates ({input_data['latitude']}, {input_data['longitude']})"
        )

        return PredictionOutput(**result)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid batch input data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        413: {"model": ErrorResponse, "description": "Batch size too large"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    summary="Make batch flood risk predictions",
    description="Predict flood risk for multiple locations in a single request for improved efficiency.",
)
@timing_middleware
async def predict_batch_flood_risk(
    batch_input: BatchPredictionInput,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> BatchPredictionOutput:
    """Make batch flood risk predictions."""
    start_time = time.time()

    try:
        predictor = get_predictor()

        if len(batch_input.predictions) > settings.max_prediction_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size exceeds maximum of {settings.max_prediction_batch_size}",
            )

        # Convert Pydantic models to dicts
        input_data_list = [pred.dict() for pred in batch_input.predictions]

        # Make batch predictions
        results = predictor.predict_batch(
            input_data_list, include_confidence=batch_input.include_confidence
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Track metrics in background
        background_tasks.add_task(track_prediction)

        logger.info(
            f"Batch prediction completed for {len(results)} locations in {processing_time_ms:.2f}ms"
        )

        # Convert results to Pydantic models
        prediction_outputs = [PredictionOutput(**result) for result in results]

        return BatchPredictionOutput(
            predictions=prediction_outputs,
            total_processed=len(prediction_outputs),
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post(
    "/validate",
    response_model=ValidationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid validation data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
    summary="Validate prediction against actual results",
    description="Submit actual flood results to validate and improve prediction accuracy.",
)
async def validate_prediction(
    validation_request: ValidationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> ValidationResponse:
    """Validate a prediction against actual results."""
    try:
        predictor = get_predictor()

        # Process validation
        result = predictor.validate_prediction(
            prediction_id=validation_request.prediction_id or generate_id("pred"),
            actual_result=validation_request.dict(),
        )

        logger.info(f"Validation recorded: {result['validation_id']}")

        return ValidationResponse(**result)

    except Exception as e:
        logger.error(f"Validation endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the API service and its dependencies.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        model_info = predictor.get_model_info()
        model_status = "loaded" if model_info["status"] == "loaded" else "not_loaded"

        dependencies = {
            "model": model_status,
            "memory": "ok" if get_current_memory_usage() > 0 else "unknown",
        }

        # Add database check if configured
        if settings.database_url:
            dependencies["database"] = "connected"  # Placeholder

        overall_status = (
            "healthy"
            if all(
                status in ["ok", "loaded", "connected"]
                for status in dependencies.values()
            )
            else "degraded"
        )

        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            model_status=model_status,
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            model_status="error",
            dependencies={"error": str(e)},
        )


@app.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description="Get Prometheus-formatted metrics for monitoring.",
)
async def get_metrics() -> Response:
    """Get Prometheus metrics."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")

    try:
        metrics_data = get_prometheus_metrics()
        return Response(content=metrics_data, media_type="text/plain")

    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.get(
    "/metrics/json",
    response_model=MetricsResponse,
    summary="API metrics endpoint (JSON)",
    description="Get performance and usage metrics for the API service in JSON format.",
)
async def get_metrics_json() -> MetricsResponse:
    """Get API performance metrics in JSON format."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")

    try:
        return MetricsResponse(
            total_predictions=prediction_count,
            predictions_last_hour=get_predictions_last_hour(),
            average_response_time_ms=get_average_response_time(),
            model_accuracy=None,  # Would need to implement accuracy tracking
            uptime_seconds=get_uptime_seconds(),
            memory_usage_mb=get_current_memory_usage(),
        )

    except Exception as e:
        logger.error(f"Metrics endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.get(
    "/model/info",
    response_model=Dict[str, Any],
    summary="Model information",
    description="Get detailed information about the loaded prediction model.",
)
async def get_model_info(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        predictor = get_predictor()
        return predictor.get_model_info()

    except Exception as e:
        logger.error(f"Model info endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.post(
    "/model/reload",
    response_model=Dict[str, str],
    summary="Reload model",
    description="Reload the prediction model (admin endpoint).",
)
async def reload_model(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> Dict[str, str]:
    """Reload the prediction model."""
    try:
        from .inference import reload_model

        success = reload_model()

        if success:
            logger.info("Model reloaded successfully")
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")

    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


# Additional utility endpoints


@app.get("/version", response_model=Dict[str, str])
async def get_version():
    """Get API version information."""
    return {
        "version": settings.app_version,
        "app_name": settings.app_name,
        "environment": "development" if settings.debug else "production",
    }


@app.get("/status", response_model=Dict[str, str])
async def get_status():
    """Simple status endpoint."""
    return {"status": "operational", "timestamp": time.time()}


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
