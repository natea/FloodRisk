"""
Prometheus metrics for FloodRisk API
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from fastapi import Response
import time
import psutil

# Create a custom registry
REGISTRY = CollectorRegistry()

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

active_requests = Gauge(
    'http_requests_active',
    'Number of active HTTP requests',
    registry=REGISTRY
)

prediction_count = Counter(
    'predictions_total',
    'Total number of flood predictions',
    ['risk_level'],
    registry=REGISTRY
)

model_inference_time = Histogram(
    'model_inference_seconds',
    'Model inference time in seconds',
    registry=REGISTRY
)

# System metrics
cpu_usage = Gauge(
    'process_cpu_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

memory_usage = Gauge(
    'process_memory_bytes',
    'Memory usage in bytes',
    registry=REGISTRY
)

def update_system_metrics():
    """Update system resource metrics"""
    try:
        process = psutil.Process()
        cpu_usage.set(process.cpu_percent())
        memory_usage.set(process.memory_info().rss)
    except Exception:
        pass

def get_metrics():
    """Generate Prometheus metrics"""
    update_system_metrics()
    return generate_latest(REGISTRY)

class MetricsMiddleware:
    """Middleware to track request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            method = scope["method"]
            
            # Skip metrics endpoint
            if path == "/metrics":
                await self.app(scope, receive, send)
                return
            
            # Track active requests
            active_requests.inc()
            start_time = time.time()
            
            # Capture status code
            status_code = 200
            
            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)
            
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                # Record metrics
                duration = time.time() - start_time
                request_count.labels(
                    method=method,
                    endpoint=path,
                    status=status_code
                ).inc()
                request_duration.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)
                active_requests.dec()
        else:
            await self.app(scope, receive, send)