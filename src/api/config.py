"""
Configuration management for the FastAPI service.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    app_name: str = "FloodRisk API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Database Configuration
    database_url: Optional[str] = None

    # Model Configuration
    model_path: str = "models/flood_risk_model.pkl"
    model_version: str = "1.0.0"
    max_prediction_batch_size: int = 100

    # Security Configuration
    api_key: Optional[str] = None
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # CORS Configuration
    cors_origins: list = ["*"]

    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"

    @validator("model_path")
    def validate_model_path(cls, v):
        """Ensure model path exists or is valid."""
        if not os.path.exists(v) and not v.startswith("http"):
            # Allow for development/testing without actual model file
            if not os.getenv("TESTING", False):
                print(f"Warning: Model path {v} does not exist")
        return v

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Environment-specific configurations
class DevelopmentConfig(Settings):
    """Development environment configuration."""

    debug: bool = True
    log_level: str = "DEBUG"


class ProductionConfig(Settings):
    """Production environment configuration."""

    debug: bool = False
    log_level: str = "WARNING"
    cors_origins: list = []  # Restrict CORS in production


class TestingConfig(Settings):
    """Testing environment configuration."""

    debug: bool = True
    database_url: str = "sqlite:///test.db"
    model_path: str = "tests/fixtures/test_model.pkl"


def get_config_by_env(env: str = None) -> Settings:
    """Get configuration based on environment."""
    env = env or os.getenv("ENVIRONMENT", "development").lower()

    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }

    return config_map.get(env, Settings)()
