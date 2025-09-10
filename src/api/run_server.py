#!/usr/bin/env python3
"""
Server startup script for FloodRisk API.
Provides convenient way to start the API server with proper configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from src.api.main import app
    from src.api.config import get_settings
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install required packages: pip install -r requirements-api.txt")
    sys.exit(1)


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            (
                logging.FileHandler("logs/api.log")
                if os.path.exists("logs")
                else logging.NullHandler()
            ),
        ],
    )


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="FloodRisk API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Get settings
    settings = get_settings()

    # Override settings with command line arguments
    host = args.host or settings.host
    port = args.port or settings.port

    print(f"Starting FloodRisk API Server on {host}:{port}")
    print(f"Environment: {'development' if args.debug else 'production'}")
    print(f"Debug mode: {args.debug}")
    print(f"Auto-reload: {args.reload}")
    print("---")

    # Configure uvicorn
    config = {
        "app": "src.api.main:app",
        "host": host,
        "port": port,
        "log_level": args.log_level.lower(),
        "reload": args.reload,
        "workers": 1 if args.reload else args.workers,  # Can't use workers with reload
    }

    # Add SSL configuration if certificates are available
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")

    if ssl_keyfile and ssl_certfile:
        if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
            config.update(
                {
                    "ssl_keyfile": ssl_keyfile,
                    "ssl_certfile": ssl_certfile,
                }
            )
            print(f"SSL enabled with cert: {ssl_certfile}")

    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
