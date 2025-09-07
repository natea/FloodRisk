#!/usr/bin/env python
"""
Simple local runner for the FloodRisk API
Run with: python run_local.py
"""

import os
import sys

# Set environment variables for local development
os.environ['DATABASE_URL'] = 'postgresql://flooduser:password123@localhost:5433/floodrisk'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'true'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import uvicorn
    from src.api.main import app
    
    print("Starting FloodRisk API server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
except ImportError as e:
    print(f"\nError: Missing dependencies - {e}")
    print("\nTo run the API locally, you need to install minimal dependencies:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install FastAPI and core deps: pip install fastapi uvicorn sqlalchemy redis psycopg2-binary")
    print("\nFor full functionality with ML models, use Docker: make dev-up")