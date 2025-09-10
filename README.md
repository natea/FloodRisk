# FloodRisk - Flood Depth Prediction System

A comprehensive flood risk prediction system using machine learning and hydrological modeling. This project combines physics-informed neural networks with multi-scale CNN architectures to predict flood depths and assess flood risks.

## Features

- **Multi-Scale CNN Architecture**: U-Net based encoder-decoder with multi-scale input processing
- **Physics-Informed Constraints**: Mass conservation and hydrological physics integration
- **Geospatial Processing**: DEM processing, terrain feature extraction, and hydrological conditioning
- **Real-time Prediction API**: FastAPI-based REST API for flood predictions
- **Scalable Architecture**: Docker containerization with Redis caching and PostgreSQL storage
- **Monitoring & Observability**: Integrated Prometheus metrics and Grafana dashboards

<img width="3376" height="1906" alt="CleanShot 2025-09-07 at 17 27 57@2x" src="https://github.com/user-attachments/assets/90cf3560-b428-49b7-9114-038866c5079c" />

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FloodRisk
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the development environment**
   ```bash
   make dev-up
   ```

4. **Run database migrations**
   ```bash
   make db-migrate
   ```

5. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Grafana Dashboard: http://localhost:3000 (admin/admin123)
   - Jupyter Notebook: http://localhost:8888

## Development Setup

### Local Development

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests**
   ```bash
   make test
   ```

### Docker Development

```bash
# Start all services
make dev-up

# View logs
make logs

# Stop services
make dev-down

# Rebuild containers
make build
```

## Project Structure

```
FloodRisk/
├── src/                          # Source code
│   ├── models/                   # ML models and architectures
│   │   ├── flood_cnn.py         # Multi-scale CNN implementation
│   │   └── __init__.py
│   ├── preprocessing/            # Data preprocessing modules
│   │   ├── dem/                 # DEM processing
│   │   └── terrain/             # Terrain feature extraction
│   ├── api/                     # FastAPI application
│   ├── tasks/                   # Background tasks (Celery)
│   └── utils/                   # Utility functions
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Data directory
├── models/                      # Trained model artifacts
├── logs/                        # Application logs
├── docker-compose.yml           # Docker services configuration
├── Dockerfile                   # Docker image definition
├── requirements.txt             # Python dependencies
├── Makefile                     # Development commands
└── pytest.ini                  # Test configuration
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Flood Depth
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "elevation_data": [...],
       "rainfall_data": [...],
       "terrain_features": {...}
     }'
```

### Upload DEM for Processing
```bash
curl -X POST "http://localhost:8000/api/v1/dem/upload" \
     -F "file=@path/to/dem.tif"
```

## Model Architecture

The system uses a multi-scale CNN architecture with:

- **Input Scales**: 256m high-resolution + 512m/1024m context
- **Physics Constraints**: Mass conservation loss functions
- **Attention Mechanisms**: Multi-scale feature fusion
- **Dimensionless Features**: Normalized terrain characteristics

### Key Components

1. **PhysicsInformedLoss**: Incorporates hydrological constraints
2. **MultiScaleEncoder**: Processes multiple resolution inputs
3. **AttentionDecoder**: Fuses multi-scale features
4. **RainfallScaling**: Handles dynamic rainfall inputs

## Data Processing Pipeline

1. **DEM Preprocessing**
   - Hydrological conditioning
   - Sink removal and filling
   - Flow direction calculation

2. **Feature Extraction**
   - Slope and curvature calculation
   - Flow accumulation analysis
   - Height Above Nearest Drainage (HAND)

3. **Multi-scale Preparation**
   - Resampling to multiple resolutions
   - Normalization and standardization

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test module
pytest tests/test_models.py

# Run integration tests
pytest tests/integration/
```

## Monitoring and Observability

### Metrics

The application exposes Prometheus metrics at `/metrics`:

- Request latency and throughput
- Model prediction accuracy
- Resource utilization
- Background task status

### Logging

Structured logging with configurable levels:
- Application logs: `./logs/floodrisk.log`
- Error tracking with Sentry (optional)

### Health Checks

- Application health: `/health`
- Database connectivity: `/health/db`
- Redis connectivity: `/health/redis`

## Production Deployment

### Environment Setup

1. **Set production environment variables**
   ```bash
   export ENVIRONMENT=production
   export DEBUG=false
   export DATABASE_URL=postgresql://...
   ```

2. **Use production Docker image**
   ```bash
   docker build --target production -t floodrisk:prod .
   ```

3. **Configure SSL and security headers**

### Security Considerations

- Use strong secret keys
- Enable SSL/TLS termination
- Configure CORS appropriately
- Implement rate limiting
- Regular security updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use Black for formatting
- Sort imports with isort
- Type hints required
- Docstrings for all functions

## License

[License information]

## Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

## Acknowledgments

- LISFLOOD-FP modeling framework
- PyTorch community

- Geospatial Python ecosystem
