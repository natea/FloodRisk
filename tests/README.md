# FloodRisk Integration Test Suite

This directory contains comprehensive integration tests for the FloodRisk flood prediction system, designed to validate production readiness and ensure no mock implementations remain in the final codebase.

## Test Structure

### Core Test Files

1. **`test_preprocessing.py`** - DEM Processing & Feature Extraction
   - Hydrological conditioning workflows
   - Terrain feature extraction validation
   - Real data processing scenarios
   - File I/O operations with actual GeoTIFF files
   - Quality validation metrics

2. **`test_model.py`** - CNN Model Architecture & Training
   - Model initialization and forward pass validation
   - Physics-informed loss function testing
   - Multi-scale input processing
   - Training loop components
   - Memory efficiency and numerical stability

3. **`test_validation.py`** - Model Validation & Metrics
   - Comprehensive evaluation metrics
   - Cross-validation workflows
   - Performance benchmarking
   - Uncertainty quantification validation
   - Production accuracy requirements

4. **`test_api.py`** - REST API Endpoints & Web Services
   - Complete FastAPI application testing
   - File upload/download functionality
   - Request/response validation
   - Error handling and edge cases
   - Production deployment scenarios

5. **`test_e2e.py`** - End-to-End Pipeline Testing
   - Complete workflow from raw DEM to predictions
   - Integration between all components
   - Production performance validation
   - Resource usage monitoring
   - System scalability testing

### Supporting Files

- **`conftest.py`** - Shared fixtures and test configuration
- **`requirements.txt`** - Test dependencies
- **`pytest.ini`** - Pytest configuration with coverage settings
- **`__init__.py`** - Test package initialization

## Production Validation Features

### Real Implementation Testing
- ✅ No mock implementations in production paths
- ✅ Real database/file system operations
- ✅ Actual model inference and training
- ✅ Production-scale data processing
- ✅ End-to-end workflow validation

### Performance Validation
- ✅ Memory usage monitoring
- ✅ Processing time benchmarks
- ✅ Concurrent request handling
- ✅ Resource leak detection
- ✅ Scalability testing

### Accuracy Validation
- ✅ Comprehensive metrics calculation
- ✅ Cross-validation procedures
- ✅ Spatial pattern analysis
- ✅ Uncertainty quantification
- ✅ Production accuracy thresholds

## Quick Start

### Installation
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_preprocessing.py  # Data processing
pytest tests/test_model.py          # Model architecture
pytest tests/test_validation.py     # Model validation
pytest tests/test_api.py            # API endpoints
pytest tests/test_e2e.py            # End-to-end workflows

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest tests/ -m "not slow"

# Run production validation tests
pytest tests/ -m "production"

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

### Test Markers

Tests are categorized with markers for selective execution:

- `@pytest.mark.slow` - Long-running tests (>10 seconds)
- `@pytest.mark.gpu` - Tests requiring GPU acceleration
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.production` - Production validation tests
- `@pytest.mark.preprocessing` - Data preprocessing tests
- `@pytest.mark.model` - Model-specific tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.validation` - Validation and metrics tests
- `@pytest.mark.e2e` - End-to-end workflow tests

## Test Design Principles

### 1. No Mock Dependencies in Production Paths
- Real file I/O operations with actual GeoTIFF files
- Actual model training and inference
- Real database connections and operations
- Production-like data processing workflows

### 2. Comprehensive Validation
- Multiple validation scenarios for each component
- Edge case and error condition testing
- Performance and resource usage validation
- Cross-validation and statistical testing

### 3. Production Readiness
- Tests mirror production deployment conditions
- Realistic data sizes and processing loads
- Performance benchmarks and thresholds
- Security and input validation testing

### 4. Maintainability
- Clear test structure and documentation
- Shared fixtures for common operations
- Parametrized tests for multiple scenarios
- Comprehensive error reporting

## Key Validation Areas

### Data Processing Pipeline
- ✅ DEM loading and validation
- ✅ Hydrological conditioning algorithms
- ✅ Terrain feature extraction
- ✅ Multi-scale data processing
- ✅ File format handling (GeoTIFF, NetCDF)

### Model Architecture
- ✅ U-Net encoder-decoder structure
- ✅ Multi-scale input processing
- ✅ Physics-informed loss functions
- ✅ Attention mechanisms
- ✅ Uncertainty quantification

### Training & Validation
- ✅ Training loop stability
- ✅ Gradient computation and optimization
- ✅ Cross-validation procedures
- ✅ Metric calculation accuracy
- ✅ Model serialization/loading

### API Integration
- ✅ REST endpoint functionality
- ✅ File upload handling
- ✅ Request/response validation
- ✅ Error handling and logging
- ✅ Security and input sanitization

### System Integration
- ✅ End-to-end workflow execution
- ✅ Component integration
- ✅ Resource management
- ✅ Performance monitoring
- ✅ Production deployment readiness

## Performance Benchmarks

### Processing Time Thresholds
- DEM preprocessing: < 30 seconds (for 1km² at 10m resolution)
- Model inference: < 5 seconds (for 256x256 prediction)
- Full E2E pipeline: < 5 minutes (small watershed)

### Memory Usage Limits
- Peak memory usage: < 2GB (for standard workflows)
- Memory leaks: < 50MB increase over 100 operations
- GPU memory: Efficient cleanup between operations

### Accuracy Requirements
- RMSE: < 0.8 meters for flood depth predictions
- MAE: < 0.5 meters for typical scenarios
- R²: > 0.7 for model performance
- Flood detection accuracy: > 80%

## Continuous Integration

The test suite is designed for automated CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Integration Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r tests/requirements.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

1. **GDAL/Rasterio Installation**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gdal-bin libgdal-dev
   
   # macOS
   brew install gdal
   
   # Install Python bindings
   pip install rasterio --no-binary rasterio
   ```

2. **PyTorch GPU Support**
   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**
   ```bash
   # Run with reduced parallelism
   pytest tests/ -n 1
   
   # Skip memory-intensive tests
   pytest tests/ -m "not slow"
   ```

### Test Debugging

```bash
# Run with maximum verbosity
pytest tests/ -vvv --tb=long

# Run specific test with debugging
pytest tests/test_model.py::TestModelValidation::test_forward_pass -s

# Profile test execution
pytest tests/ --durations=20

# Generate detailed coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Include both positive and negative test cases
3. Add appropriate markers for test categorization
4. Ensure tests are deterministic and reproducible
5. Include comprehensive docstrings
6. Validate against production requirements
7. Test with realistic data sizes and scenarios

## License

This test suite is part of the FloodRisk project and follows the same licensing terms.