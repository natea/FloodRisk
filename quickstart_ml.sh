#!/bin/bash
# FloodRisk ML Quick Start Script
# This script automates the setup of the production ML pipeline

set -e  # Exit on error

echo "======================================"
echo "🌊 FloodRisk ML Pipeline Quick Start"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.11+ required. Found: Python $python_version"
    exit 1
fi
print_status "Python $python_version detected"

# Check if we're on the right branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "ml-training" ]; then
    print_warning "Not on ml-training branch. Switching..."
    git checkout ml-training
    git pull origin ml-training
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_ml" ]; then
    print_status "Creating ML virtual environment..."
    python3 -m venv venv_ml
else
    print_status "ML virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv_ml/bin/activate

# Install basic dependencies first
print_status "Installing core dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install numpy first (required by many packages)
pip install numpy > /dev/null 2>&1

# Try to install GDAL (may fail on some systems)
print_status "Attempting to install GDAL..."
if pip install GDAL 2>/dev/null; then
    print_status "GDAL installed successfully"
else
    print_warning "GDAL installation failed - some features may be limited"
    print_warning "For full functionality, install GDAL system libraries:"
    print_warning "  macOS: brew install gdal"
    print_warning "  Ubuntu: sudo apt-get install gdal-bin libgdal-dev"
fi

# Install core ML dependencies (without GDAL-dependent packages)
print_status "Installing ML dependencies..."
pip install torch torchvision pytorch-lightning segmentation-models-pytorch > /dev/null 2>&1
pip install pandas scikit-learn matplotlib seaborn > /dev/null 2>&1
pip install fastapi uvicorn pydantic pydantic-settings > /dev/null 2>&1
pip install hydra-core wandb tqdm rich > /dev/null 2>&1

# Try to install geospatial packages
print_status "Attempting geospatial packages..."
pip install shapely pyproj 2>/dev/null || print_warning "Some geospatial packages failed"

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data/nashville/{dem,rainfall,labels,processed}
mkdir -p outputs/baseline/{checkpoints,logs}
mkdir -p results/nashville_demo

# Create a minimal test DEM if no real data available
if [ ! -f "data/nashville/test_dem.tif" ]; then
    print_status "Creating test DEM for demonstration..."
    python3 << 'EOF'
import numpy as np
import pickle

# Create a simple synthetic DEM
dem = np.random.randn(1024, 1024) * 10 + 100
dem = np.maximum(dem, 0)  # Ensure positive elevations

# Save as numpy array (since GDAL might not be available)
np.save('data/nashville/test_dem.npy', dem)
print("Test DEM created: data/nashville/test_dem.npy")

# Also create a simple config
config = {
    'dem_file': 'data/nashville/test_dem.npy',
    'rainfall_100yr': 150.0,
    'rainfall_500yr': 200.0,
    'region': 'nashville_test'
}

with open('data/nashville/test_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("Test configuration created: data/nashville/test_config.pkl")
EOF
fi

# Create a simple training script that works without GDAL
print_status "Creating simplified training script..."
cat > train_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simplified training script that works without GDAL dependencies.
"""

import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(n_samples=100, tile_size=512, n_channels=2):
    """Create dummy training data for testing."""
    data = []
    for i in range(n_samples):
        # Random input (DEM + rainfall)
        x = torch.randn(n_channels, tile_size, tile_size)
        # Random binary labels
        y = torch.randint(0, 2, (1, tile_size, tile_size)).float()
        data.append((x, y))
    return data

def train_simple_model():
    """Train a simple model with dummy data."""
    logger.info("Starting simplified training (no GDAL required)...")
    
    # Create dummy data
    logger.info("Creating dummy training data...")
    train_data = create_dummy_data(100)
    val_data = create_dummy_data(20)
    
    # Create simple UNet model
    logger.info("Creating model...")
    try:
        from src.ml.models.flood_unet import FloodUNet
        model = FloodUNet(in_channels=2, num_classes=1)
    except ImportError:
        logger.warning("Could not import FloodUNet, using simple Conv2d")
        model = torch.nn.Conv2d(2, 1, kernel_size=3, padding=1)
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    logger.info("Training for 10 epochs...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in train_data[:10]:  # Train on subset for speed
            optimizer.zero_grad()
            pred = model(x.unsqueeze(0))
            loss = criterion(pred, y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        logger.info(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "simple_model.pth")
    logger.info(f"Model saved to {output_dir / 'simple_model.pth'}")
    
    logger.info("Training complete!")
    logger.info("\nNext steps:")
    logger.info("1. Install GDAL for real data processing")
    logger.info("2. Download real DEM data with scripts/data_acquisition/download_nashville_data.py")
    logger.info("3. Run full training with scripts/train_model.py")

if __name__ == "__main__":
    train_simple_model()
EOF

chmod +x train_simple.py

# Run a quick test
print_status "Running quick training test..."
if python train_simple.py; then
    print_status "Test training completed successfully!"
else
    print_warning "Test training failed - check error messages above"
fi

echo ""
echo "======================================"
echo "🎉 Quick Start Complete!"
echo "======================================"
echo ""
echo "✅ What's been set up:"
echo "  - ML virtual environment (venv_ml)"
echo "  - Core ML dependencies"
echo "  - Directory structure"
echo "  - Test data for demonstration"
echo "  - Simple training script"
echo ""

if ! pip show GDAL > /dev/null 2>&1; then
    echo "⚠️  GDAL not installed - Limited functionality"
    echo ""
    echo "To enable full features, install GDAL:"
    echo "  macOS:  brew install gdal && pip install GDAL"
    echo "  Ubuntu: sudo apt-get install gdal-bin libgdal-dev && pip install GDAL"
    echo ""
fi

echo "📚 Next steps:"
echo ""
echo "1. For quick demo (no GDAL required):"
echo "   python train_simple.py"
echo ""
echo "2. For production setup (requires GDAL):"
echo "   # Install system dependencies first"
echo "   # Then run:"
echo "   python scripts/data_acquisition/download_nashville_data.py"
echo "   python scripts/train_model.py --phase baseline"
echo ""
echo "3. Check the documentation:"
echo "   cat SETUP_PRODUCTION_ML.md"
echo ""
echo "Environment activated. You're ready to go!"
echo "To deactivate later: deactivate"