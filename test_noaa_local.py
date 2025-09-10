#!/usr/bin/env python3
"""Test the local NOAA Atlas 14 CSV loader."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.sources.noaa_atlas14_local import NOAAAtlas14LocalLoader

def test_local_loader():
    """Test loading local NOAA Atlas 14 CSV data."""
    
    csv_path = Path("/Users/nateaune/Documents/code/FloodRisk/data/regions/nashville/rainfall/All_Depth_English_PDS.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"✅ Found CSV file: {csv_path}")
    print("=" * 60)
    
    # Load the data
    loader = NOAAAtlas14LocalLoader(csv_path)
    
    # Display metadata
    print("\n📍 Location Information:")
    metadata = loader.data['metadata']
    print(f"  Station: {metadata.get('station', 'Unknown')}")
    print(f"  Latitude: {metadata.get('latitude', 'N/A')}°")
    print(f"  Longitude: {metadata.get('longitude', 'N/A')}°")
    print(f"  Elevation: {metadata.get('elevation_ft', 'N/A')} ft")
    
    # Get 24-hour precipitation for key return periods
    print("\n🌧️  24-Hour Precipitation Depths:")
    print("-" * 40)
    
    return_periods = [10, 25, 100, 500]
    for rp in return_periods:
        try:
            depth_inches = loader.get_precipitation_depth('24-hr', rp)
            depth_mm = depth_inches * 25.4
            print(f"  {rp:4d}-year: {depth_inches:6.2f} inches = {depth_mm:7.1f} mm")
        except ValueError as e:
            print(f"  {rp:4d}-year: Error - {e}")
    
    # Test creating rainfall configuration
    print("\n⚙️  Creating Rainfall Configuration...")
    config = loader.create_rainfall_config()
    
    print(f"  ✓ Configuration created with {len(config['durations'])} durations")
    print(f"  ✓ Return periods: {config['return_periods']}")
    print(f"  ✓ Source: {config['source']}")
    
    # Display precipitation table
    print("\n📊 Full Precipitation Table (first 5 durations):")
    print("-" * 60)
    df = loader.get_precipitation_table()
    print(df.head())
    
    print("\n✅ Local NOAA data loader test completed successfully!")

if __name__ == "__main__":
    test_local_loader()