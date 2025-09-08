#!/usr/bin/env python3
"""
Download additional data required for Implementation Plan v2.
This includes spatial precipitation grids, temporal patterns, and validation data.
"""

import os
import sys
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class V2DataDownloader:
    """Download additional data for v2 implementation plan."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize downloader with output directory."""
        self.output_dir = output_dir or Path("data/v2_additional")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nashville bounding box
        self.nashville_bbox = {
            'north': 36.3,
            'south': 35.9,
            'east': -86.5,
            'west': -87.1
        }
        
    def download_spatial_precipitation_grids(self):
        """Download NOAA spatial precipitation grids."""
        logger.info("Downloading spatial precipitation grids...")
        
        # Create output directory
        grid_dir = self.output_dir / "precipitation_grids"
        grid_dir.mkdir(exist_ok=True)
        
        # Key return periods and durations for v2
        return_periods = [2, 10, 25, 100, 500]
        durations = ['5min', '1hr', '6hr', '24hr', '48hr']
        
        # Note: Actual NOAA grid URLs would need to be obtained from:
        # https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html
        
        instructions_file = grid_dir / "DOWNLOAD_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write("""# NOAA Spatial Precipitation Grid Download Instructions

## Manual Download Required
The NOAA precipitation grids need to be downloaded manually from:
https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

## Steps:
1. Navigate to the PFDS GIS Data page
2. Select "Southeastern States (NOAA Atlas 14 Volume 2)"
3. Download the following GeoTIFF files for Tennessee:

### Required Files:
- se_vol2_2yr_24hr.tif (2-year 24-hour)
- se_vol2_10yr_24hr.tif (10-year 24-hour)
- se_vol2_25yr_24hr.tif (25-year 24-hour)
- se_vol2_100yr_24hr.tif (100-year 24-hour)
- se_vol2_500yr_24hr.tif (500-year 24-hour)

### Additional Durations (if available):
- 1-hour, 6-hour, 48-hour for each return period

## Bounding Box for Nashville:
- North: 36.3°
- South: 35.9°
- East: -86.5°
- West: -87.1°

## Save Location:
Place downloaded files in: data/v2_additional/precipitation_grids/
""")
        
        logger.info(f"Instructions saved to {instructions_file}")
        return instructions_file
    
    def download_scs_temporal_patterns(self):
        """Download/generate SCS Type II temporal distribution patterns."""
        logger.info("Generating SCS Type II temporal patterns...")
        
        patterns_dir = self.output_dir / "temporal_patterns"
        patterns_dir.mkdir(exist_ok=True)
        
        # SCS Type II 24-hour distribution (appropriate for Tennessee)
        # Time ratios and cumulative precipitation ratios
        scs_type_ii_data = {
            'time_ratio': [0.0, 0.083, 0.167, 0.25, 0.333, 0.375, 0.417, 0.458, 
                          0.5, 0.542, 0.583, 0.625, 0.667, 0.708, 0.75, 0.792, 
                          0.833, 0.875, 0.917, 0.958, 1.0],
            'cumulative_rainfall_ratio': [0.0, 0.011, 0.022, 0.035, 0.048, 0.057, 
                                         0.066, 0.076, 0.089, 0.115, 0.283, 0.663, 
                                         0.735, 0.772, 0.799, 0.820, 0.838, 0.854, 
                                         0.870, 0.884, 1.0]
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(scs_type_ii_data)
        
        # Calculate incremental rainfall
        df['incremental_rainfall_ratio'] = df['cumulative_rainfall_ratio'].diff().fillna(0)
        
        # Save to CSV
        output_file = patterns_dir / "scs_type_ii_24hr.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"SCS Type II pattern saved to {output_file}")
        
        # Generate nested storm patterns for other durations
        self.generate_nested_patterns(patterns_dir)
        
        return output_file
    
    def generate_nested_patterns(self, output_dir: Path):
        """Generate nested storm patterns for multiple durations."""
        logger.info("Generating nested storm patterns...")
        
        # Nested storm approach: shorter durations peak at same time as 24-hour
        durations = [1, 2, 3, 6, 12, 24]  # hours
        
        nested_data = {}
        for duration in durations:
            if duration == 24:
                # Use full SCS Type II
                time_steps = 24
                peak_time = 11.5  # Peak at ~11.5 hours for Type II
            else:
                time_steps = duration * 4  # 15-minute intervals
                # Center the storm around the 24-hour peak
                start_time = max(0, 11.5 - duration/2)
                peak_time = duration/2
            
            nested_data[f'{duration}hr'] = {
                'duration_hours': duration,
                'time_steps': time_steps,
                'peak_time': peak_time
            }
        
        # Save nested pattern configuration
        import json
        config_file = output_dir / "nested_patterns_config.json"
        with open(config_file, 'w') as f:
            json.dump(nested_data, f, indent=2)
        
        logger.info(f"Nested patterns configuration saved to {config_file}")
        
    def download_area_reduction_factors(self):
        """Download/generate area reduction factors."""
        logger.info("Generating area reduction factors...")
        
        arf_dir = self.output_dir / "area_reduction_factors"
        arf_dir.mkdir(exist_ok=True)
        
        # TP-40 based ARF values (approximate)
        # Area (sq mi) vs reduction factor for different durations
        arf_data = {
            'area_sq_mi': [0, 10, 25, 50, 100, 200, 500, 1000],
            '30min': [1.0, 0.98, 0.96, 0.93, 0.89, 0.86, 0.81, 0.77],
            '1hr': [1.0, 0.99, 0.97, 0.95, 0.92, 0.89, 0.85, 0.81],
            '3hr': [1.0, 0.99, 0.98, 0.96, 0.94, 0.92, 0.89, 0.86],
            '6hr': [1.0, 0.99, 0.98, 0.97, 0.96, 0.94, 0.91, 0.89],
            '24hr': [1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.94, 0.92],
        }
        
        df = pd.DataFrame(arf_data)
        output_file = arf_dir / "area_reduction_factors.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Area reduction factors saved to {output_file}")
        return output_file
    
    def download_2010_flood_validation_data(self):
        """Download/prepare 2010 Nashville flood validation data."""
        logger.info("Preparing 2010 flood validation data...")
        
        validation_dir = self.output_dir / "2010_flood_validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Create instructions for manual download
        instructions = validation_dir / "DOWNLOAD_INSTRUCTIONS.md"
        with open(instructions, 'w') as f:
            f.write("""# 2010 Nashville Flood Validation Data

## Required Data Sources:

### 1. Stage IV Radar Rainfall Data
- **Dates**: April 30 - May 3, 2010
- **Download from**: https://data.eol.ucar.edu/dataset/21.006
- **Format**: GRIB2 files (4km resolution)
- **Coverage**: Southeastern US including Nashville

### 2. USGS High Water Marks
- **Source**: https://water.usgs.gov/floods/events/2010/nashville/
- **Data includes**:
  - GPS coordinates of high water marks
  - Measured water surface elevations
  - Time of peak (if available)

### 3. USGS Stream Gauge Data
- **Stations**: Cumberland River and tributaries
- **Download from**: https://waterdata.usgs.gov/nwis
- **Key stations**:
  - 03431500 - Cumberland River at Nashville
  - 03430550 - Mill Creek near Nashville
  - 03431060 - Harpeth River near Kingston Springs

### 4. Observed Flood Extent
- **Source**: USGS or local emergency management
- **Format**: Shapefile or GeoTIFF of flooded areas

### 5. Rainfall Observations
- **Nashville Airport (KBNA)**: Total of 13.57 inches May 1-2
- **Download hourly data from**: https://www.ncdc.noaa.gov/

## Save Location:
Place all downloaded files in: data/v2_additional/2010_flood_validation/
""")
        
        # Create sample validation metadata
        metadata = {
            'event_name': 'May 2010 Tennessee Floods',
            'dates': ['2010-04-30', '2010-05-03'],
            'peak_date': '2010-05-02',
            'nashville_airport_total': 343.9,  # mm (13.54 inches)
            '48hr_total': 343.9,
            'estimated_return_period': '>1000 years',
            'fatalities': 31,
            'damage_usd': 2000000000,
            'key_locations': {
                'Nashville Airport': {'lat': 36.1253, 'lon': -86.6764, 'rainfall_mm': 343.9},
                'Downtown Nashville': {'lat': 36.1627, 'lon': -86.7816, 'flood_depth_m': 3.0},
                'Opryland': {'lat': 36.2051, 'lon': -86.6889, 'flood_depth_m': 2.5}
            }
        }
        
        import json
        metadata_file = validation_dir / "2010_flood_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Validation metadata saved to {metadata_file}")
        return instructions
    
    def create_download_summary(self):
        """Create a summary of all required downloads."""
        summary_file = self.output_dir / "DOWNLOAD_SUMMARY.md"
        
        with open(summary_file, 'w') as f:
            f.write("""# FloodRisk v2 Additional Data Requirements

## Data Successfully Generated:
✅ SCS Type II temporal patterns
✅ Nested storm configurations  
✅ Area reduction factors
✅ 2010 flood metadata

## Data Requiring Manual Download:

### 1. NOAA Spatial Precipitation Grids (CRITICAL)
- [ ] 30-arc-second resolution grids for Nashville region
- [ ] Multiple return periods (2, 10, 25, 100, 500-year)
- [ ] Multiple durations (1hr, 6hr, 24hr minimum)
- **Download from**: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

### 2. Stage IV Radar Data (HIGH PRIORITY)
- [ ] April 30 - May 3, 2010 coverage
- [ ] 4km resolution GRIB2 files
- **Download from**: https://data.eol.ucar.edu/dataset/21.006

### 3. USGS High Water Marks (HIGH PRIORITY)
- [ ] 2010 Nashville flood observations
- [ ] GPS coordinates and elevations
- **Download from**: https://water.usgs.gov/floods/events/2010/nashville/

### 4. Orographic Factors (MEDIUM PRIORITY)
- [ ] NOAA Atlas 14 Volume 2 appendices
- [ ] Tennessee-specific elevation adjustments
- **Source**: NOAA Atlas 14 documentation

## Next Steps:
1. Download spatial precipitation grids (most critical)
2. Obtain 2010 flood validation data
3. Review generated temporal patterns
4. Test data integration with existing pipeline

## Directory Structure:
```
data/v2_additional/
├── precipitation_grids/     # NOAA spatial grids (manual download)
├── temporal_patterns/       # SCS patterns (generated)
├── area_reduction_factors/  # ARF curves (generated)
├── 2010_flood_validation/   # Historical event data (manual download)
└── DOWNLOAD_SUMMARY.md      # This file
```
""")
        
        logger.info(f"Download summary created at {summary_file}")
        return summary_file
    
    def run_all_downloads(self):
        """Run all download/generation tasks."""
        logger.info("Starting v2 data acquisition...")
        
        # Generate what we can programmatically
        self.download_scs_temporal_patterns()
        self.download_area_reduction_factors()
        
        # Create instructions for manual downloads
        self.download_spatial_precipitation_grids()
        self.download_2010_flood_validation_data()
        
        # Create summary
        summary = self.create_download_summary()
        
        logger.info("=" * 60)
        logger.info("V2 Data Acquisition Complete!")
        logger.info(f"Review instructions at: {summary}")
        logger.info("=" * 60)
        
        return summary


if __name__ == "__main__":
    downloader = V2DataDownloader()
    downloader.run_all_downloads()