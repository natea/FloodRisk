#!/usr/bin/env python3
"""
Download NOAA Atlas 14 spatial precipitation grids for Nashville region.
This script automates the download of gridded precipitation data.
"""

import os
import sys
import logging
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NOAAGridDownloader:
    """Download NOAA Atlas 14 precipitation grids."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize downloader."""
        self.output_dir = output_dir or Path("data/v2_additional/precipitation_grids")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nashville bounding box
        self.nashville_bbox = {
            'north': 36.3,
            'south': 35.9,
            'east': -86.5,
            'west': -87.1
        }
        
        # Base URLs for NOAA data
        self.base_urls = {
            'atlas14': 'https://hdsc.nws.noaa.gov/hdsc/pfds/',
            'gis': 'https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html',
            'direct': 'https://hdsc.nws.noaa.gov/pub/hdsc/data/'
        }
        
    def get_grid_urls(self) -> Dict[str, str]:
        """
        Get direct download URLs for NOAA Atlas 14 Volume 2 (Southeast) grids.
        
        These are the actual grid files for Tennessee region.
        """
        # NOAA Atlas 14 Volume 2 covers Tennessee
        # These URLs are for the GIS data files
        
        grid_urls = {}
        
        # The actual download pattern for NOAA Atlas 14 Volume 2 GIS data
        # Note: These are example patterns - actual URLs may vary
        base_gis = "https://hdsc.nws.noaa.gov/pub/hdsc/data/gis/"
        
        # Return periods and durations we need
        return_periods = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr', '500yr', '1000yr']
        durations = ['5min', '10min', '15min', '30min', '60min', '2hr', '3hr', 
                    '6hr', '12hr', '24hr', '48hr', '72hr', '4day', '7day', '10day', 
                    '20day', '30day', '45day', '60day']
        
        # Key files for v2 implementation (focusing on most important)
        priority_configs = [
            ('2yr', '24hr'),
            ('10yr', '24hr'),
            ('25yr', '24hr'),
            ('100yr', '24hr'),
            ('500yr', '24hr'),
            ('100yr', '6hr'),
            ('100yr', '1hr'),
        ]
        
        # Build URLs for priority configurations
        for rp, dur in priority_configs:
            # Pattern: se_vol2_[return_period]_[duration].zip
            filename = f"se_vol2_{rp}_{dur}"
            # These would be the actual file patterns
            grid_urls[f"{rp}_{dur}"] = {
                'filename': filename,
                'possible_urls': [
                    f"{base_gis}se/{filename}.zip",
                    f"{base_gis}se_vol2/{filename}.zip",
                    f"{base_gis}vol2/{filename}.zip",
                ]
            }
        
        return grid_urls
    
    def download_file(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """
        Download a file with retry logic.
        
        Args:
            url: URL to download
            output_path: Where to save the file
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Write file in chunks
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"Successfully downloaded to {output_path}")
                    return True
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract a zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Extracted {zip_path} to {extract_to}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    def download_sample_grids(self):
        """
        Download sample grid data or create instructions for manual download.
        
        Since NOAA Atlas 14 grids require navigation through their web interface,
        this provides clear instructions and attempts to download if URLs are accessible.
        """
        logger.info("Attempting to download NOAA precipitation grids...")
        
        # Try to download using known patterns
        grid_urls = self.get_grid_urls()
        
        successful_downloads = []
        failed_downloads = []
        
        for grid_name, grid_info in grid_urls.items():
            output_file = self.output_dir / f"{grid_info['filename']}.zip"
            
            if output_file.exists():
                logger.info(f"Already exists: {output_file}")
                successful_downloads.append(grid_name)
                continue
            
            # Try each possible URL
            downloaded = False
            for url in grid_info['possible_urls']:
                if self.download_file(url, output_file):
                    # Extract if it's a zip file
                    if output_file.suffix == '.zip':
                        extract_dir = self.output_dir / grid_info['filename']
                        if self.extract_zip(output_file, extract_dir):
                            successful_downloads.append(grid_name)
                            downloaded = True
                            break
                    else:
                        successful_downloads.append(grid_name)
                        downloaded = True
                        break
            
            if not downloaded:
                failed_downloads.append(grid_name)
                logger.warning(f"Could not download {grid_name}")
        
        # Create detailed instructions for manual download
        self.create_manual_instructions(failed_downloads)
        
        return successful_downloads, failed_downloads
    
    def create_manual_instructions(self, failed_downloads: List[str]):
        """Create detailed manual download instructions."""
        
        instructions_path = self.output_dir / "MANUAL_DOWNLOAD_INSTRUCTIONS.md"
        
        with open(instructions_path, 'w') as f:
            f.write("""# NOAA Atlas 14 Spatial Grid Manual Download Instructions

## Important Note
The NOAA precipitation grids are large files that may require manual download
through the NOAA HDSC web interface due to access restrictions.

## Step-by-Step Download Process:

### 1. Navigate to NOAA PFDS GIS Page
Go to: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

### 2. Select Region
- Click on "Southeastern States (NOAA Atlas 14 Volume 2 Version 3)"
- This covers Tennessee including Nashville

### 3. Download Options

#### Option A: Download Complete Dataset (Recommended)
1. Look for "Download all GIS files" or similar option
2. This will give you a large ZIP file with all return periods and durations
3. File size: ~500MB - 1GB

#### Option B: Download Individual Files
Download these specific files for Nashville flood modeling:

**Critical Files (24-hour duration):**
- `se_2yr_24hr_asc.zip` - 2-year 24-hour
- `se_10yr_24hr_asc.zip` - 10-year 24-hour
- `se_25yr_24hr_asc.zip` - 25-year 24-hour
- `se_100yr_24hr_asc.zip` - 100-year 24-hour
- `se_500yr_24hr_asc.zip` - 500-year 24-hour

**Additional Important Files:**
- `se_100yr_6hr_asc.zip` - 100-year 6-hour
- `se_100yr_1hr_asc.zip` - 100-year 1-hour
- `se_100yr_48hr_asc.zip` - 100-year 48-hour

### 4. File Format Options
- **ASCII Grid (.asc)**: Easiest to work with, human-readable
- **GeoTIFF (.tif)**: More compact, maintains projection info
- **NetCDF (.nc)**: Good for time series data

### 5. After Download
1. Extract ZIP files to: `data/v2_additional/precipitation_grids/`
2. Each ZIP contains:
   - `.asc` or `.tif` file with precipitation depths
   - `.prj` file with projection information
   - `.txt` metadata file

### 6. Alternative Data Sources

If the NOAA site is unavailable, try:

#### USGS Data Portal
https://www.sciencebase.gov/catalog/

Search for: "NOAA Atlas 14 precipitation frequency"

#### NOAA Climate Data Online
https://www.ncdc.noaa.gov/cdo-web/

#### Direct FTP (if available)
ftp://hdsc.nws.noaa.gov/pub/hdsc/data/

## Nashville Specific Information

**Bounding Box for Data Extraction:**
- North: 36.3°N
- South: 35.9°N  
- East: -86.5°W
- West: -87.1°W

**Projection Information:**
- Geographic Coordinate System: GCS_North_American_1983
- Datum: D_North_American_1983
- Prime Meridian: Greenwich
- Angular Unit: Degree

## Verification

After downloading, verify files contain:
1. Header with grid dimensions and cell size
2. NODATA value (usually -999 or -9999)
3. Precipitation values in inches (need to convert to mm)

## Failed Automatic Downloads:
""")
            
            if failed_downloads:
                f.write("\nThe following files could not be downloaded automatically:\n")
                for item in failed_downloads:
                    f.write(f"- {item}\n")
            else:
                f.write("\nAll priority files were downloaded successfully!\n")
            
            f.write("""
## Contact Information

If you have issues accessing the data:
- NOAA HDSC Help: hdsc.questions@noaa.gov
- Phone: 301-713-1677 x127

## Data Citation

When using this data, cite as:
"NOAA Atlas 14, Volume 2, Version 3: Precipitation-Frequency Atlas of the United States"
Available at: https://hdsc.nws.noaa.gov/hdsc/pfds/
""")
        
        logger.info(f"Manual download instructions saved to: {instructions_path}")
        return instructions_path
    
    def create_sample_grid(self):
        """
        Create a sample grid file to test the pipeline while waiting for real data.
        """
        import numpy as np
        
        logger.info("Creating sample grid for testing...")
        
        # Create a sample ASCII grid file
        sample_file = self.output_dir / "sample_100yr_24hr.asc"
        
        # Grid parameters for Nashville area
        ncols = 100
        nrows = 80
        xllcorner = -87.1
        yllcorner = 35.9
        cellsize = 0.006  # ~600m
        nodata_value = -9999
        
        # Create sample precipitation data (inches)
        # Center will have higher precipitation
        center_x, center_y = ncols // 2, nrows // 2
        grid_data = np.zeros((nrows, ncols))
        
        for i in range(nrows):
            for j in range(ncols):
                # Distance from center
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                # Precipitation decreases with distance (6-8 inches for 100-year)
                base_precip = 7.0  # inches
                grid_data[i, j] = base_precip * np.exp(-dist / 30) + np.random.normal(0, 0.1)
                grid_data[i, j] = max(5.0, min(9.0, grid_data[i, j]))  # Clip to reasonable range
        
        # Write ASCII grid file
        with open(sample_file, 'w') as f:
            f.write(f"ncols {ncols}\n")
            f.write(f"nrows {nrows}\n")
            f.write(f"xllcorner {xllcorner}\n")
            f.write(f"yllcorner {yllcorner}\n")
            f.write(f"cellsize {cellsize}\n")
            f.write(f"NODATA_value {nodata_value}\n")
            
            # Write grid data
            for i in range(nrows):
                row_data = ' '.join([f"{val:.3f}" for val in grid_data[i, :]])
                f.write(row_data + '\n')
        
        logger.info(f"Sample grid created: {sample_file}")
        
        # Create projection file
        prj_file = self.output_dir / "sample_100yr_24hr.prj"
        with open(prj_file, 'w') as f:
            f.write('GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",')
            f.write('SPHEROID["GRS_1980",6378137,298.257222101]],')
            f.write('PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        
        return sample_file
    
    def run_download(self):
        """Run the complete download process."""
        logger.info("=" * 60)
        logger.info("NOAA Spatial Precipitation Grid Downloader")
        logger.info("=" * 60)
        
        # Try automatic download
        successful, failed = self.download_sample_grids()
        
        # Create sample grid for testing
        sample_grid = self.create_sample_grid()
        
        # Summary
        logger.info("=" * 60)
        logger.info("Download Summary:")
        logger.info(f"Successfully downloaded: {len(successful)} files")
        logger.info(f"Failed/Manual required: {len(failed)} files")
        logger.info(f"Sample grid created: {sample_grid}")
        logger.info("=" * 60)
        
        if failed:
            logger.info("⚠️  Some files require manual download.")
            logger.info(f"See instructions: {self.output_dir}/MANUAL_DOWNLOAD_INSTRUCTIONS.md")
        
        logger.info("\nNext steps:")
        logger.info("1. If automatic download failed, follow manual instructions")
        logger.info("2. Use sample grid for testing: sample_100yr_24hr.asc")
        logger.info("3. Process grids with the data pipeline")
        
        return successful, failed


if __name__ == "__main__":
    downloader = NOAAGridDownloader()
    downloader.run_download()