#!/usr/bin/env python3
"""
Download NOAA Atlas 14 data from AWS S3 Big Data Program.
NOAA hosts datasets on AWS S3 as part of their Big Data Program.
"""

import os
import sys
import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
from pathlib import Path
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NOAABigDataDownloader:
    """Download NOAA data from AWS S3 Big Data Program."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize downloader."""
        self.output_dir = output_dir or Path("data/v2_additional/precipitation_grids")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NOAA Big Data buckets
        self.buckets = {
            'noaa-atlas14': 's3://noaa-atlas14/',
            'noaa-nws-ofs-pds': 's3://noaa-nws-ofs-pds/',
            'noaa-nexrad-level2': 's3://noaa-nexrad-level2/',
            'noaa-gfs-bdp-pds': 's3://noaa-gfs-bdp-pds/',
        }
        
        # Create S3 client without credentials (public buckets)
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED),
            region_name='us-east-1'
        )
        
    def list_bucket_contents(self, bucket_name: str, prefix: str = '') -> List[str]:
        """List contents of an S3 bucket."""
        contents = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': 100}
            )
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        contents.append(obj['Key'])
                        
        except Exception as e:
            logger.error(f"Error listing bucket {bucket_name}: {e}")
            
        return contents
    
    def download_from_s3(self, bucket: str, key: str, output_path: Path) -> bool:
        """Download a file from S3."""
        try:
            logger.info(f"Downloading s3://{bucket}/{key}")
            self.s3_client.download_file(bucket, key, str(output_path))
            logger.info(f"Downloaded to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def search_noaa_registry(self):
        """Search NOAA data registry for Atlas 14 datasets."""
        logger.info("Searching NOAA data registry...")
        
        # NOAA data catalog API
        registry_urls = [
            "https://data.noaa.gov/dataset/dataset.json",
            "https://catalog.data.gov/api/3/action/package_search?q=atlas+14+precipitation",
            "https://www.ncei.noaa.gov/data/",
        ]
        
        datasets = []
        
        for url in registry_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Found registry data from {url}")
                    
                    # Parse based on format
                    if 'result' in data:  # data.gov format
                        results = data.get('result', {}).get('results', [])
                        for result in results[:5]:
                            if 'atlas' in result.get('title', '').lower():
                                datasets.append({
                                    'title': result.get('title'),
                                    'url': result.get('url'),
                                    'resources': result.get('resources', [])
                                })
                                
            except Exception as e:
                logger.warning(f"Registry search failed for {url}: {e}")
                
        return datasets
    
    def try_aws_download(self):
        """Try to download from AWS S3 buckets."""
        logger.info("Attempting AWS S3 download...")
        
        # Check known NOAA buckets
        test_buckets = [
            ('noaa-atlas14', ''),
            ('noaa-nws-ofs-pds', 'atlas14/'),
            ('noaa-gfs-bdp-pds', 'precipitation/'),
        ]
        
        found_files = []
        
        for bucket, prefix in test_buckets:
            logger.info(f"Checking bucket: {bucket} with prefix: {prefix}")
            
            try:
                # Try to list bucket contents
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    MaxKeys=10
                )
                
                if 'Contents' in response:
                    logger.info(f"Found {len(response['Contents'])} objects in {bucket}")
                    for obj in response['Contents']:
                        key = obj['Key']
                        size = obj['Size'] / (1024 * 1024)  # MB
                        logger.info(f"  - {key} ({size:.2f} MB)")
                        found_files.append((bucket, key))
                else:
                    logger.info(f"No objects found in {bucket}/{prefix}")
                    
            except self.s3_client.exceptions.NoSuchBucket:
                logger.warning(f"Bucket {bucket} does not exist")
            except Exception as e:
                logger.warning(f"Error accessing {bucket}: {e}")
                
        return found_files
    
    def download_alternative_sources(self):
        """Try alternative precipitation data sources."""
        logger.info("Trying alternative data sources...")
        
        alternatives = {
            # PRISM Climate Data (alternative to NOAA Atlas 14)
            "PRISM 30-year normals": {
                "url": "https://prism.oregonstate.edu/normals/",
                "description": "30-year precipitation normals at 4km resolution"
            },
            
            # NASA GPM (Global Precipitation Measurement)
            "NASA GPM IMERG": {
                "url": "https://gpm.nasa.gov/data/imerg",
                "description": "30-minute precipitation estimates"
            },
            
            # NOAA CPC (Climate Prediction Center)
            "CPC Unified Precipitation": {
                "url": "https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html",
                "description": "Daily gridded precipitation for CONUS"
            },
            
            # Stage IV Radar Data
            "Stage IV Precipitation": {
                "url": "https://data.eol.ucar.edu/dataset/21.006",
                "description": "Hourly multi-sensor precipitation analysis"
            }
        }
        
        # Try to download sample data from alternatives
        sample_urls = [
            # CPC Unified daily precipitation
            "https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/RT/precip.V1.0.2020.nc",
            # PRISM sample
            "https://services.nacse.org/prism/data/public/4km/ppt/20200101",
        ]
        
        downloaded = []
        
        for url in sample_urls:
            try:
                filename = url.split('/')[-1] or 'sample_data'
                output_path = self.output_dir / f"alternative_{filename}"
                
                logger.info(f"Trying: {url}")
                response = requests.head(url, timeout=5)
                
                if response.status_code == 200:
                    size = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                    logger.info(f"  Available: {size:.2f} MB")
                    
                    # Download if small enough
                    if size < 50:  # Less than 50MB
                        response = requests.get(url, timeout=30)
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"  Downloaded to: {output_path}")
                            downloaded.append(output_path)
                            
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                
        # Report alternatives
        logger.info("\n" + "=" * 60)
        logger.info("Alternative Data Sources Available:")
        logger.info("=" * 60)
        
        for name, info in alternatives.items():
            logger.info(f"\n{name}:")
            logger.info(f"  URL: {info['url']}")
            logger.info(f"  Description: {info['description']}")
            
        return downloaded
    
    def create_download_instructions(self):
        """Create comprehensive download instructions."""
        instructions_path = self.output_dir / "AWS_DOWNLOAD_INSTRUCTIONS.md"
        
        with open(instructions_path, 'w') as f:
            f.write("""# NOAA Atlas 14 Data Download Options

## Option 1: AWS CLI (Recommended)
If you have AWS CLI installed, you can try:

```bash
# Install AWS CLI if needed
pip install awscli

# Configure for public access
aws configure set aws_access_key_id ""
aws configure set aws_secret_access_key ""

# Try to list NOAA buckets
aws s3 ls s3://noaa-atlas14/ --no-sign-request
aws s3 ls s3://noaa-nws-ofs-pds/atlas14/ --no-sign-request

# Download specific files (if found)
aws s3 cp s3://noaa-atlas14/se/100yr_24hr.tif . --no-sign-request
```

## Option 2: Direct HTTP Download
Some NOAA data is available via HTTP:

```bash
# NOAA NCEI Archive
wget https://www.ncei.noaa.gov/data/precipitation-frequency/access/

# NOAA PSL (Physical Sciences Laboratory)
wget https://downloads.psl.noaa.gov/Datasets/
```

## Option 3: THREDDS Data Server
NOAA operates THREDDS servers for data access:

- https://www.ncei.noaa.gov/thredds/catalog.html
- https://psl.noaa.gov/thredds/catalog.html

## Option 4: Google Earth Engine
NOAA Atlas 14 may be available on Google Earth Engine:

```python
import ee
ee.Initialize()

# Search for NOAA Atlas 14
noaa = ee.ImageCollection('NOAA/ATLAS14')
```

## Option 5: Manual Web Download
As a last resort, use the web interface:

1. Go to: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html
2. Select "Southeastern States (NOAA Atlas 14 Volume 2)"
3. Look for "GIS Data" or "Download" sections
4. Download files for Tennessee region

## Alternative Precipitation Datasets

### PRISM Climate Data
- URL: https://prism.oregonstate.edu/
- Resolution: 4km
- Coverage: CONUS
- Format: BIL, ASCII Grid

### NASA GPM IMERG
- URL: https://gpm.nasa.gov/data/imerg
- Resolution: 0.1 degree
- Coverage: Global
- Format: HDF5, NetCDF

### Stage IV Radar
- URL: https://data.eol.ucar.edu/dataset/21.006
- Resolution: 4km
- Coverage: CONUS
- Format: GRIB2

### CPC Unified Precipitation
- URL: https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html
- Resolution: 0.25 degree
- Coverage: CONUS
- Format: NetCDF

## Contact for Data Access
- NOAA HDSC: hdsc.questions@noaa.gov
- NOAA Big Data Program: noaa.bigdata@noaa.gov
""")
        
        logger.info(f"Instructions saved to: {instructions_path}")
        return instructions_path
    
    def run_download(self):
        """Run the complete download process."""
        logger.info("=" * 60)
        logger.info("NOAA Big Data Download Attempt")
        logger.info("=" * 60)
        
        # Search NOAA registry
        datasets = self.search_noaa_registry()
        if datasets:
            logger.info(f"Found {len(datasets)} datasets in registry")
            for ds in datasets[:3]:
                logger.info(f"  - {ds.get('title', 'Unknown')}")
        
        # Try AWS S3
        aws_files = self.try_aws_download()
        if aws_files:
            logger.info(f"Found {len(aws_files)} files on AWS")
        
        # Try alternative sources
        alt_files = self.download_alternative_sources()
        if alt_files:
            logger.info(f"Downloaded {len(alt_files)} alternative files")
        
        # Create instructions
        self.create_download_instructions()
        
        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info("- AWS S3 access attempted")
        logger.info("- Alternative sources checked")
        logger.info("- Instructions created")
        logger.info("- Sample grid available for testing")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Install boto3 if needed
    try:
        import boto3
    except ImportError:
        import subprocess
        logger.info("Installing boto3...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
        import boto3
    
    downloader = NOAABigDataDownloader()
    downloader.run_download()