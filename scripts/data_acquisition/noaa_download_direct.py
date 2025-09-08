#!/usr/bin/env python3
"""
Direct NOAA Atlas 14 download using known direct URLs.
This script attempts to download files from NOAA's direct data repository.
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectNOAADownloader:
    """Download NOAA Atlas 14 files using direct URLs."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize downloader."""
        self.output_dir = output_dir or Path("data/v2_additional/precipitation_grids")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Direct download URLs for NOAA Atlas 14 Volume 2 (Southeast)
        # These are the actual file locations on NOAA servers
        self.direct_urls = {
            # ASCII grid format files
            "se_24hr_100yr": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr_24hr_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds2/orb/se/se_100yr24h_asc.zip",
            ],
            "se_24hr_500yr": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr_24hr_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_500yr24h_asc.zip",
            ],
            "se_24hr_25yr": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr_24hr_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_25yr24h_asc.zip",
            ],
            "se_24hr_10yr": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr_24hr_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_10yr24h_asc.zip",
            ],
            "se_24hr_2yr": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr24h_asc.zip",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr_24hr_asc.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_2yr24h_asc.zip",
            ],
            # GeoTIFF format alternatives
            "se_24hr_100yr_tif": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h.zip",
                "https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h.zip",
            ],
        }
        
        # Try NetCDF endpoints
        self.netcdf_urls = {
            "atlas14_se": [
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/atlas14_se.nc",
                "https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_atlas14_v2.nc",
            ]
        }
        
    def download_with_retry(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """
        Download a file with retry logic.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting download: {url}")
                logger.info(f"  Attempt {attempt + 1}/{max_retries}")
                
                # First check if URL is accessible
                response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                logger.info(f"  HEAD response: {response.status_code}")
                
                if response.status_code == 200:
                    # Try to download
                    response = requests.get(url, headers=headers, stream=True, timeout=30)
                    
                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        logger.info(f"  File size: {total_size / 1024 / 1024:.2f} MB")
                        
                        # Write file
                        with open(output_path, 'wb') as f:
                            downloaded = 0
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                                            logger.info(f"  Progress: {progress:.1f}%")
                        
                        logger.info(f"✓ Downloaded successfully: {output_path}")
                        return True
                    else:
                        logger.warning(f"  HTTP {response.status_code} for GET request")
                else:
                    logger.warning(f"  URL not accessible (HTTP {response.status_code})")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"  Timeout error")
            except requests.exceptions.ConnectionError:
                logger.warning(f"  Connection error")
            except Exception as e:
                logger.warning(f"  Error: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        return False
    
    def extract_zip(self, zip_path: Path) -> bool:
        """Extract a zip file."""
        try:
            extract_dir = self.output_dir / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted to: {extract_dir}")
            
            # List extracted files
            extracted_files = list(extract_dir.glob("*"))
            logger.info(f"Extracted files: {[f.name for f in extracted_files[:5]]}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    def run_download(self):
        """Run the download process."""
        logger.info("=" * 60)
        logger.info("Direct NOAA Download Attempt")
        logger.info("=" * 60)
        
        successful = []
        failed = []
        
        # Try direct URLs
        for file_id, urls in self.direct_urls.items():
            output_file = self.output_dir / f"{file_id}.zip"
            
            if output_file.exists():
                logger.info(f"Already exists: {output_file}")
                successful.append(file_id)
                continue
            
            downloaded = False
            for url in urls:
                if self.download_with_retry(url, output_file, max_retries=2):
                    # Extract if successful
                    if self.extract_zip(output_file):
                        successful.append(file_id)
                    downloaded = True
                    break
            
            if not downloaded:
                failed.append(file_id)
        
        # Try NetCDF files
        for file_id, urls in self.netcdf_urls.items():
            output_file = self.output_dir / f"{file_id}.nc"
            
            if output_file.exists():
                logger.info(f"Already exists: {output_file}")
                continue
            
            for url in urls:
                if self.download_with_retry(url, output_file, max_retries=2):
                    successful.append(file_id)
                    break
        
        # Summary
        logger.info("=" * 60)
        logger.info("Download Summary:")
        logger.info(f"Successful: {len(successful)}")
        for s in successful:
            logger.info(f"  ✓ {s}")
        logger.info(f"Failed: {len(failed)}")
        for f in failed:
            logger.info(f"  ✗ {f}")
        logger.info("=" * 60)
        
        # If all failed, provide alternative instructions
        if not successful:
            self.create_wget_script()
            self.check_alternative_sources()
        
        return successful, failed
    
    def create_wget_script(self):
        """Create a wget script for manual download attempts."""
        script_path = self.output_dir / "download_with_wget.sh"
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# NOAA Atlas 14 Download Script\n")
            f.write("# Run this script to attempt downloads using wget\n\n")
            
            f.write("mkdir -p grids\n")
            f.write("cd grids\n\n")
            
            # Add wget commands for each URL
            for file_id, urls in self.direct_urls.items():
                f.write(f"# {file_id}\n")
                for url in urls:
                    f.write(f"wget -nc -t 3 --user-agent='Mozilla/5.0' '{url}' || echo 'Failed: {url}'\n")
                f.write("\n")
            
            f.write("echo 'Download attempts complete'\n")
            f.write("ls -la *.zip 2>/dev/null || echo 'No files downloaded'\n")
        
        script_path.chmod(0o755)
        logger.info(f"Created wget script: {script_path}")
        logger.info("Run with: bash " + str(script_path))
    
    def check_alternative_sources(self):
        """Check and report alternative data sources."""
        logger.info("\n" + "=" * 60)
        logger.info("Alternative Data Sources:")
        logger.info("=" * 60)
        
        alternatives = {
            "USGS ScienceBase": {
                "url": "https://www.sciencebase.gov/catalog/",
                "search": "NOAA Atlas 14 precipitation frequency Tennessee",
                "note": "May have pre-processed regional datasets"
            },
            "NOAA Climate Data Online": {
                "url": "https://www.ncdc.noaa.gov/cdo-web/datasets",
                "search": "Precipitation frequency",
                "note": "Historical precipitation data"
            },
            "NOAA Big Data Program": {
                "url": "https://www.noaa.gov/organization/information-technology/big-data-program",
                "search": "Atlas 14",
                "note": "Cloud-hosted datasets (AWS, Google Cloud)"
            },
            "HydroShare": {
                "url": "https://www.hydroshare.org/",
                "search": "NOAA Atlas 14 Tennessee",
                "note": "Community hydrologic data repository"
            },
            "Earth Explorer (USGS)": {
                "url": "https://earthexplorer.usgs.gov/",
                "search": "Precipitation",
                "note": "Satellite-based precipitation products"
            }
        }
        
        for source, info in alternatives.items():
            logger.info(f"\n{source}:")
            logger.info(f"  URL: {info['url']}")
            logger.info(f"  Search: {info['search']}")
            logger.info(f"  Note: {info['note']}")
        
        logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    downloader = DirectNOAADownloader()
    downloader.run_download()