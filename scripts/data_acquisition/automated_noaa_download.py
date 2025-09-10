#!/usr/bin/env python3
"""
Automated NOAA Atlas 14 spatial precipitation grid downloader using Playwright.
This script automates the web interface navigation to download gridded precipitation data.
"""

import os
import sys
import logging
import asyncio
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from playwright.async_api import async_playwright, Page, Download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedNOAADownloader:
    """Automate NOAA Atlas 14 precipitation grid downloads using browser automation."""
    
    def __init__(self, output_dir: Path = None, headless: bool = False):
        """
        Initialize the automated downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            headless: Run browser in headless mode (False for debugging)
        """
        self.output_dir = output_dir or Path("data/v2_additional/precipitation_grids")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        
        # NOAA HDSC URLs
        self.base_url = "https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html"
        self.southeast_url = "https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_map_se.html"
        
        # Target files to download (Tennessee/Nashville region)
        self.target_files = [
            # Priority files for Nashville flood modeling
            "se_2yr_24hr",
            "se_10yr_24hr", 
            "se_25yr_24hr",
            "se_100yr_24hr",
            "se_500yr_24hr",
            # Additional durations
            "se_100yr_1hr",
            "se_100yr_6hr",
            "se_100yr_48hr",
        ]
        
    async def navigate_to_downloads(self, page: Page) -> bool:
        """
        Navigate to the NOAA HDSC download page for Southeast region.
        
        Args:
            page: Playwright page object
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            logger.info(f"Navigating to NOAA HDSC main page: {self.base_url}")
            await page.goto(self.base_url, wait_until="networkidle")
            
            # Look for Southeast region link
            southeast_link = page.locator('a:has-text("Southeastern")')
            if await southeast_link.count() > 0:
                logger.info("Found Southeastern States link, clicking...")
                await southeast_link.first.click()
                await page.wait_for_load_state("networkidle")
                
                # Look for GIS data section
                gis_section = page.locator('text=/GIS data/i')
                if await gis_section.count() > 0:
                    logger.info("Found GIS data section")
                    return True
                else:
                    # Try direct navigation to Southeast page
                    logger.info("GIS section not found, trying direct URL...")
                    await page.goto(self.southeast_url, wait_until="networkidle")
                    return True
            else:
                # Try alternative navigation
                logger.info("Southeast link not found, trying alternative navigation...")
                
                # Look for Volume 2 (covers Tennessee)
                vol2_link = page.locator('a:has-text("Volume 2")')
                if await vol2_link.count() > 0:
                    await vol2_link.first.click()
                    await page.wait_for_load_state("networkidle")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False
    
    async def find_download_links(self, page: Page) -> Dict[str, str]:
        """
        Find download links for target precipitation grid files.
        
        Args:
            page: Playwright page object
            
        Returns:
            Dictionary mapping file names to download URLs
        """
        download_links = {}
        
        try:
            # Common patterns for NOAA download links
            patterns = [
                # ASCII grid files
                'a[href*=".zip"][href*="asc"]',
                'a[href*=".zip"][href*="24hr"]',
                'a[href*=".zip"][href*="1hr"]',
                'a[href*=".zip"][href*="6hr"]',
                'a[href*=".zip"][href*="48hr"]',
                # General zip files
                'a[href$=".zip"]',
                # Download buttons/links
                'a:has-text("Download")',
                'button:has-text("Download")',
            ]
            
            for pattern in patterns:
                links = await page.locator(pattern).all()
                for link in links:
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    
                    if href:
                        # Check if this is one of our target files
                        for target in self.target_files:
                            if target in href.lower() or target in (text or "").lower():
                                # Make absolute URL if relative
                                if not href.startswith("http"):
                                    base = page.url.rsplit("/", 1)[0]
                                    href = f"{base}/{href}"
                                    
                                download_links[target] = href
                                logger.info(f"Found download link for {target}: {href}")
                                break
            
            # Alternative: Look for data tables
            if not download_links:
                logger.info("Looking for data in tables...")
                tables = await page.locator("table").all()
                for table in tables:
                    rows = await table.locator("tr").all()
                    for row in rows:
                        cells = await row.locator("td").all()
                        for cell in cells:
                            links = await cell.locator("a[href$='.zip']").all()
                            for link in links:
                                href = await link.get_attribute("href")
                                text = await link.text_content()
                                
                                for target in self.target_files:
                                    if target in (text or "").lower():
                                        if not href.startswith("http"):
                                            base = page.url.rsplit("/", 1)[0]
                                            href = f"{base}/{href}"
                                        download_links[target] = href
                                        logger.info(f"Found in table: {target}")
                                        break
                                        
        except Exception as e:
            logger.error(f"Error finding download links: {e}")
            
        return download_links
    
    async def download_file(self, page: Page, url: str, filename: str) -> Optional[Path]:
        """
        Download a file from the given URL.
        
        Args:
            page: Playwright page object
            url: URL to download from
            filename: Name to save the file as
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            output_path = self.output_dir / f"{filename}.zip"
            
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                return output_path
            
            logger.info(f"Downloading {filename} from {url}")
            
            # Start download
            async with page.expect_download() as download_info:
                await page.goto(url)
            
            download = await download_info.value
            
            # Save the file
            await download.save_as(output_path)
            logger.info(f"Downloaded successfully: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            return None
    
    async def extract_zip(self, zip_path: Path) -> bool:
        """Extract a zip file."""
        try:
            extract_dir = self.output_dir / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted {zip_path} to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    async def run_automated_download(self):
        """
        Run the complete automated download process.
        """
        logger.info("=" * 60)
        logger.info("Automated NOAA Precipitation Grid Downloader")
        logger.info("=" * 60)
        
        successful_downloads = []
        failed_downloads = []
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                # Navigate to NOAA site
                if await self.navigate_to_downloads(page):
                    logger.info("Successfully navigated to NOAA downloads page")
                    
                    # Wait for page to fully load
                    await page.wait_for_timeout(3000)
                    
                    # Find download links
                    download_links = await self.find_download_links(page)
                    
                    if download_links:
                        logger.info(f"Found {len(download_links)} download links")
                        
                        # Download each file
                        for filename, url in download_links.items():
                            downloaded_path = await self.download_file(page, url, filename)
                            
                            if downloaded_path:
                                # Extract if it's a zip
                                if downloaded_path.suffix == '.zip':
                                    if await self.extract_zip(downloaded_path):
                                        successful_downloads.append(filename)
                                    else:
                                        failed_downloads.append(filename)
                                else:
                                    successful_downloads.append(filename)
                            else:
                                failed_downloads.append(filename)
                            
                            # Small delay between downloads
                            await page.wait_for_timeout(2000)
                    else:
                        logger.warning("No download links found")
                        
                        # Take screenshot for debugging
                        screenshot_path = self.output_dir / "debug_screenshot.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Screenshot saved for debugging: {screenshot_path}")
                        
                        # Try alternative approach - look for direct file listing
                        await self.try_alternative_download(page)
                else:
                    logger.error("Failed to navigate to downloads page")
                    
            except Exception as e:
                logger.error(f"Automation error: {e}")
                
            finally:
                await browser.close()
        
        # Summary
        logger.info("=" * 60)
        logger.info("Download Summary:")
        logger.info(f"Successfully downloaded: {len(successful_downloads)} files")
        if successful_downloads:
            for f in successful_downloads:
                logger.info(f"  ✓ {f}")
        
        logger.info(f"Failed/Manual required: {len(failed_downloads)} files")
        if failed_downloads:
            for f in failed_downloads:
                logger.info(f"  ✗ {f}")
        
        # Add files not attempted
        not_attempted = [f for f in self.target_files 
                        if f not in successful_downloads and f not in failed_downloads]
        if not_attempted:
            logger.info(f"Not found/attempted: {len(not_attempted)} files")
            for f in not_attempted:
                logger.info(f"  - {f}")
        
        logger.info("=" * 60)
        
        return successful_downloads, failed_downloads
    
    async def try_alternative_download(self, page: Page):
        """
        Try alternative download methods if standard approach fails.
        """
        logger.info("Trying alternative download approach...")
        
        try:
            # Try FTP links
            ftp_url = "https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html"
            await page.goto(ftp_url, wait_until="networkidle")
            
            # Look for FTP or direct download sections
            ftp_links = await page.locator('a[href*="ftp://"]').all()
            if ftp_links:
                logger.info(f"Found {len(ftp_links)} FTP links")
                for link in ftp_links[:5]:  # Check first 5
                    href = await link.get_attribute("href")
                    text = await link.text_content()
                    logger.info(f"FTP link: {text} -> {href}")
            
            # Look for any download instructions
            instructions = await page.locator('text=/download|instructions/i').all()
            if instructions:
                logger.info("Found download instructions on page")
                for inst in instructions[:3]:
                    text = await inst.text_content()
                    logger.info(f"Instruction: {text[:100]}...")
                    
        except Exception as e:
            logger.error(f"Alternative download failed: {e}")


async def main():
    """Main entry point."""
    # Run with headless=False initially to see what's happening
    downloader = AutomatedNOAADownloader(headless=False)
    await downloader.run_automated_download()


if __name__ == "__main__":
    asyncio.run(main())