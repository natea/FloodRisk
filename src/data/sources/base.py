"""Base class and utilities for data source implementations."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter, Retry
import logging

from ..config import DataConfig


logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """Exception raised by data source operations."""
    pass


class BaseDataSource(ABC):
    """Base class for data source implementations.
    
    Provides common functionality for caching, HTTP requests, validation,
    and error handling that can be reused across different data sources.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize base data source.
        
        Args:
            config: Data configuration object. If None, uses default config.
        """
        self.config = config or DataConfig()
        self._session = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up logging for data source operations."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session with retry strategy."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            # Set timeout and headers
            self._session.timeout = self.config.timeout_seconds
            self._session.headers.update({
                'User-Agent': 'FloodRisk Data Acquisition v0.1.0'
            })
            
        return self._session
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters.
        
        Args:
            **kwargs: Parameters to include in cache key
            
        Returns:
            SHA256 hash of serialized parameters
        """
        # Sort parameters for consistent hashing
        sorted_params = dict(sorted(kwargs.items()))
        param_str = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, extension: str = ".json") -> Path:
        """Get cache file path for given key.
        
        Args:
            cache_key: Cache key identifier
            extension: File extension for cache file
            
        Returns:
            Path to cache file
        """
        cache_subdir = self.config.cache_dir / self.__class__.__name__.lower()
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{cache_key}{extension}"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file is still valid.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
            
        # Check file size
        if cache_path.stat().st_size < self.config.min_file_size_bytes:
            return False
            
        # Check expiry
        file_age = time.time() - cache_path.stat().st_mtime
        max_age = self.config.cache_expiry_days * 24 * 3600
        
        return file_age < max_age
    
    def _save_to_cache(self, cache_path: Path, data: Any) -> None:
        """Save data to cache file.
        
        Args:
            cache_path: Path to cache file
            data: Data to cache (must be JSON serializable)
        """
        if not self.config.enable_caching:
            return
            
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_path}: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Any]:
        """Load data from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Cached data or None if cache miss/invalid
        """
        if not self.config.enable_caching or not self._is_cache_valid(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            self.logger.debug(f"Loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_path}: {e}")
            return None
    
    def _download_file(
        self, 
        url: str, 
        output_path: Path,
        resume: bool = True,
        validate: bool = True
    ) -> Path:
        """Download file with resume capability and validation.
        
        Args:
            url: URL to download
            output_path: Path to save downloaded file
            resume: Whether to resume partial downloads
            validate: Whether to validate download
            
        Returns:
            Path to downloaded file
            
        Raises:
            DataSourceError: If download fails or validation fails
        """
        self.logger.info(f"Downloading: {url}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing partial file
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        headers = {}
        
        if resume and temp_path.exists():
            headers['Range'] = f'bytes={temp_path.stat().st_size}-'
            self.logger.info(f"Resuming download from byte {temp_path.stat().st_size}")
        
        try:
            response = self.session.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Determine write mode
            mode = 'ab' if resume and temp_path.exists() and response.status_code == 206 else 'wb'
            
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                    if chunk:
                        f.write(chunk)
            
            # Move temp file to final location
            temp_path.rename(output_path)
            
            # Validate download if requested
            if validate and self.config.validate_downloads:
                self._validate_download(output_path, response.headers)
            
            self.logger.info(f"Download completed: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            self.logger.error(f"Download failed for {url}: {e}")
            raise DataSourceError(f"Failed to download {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            raise DataSourceError(f"Download error: {e}")
    
    def _validate_download(self, file_path: Path, headers: Dict[str, str]) -> None:
        """Validate downloaded file.
        
        Args:
            file_path: Path to downloaded file
            headers: HTTP response headers
            
        Raises:
            DataSourceError: If validation fails
        """
        if not file_path.exists():
            raise DataSourceError(f"Downloaded file not found: {file_path}")
        
        file_size = file_path.stat().st_size
        
        # Check minimum size
        if file_size < self.config.min_file_size_bytes:
            raise DataSourceError(f"Downloaded file too small: {file_size} bytes")
        
        # Check content length if available
        if 'content-length' in headers:
            expected_size = int(headers['content-length'])
            if file_size != expected_size:
                raise DataSourceError(
                    f"File size mismatch: expected {expected_size}, got {file_size}"
                )
        
        self.logger.debug(f"Download validation passed: {file_path} ({file_size} bytes)")
    
    @abstractmethod
    def get_available_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Get list of available data products.
        
        Returns:
            List of available data products with metadata
        """
        pass
    
    @abstractmethod  
    def download_data(self, **kwargs) -> List[Path]:
        """Download data products.
        
        Returns:
            List of paths to downloaded files
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            self._session.close()
            self._session = None