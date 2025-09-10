"""NOAA Atlas 14 precipitation frequency data acquisition module.

This module provides functionality for downloading precipitation frequency
estimates from NOAA Atlas 14 for multiple return periods and durations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlencode
import requests
from pyproj import CRS, Transformer
import logging

from .base import BaseDataSource, DataSourceError
from ..config import BoundingBox, DataConfig
from .noaa_atlas14_local import NOAAAtlas14LocalLoader

logger = logging.getLogger(__name__)


class NOAAAtlas14Fetcher(BaseDataSource):
    """NOAA Atlas 14 precipitation frequency data fetcher.
    
    Downloads precipitation frequency estimates for specified locations,
    return periods, and durations from NOAA's Precipitation Frequency
    Data Server (PFDS).
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize NOAA Atlas 14 fetcher.
        
        Args:
            config: Data configuration object
        """
        super().__init__(config)
        self.pfds_url = self.config.noaa_pfds_base_url
        self.api_url = self.config.noaa_api_base_url
        
        # Standard durations in minutes (NOAA uses minutes)
        self.duration_minutes = {
            5: "5-min", 10: "10-min", 15: "15-min", 30: "30-min",
            60: "60-min", 120: "2-hr", 180: "3-hr", 360: "6-hr",
            720: "12-hr", 1440: "24-hr", 2880: "48-hr", 4320: "3-day",
            5760: "4-day", 7200: "5-day", 8640: "6-day", 8760: "7-day",
            10080: "10-day", 20160: "20-day", 30240: "30-day", 43200: "45-day", 
            60480: "60-day"
        }
        
        # Convert config durations to minutes
        self.config_durations_min = [int(h * 60) for h in self.config.durations_hours]
    
    def get_available_data(
        self,
        location: Union[Tuple[float, float], Dict[str, float]],
        **kwargs
    ) -> Dict[str, Any]:
        """Get available precipitation frequency data for location.
        
        Args:
            location: (longitude, latitude) tuple or dict with 'lon', 'lat' keys
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with available return periods and durations
            
        Raises:
            DataSourceError: If location query fails
        """
        if isinstance(location, (tuple, list)):
            lon, lat = location
        else:
            lon, lat = location['lon'], location['lat']
            
        cache_key = self._generate_cache_key(
            lon=lon, lat=lat, action="available_data"
        )
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache first
        if cached_data := self._load_from_cache(cache_path):
            return cached_data
        
        self.logger.info(f"Querying available NOAA Atlas 14 data for ({lon:.4f}, {lat:.4f})")
        
        try:
            # Query PFDS API for point data
            params = {
                'lat': lat,
                'lon': lon,
                'units': 'english',  # Can be changed to 'metric' if needed
                'series': 'pds'  # Partial duration series
            }
            
            # Try the API endpoint first
            api_url = f"{self.api_url}/pfds_api.php"
            response = self.session.get(api_url, params=params)
            
            if response.status_code == 200:
                # Parse API response
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                
                available_data = {
                    'location': {'longitude': lon, 'latitude': lat},
                    'return_periods': self.config.return_periods_years,
                    'durations_minutes': list(self.duration_minutes.keys()),
                    'durations_labels': list(self.duration_minutes.values()),
                    'units': params['units'],
                    'series': params['series'],
                    'source': 'NOAA Atlas 14',
                    'api_response': data
                }
                
            else:
                # Fallback to basic structure if API unavailable
                self.logger.warning("NOAA API not accessible, using default structure")
                available_data = {
                    'location': {'longitude': lon, 'latitude': lat},
                    'return_periods': self.config.return_periods_years,
                    'durations_minutes': list(self.duration_minutes.keys()),
                    'durations_labels': list(self.duration_minutes.values()),
                    'units': 'english',
                    'series': 'pds',
                    'source': 'NOAA Atlas 14'
                }
            
            # Cache results
            self._save_to_cache(cache_path, available_data)
            
            return available_data
            
        except Exception as e:
            raise DataSourceError(f"Failed to query NOAA Atlas 14 availability: {e}")
    
    def download_data(self, **kwargs) -> List[Path]:
        """Download data products (implementation of abstract method).
        
        Args:
            **kwargs: Download parameters
            
        Returns:
            List of paths to downloaded files
        """
        # Delegate to specific download methods based on parameters
        if 'location' in kwargs:
            # Point data download
            result = self.download_point_data(**kwargs)
            return [result] if isinstance(result, Path) else result
        elif 'bbox' in kwargs or 'region_name' in kwargs:
            # Regional data download
            if 'region_name' in kwargs:
                return self.download_region(**kwargs)
            else:
                return self.download_region_data(**kwargs)
        else:
            raise DataSourceError("Must specify either 'location' for point data or 'bbox'/'region_name' for regional data")
    
    def download_point_data(
        self,
        location: Union[Tuple[float, float], Dict[str, float]],
        return_periods: Optional[List[int]] = None,
        durations_hours: Optional[List[float]] = None,
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """Download precipitation frequency data for a point location.
        
        NOTE: The original NOAA PFDS CSV API has been deprecated. This implementation
        generates sample data based on typical precipitation frequency patterns for
        the southeastern United States. For production use, implement proper GIS
        ASCII grid file processing or use the interactive PFDS web interface.
        
        Args:
            location: (longitude, latitude) tuple or dict with 'lon', 'lat' keys
            return_periods: Return periods in years (default: from config)
            durations_hours: Durations in hours (default: from config)
            output_dir: Directory to save data file
            **kwargs: Additional parameters
            
        Returns:
            Path to downloaded data file (CSV format)
            
        Raises:
            DataSourceError: If download fails
        """
        if isinstance(location, (tuple, list)):
            lon, lat = location
        else:
            lon, lat = location['lon'], location['lat']
        
        if return_periods is None:
            return_periods = self.config.return_periods_years
        if durations_hours is None:
            durations_hours = self.config.durations_hours
            
        durations_min = [int(h * 60) for h in durations_hours]
        
        if output_dir is None:
            output_dir = self.config.data_dir / "rainfall" / "atlas14"
        
        # Generate filename
        filename = f"atlas14_point_{lat:.4f}_{lon:.4f}.csv"
        output_path = output_dir / filename
        
        cache_key = self._generate_cache_key(
            lon=lon, lat=lat,
            return_periods=return_periods,
            durations=durations_min
        )
        cache_path = self._get_cache_path(cache_key, ".csv")
        
        # Check if we have cached data
        if self._is_cache_valid(cache_path):
            self.logger.info(f"Using cached Atlas 14 data: {cache_path}")
            if cache_path != output_path:
                cache_path.replace(output_path)
            return output_path
        
        # Check for local CSV files first
        local_csv_paths = [
            self.config.data_dir / "regions" / "nashville" / "rainfall" / "All_Depth_English_PDS.csv",
            Path("/Users/nateaune/Documents/code/FloodRisk/data/regions/nashville/rainfall/All_Depth_English_PDS.csv"),
            output_dir / "All_Depth_English_PDS.csv"
        ]
        
        local_csv = None
        for csv_path in local_csv_paths:
            if csv_path.exists():
                local_csv = csv_path
                self.logger.info(f"Found local NOAA Atlas 14 CSV: {csv_path}")
                break
        
        if local_csv:
            # Use local CSV data
            try:
                loader = NOAAAtlas14LocalLoader(local_csv)
                
                # Get precipitation depths for requested return periods and durations
                data_rows = []
                for rp in return_periods:
                    row = {'return_period_years': rp}
                    for duration_min in durations_min:
                        # Convert minutes to NOAA duration string
                        duration_str = self._minutes_to_duration_string(duration_min)
                        try:
                            depth_inches = loader.get_precipitation_depth(duration_str, rp)
                            # Convert inches to mm
                            depth_mm = depth_inches * 25.4
                            row[f'duration_{duration_min}_min'] = depth_mm
                        except (KeyError, ValueError) as e:
                            self.logger.warning(f"Could not get {duration_str} {rp}-year data: {e}")
                            # Use approximate value
                            row[f'duration_{duration_min}_min'] = self._estimate_precipitation(rp, duration_min)
                    data_rows.append(row)
                
                # Add metadata and apply spatial variation
                metadata = loader.data.get('metadata', {})
                station_lat = metadata.get('latitude', 36.1253)
                station_lon = metadata.get('longitude', -86.6764)
                
                # Calculate distance from station to grid point
                distance_km = self._calculate_distance(station_lat, station_lon, lat, lon)
                
                # Apply spatial variation based on distance (simple linear interpolation)
                # Precipitation typically varies by ~1-2% per 10 km
                variation_factor = 1.0 + (distance_km * 0.001)  # 0.1% per km
                
                for row in data_rows:
                    # Apply spatial variation to precipitation values
                    for key in row.keys():
                        if key.startswith('duration_'):
                            row[key] *= variation_factor
                    
                    row['latitude'] = lat  # Use actual grid point coordinates
                    row['longitude'] = lon
                    row['station'] = f"{metadata.get('station', 'Nashville WSO Airport')} (interpolated)"
                    row['source'] = 'NOAA Atlas 14 Local CSV'
                
                df = pd.DataFrame(data_rows)
                self.logger.info(f"Loaded precipitation data from local CSV for {len(return_periods)} return periods")
                
            except Exception as e:
                self.logger.warning(f"Failed to load local CSV: {e}. Falling back to synthetic data.")
                df = self._generate_sample_atlas14_data(return_periods, durations_min, lat, lon)
        else:
            self.logger.warning(f"No local NOAA Atlas 14 CSV found. Generating sample precipitation data for point ({lon:.4f}, {lat:.4f})")
            # Generate sample precipitation frequency data
            # These are approximate values for the southeastern US (Nashville area)
            df = self._generate_sample_atlas14_data(return_periods, durations_min, lat, lon)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            # Cache the CSV
            if cache_path != output_path:
                df.to_csv(cache_path, index=False)
            
            self.logger.info(f"Saved Atlas 14 data: {output_path}")
            return output_path
                
        except Exception as e:
            raise DataSourceError(f"Error saving Atlas 14 data: {e}")
    
    def download_region_data(
        self,
        bbox: Union[BoundingBox, Dict[str, float]],
        grid_spacing: float = 0.01,  # ~1km spacing
        return_periods: Optional[List[int]] = None,
        durations_hours: Optional[List[float]] = None,
        output_dir: Optional[Path] = None,
        region_name: Optional[str] = None,
        **kwargs
    ) -> List[Path]:
        """Download precipitation data for a region using grid points.
        
        Args:
            bbox: Geographic bounding box
            grid_spacing: Grid spacing in decimal degrees
            return_periods: Return periods in years
            durations_hours: Durations in hours  
            output_dir: Directory to save data files
            region_name: Name for the region
            **kwargs: Additional parameters
            
        Returns:
            List of paths to downloaded data files
            
        Raises:
            DataSourceError: If download fails
        """
        if isinstance(bbox, dict):
            bbox = BoundingBox(**bbox)
        
        if return_periods is None:
            return_periods = self.config.return_periods_years
        if durations_hours is None:
            durations_hours = self.config.durations_hours
            
        if output_dir is None:
            if region_name:
                output_dir = self.config.data_dir / "regions" / region_name / "rainfall" / "atlas14"
            else:
                output_dir = self.config.data_dir / "rainfall" / "atlas14" / "region"
        
        # Generate grid points
        lon_points = []
        lat_points = []
        
        lon = bbox.west
        while lon <= bbox.east:
            lat = bbox.south
            while lat <= bbox.north:
                lon_points.append(lon)
                lat_points.append(lat)
                lat += grid_spacing
            lon += grid_spacing
        
        self.logger.info(f"Downloading Atlas 14 data for {len(lon_points)} grid points")
        
        downloaded_files = []
        successful_points = 0
        max_points = 50  # Limit to reasonable number for sample data
        
        for i, (lon, lat) in enumerate(zip(lon_points, lat_points)):
            if successful_points >= max_points:
                self.logger.info(f"Reached maximum sample points limit ({max_points}), stopping...")
                break
                
            try:
                self.logger.info(f"Processing grid point {i+1}/{len(lon_points)}: ({lon:.4f}, {lat:.4f})")
                
                point_file = self.download_point_data(
                    location=(lon, lat),
                    return_periods=return_periods,
                    durations_hours=durations_hours,
                    output_dir=output_dir
                )
                
                downloaded_files.append(point_file)
                successful_points += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to download data for point ({lon:.4f}, {lat:.4f}): {e}")
                continue
        
        if not downloaded_files:
            raise DataSourceError("No precipitation data was successfully downloaded")
        
        self.logger.info(f"Successfully downloaded {len(downloaded_files)} precipitation data files")
        return downloaded_files
    
    def _parse_atlas14_response(
        self,
        response_data: Dict[str, Any],
        return_periods: List[int],
        durations_min: List[int]
    ) -> pd.DataFrame:
        """Parse NOAA Atlas 14 API response into structured DataFrame.
        
        Args:
            response_data: JSON response from NOAA API
            return_periods: Requested return periods
            durations_min: Requested durations in minutes
            
        Returns:
            DataFrame with precipitation frequency estimates
        """
        records = []
        
        # Extract precipitation estimates from response
        # This is a simplified parser - actual NOAA response structure may vary
        for duration_min in durations_min:
            duration_label = self.duration_minutes.get(duration_min, f"{duration_min}-min")
            
            for return_period in return_periods:
                record = {
                    'duration_minutes': duration_min,
                    'duration_label': duration_label,
                    'return_period_years': return_period,
                    'precipitation_inches': 0.0,  # Placeholder
                    'lower_bound': 0.0,  # 90% confidence interval lower
                    'upper_bound': 0.0   # 90% confidence interval upper
                }
                
                # Try to extract actual values from response
                if 'data' in response_data:
                    data_key = f"{return_period}yr_{duration_label}"
                    if data_key in response_data['data']:
                        record['precipitation_inches'] = response_data['data'][data_key].get('estimate', 0.0)
                        record['lower_bound'] = response_data['data'][data_key].get('lower', 0.0)
                        record['upper_bound'] = response_data['data'][data_key].get('upper', 0.0)
                
                records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    def _minutes_to_duration_string(self, minutes: int) -> str:
        """Convert minutes to NOAA duration string format.
        
        Args:
            minutes: Duration in minutes
            
        Returns:
            Duration string (e.g., '24-hr', '60-min')
        """
        duration_map = {
            5: '5-min', 10: '10-min', 15: '15-min', 30: '30-min',
            60: '60-min', 120: '2-hr', 180: '3-hr', 360: '6-hr',
            720: '12-hr', 1440: '24-hr', 2880: '2-day', 4320: '3-day',
            5760: '4-day', 10080: '7-day', 14400: '10-day', 28800: '20-day',
            43200: '30-day', 64800: '45-day', 86400: '60-day'
        }
        return duration_map.get(minutes, f'{minutes}-min')
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        import math
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _estimate_precipitation(self, return_period: int, duration_min: int) -> float:
        """Estimate precipitation depth based on return period and duration.
        
        Args:
            return_period: Return period in years
            duration_min: Duration in minutes
            
        Returns:
            Estimated precipitation depth in mm
        """
        # Base values for 24-hour duration (mm)
        base_24hr = {
            10: 100, 25: 125, 50: 150, 100: 175, 500: 225, 1000: 250
        }
        
        # Get base value or interpolate
        if return_period in base_24hr:
            base_value = base_24hr[return_period]
        else:
            # Simple linear interpolation
            base_value = 100 + (return_period - 10) * 1.5
        
        # Adjust for duration
        duration_factor = (duration_min / 1440) ** 0.5  # 1440 minutes = 24 hours
        return base_value * duration_factor
    
    def _generate_sample_atlas14_data(
        self,
        return_periods: List[int],
        durations_min: List[int],
        lat: float,
        lon: float
    ) -> pd.DataFrame:
        """Generate sample NOAA Atlas 14 precipitation frequency data.
        
        This generates reasonable precipitation frequency estimates for the 
        southeastern United States based on typical patterns. Values are
        approximate and should be replaced with actual NOAA data for production use.
        
        Args:
            return_periods: Return periods in years
            durations_min: Durations in minutes
            lat: Latitude
            lon: Longitude
            
        Returns:
            DataFrame with sample precipitation frequency estimates
        """
        import math
        
        records = []
        
        # Base 24-hour precipitation depths for Nashville area (inches)
        base_24hr_depths = {
            10: 4.5,
            25: 5.2,
            50: 5.8,
            100: 6.4,
            500: 7.8
        }
        
        # Duration adjustment factors (relative to 24-hour)
        duration_factors = {
            60: 0.35,    # 1-hour
            120: 0.50,   # 2-hour
            180: 0.60,   # 3-hour
            360: 0.75,   # 6-hour
            720: 0.90,   # 12-hour
            1440: 1.00,  # 24-hour
        }
        
        for duration_min in durations_min:
            duration_label = self.duration_minutes.get(duration_min, f"{duration_min}-min")
            
            # Get duration factor (interpolate if not exact match)
            if duration_min in duration_factors:
                duration_factor = duration_factors[duration_min]
            else:
                # Simple linear interpolation between closest values
                sorted_durations = sorted(duration_factors.keys())
                if duration_min < sorted_durations[0]:
                    duration_factor = duration_factors[sorted_durations[0]] * 0.8
                elif duration_min > sorted_durations[-1]:
                    duration_factor = duration_factors[sorted_durations[-1]] * 1.1
                else:
                    # Find bounding values and interpolate
                    lower_dur = max(d for d in sorted_durations if d <= duration_min)
                    upper_dur = min(d for d in sorted_durations if d >= duration_min)
                    if lower_dur == upper_dur:
                        duration_factor = duration_factors[lower_dur]
                    else:
                        ratio = (duration_min - lower_dur) / (upper_dur - lower_dur)
                        duration_factor = (duration_factors[lower_dur] * (1 - ratio) + 
                                         duration_factors[upper_dur] * ratio)
            
            for return_period in return_periods:
                # Get base depth for return period (interpolate if needed)
                if return_period in base_24hr_depths:
                    base_depth = base_24hr_depths[return_period]
                else:
                    # Log-linear interpolation for return periods
                    sorted_rps = sorted(base_24hr_depths.keys())
                    if return_period < sorted_rps[0]:
                        base_depth = base_24hr_depths[sorted_rps[0]] * 0.9
                    elif return_period > sorted_rps[-1]:
                        # Extrapolate using log relationship
                        base_depth = base_24hr_depths[sorted_rps[-1]] * (
                            math.log(return_period) / math.log(sorted_rps[-1])
                        )
                    else:
                        # Log-linear interpolation
                        lower_rp = max(rp for rp in sorted_rps if rp <= return_period)
                        upper_rp = min(rp for rp in sorted_rps if rp >= return_period)
                        if lower_rp == upper_rp:
                            base_depth = base_24hr_depths[lower_rp]
                        else:
                            log_ratio = (math.log(return_period) - math.log(lower_rp)) / (
                                math.log(upper_rp) - math.log(lower_rp)
                            )
                            base_depth = (base_24hr_depths[lower_rp] * (1 - log_ratio) + 
                                        base_24hr_depths[upper_rp] * log_ratio)
                
                # Calculate precipitation depth
                precipitation_inches = base_depth * duration_factor
                
                # Add some uncertainty bounds (±15%)
                lower_bound = precipitation_inches * 0.85
                upper_bound = precipitation_inches * 1.15
                
                record = {
                    'duration_minutes': duration_min,
                    'duration_label': duration_label,
                    'return_period_years': return_period,
                    'precipitation_inches': round(precipitation_inches, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    def get_rainfall_depths(
        self,
        location: Union[Tuple[float, float], Dict[str, float]],
        return_periods: Optional[List[int]] = None,
        duration_hours: float = 24,
        **kwargs
    ) -> Dict[int, float]:
        """Get 24-hour (or specified duration) rainfall depths for return periods.
        
        Args:
            location: (longitude, latitude) tuple or dict
            return_periods: Return periods in years
            duration_hours: Storm duration in hours (default: 24)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping return period to rainfall depth in inches
            
        Raises:
            DataSourceError: If data retrieval fails
        """
        if return_periods is None:
            return_periods = self.config.return_periods_years
        
        # Download point data
        data_file = self.download_point_data(
            location=location,
            return_periods=return_periods,
            durations_hours=[duration_hours],
            **kwargs
        )
        
        # Read and parse the data
        try:
            df = pd.read_csv(data_file)
            
            # Filter for requested duration
            duration_min = int(duration_hours * 60)
            duration_data = df[df['duration_minutes'] == duration_min]
            
            # Extract rainfall depths by return period
            rainfall_depths = {}
            for _, row in duration_data.iterrows():
                return_period = row['return_period_years']
                if return_period in return_periods:
                    rainfall_depths[return_period] = row['precipitation_inches']
            
            return rainfall_depths
            
        except Exception as e:
            raise DataSourceError(f"Failed to parse rainfall depth data: {e}")
    
    def download_region(
        self,
        region_name: str,
        grid_spacing: float = 0.01,
        return_periods: Optional[List[int]] = None,
        durations_hours: Optional[List[float]] = None,
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> List[Path]:
        """Download precipitation data for a predefined region.
        
        Args:
            region_name: Name of predefined region (e.g., 'nashville', 'tennessee')
            grid_spacing: Grid spacing in decimal degrees
            return_periods: Return periods in years
            durations_hours: Durations in hours
            output_dir: Directory to save data files
            **kwargs: Additional parameters
            
        Returns:
            List of paths to downloaded data files
            
        Raises:
            DataSourceError: If region not found or download fails
        """
        bbox = self.config.get_region_bbox(region_name)
        if bbox is None:
            available_regions = list(self.config.regions.keys())
            raise DataSourceError(
                f"Region '{region_name}' not found. "
                f"Available regions: {available_regions}"
            )
        
        if output_dir is None:
            output_dir = self.config.data_dir / "regions" / region_name / "rainfall" / "atlas14"
        
        self.logger.info(f"Downloading NOAA Atlas 14 data for region: {region_name}")
        
        return self.download_region_data(
            bbox=bbox,
            grid_spacing=grid_spacing,
            return_periods=return_periods,
            durations_hours=durations_hours,
            output_dir=output_dir,
            region_name=region_name,
            **kwargs
        )