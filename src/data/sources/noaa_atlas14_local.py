"""
NOAA Atlas 14 data loader for local CSV files.
Reads precipitation frequency estimates from downloaded NOAA Atlas 14 CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class NOAAAtlas14LocalLoader:
    """Load NOAA Atlas 14 precipitation data from local CSV files."""
    
    def __init__(self, csv_path: Path):
        """
        Initialize local NOAA Atlas 14 data loader.
        
        Args:
            csv_path: Path to the NOAA Atlas 14 CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"NOAA Atlas 14 CSV not found: {csv_path}")
        
        self.data = self._parse_csv()
        logger.info(f"Loaded NOAA Atlas 14 data from {csv_path}")
        
    def _parse_csv(self) -> Dict:
        """
        Parse NOAA Atlas 14 CSV file format.
        
        Returns:
            Dictionary with precipitation data organized by duration and return period
        """
        data = {
            'metadata': {},
            'estimates': {},
            'upper_bound': {},
            'lower_bound': {}
        }
        
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()
        
        # Parse metadata from header
        for i, line in enumerate(lines[:15]):
            if 'Station Name:' in line:
                data['metadata']['station'] = line.split(':', 1)[1].strip()
            elif 'Latitude:' in line:
                data['metadata']['latitude'] = float(line.split(':')[1].split()[0])
            elif 'Longitude:' in line:
                data['metadata']['longitude'] = float(line.split(':')[1].split()[0])
            elif 'Elevation' in line:
                data['metadata']['elevation_ft'] = float(line.split(':')[1].split()[0])
        
        # Find the three data sections
        sections = {
            'estimates': None,
            'upper_bound': None,
            'lower_bound': None
        }
        
        for i, line in enumerate(lines):
            if 'PRECIPITATION FREQUENCY ESTIMATES' in line and 'UPPER BOUND' not in line and 'LOWER BOUND' not in line:
                sections['estimates'] = i + 1
            elif 'UPPER BOUND OF 90% CONFIDENCE INTERVAL' in line:
                sections['upper_bound'] = i + 1
            elif 'LOWER BOUND OF 90% CONFIDENCE INTERVAL' in line:
                sections['lower_bound'] = i + 1
        
        # Parse each section
        for section_name, start_idx in sections.items():
            if start_idx is None:
                continue
                
            # Parse header line with return periods
            header_line = lines[start_idx].strip()
            # Format: "by duration for ARI (years):, 1,2,5,10,25,50,100,200,500,1000"
            return_periods = [int(x.strip()) for x in header_line.split(':,')[1].split(',')]
            
            # Parse data lines
            i = start_idx + 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('Date/time') or line.startswith('pyRunTime'):
                    break
                if ':' not in line:
                    break
                    
                parts = line.split(':,')
                if len(parts) == 2:
                    duration = parts[0].strip()
                    values = [float(x.strip()) for x in parts[1].split(',')]
                    
                    if duration not in data[section_name]:
                        data[section_name][duration] = {}
                    
                    for rp, val in zip(return_periods, values):
                        data[section_name][duration][rp] = val
                
                i += 1
        
        logger.info(f"Parsed {len(data['estimates'])} durations with {len(return_periods)} return periods")
        
        return data
    
    def get_precipitation_depth(
        self, 
        duration: str, 
        return_period: int,
        confidence: str = 'estimates'
    ) -> float:
        """
        Get precipitation depth for specific duration and return period.
        
        Args:
            duration: Duration string (e.g., '24-hr', '6-hr', '60-min')
            return_period: Return period in years (e.g., 100, 500)
            confidence: Which estimates to use ('estimates', 'upper_bound', 'lower_bound')
            
        Returns:
            Precipitation depth in inches
        """
        if confidence not in self.data:
            raise ValueError(f"Invalid confidence level: {confidence}")
            
        if duration not in self.data[confidence]:
            available = list(self.data[confidence].keys())
            raise ValueError(f"Duration '{duration}' not found. Available: {available}")
            
        if return_period not in self.data[confidence][duration]:
            available = list(self.data[confidence][duration].keys())
            raise ValueError(f"Return period {return_period} not found. Available: {available}")
            
        return self.data[confidence][duration][return_period]
    
    def get_24hr_precipitation(self, return_periods: List[int] = None) -> Dict[int, float]:
        """
        Get 24-hour precipitation depths for multiple return periods.
        
        Args:
            return_periods: List of return periods in years (default: [10, 25, 100, 500])
            
        Returns:
            Dictionary mapping return period to precipitation depth in mm
        """
        if return_periods is None:
            return_periods = [10, 25, 100, 500]
        
        results = {}
        for rp in return_periods:
            try:
                # Get 24-hour precipitation in inches
                depth_inches = self.get_precipitation_depth('24-hr', rp)
                # Convert to mm (1 inch = 25.4 mm)
                depth_mm = depth_inches * 25.4
                results[rp] = depth_mm
                logger.info(f"{rp}-year 24-hr precipitation: {depth_inches:.2f} in ({depth_mm:.1f} mm)")
            except ValueError as e:
                logger.warning(f"Could not get {rp}-year precipitation: {e}")
        
        return results
    
    def get_precipitation_table(self) -> pd.DataFrame:
        """
        Get full precipitation frequency table as DataFrame.
        
        Returns:
            DataFrame with durations as rows and return periods as columns
        """
        df_data = []
        
        for duration in self.data['estimates']:
            row = {'Duration': duration}
            for rp in sorted(self.data['estimates'][duration].keys()):
                row[f'{rp}yr'] = self.data['estimates'][duration][rp]
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by duration order (convert to minutes for sorting)
        duration_order = {
            '5-min': 5,
            '10-min': 10,
            '15-min': 15,
            '30-min': 30,
            '60-min': 60,
            '2-hr': 120,
            '3-hr': 180,
            '6-hr': 360,
            '12-hr': 720,
            '24-hr': 1440,
            '2-day': 2880,
            '3-day': 4320,
            '4-day': 5760,
            '7-day': 10080,
            '10-day': 14400,
            '20-day': 28800,
            '30-day': 43200,
            '45-day': 64800,
            '60-day': 86400
        }
        
        df['sort_order'] = df['Duration'].map(duration_order)
        df = df.sort_values('sort_order').drop('sort_order', axis=1)
        df = df.set_index('Duration')
        
        return df
    
    def create_rainfall_config(self, output_path: Path = None) -> Dict:
        """
        Create rainfall configuration for ML pipeline.
        
        Args:
            output_path: Optional path to save configuration as JSON
            
        Returns:
            Configuration dictionary
        """
        config = {
            'source': 'NOAA Atlas 14',
            'station': self.data['metadata'].get('station', 'Unknown'),
            'location': {
                'latitude': self.data['metadata'].get('latitude'),
                'longitude': self.data['metadata'].get('longitude'),
                'elevation_ft': self.data['metadata'].get('elevation_ft')
            },
            'precipitation_24hr_mm': self.get_24hr_precipitation(),
            'precipitation_depths_inches': {},
            'durations': list(self.data['estimates'].keys()),
            'return_periods': sorted(list(self.data['estimates']['24-hr'].keys()))
        }
        
        # Add all precipitation depths
        for duration in self.data['estimates']:
            config['precipitation_depths_inches'][duration] = self.data['estimates'][duration]
        
        if output_path:
            import json
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved rainfall configuration to {output_path}")
        
        return config


def update_nashville_rainfall_config():
    """Update Nashville configuration with real NOAA Atlas 14 data."""
    
    # Path to the downloaded CSV
    csv_path = Path("/Users/nateaune/Documents/code/FloodRisk/data/regions/nashville/rainfall/All_Depth_English_PDS.csv")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Load the data
    loader = NOAAAtlas14LocalLoader(csv_path)
    
    # Display precipitation table
    print("\n" + "="*60)
    print("NOAA Atlas 14 Precipitation Frequency Estimates")
    print("Nashville WSO Airport")
    print("="*60)
    
    df = loader.get_precipitation_table()
    print(df)
    
    # Get 24-hour precipitation for key return periods
    print("\n" + "="*60)
    print("24-Hour Precipitation Depths for ML Training")
    print("="*60)
    
    precip_24hr = loader.get_24hr_precipitation([10, 25, 100, 500])
    for rp, depth_mm in precip_24hr.items():
        depth_in = depth_mm / 25.4
        print(f"{rp:3d}-year: {depth_in:6.2f} inches = {depth_mm:6.1f} mm")
    
    # Create and save configuration
    config_path = Path("/Users/nateaune/Documents/code/FloodRisk/data/nashville/rainfall_config.json")
    config = loader.create_rainfall_config(config_path)
    
    print(f"\n✅ Configuration saved to: {config_path}")
    
    return loader


if __name__ == "__main__":
    # Test with Nashville data
    update_nashville_rainfall_config()