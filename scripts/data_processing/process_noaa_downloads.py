#!/usr/bin/env python3
"""
Process all downloaded NOAA Atlas 14 precipitation grids.
Automatically detects and processes ASCII grid files in the data directory.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import xarray as xr
from scipy import interpolate

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.process_precipitation_grids import PrecipitationGridProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NOAAGridBatchProcessor(PrecipitationGridProcessor):
    """Process all downloaded NOAA grids."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize batch processor."""
        super().__init__(data_dir)
        self.noaa_grids = []
        
    def find_noaa_grids(self) -> List[Dict]:
        """Find all NOAA ASCII grid files in the data directory."""
        grids = []
        
        # Look for NOAA directories (pattern: se*yr*h*)
        for dir_path in self.data_dir.glob("se*yr*h*"):
            if dir_path.is_dir():
                # Look for ASCII grid files
                asc_files = list(dir_path.glob("*.asc"))
                
                for asc_file in asc_files:
                    # Parse filename to extract metadata
                    filename = asc_file.stem
                    
                    # Extract return period and duration from filename
                    # Pattern: se[return_period]yr[duration]ha
                    parts = filename.replace('se', '').replace('ha', '').split('yr')
                    
                    if len(parts) == 2:
                        try:
                            return_period = int(parts[0])
                            duration_str = parts[1]
                            
                            # Convert duration to hours
                            if duration_str.isdigit():
                                duration_hours = int(duration_str)
                            else:
                                duration_hours = duration_str
                            
                            grid_info = {
                                'file_path': asc_file,
                                'dir_path': dir_path,
                                'filename': filename,
                                'return_period': return_period,
                                'duration_hours': duration_hours,
                                'prj_file': dir_path / f"{filename}.prj",
                                'xml_file': dir_path / f"{filename}.xml",
                            }
                            
                            grids.append(grid_info)
                            logger.info(f"Found grid: {return_period}-year {duration_hours}-hour")
                            
                        except Exception as e:
                            logger.warning(f"Could not parse filename {filename}: {e}")
        
        # Sort by return period and duration
        grids.sort(key=lambda x: (x['return_period'], str(x['duration_hours'])))
        
        return grids
    
    def process_noaa_grid(self, grid_info: Dict) -> Dict:
        """Process a single NOAA grid file."""
        logger.info(f"\nProcessing {grid_info['filename']}...")
        
        try:
            # Read ASCII grid
            data, metadata = self.read_ascii_grid(grid_info['file_path'])
            
            # The NOAA grids are typically in inches, convert to mm
            data_mm = self.convert_units(data, 'inches', 'mm')
            
            # Clip to Nashville bbox
            clipped_data, clipped_metadata = self.clip_to_bbox(data_mm, metadata)
            
            # Resample to target resolution (1km)
            resampled_data, resampled_metadata = self.resample_grid(
                clipped_data, clipped_metadata
            )
            
            # Create ensemble with uncertainty (using NOAA 90% confidence intervals)
            # Typically, the 90% CI is about ±20% of the estimate
            ensemble = self.create_ensemble_grids(resampled_data, num_members=20, cv=0.15)
            
            # Save processed data
            output_filename = f"noaa_{grid_info['return_period']}yr_{grid_info['duration_hours']}hr.npz"
            output_file = self.output_dir / output_filename
            
            np.savez_compressed(
                output_file,
                mean=resampled_data,
                ensemble=np.array(ensemble),
                metadata=resampled_metadata,
                bbox=self.nashville_bbox,
                return_period=grid_info['return_period'],
                duration_hours=grid_info['duration_hours']
            )
            
            logger.info(f"Saved: {output_file}")
            
            # Calculate statistics
            stats = {
                'file': output_filename,
                'return_period_years': grid_info['return_period'],
                'duration_hours': grid_info['duration_hours'],
                'min_precipitation_mm': float(np.nanmin(resampled_data)),
                'max_precipitation_mm': float(np.nanmax(resampled_data)),
                'mean_precipitation_mm': float(np.nanmean(resampled_data)),
                'std_precipitation_mm': float(np.nanstd(resampled_data)),
                'p10_mm': float(np.nanpercentile(resampled_data[~np.isnan(resampled_data)], 10)),
                'p50_mm': float(np.nanpercentile(resampled_data[~np.isnan(resampled_data)], 50)),
                'p90_mm': float(np.nanpercentile(resampled_data[~np.isnan(resampled_data)], 90)),
                'grid_shape': resampled_data.shape,
                'resolution_degrees': resampled_metadata['cellsize'],
                'ensemble_members': len(ensemble),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process {grid_info['filename']}: {e}")
            return None
    
    def create_summary_report(self, all_stats: List[Dict]):
        """Create a summary report of all processed grids."""
        
        # Save individual statistics
        stats_file = self.output_dir / "noaa_grid_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Create summary DataFrame
        df = pd.DataFrame(all_stats)
        
        # Save as CSV for easy viewing
        csv_file = self.output_dir / "noaa_grid_summary.csv"
        df.to_csv(csv_file, index=False)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("NOAA PRECIPITATION GRID PROCESSING SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\nProcessed {len(all_stats)} grids successfully")
        
        logger.info("\nPrecipitation Depths by Return Period (24-hour duration):")
        df_24hr = df[df['duration_hours'] == 24].sort_values('return_period_years')
        
        for _, row in df_24hr.iterrows():
            rp = row['return_period_years']
            mean_mm = row['mean_precipitation_mm']
            mean_in = mean_mm / 25.4
            max_mm = row['max_precipitation_mm']
            max_in = max_mm / 25.4
            
            logger.info(f"  {rp:3d}-year: Mean={mean_mm:6.1f}mm ({mean_in:4.2f}\") | Max={max_mm:6.1f}mm ({max_in:4.2f}\")")
        
        logger.info("\nAll Duration Statistics:")
        for _, row in df.iterrows():
            rp = row['return_period_years']
            dur = row['duration_hours']
            mean_mm = row['mean_precipitation_mm']
            
            logger.info(f"  {rp:3d}-year {dur:2}-hour: {mean_mm:6.1f}mm")
        
        logger.info(f"\nOutput files saved to: {self.output_dir}")
        logger.info(f"  - Individual grids: noaa_*yr_*hr.npz")
        logger.info(f"  - Statistics: {stats_file}")
        logger.info(f"  - Summary CSV: {csv_file}")
        
        return df
    
    def create_combined_dataset(self, all_stats: List[Dict]):
        """Create a combined dataset with all return periods and durations."""
        
        logger.info("\nCreating combined dataset...")
        
        # Group by duration
        durations = {}
        for stat in all_stats:
            dur = stat['duration_hours']
            if dur not in durations:
                durations[dur] = []
            durations[dur].append(stat)
        
        # Create combined arrays for each duration
        for duration_hours, stats_list in durations.items():
            logger.info(f"Combining {duration_hours}-hour precipitation grids...")
            
            # Sort by return period
            stats_list.sort(key=lambda x: x['return_period_years'])
            
            # Load all grids for this duration
            return_periods = []
            grids = []
            
            for stat in stats_list:
                rp = stat['return_period_years']
                filename = stat['file']
                
                data = np.load(self.output_dir / filename)
                grids.append(data['mean'])
                return_periods.append(rp)
            
            # Stack into 3D array (return_period, lat, lon)
            combined = np.stack(grids, axis=0)
            
            # Save combined dataset
            output_file = self.output_dir / f"combined_{duration_hours}hr.npz"
            np.savez_compressed(
                output_file,
                data=combined,
                return_periods=return_periods,
                duration_hours=duration_hours,
                bbox=self.nashville_bbox
            )
            
            logger.info(f"  Saved: {output_file}")
            logger.info(f"  Shape: {combined.shape}")
            logger.info(f"  Return periods: {return_periods}")
    
    def run_batch_processing(self):
        """Run the complete batch processing pipeline."""
        logger.info("=" * 70)
        logger.info("NOAA ATLAS 14 BATCH PROCESSING PIPELINE")
        logger.info("=" * 70)
        
        # Find all NOAA grids
        logger.info("\nSearching for NOAA grids...")
        grids = self.find_noaa_grids()
        
        if not grids:
            logger.error("No NOAA grids found!")
            logger.info("Please ensure files are in directories named like: se100yr24ha/")
            return None
        
        logger.info(f"\nFound {len(grids)} NOAA grids to process")
        
        # Process each grid
        all_stats = []
        for grid_info in grids:
            stats = self.process_noaa_grid(grid_info)
            if stats:
                all_stats.append(stats)
        
        if all_stats:
            # Create summary report
            df = self.create_summary_report(all_stats)
            
            # Create combined datasets
            self.create_combined_dataset(all_stats)
            
            logger.info("\n" + "=" * 70)
            logger.info("PROCESSING COMPLETE!")
            logger.info("=" * 70)
            logger.info("\nThe NOAA precipitation grids have been successfully processed")
            logger.info("and are ready for integration with the flood risk ML model.")
            
            return df
        else:
            logger.error("No grids were successfully processed")
            return None


if __name__ == "__main__":
    processor = NOAAGridBatchProcessor()
    processor.run_batch_processing()