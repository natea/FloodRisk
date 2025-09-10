#!/usr/bin/env python3
"""
Analyze NOAA ASCII grid to find data coverage and Nashville region
"""

import numpy as np
from pathlib import Path
import sys

def read_ascii_grid(file_path):
    """Read ASCII grid file and return data with metadata."""
    with open(file_path, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1]) if len(line) > 1 else line[1]
        
        # Read data
        data = []
        for line in f:
            row = [float(x) for x in line.strip().split()]
            data.append(row)
    
    return np.array(data), header

def analyze_grid_coverage(data, header, nodata_value=-9):
    """Analyze grid coverage and data distribution."""
    nrows = int(header['nrows'])
    ncols = int(header['ncols'])
    xll = header['xllcorner']
    yll = header['yllcorner']
    cellsize = header['cellsize']
    
    print(f"\n=== GRID METADATA ===")
    print(f"Grid dimensions: {nrows} rows x {ncols} cols")
    print(f"Lower-left corner: ({xll:.4f}, {yll:.4f})")
    print(f"Upper-right corner: ({xll + ncols * cellsize:.4f}, {yll + nrows * cellsize:.4f})")
    print(f"Cell size: {cellsize:.6f} degrees")
    
    # Find data coverage
    valid_mask = data != nodata_value
    valid_count = np.sum(valid_mask)
    total_cells = nrows * ncols
    
    print(f"\n=== DATA COVERAGE ===")
    print(f"Total cells: {total_cells:,}")
    print(f"Valid data cells: {valid_count:,} ({100*valid_count/total_cells:.1f}%)")
    print(f"NoData cells: {total_cells - valid_count:,} ({100*(total_cells - valid_count)/total_cells:.1f}%)")
    
    if valid_count > 0:
        valid_data = data[valid_mask]
        print(f"\n=== PRECIPITATION VALUES (where data exists) ===")
        print(f"Min: {np.min(valid_data):.1f}")
        print(f"Max: {np.max(valid_data):.1f}")
        print(f"Mean: {np.mean(valid_data):.1f}")
        print(f"Median: {np.median(valid_data):.1f}")
        print(f"Std Dev: {np.std(valid_data):.1f}")
        
        # Find bounding box of valid data
        valid_rows, valid_cols = np.where(valid_mask)
        min_row, max_row = valid_rows.min(), valid_rows.max()
        min_col, max_col = valid_cols.min(), valid_cols.max()
        
        # Convert to geographic coordinates
        # Remember: row 0 is at the TOP (north)
        data_west = xll + min_col * cellsize
        data_east = xll + (max_col + 1) * cellsize
        data_north = yll + (nrows - min_row) * cellsize
        data_south = yll + (nrows - max_row - 1) * cellsize
        
        print(f"\n=== VALID DATA EXTENT ===")
        print(f"Geographic bounds of valid data:")
        print(f"  West: {data_west:.4f}°")
        print(f"  East: {data_east:.4f}°")
        print(f"  South: {data_south:.4f}°")
        print(f"  North: {data_north:.4f}°")
        print(f"Grid indices with data:")
        print(f"  Rows: {min_row} to {max_row}")
        print(f"  Cols: {min_col} to {max_col}")
    
    return data, header

def check_nashville_region(data, header, nodata_value=-9):
    """Check if Nashville region has data."""
    # Nashville bounding box
    nashville_bbox = {
        'west': -87.1,
        'east': -86.5,
        'south': 35.9,
        'north': 36.3
    }
    
    nrows = int(header['nrows'])
    ncols = int(header['ncols'])
    xll = header['xllcorner']
    yll = header['yllcorner']
    cellsize = header['cellsize']
    
    # Calculate Nashville's position in the grid
    col_start = int((nashville_bbox['west'] - xll) / cellsize)
    col_end = int((nashville_bbox['east'] - xll) / cellsize) + 1
    
    # Row calculation (row 0 is north)
    row_start_from_bottom = int((nashville_bbox['south'] - yll) / cellsize)
    row_end_from_bottom = int((nashville_bbox['north'] - yll) / cellsize) + 1
    
    row_start = nrows - row_end_from_bottom
    row_end = nrows - row_start_from_bottom
    
    # Ensure indices are within bounds
    row_start = max(0, row_start)
    row_end = min(nrows, row_end)
    col_start = max(0, col_start)
    col_end = min(ncols, col_end)
    
    print(f"\n=== NASHVILLE REGION ===")
    print(f"Nashville bbox: {nashville_bbox}")
    print(f"Grid indices for Nashville:")
    print(f"  Rows: {row_start} to {row_end}")
    print(f"  Cols: {col_start} to {col_end}")
    
    if row_start < row_end and col_start < col_end:
        nashville_data = data[row_start:row_end, col_start:col_end]
        valid_mask = nashville_data != nodata_value
        valid_count = np.sum(valid_mask)
        
        print(f"\nNashville region grid size: {nashville_data.shape}")
        print(f"Valid data cells in Nashville: {valid_count} out of {nashville_data.size}")
        
        if valid_count > 0:
            valid_data = nashville_data[valid_mask]
            print(f"\nNashville precipitation statistics:")
            print(f"  Min: {np.min(valid_data):.1f}")
            print(f"  Max: {np.max(valid_data):.1f}")
            print(f"  Mean: {np.mean(valid_data):.1f}")
            print(f"  Median: {np.median(valid_data):.1f}")
            
            # Show sample values
            print(f"\nSample values from Nashville region:")
            sample_values = valid_data[:10] if len(valid_data) > 10 else valid_data
            print(f"  {sample_values}")
        else:
            print(f"\n⚠️ WARNING: Nashville region contains only NODATA values!")
            print(f"The downloaded grid may not cover Nashville with valid data.")
    else:
        print(f"\n❌ ERROR: Nashville region is outside the grid bounds!")

def main():
    # Check for the 100yr 24hr grid
    grid_file = Path("data/v2_additional/precipitation_grids/se100yr24ha/se100yr24ha.asc")
    
    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        return 1
    
    print(f"Analyzing NOAA grid: {grid_file}")
    
    # Read and analyze the grid
    data, header = read_ascii_grid(grid_file)
    analyze_grid_coverage(data, header)
    check_nashville_region(data, header)
    
    # Visual representation of data coverage
    print(f"\n=== DATA COVERAGE MAP ===")
    print("Creating a simple ASCII visualization (# = data, . = nodata):")
    
    # Downsample for visualization
    step = max(data.shape[0] // 50, data.shape[1] // 100, 1)
    downsampled = data[::step, ::step]
    
    for row in downsampled[:30]:  # Show first 30 rows
        line = ""
        for val in row[:100]:  # Show first 100 cols
            line += "#" if val != -9 else "."
        print(line)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())