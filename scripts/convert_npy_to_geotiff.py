#!/usr/bin/env python
"""
Convert NumPy array (.npy) files to GeoTIFF format for use with rasterio.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_npy_to_geotiff(
    npy_path: str,
    output_path: str,
    bounds: tuple = None,
    crs: str = 'EPSG:4326'
):
    """
    Convert a NumPy array to GeoTIFF format.
    
    Args:
        npy_path: Path to input .npy file
        output_path: Path for output GeoTIFF
        bounds: Tuple of (west, south, east, north) in CRS units
        crs: Coordinate reference system (default: WGS84)
    """
    logger.info(f"Loading NumPy array from {npy_path}")
    data = np.load(npy_path)
    
    # Handle different array dimensions
    if data.ndim == 2:
        height, width = data.shape
        count = 1
        data = data.reshape(1, height, width)
    elif data.ndim == 3:
        # Assume shape is (bands, height, width) or (height, width, bands)
        if data.shape[0] <= 4:  # Likely (bands, height, width)
            count, height, width = data.shape
        else:  # Likely (height, width, bands)
            height, width, count = data.shape
            data = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")
    
    logger.info(f"Array shape: {data.shape} (bands={count}, height={height}, width={width})")
    
    # Calculate transform from bounds if provided
    if bounds:
        west, south, east, north = bounds
        pixel_width = (east - west) / width
        pixel_height = (north - south) / height
        transform = from_origin(west, north, pixel_width, pixel_height)
        logger.info(f"Using provided bounds: {bounds}")
    else:
        # Default to 1 unit per pixel, centered at origin
        # For Nashville area, we'll use approximate coordinates
        # Nashville is approximately at 36.16°N, -86.78°W
        west = -87.1284  # Approximate west bound
        north = 36.2178  # Approximate north bound
        pixel_width = 0.001  # ~111 meters per pixel at this latitude
        pixel_height = 0.001
        transform = from_origin(west, north, pixel_width, pixel_height)
        logger.info(f"Using default Nashville area bounds with pixel size {pixel_width}")
    
    # Determine appropriate data type
    if data.dtype == np.float64:
        dtype = rasterio.float32
    elif data.dtype == np.float32:
        dtype = rasterio.float32
    elif data.dtype == np.int32:
        dtype = rasterio.int32
    elif data.dtype == np.int16:
        dtype = rasterio.int16
    else:
        dtype = rasterio.float32
        data = data.astype(np.float32)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to GeoTIFF
    logger.info(f"Writing GeoTIFF to {output_path}")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress='deflate'
    ) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)
    
    logger.info(f"Successfully converted {npy_path} to {output_path}")
    
    # Verify the output
    with rasterio.open(output_path) as src:
        logger.info(f"Output GeoTIFF info:")
        logger.info(f"  - Shape: {src.shape}")
        logger.info(f"  - Bands: {src.count}")
        logger.info(f"  - CRS: {src.crs}")
        logger.info(f"  - Bounds: {src.bounds}")
        logger.info(f"  - Data type: {src.dtypes[0]}")

def main():
    parser = argparse.ArgumentParser(description='Convert NumPy array to GeoTIFF')
    parser.add_argument('input', help='Input .npy file path')
    parser.add_argument('output', help='Output GeoTIFF file path')
    parser.add_argument('--west', type=float, help='Western bound')
    parser.add_argument('--south', type=float, help='Southern bound')
    parser.add_argument('--east', type=float, help='Eastern bound')
    parser.add_argument('--north', type=float, help='Northern bound')
    parser.add_argument('--crs', default='EPSG:4326', help='Coordinate reference system')
    
    args = parser.parse_args()
    
    bounds = None
    if all([args.west, args.south, args.east, args.north]):
        bounds = (args.west, args.south, args.east, args.north)
    
    convert_npy_to_geotiff(
        args.input,
        args.output,
        bounds=bounds,
        crs=args.crs
    )

if __name__ == '__main__':
    main()