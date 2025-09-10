# 2010 Nashville Flood Validation Data

## Required Data Sources:

### 1. Stage IV Radar Rainfall Data
- **Dates**: April 30 - May 3, 2010
- **Download from**: https://data.eol.ucar.edu/dataset/21.006
- **Format**: GRIB2 files (4km resolution)
- **Coverage**: Southeastern US including Nashville

### 2. USGS High Water Marks
- **Source**: https://water.usgs.gov/floods/events/2010/nashville/
- **Data includes**:
  - GPS coordinates of high water marks
  - Measured water surface elevations
  - Time of peak (if available)

### 3. USGS Stream Gauge Data
- **Stations**: Cumberland River and tributaries
- **Download from**: https://waterdata.usgs.gov/nwis
- **Key stations**:
  - 03431500 - Cumberland River at Nashville
  - 03430550 - Mill Creek near Nashville
  - 03431060 - Harpeth River near Kingston Springs

### 4. Observed Flood Extent
- **Source**: USGS or local emergency management
- **Format**: Shapefile or GeoTIFF of flooded areas

### 5. Rainfall Observations
- **Nashville Airport (KBNA)**: Total of 13.57 inches May 1-2
- **Download hourly data from**: https://www.ncdc.noaa.gov/

## Save Location:
Place all downloaded files in: data/v2_additional/2010_flood_validation/
