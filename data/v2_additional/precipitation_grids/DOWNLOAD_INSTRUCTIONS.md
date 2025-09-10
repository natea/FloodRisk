# NOAA Spatial Precipitation Grid Download Instructions

## Manual Download Required
The NOAA precipitation grids need to be downloaded manually from:
https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

## Steps:
1. Navigate to the PFDS GIS Data page
2. Select "Southeastern States (NOAA Atlas 14 Volume 2)"
3. Download the following GeoTIFF files for Tennessee:

### Required Files:
- se_vol2_2yr_24hr.tif (2-year 24-hour)
- se_vol2_10yr_24hr.tif (10-year 24-hour)
- se_vol2_25yr_24hr.tif (25-year 24-hour)
- se_vol2_100yr_24hr.tif (100-year 24-hour)
- se_vol2_500yr_24hr.tif (500-year 24-hour)

### Additional Durations (if available):
- 1-hour, 6-hour, 48-hour for each return period

## Bounding Box for Nashville:
- North: 36.3°
- South: 35.9°
- East: -86.5°
- West: -87.1°

## Save Location:
Place downloaded files in: data/v2_additional/precipitation_grids/
