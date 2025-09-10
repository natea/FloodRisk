# NOAA Atlas 14 Spatial Grid Manual Download Instructions

## Important Note
The NOAA precipitation grids are large files that may require manual download
through the NOAA HDSC web interface due to access restrictions.

## Step-by-Step Download Process:

### 1. Navigate to NOAA PFDS GIS Page
Go to: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

### 2. Select Region
- Click on "Southeastern States (NOAA Atlas 14 Volume 2 Version 3)"
- This covers Tennessee including Nashville

### 3. Download Options

#### Option A: Download Complete Dataset (Recommended)
1. Look for "Download all GIS files" or similar option
2. This will give you a large ZIP file with all return periods and durations
3. File size: ~500MB - 1GB

#### Option B: Download Individual Files
Download these specific files for Nashville flood modeling:

**Critical Files (24-hour duration):**
- `se_2yr_24hr_asc.zip` - 2-year 24-hour
- `se_10yr_24hr_asc.zip` - 10-year 24-hour
- `se_25yr_24hr_asc.zip` - 25-year 24-hour
- `se_100yr_24hr_asc.zip` - 100-year 24-hour
- `se_500yr_24hr_asc.zip` - 500-year 24-hour

**Additional Important Files:**
- `se_100yr_6hr_asc.zip` - 100-year 6-hour
- `se_100yr_1hr_asc.zip` - 100-year 1-hour
- `se_100yr_48hr_asc.zip` - 100-year 48-hour

### 4. File Format Options
- **ASCII Grid (.asc)**: Easiest to work with, human-readable
- **GeoTIFF (.tif)**: More compact, maintains projection info
- **NetCDF (.nc)**: Good for time series data

### 5. After Download
1. Extract ZIP files to: `data/v2_additional/precipitation_grids/`
2. Each ZIP contains:
   - `.asc` or `.tif` file with precipitation depths
   - `.prj` file with projection information
   - `.txt` metadata file

### 6. Alternative Data Sources

If the NOAA site is unavailable, try:

#### USGS Data Portal
https://www.sciencebase.gov/catalog/

Search for: "NOAA Atlas 14 precipitation frequency"

#### NOAA Climate Data Online
https://www.ncdc.noaa.gov/cdo-web/

#### Direct FTP (if available)
ftp://hdsc.nws.noaa.gov/pub/hdsc/data/

## Nashville Specific Information

**Bounding Box for Data Extraction:**
- North: 36.3°N
- South: 35.9°N  
- East: -86.5°W
- West: -87.1°W

**Projection Information:**
- Geographic Coordinate System: GCS_North_American_1983
- Datum: D_North_American_1983
- Prime Meridian: Greenwich
- Angular Unit: Degree

## Verification

After downloading, verify files contain:
1. Header with grid dimensions and cell size
2. NODATA value (usually -999 or -9999)
3. Precipitation values in inches (need to convert to mm)

## Failed Automatic Downloads:

The following files could not be downloaded automatically:
- 2yr_24hr
- 10yr_24hr
- 25yr_24hr
- 100yr_24hr
- 500yr_24hr
- 100yr_6hr
- 100yr_1hr

## Contact Information

If you have issues accessing the data:
- NOAA HDSC Help: hdsc.questions@noaa.gov
- Phone: 301-713-1677 x127

## Data Citation

When using this data, cite as:
"NOAA Atlas 14, Volume 2, Version 3: Precipitation-Frequency Atlas of the United States"
Available at: https://hdsc.nws.noaa.gov/hdsc/pfds/
