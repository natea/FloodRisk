# NOAA Atlas 14 Data Download Options

## Option 1: AWS CLI (Recommended)
If you have AWS CLI installed, you can try:

```bash
# Install AWS CLI if needed
pip install awscli

# Configure for public access
aws configure set aws_access_key_id ""
aws configure set aws_secret_access_key ""

# Try to list NOAA buckets
aws s3 ls s3://noaa-atlas14/ --no-sign-request
aws s3 ls s3://noaa-nws-ofs-pds/atlas14/ --no-sign-request

# Download specific files (if found)
aws s3 cp s3://noaa-atlas14/se/100yr_24hr.tif . --no-sign-request
```

## Option 2: Direct HTTP Download
Some NOAA data is available via HTTP:

```bash
# NOAA NCEI Archive
wget https://www.ncei.noaa.gov/data/precipitation-frequency/access/

# NOAA PSL (Physical Sciences Laboratory)
wget https://downloads.psl.noaa.gov/Datasets/
```

## Option 3: THREDDS Data Server
NOAA operates THREDDS servers for data access:

- https://www.ncei.noaa.gov/thredds/catalog.html
- https://psl.noaa.gov/thredds/catalog.html

## Option 4: Google Earth Engine
NOAA Atlas 14 may be available on Google Earth Engine:

```python
import ee
ee.Initialize()

# Search for NOAA Atlas 14
noaa = ee.ImageCollection('NOAA/ATLAS14')
```

## Option 5: Manual Web Download
As a last resort, use the web interface:

1. Go to: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html
2. Select "Southeastern States (NOAA Atlas 14 Volume 2)"
3. Look for "GIS Data" or "Download" sections
4. Download files for Tennessee region

## Alternative Precipitation Datasets

### PRISM Climate Data
- URL: https://prism.oregonstate.edu/
- Resolution: 4km
- Coverage: CONUS
- Format: BIL, ASCII Grid

### NASA GPM IMERG
- URL: https://gpm.nasa.gov/data/imerg
- Resolution: 0.1 degree
- Coverage: Global
- Format: HDF5, NetCDF

### Stage IV Radar
- URL: https://data.eol.ucar.edu/dataset/21.006
- Resolution: 4km
- Coverage: CONUS
- Format: GRIB2

### CPC Unified Precipitation
- URL: https://psl.noaa.gov/data/gridded/data.unified.daily.conus.html
- Resolution: 0.25 degree
- Coverage: CONUS
- Format: NetCDF

## Contact for Data Access
- NOAA HDSC: hdsc.questions@noaa.gov
- NOAA Big Data Program: noaa.bigdata@noaa.gov
