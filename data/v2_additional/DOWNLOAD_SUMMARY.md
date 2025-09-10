# FloodRisk v2 Additional Data Requirements

## Data Successfully Generated:
✅ SCS Type II temporal patterns
✅ Nested storm configurations  
✅ Area reduction factors
✅ 2010 flood metadata

## Data Requiring Manual Download:

### 1. NOAA Spatial Precipitation Grids (CRITICAL)
- [ ] 30-arc-second resolution grids for Nashville region
- [ ] Multiple return periods (2, 10, 25, 100, 500-year)
- [ ] Multiple durations (1hr, 6hr, 24hr minimum)
- **Download from**: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html

### 2. Stage IV Radar Data (HIGH PRIORITY)
- [ ] April 30 - May 3, 2010 coverage
- [ ] 4km resolution GRIB2 files
- **Download from**: https://data.eol.ucar.edu/dataset/21.006

### 3. USGS High Water Marks (HIGH PRIORITY)
- [ ] 2010 Nashville flood observations
- [ ] GPS coordinates and elevations
- **Download from**: https://water.usgs.gov/floods/events/2010/nashville/

### 4. Orographic Factors (MEDIUM PRIORITY)
- [ ] NOAA Atlas 14 Volume 2 appendices
- [ ] Tennessee-specific elevation adjustments
- **Source**: NOAA Atlas 14 documentation

## Next Steps:
1. Download spatial precipitation grids (most critical)
2. Obtain 2010 flood validation data
3. Review generated temporal patterns
4. Test data integration with existing pipeline

## Directory Structure:
```
data/v2_additional/
├── precipitation_grids/     # NOAA spatial grids (manual download)
├── temporal_patterns/       # SCS patterns (generated)
├── area_reduction_factors/  # ARF curves (generated)
├── 2010_flood_validation/   # Historical event data (manual download)
└── DOWNLOAD_SUMMARY.md      # This file
```
