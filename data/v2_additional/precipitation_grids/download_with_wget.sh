#!/bin/bash
# NOAA Atlas 14 Download Script
# Run this script to attempt downloads using wget

mkdir -p grids
cd grids

# se_24hr_100yr
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr_24hr_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr_24hr_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds2/orb/se/se_100yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds2/orb/se/se_100yr24h_asc.zip'

# se_24hr_500yr
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr_24hr_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_500yr_24hr_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_500yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_500yr24h_asc.zip'

# se_24hr_25yr
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr_24hr_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_25yr_24hr_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_25yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_25yr24h_asc.zip'

# se_24hr_10yr
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr_24hr_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_10yr_24hr_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_10yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_10yr24h_asc.zip'

# se_24hr_2yr
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr24h_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr_24hr_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_2yr_24hr_asc.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_2yr24h_asc.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_2yr24h_asc.zip'

# se_24hr_100yr_tif
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/pub/hdsc/data/se/se_100yr24h.zip'
wget -nc -t 3 --user-agent='Mozilla/5.0' 'https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h.zip' || echo 'Failed: https://hdsc.nws.noaa.gov/hdsc/pfds/orb/se/se_100yr24h.zip'

echo 'Download attempts complete'
ls -la *.zip 2>/dev/null || echo 'No files downloaded'
