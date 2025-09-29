# manuscript_ID_2025_09_23409_code
code repository for manuscript entitled "Cloud Changes Weaken Atmospheric Solar Absorption Over Land as Climate Warms" (manuscript ID: 2025-09-23409)

DATA availability:

(1)The CERES EBAF4.2.1 radiation data are available through: https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAF421Selection.jsp

(2)The CERES SSF1DEG level3 cloud cover product can be found: https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SSF1degEd42Selection.jsp

(3)The radiation and cloud cover data from ERA5 can be accessed here: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview

(4)The water vapor profile data from ERA5 can be accessed here: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means?tab=overview

(5)The MERRA-2 aerosols radiative forcing data is available here: https://disc.gsfc.nasa.gov/datasets?project=MERRA-2

(6)The CESM-CAM5 radiative kernel can be accessed through: https://zenodo.org/records/997902


Requirements:

numpy, xarray, pandas, cartopy, matplotlib, scipy, pymannkendall, netCDF4, glob, seaborn

Instructions:

The analysis codes based on CERES, ERA5, and CMIP6 data are housed in the CERES, ERA5, and CMIP6 folders respectively, while the code for radiative kernel analysis is located in the radiative_kernel folder.  
The CERES analysis code includes scripts for multi-year mean analysis, trend analysis of annual means, spatial trend distribution analysis and relationship analysis between radiative impact and cloud changes.  
The ERA5 analysis code provides scripts for analyzing spatial trend distributions.  
The CMIP6 analysis code offers scripts for linear trend estimation analysis of time series data.
