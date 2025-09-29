import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.stats import theilslopes, kendalltau, norm

year_num = 13
year_start = 2001
# Define file paths
file1 = '../data/CERES_SSF1deg-Month_Terra-MODIS_Ed4.1_Subset_200003-202407.nc'
file3 = '../../CERES/data/CERES_grid_landmask_180x360.nc'

ds1 = xr.open_dataset(file1)
ds3 = xr.open_dataset(file3)
of = np.array(ds3['landmask'])

# Extract target time period
target_start = '2011-01-01'
target_end = '2023-12-31'
cf_high = ds1['cldarea_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midhigh = ds1['cldarea_mid_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_low = ds1['cldarea_low_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midlow = ds1['cldarea_mid_low_daynight_mon'].sel(time=slice(target_start, target_end))
cf = ds1['cldarea_total_daynight_mon'].sel(time=slice(target_start, target_end))

cf_high = np.array(cf_high).reshape((year_num, 12, 180, 360))
cf_low = np.array(cf_low).reshape((year_num, 12, 180, 360))
cf_mid = np.array(cf_midhigh+cf_midlow).reshape((year_num, 12, 180, 360))
cf = np.array(cf).reshape((year_num, 12, 180, 360))
total_absoprtion = np.nanmean(cf_low+cf_mid, axis=1)

# Initialize result arrays
beta_grid = np.zeros((180, 360))  # Slope grid

lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
for i in range(180):  # Latitude loop
    for j in range(360):  # Longitude loop
        ts = (total_absoprtion[:, i, j])
        if np.isnan(ts).any() or of[i, j] < 90:  # Exclude ocean points
            beta_grid[i, j] = np.nan
            continue

        # Theil-Sen slope calculation
        years = np.arange(year_start, year_start + year_num)  # 2001-2013
        slope, intercept, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10  # Convert to decadal trend (13 years → 13 years, decade = 10 years)



# ===== Visualization settings =====
# Custom colormap
colors = ["#015493", "#FFFFFF", "#C5272D"]  # Dark blue, pure white, bright red
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

# Set fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create figure
fig = plt.figure(figsize=(6, 4), dpi=500)  # Increase height to accommodate colorbar

# Create map projection
proj = ccrs.Robinson(central_longitude=180)
ax_map = fig.add_subplot(1, 1, 1, projection=proj)

# ===== Plot map =====
# Set data range
vmin, vmax = -6, 6

# Plot filled contours
im1 = ax_map.pcolormesh(
    lon, lat, beta_grid,
    vmin=vmin, vmax=vmax,
    cmap=cmap_custom,
    shading='auto',
    transform=ccrs.PlateCarree()
)

# Add geographic features
ax_map.coastlines(linewidth=0.5, color='black')

# Add gridlines (only show left and top labels)
gl = ax_map.gridlines(
    draw_labels=True,
    linestyle='--',
    alpha=0.0,
    color='gray'
)

# Only show top and left labels
gl.top_labels = True
gl.left_labels = True
gl.bottom_labels = False
gl.right_labels = False

# Set gridline positions
gl.xlocator = plt.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

# ===== Add colorbar =====
cbar_ax = fig.add_axes([0.22, 0.125, 0.6, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(
    im1,
    cax=cbar_ax,
    orientation='horizontal',
    extend='both',
    label='Trend (% per decade)'
)
cbar.set_ticks([vmin, vmin/2, 0, vmax/2, vmax])  # Set tick positions

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Make space for colorbar
plt.show()