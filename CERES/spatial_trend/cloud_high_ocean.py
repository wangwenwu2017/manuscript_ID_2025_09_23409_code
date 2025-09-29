import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.stats import theilslopes, kendalltau, norm

year_num = 13
year_start = 2011
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
cf_mid = np.array(cf_midlow+cf_midhigh).reshape((year_num, 12, 180, 360))
cf = np.array(cf).reshape((year_num, 12, 180, 360))
total_absoprtion = np.nanmean(cf_high, axis=1)

# Initialize result arrays
beta_grid = np.zeros((180, 360))  # Slope grid

lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
for i in range(180):  # Latitude loop
    for j in range(360):  # Longitude loop
        ts = (total_absoprtion[:, i, j])
        if np.isnan(ts).any() or of[i, j] >= 90:  # Skip ocean points
            beta_grid[i, j] = np.nan
            continue

        # Theil-Sen slope calculation
        years = np.arange(year_start, year_start + year_num)  # 2011-2023
        slope, intercept, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10  # Convert to decadal trend (13 years → 13 years, decade = 10 years)

# Define custom color scheme
colors = ["#1E78B5", "#FFFFFF", "#FF0000"]  # Dark blue, pure white, bright red

# Create custom colormap
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

# Set font styles
plt.rcParams['font.family'] = 'sans-serif'  # Specify sans-serif font family
plt.rcParams['font.sans-serif'] = ['Arial']  # Prioritize Arial font
plt.rcParams['axes.unicode_minus'] = False   # Fix minus sign display issue
# Optional: Unified font size settings
plt.rcParams['font.size'] = 12             # Global base font size
plt.rcParams['axes.titlesize'] = 14         # Title font size
plt.rcParams['axes.labelsize'] = 12         # Axis label font size
plt.rcParams['xtick.labelsize'] = 10        # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 10        # Y-axis tick font size
fig = plt.figure(figsize=(7, 4), dpi=500)  # Increase width to accommodate two subplots

# Create asymmetric layout using GridSpec
gs = GridSpec(20, 1, figure=fig)  # 1 row, 10 columns
# Right subplot: World map (occupies 8/10 width)
proj = ccrs.Robinson(central_longitude=180)
ax_map = fig.add_subplot(gs[0:17, 0], projection=proj)


# ====================== Right: World Map ======================
# Create map
levels = np.linspace(-8, 8, 41)  # 41 levels
im1 = ax_map.pcolormesh(lon, lat, beta_grid,
                        vmin=-8, vmax=8,
                        cmap=cmap_custom,
                        shading='auto',
                        transform=ccrs.PlateCarree(central_longitude=0))

# Geographic features
ax_map.coastlines(linewidth=0.5, color='black')
gl = ax_map.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
gl.bottom_labels = False
gl.right_labels = False
gl.xlocator = plt.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

# Add title
ax_map.set_title(' ', fontsize=12, pad=10)

# Adjust layout
plt.tight_layout()  # Reserve space for top title

# ====================== Colorbar Settings ======================
# Add horizontal colorbar at the bottom
cbar_ax = fig.add_axes([0.22, 0.12, 0.6, 0.025])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal',
                    extend='neither', label='% per decade')
cbar.set_ticks([-8, -4, 0, 4, 8])  # Set tick positions
# Add overall title
plt.suptitle(' ', fontsize=14, y=0.98)
plt.show()