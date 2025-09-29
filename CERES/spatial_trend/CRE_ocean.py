import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
from scipy.stats import theilslopes, kendalltau, norm

year_num = 13
year_start = 2011
# Define file paths
file1 = '../../CERES/data/CERES_EBAF_Ed4.2.1_Subset_200101-202412.nc'
file3 = '../../CERES/data/CERES_grid_landmask_180x360.nc'
combined_ds = xr.open_dataset(file1)
ds3 = xr.open_dataset(file3)
of = np.array(ds3['landmask'])

target_start = '2011-01-01'
target_end = '2023-12-31'

# Extract target time period (January 2004 to December 2013)
clear_toa_down = combined_ds['solar_mon'].sel(time=slice(target_start, target_end))
clear_toa_up = combined_ds['toa_sw_clr_t_mon'].sel(time=slice(target_start, target_end))
all_toa_down = combined_ds['solar_mon'].sel(time=slice(target_start, target_end))
all_toa_up = combined_ds['toa_sw_all_mon'].sel(time=slice(target_start, target_end))
clear_surface_down = combined_ds['sfc_sw_down_clr_t_mon'].sel(time=slice(target_start, target_end))
clear_surface_up = combined_ds['sfc_sw_up_clr_t_mon'].sel(time=slice(target_start, target_end))
all_surface_down = combined_ds['sfc_sw_down_all_mon'].sel(time=slice(target_start, target_end))
all_surface_up = combined_ds['sfc_sw_up_all_mon'].sel(time=slice(target_start, target_end))

toa_below = ((all_toa_down - all_toa_up) - (clear_toa_down - clear_toa_up))
surface_below = ((all_surface_down - all_surface_up) - (clear_surface_down - clear_surface_up))
clear_absorption = ((clear_toa_down - clear_toa_up) - (clear_surface_down - clear_surface_up))
all_absorption = ((all_toa_down - all_toa_up) - (all_surface_down - all_surface_up))
cre = all_absorption - clear_absorption
total_absoprtion = np.array(cre).reshape((year_num, 12, 180, 360))
total_absoprtion = np.nanmean(total_absoprtion, axis=1)

# Initialize result array
beta_grid = np.zeros((180, 360))  # Slope grid
lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
for i in range(180):  # Latitude loop
    for j in range(360):  # Longitude loop
        ts = (total_absoprtion[:, i, j])
        if np.isnan(ts).any() or of[i, j] >= 90:
            # if np.isnan(ts).any():
            beta_grid[i, j] = np.nan
            continue

        # Theil-Sen slope calculation
        years = np.arange(year_start, year_start + year_num)
        slope, intercept, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10  # Convert to decadal trend (20 years → 240 months, decade = 120 months)

import matplotlib.colors as mcolors

colors = ["#015493", "#FFFFFF", "#C5272D"]  # Dark blue, pure white, bright red
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

# Specify sans-serif font family
plt.rcParams['font.family'] = 'sans-serif'
# Prioritize Arial font
plt.rcParams['font.sans-serif'] = ['Arial']
# Fix minus sign display issue
plt.rcParams['axes.unicode_minus'] = False
# Optional: Unified font size settings
plt.rcParams['font.size'] = 12  # Global base font size
plt.rcParams['axes.titlesize'] = 14  # Title font size
plt.rcParams['axes.labelsize'] = 12  # Axis label font size
plt.rcParams['xtick.labelsize'] = 10  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 10  # Y-axis tick font size
fig = plt.figure(figsize=(7, 4), dpi=500)  # Increase width to accommodate two subplots

# Create asymmetric layout using GridSpec
gs = GridSpec(20, 1, figure=fig)  # 1 row, 10 columns
# Right subplot: World map (occupies 8/10 width)
proj = ccrs.Robinson(central_longitude=180)
ax_map = fig.add_subplot(gs[0:17, 0], projection=proj)

# ====================== Right: World Map ======================
# Create map
levels = np.linspace(-2, 2, 41)  # 41 levels
im1 = ax_map.pcolormesh(lon, lat, beta_grid,
                        vmin=-2, vmax=2,
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
                    extend='neither', label='W/m² per decade')
cbar.set_ticks([-2, -1, 0, 1, 2])  # Set tick positions
# Add overall title
plt.suptitle(' ', fontsize=14, y=0.98)
plt.show()