import netCDF4 as nc
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

def weighted_global_avg(data_3d, weight_2d):
    """
    Calculate weighted global average
    :param data_3d: 3D array (time, latitude, longitude)
    :param weight_2d: 2D weights (latitude, longitude)
    :return: Global average for each timestep
    """
    # Handle missing values (NaN)
    valid_mask = ~np.isnan(data_3d)

    # Calculate total valid weights
    total_weights = np.sum(weight_2d * valid_mask, axis=(1, 2))

    # Calculate weighted sum
    weighted_sum = np.nansum(data_3d * weight_2d, axis=(1, 2))

    # Calculate monthly averages
    return weighted_sum / total_weights

# Load ERA5 data
era5_data = nc.Dataset('f1f8443027384fea70532eacd14ffc1.nc')
hcc = np.array(era5_data['hcc'][204:360])  # High cloud cover
lcc = np.array(era5_data['lcc'][204:360])  # Low cloud cover
mcc = np.array(era5_data['mcc'][204:360])  # Medium cloud cover
lat = np.array(era5_data['latitude'])
lon = np.array(era5_data['longitude'])

# Process medium cloud cover data
total_absorption = (mcc).reshape(13, 12, 721, 1440)
enhanced_absorption = np.nanmean(total_absorption, axis=1)

# Load land-sea mask
era5_data_ocean = nc.Dataset('../ERA5/b2ea2f963a76db080a917ef4846c082f.nc')
lsm = np.array(era5_data_ocean['lsm'])[0]

from scipy.stats import theilslopes, kendalltau, norm

# Initialize result array
beta_grid = np.zeros((721, 1440))  # Slope grid

# Calculate trends for each grid point
for i in range(721):  # Latitude loop
    for j in range(1440):  # Longitude loop
        ts = enhanced_absorption[:, i, j] # Single point time series
        if np.isnan(ts).any() or lsm[i, j] == 0 or abs(lat[i]>70):
            beta_grid[i, j] = np.nan
            continue

        # Theil-Sen slope calculation
        years = np.arange(2011, 2024)  # 2011-2023 years
        slope, intercept, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10 * 100  # Convert to decadal trend and percentage

import matplotlib.colors as mcolors
colors = ["#1E78B5", "#FFFFFF", "#FF0000"]  # Dark blue, pure white, bright red

# Create custom colormap
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

colors = ["#015493", "#FFFFFF", "#C5272D"]  # Dark blue, pure white, bright red

# Create custom colormap
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

# Set font properties
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
levels = np.linspace(-10, 10, 41)  # 41 levels
im1 = ax_map.pcolormesh(
    lon, lat, beta_grid,
    vmin=-10, vmax=10,
    cmap=cmap_custom,
    shading='auto',
    transform=ccrs.PlateCarree(central_longitude=0)
)

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
cbar = fig.colorbar(
    im1,
    cax=cbar_ax,
    orientation='horizontal',
    extend='neither',
    label='% per decade'
)
cbar.set_ticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])  # Set tick positions
# Add overall title
plt.suptitle(' ', fontsize=14, y=0.98)
plt.show()