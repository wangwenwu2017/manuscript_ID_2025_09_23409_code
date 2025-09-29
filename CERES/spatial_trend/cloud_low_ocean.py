import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from scipy.stats import theilslopes, kendalltau, norm

year_num = 13
year_start = 2011
# Define file paths
file1 = '../data/CERES_SSF1deg-Month_Terra-MODIS_Ed4.1_Subset_200003-202407.nc'
file3 = '../../CERES/data/CERES_grid_landmask_180x360.nc'

# Load datasets
ds1 = xr.open_dataset(file1)
ds3 = xr.open_dataset(file3)
landmask = np.array(ds3['landmask'])

# Extract target time period (2011-2023)
target_start = '2011-01-01'
target_end = '2023-12-31'

# Extract cloud data variables
cf_high = ds1['cldarea_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midhigh = ds1['cldarea_mid_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_low = ds1['cldarea_low_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midlow = ds1['cldarea_mid_low_daynight_mon'].sel(time=slice(target_start, target_end))

# Reshape and combine cloud data
cf_high = np.array(cf_high).reshape((year_num, 12, 180, 360))
cf_low = np.array(cf_low).reshape((year_num, 12, 180, 360))
cf_mid = np.array(cf_midhigh + cf_midlow).reshape((year_num, 12, 180, 360))
total_cloud = np.nanmean(cf_low + cf_mid, axis=1)

# Initialize result arrays
beta_grid = np.full((180, 360), np.nan)  # Slope grid (initialize with NaN)

# Create coordinate grids
lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Create mask for processing (land points between -70 and 70 latitude)
process_mask = (landmask >= 90) & (np.abs(lat_grid) <= 70)

# Process each grid cell
for i in range(180):
    for j in range(360):
        if not process_mask[i, j]:
            continue  # Skip non-land or high-latitude points

        ts = total_cloud[:, i, j]
        if np.isnan(ts).any():
            continue  # Skip if any NaNs in time series

        # Theil-Sen slope calculation
        years = np.arange(year_start, year_start + year_num)
        slope, _, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10  # Convert to decadal trend

# ===== Visualization =====
# Custom colormap
colors = ["#015493", "#FFFFFF", "#C5272D"]  # Dark blue, white, bright red
cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_div", colors, N=101)

# Set plot styles
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Create figure and map projection
fig = plt.figure(figsize=(8, 5), dpi=500)
proj = ccrs.Robinson(central_longitude=180)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Plot cloud trend data
vmin, vmax = -6, 6
mesh = ax.pcolormesh(
    lon, lat, beta_grid,
    vmin=vmin, vmax=vmax,
    cmap=cmap_custom,
    shading='auto',
    transform=ccrs.PlateCarree()
)

# Add geographic features
ax.coastlines(linewidth=0.5, color='black')

# Add gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
gl.top_labels = True
gl.left_labels = True
gl.bottom_labels = False
gl.right_labels = False
gl.xlocator = plt.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

# Add colorbar
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Low/Mid Cloud Trend (% per decade)', fontsize=12)
cbar.set_ticks([vmin, vmin / 2, 0, vmax / 2, vmax])

# Add title
ax.set_title('Low/Mid Cloud Cover Trend Over Land (2011-2023)', fontsize=14, pad=10)

# Adjust layout and show plot
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make space for colorbar
plt.show()