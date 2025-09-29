import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

year_num = 11
year_start = 2013
# Define file paths
file1 = '../data/CERES_EBAF_Ed4.2.1_Subset_200101-202412.nc'
combined_ds = xr.open_dataset(file1)

target_start = '2003-01-01'
target_end = '2013-12-31'

# Extract target time period (January 2004 - December 2013)
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
cre = np.array(((all_toa_down - all_toa_up) - (all_surface_down - all_surface_up)-((clear_toa_down - clear_toa_up)-(clear_surface_down - clear_surface_up))))

total_absoprtion = np.nanmean(cre, axis=0)
np.save('CRE.npy', total_absoprtion)

lat = np.array(combined_ds['lat'])
lon = np.array(combined_ds['lon'])
colors = ["#015493", "#FFFFFF", "#C5272D"]
color_nodes = [-5, 0, 15]
bounds = np.array([-5, 0, 0, 15])  # Boundary list: [-5, 0, 0, 15]
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors))

# 3. Create non-uniform colormap
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div",
    [(0, "#015493"), (0.25, "#FFFFFF"), (1, "#C5272D")],  # Manually specified positions
    N=101
)

# Specify sans-serif font family
plt.rcParams['font.family'] = 'sans-serif'
# Prioritize Arial font
plt.rcParams['font.sans-serif'] = ['Arial']
# Fix minus sign display issue
plt.rcParams['axes.unicode_minus'] = False
# Optional: Unified font size settings
plt.rcParams['font.size'] = 12             # Global base font size
plt.rcParams['axes.titlesize'] = 14         # Title font size
plt.rcParams['axes.labelsize'] = 12         # Axis label font size
plt.rcParams['xtick.labelsize'] = 10        # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 10        # Y-axis tick font size
fig = plt.figure(figsize=(8, 5), dpi=500)  # Increase width to accommodate two subplots

# Right subplot: World map (occupies 8/10 width)
proj = ccrs.Robinson(central_longitude=180)
ax_map = fig.add_subplot( projection=proj)
levels = np.linspace(-5, 15, 61)  # 61 levels
np.save('ACRE_ceres_10', total_absoprtion)
# im1 = ax_map.contourf(lon, lat, total_absoprtion,
#                   levels=levels,
#                   cmap=cmap_custom,  # Reverse color scale from red to blue
#                   extend='both',  # Extend both ends
#                   transform=ccrs.PlateCarree(central_longitude=0))

im1 = ax_map.pcolormesh(lon, lat, total_absoprtion,
                  vmin=-5, vmax=15,
                  cmap=cmap_custom,  # Reverse color scale from red to blue
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
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Reserve space for top title

# ====================== Colorbar settings ======================
# Add horizontal colorbar at the bottom
cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical',
                    extend='neither', label='W/m²')
cbar.set_ticks([-5, 0, 5, 10, 15])  # Set tick positions
# Add overall title
plt.suptitle(' ', fontsize=14, y=0.98)
plt.show()