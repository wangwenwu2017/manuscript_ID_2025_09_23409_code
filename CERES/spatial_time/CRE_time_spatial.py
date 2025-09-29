import xarray as xr
import numpy as np
from scipy.stats import theilslopes, kendalltau, norm

year_num = 13
year_start = 2011
# Define file paths
file1 = '../../CERES/data/CERES_EBAF_Ed4.2.1_Subset_200101-202412.nc'  # March 2000 to January 2013
file3 = '../../CERES/data/CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-200003.nc'
combined_ds = xr.open_dataset(file1)
ds3 = xr.open_dataset(file3)
of = np.array(ds3['aux_ocean_mon'])[0]

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
cre = np.array(((all_toa_down - all_toa_up) - (all_surface_down - all_surface_up)-((clear_toa_down - clear_toa_up)-(clear_surface_down - clear_surface_up))))
total_absoprtion = np.array(cre).reshape((year_num, 12, 180, 360))
index = np.where(of>90)
total_absoprtion[:,:,index[0], index[1]] = np.nan
total_absoprtion = np.nanmean(total_absoprtion, axis=(3))

beta_grid = np.zeros((12, 60))  # Slope grid
z_grid = np.zeros((12, 60))  # Mann-Kendall test Z-value grid
p_grid = np.zeros((12, 60))

lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
for i in range(12):  # Latitude loop
    for j in range(60):  # Longitude loop
        ts = (np.mean(total_absoprtion[:, i, 3*j:3*j+3], axis=1))
        if np.isnan(ts).any():
            beta_grid[i, j] = np.nan
            z_grid[i, j] = np.nan
            continue

        # Theil-Sen slope calculation
        years = np.arange(year_start, year_start + year_num)  # 2011-2023
        slope, intercept, _, _ = theilslopes(ts, years)
        beta_grid[i, j] = slope * 10  # Convert to decadal trend (20 years → 240 months, decade = 120 months)

        # Mann-Kendall test
        tau, p_value = kendalltau(np.arange(year_num), ts)
        z_grid[i, j] = np.sign(tau) * norm.ppf(1 - p_value / 2)  # Calculate Z-value
significant_mask = np.abs(z_grid) > 1.96  # |Z|>1.96 → p<0.05
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Generate example data (12 years × 60 latitudes)
years = 12
latitudes = 60

data = beta_grid

# Set year and latitude coordinates
start_year = 2011
year_labels = [f"{start_year + i}" for i in range(years)]
lat_values = np.linspace(-90, 90, latitudes)  # From 90°S to 90°N

# Create professional heatmap
plt.figure(figsize=(4, 6), dpi=500)
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
ax = plt.gca()
import matplotlib.colors as mcolors
colors = ["#015493", "#FFFFFF", "#C5272D"]  # Dark blue, pure white, bright red
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", colors, N=101
)

# Plot heatmap using pcolormesh
cmesh = plt.pcolormesh(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], lat_values, data.T,  # Transpose data to match axes
    shading='auto',
    cmap=cmap_custom,
    vmax=1.5,
    vmin=-1.5
)

# Add colorbar
cbar = plt.colorbar(cmesh, ax=ax, pad=0.05, aspect=40)
cbar.set_label('W/m²/decade', fontsize=12)
significant_data = significant_mask
for i in range(significant_data.shape[0]):
    for j in range(significant_data.shape[1]):
        if significant_data[i, j]:
            # Calculate background brightness, choose contrasting color
            ax.scatter(i, lat_values[j], s=15, color='black', edgecolors='none', zorder=3)

# Set axis ticks and labels
plt.xlabel('Month', fontsize=12, labelpad=10)
plt.ylabel('Lat', fontsize=12, labelpad=10)

# Set x-axis ticks (months)
plt.xticks(['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec'], rotation=45)
# Set y-axis ticks (latitudes)
plt.yticks(np.arange(-90, 91, 30),
           ['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N'],
           fontsize=10)
plt.ylim(-70, 70)
# Add auxiliary grid lines
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(10))

# Add equator reference line
plt.axhline(y=0, color='k', linestyle='-', alpha=0.7, linewidth=1.5)
# Adjust layout
plt.tight_layout()

# Show plot
plt.show()