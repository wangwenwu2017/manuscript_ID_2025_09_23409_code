import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

file = '../../CERES/data/CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-200003.nc'
ds = xr.open_dataset(file)
of = np.array(ds['aux_ocean_mon'])[0]
lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
lon_grid, lat_grid = np.meshgrid(lon, lat)

CRE = np.load('CRE.npy')
Cloud_low = np.load('LC_average.npy')
Cloud_high = np.load('HC_average.npy')

index = np.where((abs(lat_grid) < 70))
CRE = CRE[index]
cloud_high = Cloud_high[index]
cloud_low = Cloud_low[index]

# Define bin boundaries
cloud_high_bins = np.arange(0, 60.01, 12)
cloud_low_bins = np.arange(0, 100.01, 20)

# Calculate bin centers (for labeling)
high_centers = (cloud_high_bins[:-1] + cloud_high_bins[1:]) / 2
low_centers = (cloud_low_bins[:-1] + cloud_low_bins[1:]) / 2

# Initialize statistical matrix (filled with NaN)
mean_grid = np.full((len(high_centers), len(low_centers)), np.nan)  # Note: dimension order swapped

# Traverse 2D bins - note: loop order swapped
for i in range(len(high_centers)):
    for j in range(len(low_centers)):
        # Filter data points in current bin
        mask = (
                (cloud_high >= cloud_high_bins[i]) &
                (cloud_high < cloud_high_bins[i + 1]) &
                (cloud_low >= cloud_low_bins[j]) &
                (cloud_low < cloud_low_bins[j + 1])
        )
        if np.sum(mask) > 100:  # Minimum sample size threshold
            mean_grid[i, j] = np.nanmean(CRE[mask])  # Note: index order swapped

# Set fonts and styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Create figure
plt.figure(figsize=(6, 3), dpi=500)  # Adjust figure size for swapped axes

# Create custom colormap
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "custom_div",
    [(0, "#015493"), (1 / 3, "#FFFFFF"), (1, "#C5272D")],
    N=101
)

# Plot heatmap - swap X and Y axes
mesh = plt.pcolormesh(
    cloud_low_bins,  # X-axis now represents low cloud cover
    cloud_high_bins,  # Y-axis now represents high cloud cover
    mean_grid,  # Data matrix remains unchanged
    shading='auto',
    cmap=cmap_custom,
    edgecolor='k',
    linewidth=0.5,
    vmin=-4, vmax=8
)

# Add colorbar
cbar = plt.colorbar(mesh)
cbar.set_label('SW CRE on $A_{atm}$ (W/$m^2$)', fontsize=15)

# Label axes - swap labels
plt.xlabel('Low/Middle cloud cover (%)', fontsize=18, fontweight='bold')
plt.ylabel('High cloud cover (%)', fontsize=18, fontweight='bold')

# Annotate each bin with sample size and mean value
for i in range(len(high_centers)):
    for j in range(len(low_centers)):
        if not np.isnan(mean_grid[i, j]):  # Note: index order swapped
            # Calculate current bin sample size
            mask = (
                    (cloud_high >= cloud_high_bins[i]) &
                    (cloud_high < cloud_high_bins[i + 1]) &
                    (cloud_low >= cloud_low_bins[j]) &
                    (cloud_low < cloud_low_bins[j + 1])
            )
            count = np.sum(mask)
            value = mean_grid[i, j]

            # Adjust text color based on background
            text_color = 'k' if value < 0 else 'k'

            # Annotation position - swap coordinates
            plt.text(
                low_centers[j],  # X position is low cloud center
                high_centers[i],  # Y position is high cloud center
                f'{value:.2f}\n(n={count})',
                ha='center', va='center',
                color=text_color, fontsize=13)

plt.tight_layout()
# plt.savefig('Cloud_CRE_Relationship.png', dpi=500, bbox_inches='tight')
plt.show()