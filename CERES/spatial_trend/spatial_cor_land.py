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

CRE = np.load('land/CRE.npy')
Cloud_low = np.load('land/Cloud_low.npy')
Cloud_high = np.load('land/Cloud_high.npy')
CRE_trend = np.load('land/CRE_trend.npy')


index = np.where((abs(lat_grid)<70)&(CRE_trend==CRE_trend))
cloud_high = Cloud_high[index]
cloud_low = Cloud_low[index]
cre_trend = CRE_trend[index]


import numpy as np
import matplotlib.pyplot as plt


# Define bin boundaries
cloud_high_bins = np.arange(-2., 2.51, 1.5)  # [-0.5, 0.0, 0.5, 1.0, 1.5]
cloud_low_bins = np.arange(-3, 3.01, 2)   # [-1.0, -0.5, 0.0, 0.5, 1.0]
# Calculate bin centers (for labeling)
high_centers = (cloud_high_bins[:-1] + cloud_high_bins[1:]) / 2
low_centers = (cloud_low_bins[:-1] + cloud_low_bins[1:]) / 2

# Initialize statistical matrix (filled with NaN)
mean_grid = np.full((len(low_centers), len(high_centers)), np.nan)

# Traverse 2D bins
for i in range(len(high_centers)):
    for j in range(len(low_centers)):
        # Filter data points in current bin
        mask = (
            (cloud_high >= cloud_high_bins[i]) &
            (cloud_high < cloud_high_bins[i+1]) &
            (cloud_low >= cloud_low_bins[j]) &
            (cloud_low < cloud_low_bins[j+1])
        )
        if np.sum(mask) > 100:  # Minimum sample size threshold=100
            mean_grid[j, i] = np.nanmean(cre_trend[mask])

# Create grid coordinates
X, Y = np.meshgrid(high_centers, low_centers)
# Specify sans-serif font family
plt.rcParams['font.family'] = 'sans-serif'
# Prioritize Arial font
plt.rcParams['font.sans-serif'] = ['Arial']
# Fix minus sign display issue
plt.rcParams['axes.unicode_minus'] = False
# Optional: Unified font size settings
plt.rcParams['font.size'] = 12             # Global base font size
plt.rcParams['axes.titlesize'] = 14         # Title font size
plt.rcParams['axes.labelsize'] = 16         # Axis label font size
plt.rcParams['xtick.labelsize'] = 14        # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 14        # Y-axis tick font size
# Plot heatmap
plt.figure(figsize=(5, 4), dpi=500)

colors = [
        (0.00, 0.25, 0.75),  # Dark navy blue
        (0.05, 0.35, 0.80),  # Navy blue
        (0.10, 0.45, 0.85),  # Dark blue
        (0.25, 0.55, 0.88),  # Blue
        (0.40, 0.65, 0.90),  # Medium blue
        (0.55, 0.75, 0.92),  # Light blue
        (0.70, 0.85, 0.95),  # Pale blue
        (0.85, 0.92, 0.98),  # Light sky blue
        (0.95, 0.98, 1.00)   # Very light blue (almost white)
    ]
colorsss = LinearSegmentedColormap.from_list("premium_blue", colors)
mesh = plt.pcolormesh(
    cloud_high_bins, cloud_low_bins, mean_grid,
    shading='auto',
    cmap=colorsss,  # Blue color scale
    edgecolor='k',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(mesh)
cbar.set_label('SW CRE on $A_{atm}$ trend', fontsize=18)

# Label axes and title
plt.xlabel('High cloud cover trend', fontsize=18, fontweight='bold')
plt.ylabel('Low/Mid cloud cover trend', fontsize=19, fontweight='bold')

# Annotate each bin with sample size and mean value (optional)
for i in range(len(high_centers)):
    for j in range(len(low_centers)):
        if not np.isnan(mean_grid[j, i]):
            # Calculate current bin sample size
            mask = (
                (cloud_high >= cloud_high_bins[i]) &
                (cloud_high < cloud_high_bins[i+1]) &
                (cloud_low >= cloud_low_bins[j]) &
                (cloud_low < cloud_low_bins[j+1])
            )
            count = np.sum(mask)
            value = mean_grid[j, i]
            # Adjust text color based on background
            text_color = 'k' if value < 0 else 'k'
            plt.text(
                high_centers[i], low_centers[j],
                f'{value:.2f}\n(n={count})',
                ha='center', va='center',
                color=text_color, fontsize=18,fontweight='bold')

plt.tight_layout()
plt.show()
