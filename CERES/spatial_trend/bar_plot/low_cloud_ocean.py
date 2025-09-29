import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# Load data (assuming these files already exist)
file = '../../data/CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-200003.nc'
ds = xr.open_dataset(file)
of = np.array(ds['aux_ocean_mon'])[0]
lon = np.linspace(0, 360, 360)
lat = np.linspace(-89.5, 89.5, 180)
lon_grid, lat_grid = np.meshgrid(lon, lat)

CRE = np.load('../land/CRE.npy')
Cloud_low = np.load('../ocean/Cloud_low_ocean.npy')
Cloud_high = np.load('../ocean/Cloud_high_ocean.npy')
CRE_trend = np.load('../ocean/CRE_trend.npy')

# Filter data (excluding high-latitude regions)
index = np.where((abs(lat_grid) < 70) & (~np.isnan(CRE_trend)))
cloud_high = Cloud_high[index]
cloud_low = Cloud_low[index]
cre_trend = CRE_trend[index]

# Set font styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define bin boundaries for low/mid cloud cover trend
cloud_low_bins = np.arange(-4, 5.01, 1.8)  # [-2.0, -1.7, -1.4, ..., 0.7, 1.0]
low_centers = (cloud_low_bins[:-1] + cloud_low_bins[1:]) / 2  # Bin center points

# Initialize statistical arrays
mean_values = np.full(len(low_centers), np.nan)  # Mean values
std_values = np.full(len(low_centers), np.nan)  # Standard deviation
count_values = np.zeros(len(low_centers), dtype=int)  # Sample count

# Traverse bins of low/mid cloud cover trend
for j in range(len(low_centers)):
    # Filter data points in the current bin
    mask = (
            (cloud_low >= cloud_low_bins[j]) &
            (cloud_low < cloud_low_bins[j + 1])
    )
    count = np.sum(mask)
    count_values[j] = count

    if count > 50:  # Minimum sample size threshold
        mean_values[j] = np.nanmean(cre_trend[mask])
        std_values[j] = np.nanstd(cre_trend[mask])

# Create figure
plt.figure(figsize=(6, 4), dpi=500)

# Plot bar chart (mean values)
bars = plt.bar(low_centers, mean_values, width=1.2,
               color='#be588d', alpha=0.4, edgecolor='#be588d')  # Use magenta for low/mid cloud

# Add mean value labels
for j, center in enumerate(low_centers):
    if not np.isnan(mean_values[j]):
        # Calculate label position (adjusted based on positive/negative value)
        if mean_values[j] >= 0:
            # Positive value: place above the bar top
            label_y = mean_values[j] + 0.04
            va = 'bottom'
        else:
            # Negative value: place below the bar top
            label_y = mean_values[j] - 0.07
            va = 'top'

        # Add mean value label
        plt.text(
            center, label_y,
            f'{mean_values[j]:.2f}',
            ha='center', va=va, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round, pad=0.2', facecolor='white', alpha=0.8)
        )

# Add sample count labels
for j, center in enumerate(low_centers):
    if not np.isnan(mean_values[j]):
        # Calculate label position (place at the bottom of the bar)
        label_y = 0.07  # Ensure at the bottom of the chart

        plt.text(
            center, label_y,
            f'n={count_values[j]}',
            ha='center', va='top', fontsize=14, fontweight='bold', color='black'
        )

# Add range labels (below the bars)
for j, center in enumerate(low_centers):
    # Get the range of the current bin
    bin_min = cloud_low_bins[j]
    bin_max = cloud_low_bins[j + 1]

    # Create range string
    range_str = f'[{bin_min:.1f}, {bin_max:.1f})'

    # Calculate label position (below the bar)
    label_y = 1.2

    # Add range label
    plt.text(
        center, label_y,
        range_str,
        ha='center', va='bottom', fontsize=14, fontweight='bold', color='black',
        rotation=0  # Can be set to 45 degrees if labels overlap
    )

# Add title and labels
plt.xlabel('Low/Mid Cloud Cover Trend', fontsize=16, fontweight='bold')
plt.ylabel('SW CRE on $A_{atm}$ trend', fontsize=16, fontweight='bold')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(0, 1.2)  # Set y-axis limits

# Add zero line
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()