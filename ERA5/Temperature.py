import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot
import matplotlib.ticker as ticker

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

# Load ERA5 temperature data
data = nc.Dataset('ERA5_temperature.nc')
tem_2m = data['t2m']
sfc_tem = data['skt']

# Process surface temperature data
sfc_tem = np.array(sfc_tem).reshape((24, 12, 1801, 3600))
sfc_tem = np.nanmean(sfc_tem, axis=1)

# Calculate latitude weights
latitudes = np.array(data['latitude'])
cos_weights = np.cos(np.deg2rad(latitudes))
cos_weights[np.where((latitudes > 70)|(latitudes < -70))] = 0
weight_2d = np.tile(cos_weights[:, np.newaxis], (1, 3600))

# Compute weighted global temperature average
t = weighted_global_avg(sfc_tem, weight_2d)
t = t - np.mean(t)  # Calculate anomalies

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

# Create figure
plt.figure(figsize=(5, 3), dpi=500)

# Add reference lines
plt.axvline(x=2011, color='#2D2D54', linestyle=':', alpha=0.8)  # Vertical line at 2011
plt.axhline(y=0.0, color='#2D2D54', linestyle=':', alpha=0.8)    # Horizontal line at zero

# Plot temperature anomalies
years = np.arange(2001, 2024)  # 2001-2023 years
plt.plot(years, t[:23], '-', color='black', alpha=0.7, linewidth=1.5)  # Line plot
plt.plot(years, t[:23], 'o', color='#a577ad', markersize=6)           # Marker plot

# Custom formatter for y-axis ticks
class FourDigitFormatter(ticker.Formatter):
    def __init__(self):
        pass

    def __call__(self, value, pos=None):
        if value == 0:
            return "0.00"
        int_digits = len(str(int(abs(value))))
        if int_digits >= 3:
            return f"{value:.0f}"
        else:
            decimal_digits = 3 - int_digits
            decimal_digits = max(0, decimal_digits)
            return f"{value:.{decimal_digits}f}"

# Apply custom formatter
plt.gca().yaxis.set_major_formatter(FourDigitFormatter())

# Set x-axis ticks and labels
plt.xticks([2001, 2006, 2011, 2016, 2021], rotation=45)  # Show selected years
plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Anomalies (°C)", fontsize=12, fontweight='bold')

# Adjust layout and display
plt.tight_layout()
plt.show()