import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes
import matplotlib.ticker as ticker
import netCDF4 as nc

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

# Process CERES data
file1 = 'data/CERES_EBAF_Ed4.2.1_Subset_200101-202412.nc'
file3 = 'data/CERES_grid_landmask_180x360.nc'

combined_ds = xr.open_dataset(file1)
ds3 = xr.open_dataset(file3)
of = np.array(ds3['landmask'])

target_start = '2001-01-01'
target_end = '2023-12-31'

clear_toa_down = combined_ds['solar_mon'].sel(time=slice(target_start, target_end))
clear_toa_up = combined_ds['toa_sw_clr_t_mon'].sel(time=slice(target_start, target_end))
all_toa_down = combined_ds['solar_mon'].sel(time=slice(target_start, target_end))
all_toa_up = combined_ds['toa_sw_all_mon'].sel(time=slice(target_start, target_end))
clear_surface_down = combined_ds['sfc_sw_down_clr_t_mon'].sel(time=slice(target_start, target_end))
clear_surface_up = combined_ds['sfc_sw_up_clr_t_mon'].sel(time=slice(target_start, target_end))
all_surface_down = combined_ds['sfc_sw_down_all_mon'].sel(time=slice(target_start, target_end))
all_surface_up = combined_ds['sfc_sw_up_all_mon'].sel(time=slice(target_start, target_end))

atm_cre = (all_toa_down - all_toa_up)-(all_surface_down - all_surface_up)-((clear_toa_down - clear_toa_up)-(clear_surface_down - clear_surface_up))

latitudes = np.array(combined_ds['lat'])
cos_weights = np.cos(np.deg2rad(latitudes))
cos_weights[np.where((latitudes > 70)|(latitudes < -70))] = 0
weight_2d = np.tile(cos_weights[:, np.newaxis], (1, 360))
weight_2d[np.where(of < 90)] = 0

d1_ceres = weighted_global_avg(atm_cre, weight_2d)
grouped_ceres = np.array(d1_ceres).reshape((23, 12))

# Calculate CERES annual averages
d1_ceres = np.mean(grouped_ceres, axis=1)

# Process ERA5 data
era5_data = nc.Dataset('../ERA5/32989b5c4c16d26409580bbcf0a2aa3f.nc')
ssrc = np.array(era5_data['avg_snswrfcs'])[12:-12]  # Clear-sky surface shortwave radiation
ssr = np.array(era5_data['avg_snswrf'])[12:-12]    # Surface shortwave radiation
tsr = np.array(era5_data['avg_tnswrf'])[12:-12]    # Top-of-atmosphere shortwave radiation
tsrc = np.array(era5_data['avg_tnswrfcs'])[12:-12]  # Clear-sky top-of-atmosphere shortwave radiation

# Calculate atmospheric absorption: ((TOA_net - surface_net))
enhanced_absorption = ((tsr-ssr - (tsrc-ssrc)))
sfc_absorption = (ssr - ssrc)
era5_data_ocean = nc.Dataset('../ERA5/b2ea2f963a76db080a917ef4846c082f.nc')
lsm = np.array(era5_data_ocean['lsm'])[0]

# Get latitude data
lats = era5_data.variables['latitude'][:] if 'latitude' in era5_data.variables else era5_data.variables['lat'][:]
lat_rad = np.deg2rad(lats)
cos_weights = np.cos(lat_rad)
cos_weights[np.where(abs(lats)>70)] = 0
weight_2d_era5 = np.tile(cos_weights[:, np.newaxis], (1, 1440))
weight_2d_era5[np.where(lsm==0)] = 0

d1_era5 = weighted_global_avg(enhanced_absorption, weight_2d_era5)
grouped_era5 = np.array(d1_era5).reshape(-1, 12)

# Calculate ERA5 annual averages
d1_era5 = np.mean(grouped_era5, axis=1)

# Set global font properties
plt.rcParams['font.family'] = 'Arial'  # Widely available font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# CERES data processing
radiation_ceres = np.array(d1_ceres)
radiation_ceres = radiation_ceres - np.mean(radiation_ceres)
years_ceres = np.arange(2001, 2024)  # 2001-2024 years

# ERA5 data processing
radiation_era5 = np.array(d1_era5)
radiation_era5 = radiation_era5 - np.mean(radiation_era5)
years_era5 = np.arange(2001, 2024)  # 2011-2024 years

# Calculate CERES trend (2011-2024)
slope_ceres, intercept_ceres, low_ceres, high_ceres = theilslopes(radiation_ceres[:], years_ceres[:])
lr_trend_ceres = intercept_ceres + slope_ceres * years_ceres[:]

tau, p_value_ceres = kendalltau(years_ceres[:], radiation_ceres[:])

# Calculate ERA5 trend (2011-2024)
slope_era5, intercept_era5, low_era5, high_era5 = theilslopes(radiation_era5[:], years_era5[:])
lr_trend_era5 = intercept_era5 + slope_era5 * years_era5[:]

tau, p_value_era5 = kendalltau(years_era5[:], radiation_era5[:])
# Create figure
plt.figure(figsize=(5, 3), dpi=500)

# Plot CERES data
plt.plot(years_ceres, radiation_ceres, '-', color='black', alpha=0.5, linewidth=1.5)
plt.plot(years_ceres, radiation_ceres, 'o', color='#23BAC5', markersize=6, label='CERES')

# Plot ERA5 data
plt.plot(years_era5, radiation_era5, '-', color='black', alpha=0.5, linewidth=1.5)
plt.plot(years_era5, radiation_era5, 'o', color='#F0A73A', markersize=6, label='ERA5')

# Plot trend lines
plt.plot(years_ceres[:], lr_trend_ceres, 'b--', linewidth=2, label='CERES trend')
plt.plot(years_era5[:], lr_trend_era5, 'r--', linewidth=2, label='ERA5 trend')

plt.axvline(x=2011, color='#2D2D54', linestyle=':', alpha=0.8)
plt.axhline(y=0.0, color='#2D2D54', linestyle=':', alpha=0.8)

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


# Set x-axis
plt.xticks([2001, 2006, 2011, 2016, 2021], rotation=45)
plt.xlim(2000, 2024)
plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Anomalies (W/m²)", fontsize=12, fontweight='bold')
plt.legend(loc='upper right', ncol=2, columnspacing=0.5, fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.ylim(-0.62, 0.4)
stat_text = (
    f"CERES trend: {slope_ceres * 10:.2f} W/m²/decade\n"
    f"CERES p-value: {p_value_ceres:.6f}\n"
    f"\n"
    f"ERA5 trend: {slope_era5 * 10:.2f} W/m²/decade\n"
    f"ERA5 p-value < 0.000001"
)
plt.text(
            0.02, 0.02,
            stat_text,
            transform=plt.gca().transAxes,
            ha='left',
            va='bottom',
            fontsize=10, fontweight='bold'
        )

plt.tight_layout()
plt.show()

# Output trend information
print("=== CERES Data (2011-2024) ===")
print(f"Annual trend: {slope_ceres:.4f} W/m²/year")
tau_ceres, p_value_ceres = kendalltau(years_ceres[:], radiation_ceres[:])
print(f"Mann-Kendall p-value: {p_value_ceres:.4f}")
print(f"Trend significance: {'Significant' if p_value_ceres < 0.05 else 'Not significant'}")

print("\n=== ERA5 Data (2011-2024) ===")
print(f"Annual trend: {slope_era5:.4f} W/m²/year")
tau_era5, p_value_era5 = kendalltau(years_era5, radiation_era5)
print(f"Mann-Kendall p-value: {p_value_era5:.4f}")
print(f"Trend significance: {'Significant' if p_value_era5 < 0.05 else 'Not significant'}")