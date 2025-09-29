import numpy as np
import xarray as xr


def weighted_global_avg(data_3d, weight_2d):
    """
    Calculate weighted global average
    :param data_3d: 3D array (time, latitude, longitude)
    :param weight_2d: 2D weights (latitude, longitude)
    :return: Global average for each timestep
    """
    # Handle missing values (NaN and Inf)
    valid_mask = (~np.isnan(data_3d)) & (~np.isinf(data_3d))

    # Calculate total valid weights
    total_weights = np.nansum(weight_2d * valid_mask, axis=(1, 2))

    # Calculate weighted sum
    weighted_sum = np.nansum(data_3d * weight_2d, axis=(1, 2))

    # Calculate monthly averages
    return weighted_sum / total_weights


# File paths (update with your actual paths)
q_file = 'demodata_2/wv_data.nc'
t_file = 'demodata_2/t_data.nc'
radiative_file = 'demodata_2/CERES_radiative_fluxes_192x288.nc'
kernel_dir = 'kernels/'

# Load kernel datasets
q_kernel_TOA_all = np.array(xr.open_dataset(f'{kernel_dir}/q.kernel.nc')['FSNT'])
alb_kernel_TOA_all = np.array(xr.open_dataset(f'{kernel_dir}/alb.kernel.nc')['FSNT'])

q_kernel_TOA_clr = np.array(xr.open_dataset(f'{kernel_dir}/q.kernel.nc')['FSNTC'])
alb_kernel_TOA_clr = np.array(xr.open_dataset(f'{kernel_dir}/alb.kernel.nc')['FSNTC'])

q_kernel_SUR_all = np.array(xr.open_dataset(f'{kernel_dir}/q.kernel.nc')['FSNS'])
alb_kernel_SUR_all = np.array(xr.open_dataset(f'{kernel_dir}/alb.kernel.nc')['FSNS'])

q_kernel_SUR_clr = np.array(xr.open_dataset(f'{kernel_dir}/q.kernel.nc')['FSNSC'])
alb_kernel_SUR_clr = np.array(xr.open_dataset(f'{kernel_dir}/alb.kernel.nc')['FSNSC'])

# Load core datasets
vapor_ds = xr.open_dataset(q_file)
radiative_ds = xr.open_dataset(radiative_file)
t_ds = xr.open_dataset(t_file)

# Albedo feedback
start_time = '2001-01'
end_time = '2023-12'
ds_all = np.array(radiative_ds['sfc_sw_down_all_mon'].sel(time=slice(start_time, end_time)))
us_all = np.array(radiative_ds['sfc_sw_up_all_mon'].sel(time=slice(start_time, end_time)))

ds_clr = np.array(radiative_ds['sfc_sw_down_clr_t_mon'].sel(time=slice(start_time, end_time)))
us_clr = np.array(radiative_ds['sfc_sw_up_clr_t_mon'].sel(time=slice(start_time, end_time)))

dt_all = np.array(radiative_ds['solar_mon'].sel(time=slice(start_time, end_time)))
ut_all = np.array(radiative_ds['toa_sw_all_mon'].sel(time=slice(start_time, end_time)))

dt_clr = np.array(radiative_ds['solar_mon'].sel(time=slice(start_time, end_time)))
ut_clr = np.array(radiative_ds['toa_sw_clr_t_mon'].sel(time=slice(start_time, end_time)))

# Initialize feedback arrays
dSW_alb_global__ = []
dSW_q_global__ = []


# Define saturation specific humidity function
def calcsatspechum(t, p):
    """
    Calculate saturation specific humidity

    Parameters:
    t : numpy.ndarray or float - temperature (units: K)
    p : numpy.ndarray or float - pressure (units: hPa)

    Returns:
    qs : numpy.ndarray or float - saturation specific humidity (dimensionless, kg/kg)

    Formula from: Buck (1981)
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    t = np.asarray(t)
    p = np.asarray(p)
    p_expanded = p.reshape(1, len(p), 1, 1)

    # Broadcast pressure to match temperature array shape
    p = np.broadcast_to(p_expanded, t.shape)

    # Calculate saturation vapor pressure for liquid water
    # Convert T to Celsius
    t_c = t - 273.15
    es_liquid = (1.0007 + (3.46e-6 * p)) * 6.1121 * np.exp(17.502 * t_c / (240.97 + t_c))

    # Calculate saturation mixing ratio relative to liquid water (kg/kg)
    wsl = 0.622 * es_liquid / (p - es_liquid)

    # Calculate saturation vapor pressure for ice
    es_ice = (1.0003 + (4.18e-6 * p)) * 6.1115 * np.exp(22.452 * t_c / (272.55 + t_c))

    # Calculate saturation mixing ratio relative to ice (kg/kg)
    wsi = 0.622 * es_ice / (p - es_ice)

    # Select liquid or ice based on temperature
    # Use liquid water for T ≥ 273.15K (0°C), otherwise use ice
    ws = np.where(t >= 273.15, wsl, wsi)

    # Convert mixing ratio to specific humidity (kg/kg)
    qs = ws / (1 + ws)

    return qs


# Precompute latitude weights and land mask
lats = xr.open_dataset(f'{kernel_dir}/q.kernel.nc')['lat'].values  # Latitude array in degrees
lat_rad = np.deg2rad(lats)
lat_weights = np.cos(lat_rad)
lat_weights[np.where((lats > 70) | (lats < -70))] = 0
weights_2d = np.tile(lat_weights[:, np.newaxis], (1, 288))

# Apply land mask
land = xr.open_dataset('sftlf_fx_CESM2-WACCM_ssp534-over_r1i1p1f1_gn.nc')
land = np.array(land['sftlf'])
weights_2d[np.where(land < 90)] = 0

# Loop through year pairs
for i in range(22):
    for j in range(i + 1, 23):
        # Calculate albedo feedback using all-sky radiation
        alb_before = us_all[i * 12:(i + 1) * 12] / ds_all[i * 12:(i + 1) * 12]
        alb_after = us_all[j * 12:(j + 1) * 12] / ds_all[j * 12:(j + 1) * 12]
        dalb = np.array(alb_after) - np.array(alb_before)
        dSW_alb = np.array(alb_kernel_TOA_all) * dalb * 100

        # Calculate water vapor feedback
        q_before = np.array(vapor_ds['interpolated'][i * 12:(i + 1) * 12])
        q_after = np.array(vapor_ds['interpolated'][j * 12:(j + 1) * 12])
        t_before = np.array(t_ds['temp'][i * 12:(i + 1) * 12])
        t_after = np.array(t_ds['temp'][j * 12:(j + 1) * 12])
        p = (vapor_ds['pressure_level'])

        # Calculate saturation specific humidity
        qs_before = np.array(calcsatspechum(t_before, p))
        qs_after = np.array(calcsatspechum(t_after, p))
        q_after = np.array(q_after)
        q_before = np.array(q_before)
        dqsdt = (qs_after - qs_before) / (t_after - t_before)
        rh = q_before / qs_before
        dqdt = rh * dqsdt

        dSW_q = (q_kernel_TOA_all / dqdt) * np.array(q_after - q_before)  # Q-change contribution
        dSW_q[np.where(np.isinf(dSW_q))] = np.nan

        # Compute global averages
        dSW_alb_global = weighted_global_avg(dSW_alb, weights_2d)
        dSW_alb_global = np.nanmean(np.array(dSW_alb_global), axis=0) * 10 / (j - i)
        dSW_alb_global__.append(dSW_alb_global)

        dSW_q_global = weighted_global_avg(np.nansum(dSW_q, axis=1), weights_2d)
        dSW_q_global = np.nanmean(np.array(dSW_q_global), axis=0) * 10 / (j - i)
        dSW_q_global__.append(dSW_q_global)

# Output results
print(f"SW albedo Feedback: " + str(np.median(dSW_alb_global__)) + " W m⁻²")
print(f"SW water vapor Feedback: " + str(np.median(dSW_q_global__)) + " W m⁻²")

# Save results
np.save('output/dSW_alb_global_toa_all', np.array(dSW_alb_global__))
np.save('output/dSW_q_global_toa_all', np.array(dSW_q_global__))