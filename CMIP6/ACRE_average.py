import netCDF4 as nc
import glob
import warnings
from pathlib import Path
import os
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def merge_cmip6_time_series(directory, output_file=None):
    """
    Minimal CMIP6 time series merging tool

    Parameters:
    directory: Directory path containing CMIP6 files
    output_file: Optional output file path

    Returns:
    merged_ds: Merged dataset
    """
    # 1. Get all NetCDF files in the directory
    file_pattern = os.path.join(directory, "*.nc")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        raise FileNotFoundError(f"No NetCDF files found in directory: {directory}")
    data = []

    print(f"Found {len(file_list)} files:")
    for i, file in enumerate(file_list):
        print(f"{i + 1}. {os.path.basename(file)}")
        data.append(xr.open_dataset(file_list[i]))

    merged_ds = xr.concat(data, dim='time', coords='minimal')

    # 2. Merge files using xarray

    # 3. Ensure correct time order
    merged_ds = merged_ds.sortby('time')

    print(f"\nMerge successful! Time range: {merged_ds.time.min().values} to {merged_ds.time.max().values}")
    print(f"Total timesteps: {len(merged_ds.time)}")

    # 4. Save result (optional)
    if output_file:
        print(f"Saving result to: {output_file}")
        merged_ds.to_netcdf(output_file)

    return merged_ds

def get_parent_directory(file_path):
    """Get parent directory of a file using pathlib"""
    return Path(file_path).parent


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


warnings.filterwarnings('ignore', category=RuntimeWarning)
# Load land-sea mask
land = xr.open_dataset('/home/wenwu/06RF/TEST2025/CESM2/sftlf_fx_CESM2-WACCM_ssp534-over_r1i1p1f1_gn.nc')
src_lon = np.array(land['lon'])
src_lat = np.array(land['lat'])
land = np.array(land['sftlf'])


def find_files_in_leaf_directory(root_path):
    """
    Find all files in the leaf directory of a chain-like directory structure

    Parameters:
    root_path: Starting directory path

    Returns:
    list - List of all file paths in the leaf directory
    """
    # Convert input path to Path object
    current_dir = Path(root_path).resolve()

    # Check if path exists and is a directory
    if not current_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {current_dir}")
    if not current_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {current_dir}")

    # Track current directory and subdirectory count
    while True:
        # Get all direct children of current directory (ignore hidden files)
        contents = [item for item in current_dir.iterdir() if not item.name.startswith('.')]

        # Find all subdirectories
        subdirs = [item for item in contents if item.is_dir()]

        # If no subdirectories, we've reached the leaf level
        if not subdirs:
            # Return all non-directory files
            return [str(item) for item in contents if item.is_file()]

        # If multiple subdirectories but only one expected, handle this case
        if len(subdirs) > 1:
            print(f"Warning: Multiple subdirectories found in {current_dir}. Selecting first subdirectory to continue.")

        # Enter the first subdirectory
        current_dir = subdirs[0]

# Test file finding function
test = find_files_in_leaf_directory('/media/wenwu/wenwu_elements_1/CMIP6/ACCESS-ESM1-5/historical_r1i1p1f1/rsds')

# Define target time period
target_start = '2004-01-01'
target_end = '2013-12-30'


# List of CMIP6 models to process
# model_name_list = ['BCC-CSM2-MR', 'CNRM-ESM2-1', 'FGOALS-g3', 'GFDL-ESM4', 'HadGEM3-GC31-LL',
#               'IITM-ESM', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
#               'NESM3', 'NorESM2-LM','CESM2']

model_name_list = ['BCC-CSM2-MR', 'CNRM-ESM2-1', 'GFDL-ESM4', 'HadGEM3-GC31-LL',
              'IITM-ESM', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
              'NESM3', 'NorESM2-LM', 'CESM2']
# List of radiation parameters to process
parameter_list = ['rsds', 'rsdscs', 'rsdt', 'rsus', 'rsuscs', 'rsut', 'rsutcs']
# Root path for CMIP6 data
root_path = '/media/wenwu/wenwu_elements_1/CMIP6/'
# Initialize result array for ACRE calculations
result_ACRE_10 = np.zeros((len(model_name_list), np.array(src_lat).shape[0], np.array(src_lon).shape[0]))
i = 0
# Process each CMIP6 model
for model_name in model_name_list:
    for simulation in ['historical*']:
        print(model_name)
        # Find path for current model and simulation
        path_1 = glob.glob(root_path + model_name + '/' + simulation, recursive=True)
        # Process rsds parameter
        path_2 = path_1[0] + '/' + parameter_list[0]
        file_path = find_files_in_leaf_directory(path_2)
        rsds = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[0]].sel(time=slice(target_start, target_end))

        # Process rsdscs parameter
        path_2 = path_1[0] + '/' + parameter_list[1]
        file_path = find_files_in_leaf_directory(path_2)
        rsdscs = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[1]].sel(time=slice(target_start, target_end))

        # Process rsdt parameter
        path_2 = path_1[0] + '/' + parameter_list[2]
        file_path = find_files_in_leaf_directory(path_2)
        rsdt = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[2]].sel(time=slice(target_start, target_end))

        # Process rsus parameter
        path_2 = path_1[0] + '/' + parameter_list[3]
        file_path = find_files_in_leaf_directory(path_2)
        rsus = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[3]].sel(time=slice(target_start, target_end))

        # Process rsuscs parameter
        path_2 = path_1[0] + '/' + parameter_list[4]
        file_path = find_files_in_leaf_directory(path_2)
        rsuscs = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[4]].sel(time=slice(target_start, target_end))

        # Process rsut parameter
        path_2 = path_1[0] + '/' + parameter_list[5]
        file_path = find_files_in_leaf_directory(path_2)
        rsut = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[5]].sel(time=slice(target_start, target_end))

        # Process rsutcs parameter
        path_2 = path_1[0] + '/' + parameter_list[6]
        file_path = find_files_in_leaf_directory(path_2)
        rsutcs = merge_cmip6_time_series(get_parent_directory(file_path[0]))[parameter_list[6]].sel(time=slice(target_start, target_end))

        # Get latitude and longitude from last processed file
        lat = np.array(nc.Dataset(file_path[0])['lat'])
        lon = np.array(nc.Dataset(file_path[0])['lon'])

        # Calculate Atmospheric Cloud Radiative Effect (ACRE)
        ACRE = np.array(((rsdt - rsut) - (rsds - rsus) - ((rsdt - rsutcs) - (rsdscs - rsuscs))))
        # Calculate temporal average
        ACRE = np.nanmean(ACRE, axis=0)

        # Create interpolator for ACRE data
        interp = RegularGridInterpolator(
            (lat, lon),
            ACRE,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

        # Create grid for target resolution
        X, Y = np.meshgrid(src_lat, src_lon, indexing='ij')
        points = np.array([X.ravel(), Y.ravel()]).T
        # Interpolate ACRE data to target grid
        result_ACRE_10[i,:,:] = interp(points).reshape(src_lat.shape[0], src_lon.shape[0])
        i += 1
# Save final results
np.save('ACRE_10_average', result_ACRE_10)