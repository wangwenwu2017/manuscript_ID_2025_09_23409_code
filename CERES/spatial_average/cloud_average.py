import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

year_num = 11
year_start = 2003
# Define file paths
file1 = '../data/CERES_SSF1deg-Month_Terra-MODIS_Ed4.1_Subset_200003-202407.nc'  # March 2000 - July 2024
ds1 = xr.open_dataset(file1)

target_start = '2003-01-01'
target_end = '2013-12-31'

# Extract target time period (January 2004 - December 2013)
cf_high = ds1['cldarea_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midhigh = ds1['cldarea_mid_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf_low = ds1['cldarea_low_daynight_mon'].sel(time=slice(target_start, target_end))
cf_midlow = ds1['cldarea_mid_high_daynight_mon'].sel(time=slice(target_start, target_end))
cf = ds1['cldarea_total_daynight_mon'].sel(time=slice(target_start, target_end))

cf_high = np.array(cf_high).reshape((year_num, 12, 180, 360))
cf_mid = np.array(cf_low+cf_midhigh+cf_midlow).reshape((year_num, 12, 180, 360))

total_absoprtion = np.nanmean(cf_mid, axis=(0,1))
# np.save('LC_average.npy', total_absoprtion)
lat = np.array(ds1['lat'])
lon = np.array(ds1['lon'])



for m in range(1):
    # Specify sans-serif font family
    plt.rcParams['font.family'] = 'sans-serif'
    # Prioritize Arial font
    plt.rcParams['font.sans-serif'] = ['Arial']
    # Fix minus sign display issue
    plt.rcParams['axes.unicode_minus'] = False
    # Optional: Unified font size settings
    plt.rcParams['font.size'] = 12  # Global base font size
    plt.rcParams['axes.titlesize'] = 14  # Title font size
    plt.rcParams['axes.labelsize'] = 12  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 10  # X-axis tick font size
    plt.rcParams['ytick.labelsize'] = 10  # Y-axis tick font size
    fig = plt.figure(figsize=(8, 5), dpi=500)  # Increase width to accommodate two subplots
    proj = ccrs.Robinson(central_longitude=180)
    ax_map = fig.add_subplot(projection=proj)
    cmap_custom = mcolors.LinearSegmentedColormap.from_list(
        "custom_div",
        [(0, "#015493"), (0.5, "#FFFFFF"), (1, "#C5272D")],  # Manually specified positions
        N=101
    )
    # im1 = ax_map.pcolormesh(lon, lat, total_absoprtion,
    #                       vmin=0,vmax=100,
    #                       cmap='Spectral_r',  # Reverse color scale from red to blue
    #                       transform=ccrs.PlateCarree(central_longitude=0))

    levels = np.linspace(0, 60, 101)  # 101 levels
    im1 = ax_map.contourf(lon, lat, total_absoprtion,
                          levels=levels,
                          cmap=cmap_custom,  # Reverse color scale from red to blue
                          extend='both',  # Extend both ends
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
                        extend='neither', label='%')
    cbar.set_ticks([0, 15, 30, 45, 60])  # Set tick positions
    # Add overall title
    plt.suptitle(' ', fontsize=14, y=0.98)
    plt.show()