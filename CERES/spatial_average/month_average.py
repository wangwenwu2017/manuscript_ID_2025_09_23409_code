import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import theilslopes, kendalltau, norm
import pymannkendall as mk

year_num = 21
year_start = 2004
# 定义文件路径
file1 = '../data/CERES_EBAF_Ed4.2.1_Subset_200101-202412.nc'  # 2000年3月-2013年1月
combined_ds = xr.open_dataset(file1)

target_start = '2004-01-01'
target_end = '2024-12-31'

# 提取目标时间段（2004-01到2013-12）
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
cre = np.array(((all_toa_down - all_toa_up) - (all_surface_down - all_surface_up)-((clear_toa_down - clear_toa_up)-(clear_surface_down - clear_surface_up)))).reshape(21, 12, 180, 360)

total_absoprtion = np.nanmean(cre, axis=(0,1))

lat = np.array(combined_ds['lat'])
lon = np.array(combined_ds['lon'])



for m in range(1):
    plt.rcParams['font.family'] = 'sans-serif'  # 指定无衬线字体族
    plt.rcParams['font.sans-serif'] = ['Arial']  # 优先使用 Arial
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 可选：统一字号设置
    plt.rcParams['font.size'] = 12  # 全局基准字号
    plt.rcParams['axes.titlesize'] = 14  # 标题字号
    plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字号
    plt.rcParams['xtick.labelsize'] = 10  # X轴刻度字号
    plt.rcParams['ytick.labelsize'] = 10  # Y轴刻度字号
    fig = plt.figure(figsize=(8, 5), dpi=500)  # 增加宽度以容纳两个子图
    proj = ccrs.PlateCarree(central_longitude=180)
    ax_map = fig.add_subplot(projection=proj)
    colors = ["#015493", "#FFFFFF", "#C5272D"]  # 深蓝、白、正红
    color_nodes = [-10, 0, 20]  # 关键：0是分界点，需在边界中重复
    bounds = np.array([-10, 0, 0, 20])  # 边界列表：[-5, 0, 0, 15]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(colors))
    cmap_custom = mcolors.LinearSegmentedColormap.from_list(
        "custom_div",
        [(0, "#015493"), (0.33, "#FFFFFF"), (1, "#C5272D")],  # 手动指定位置
        N=101
    )
    levels = np.linspace(-10, 20, 61)  # 11个级别


    im1 = ax_map.pcolormesh(lon, lat, total_absoprtion,
                          vmin=-10,vmax=20,
                          cmap=cmap_custom,  # 从红到蓝的反向色带
                          transform=ccrs.PlateCarree(central_longitude=0))


    # im1 = ax_map.contourf(lon, lat, total_absoprtion[m],
    #                       levels=levels,
    #                       cmap=cmap_custom,  # 从红到蓝的反向色带
    #                       extend='both',  # 扩展两端颜色
    #                       transform=ccrs.PlateCarree(central_longitude=0))

    # 地理要素
    ax_map.coastlines(linewidth=0.5, color='black')
    gl = ax_map.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

    # 添加标题
    ax_map.set_title(' ', fontsize=12, pad=10)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为顶部标题留出空间

    # ====================== 颜色条设置 ======================
    # 在底部添加水平颜色条
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical',
                        extend='neither', label='W/m² per decade')
    cbar.set_ticks([-5, 0, 5, 10, 15])  # 设置刻度位置
    # 添加整体标题
    plt.suptitle(' ', fontsize=14, y=0.98)
    plt.savefig('CRE_month_average_'+str(m)+'_.jpeg')
    # plt.show()