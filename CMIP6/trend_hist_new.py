import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes


def adaptive_combine_arrays(models, base_suffix='_hist.npy', method='truncate'):
    """
    Adaptively process arrays of different lengths

    Parameters:
    method: 'truncate' - Truncate to shortest length
            'pad' - Pad to longest length
            'interpolate' - Interpolate to longest length
    """
    arrays = []
    valid_models = []
    lengths = []

    # Load all data
    for model in models:
        file_path = f"{model}{base_suffix}"

        # if not os.path.exists('CMIP6_data/'+file_path):
        #     continue

        try:
            arr = np.load('CMIP6_data/' + file_path)

            # Ensure it's a 1D array
            if arr.ndim > 1:
                arr = arr.flatten()
            elif arr.ndim == 0:
                arr = np.array([arr])

            arrays.append(arr)
            valid_models.append(model)
            lengths.append(len(arr))
        except Exception as e:
            print(f"Failed to load model {model}: {str(e)}")

    if not arrays:
        return None, []

    min_len = min(lengths)
    max_len = max(lengths)

    # Handle different lengths
    if min_len != max_len:
        print(f"Array length range: {min_len} to {max_len}")

        processed_arrays = []

        for i, arr in enumerate(arrays):
            processed = arr[:min_len]
            print(f"Truncated {valid_models[i]} from {len(arr)} to {min_len}")
            processed_arrays.append(processed)

        arrays = processed_arrays

    # Combine arrays
    return np.array(arrays), valid_models


# Model list
models = [
    'BCC-CSM2-MR', 'CNRM-ESM2-1', 'GFDL-ESM4', 'HadGEM3-GC31-LL',
    'IITM-ESM', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
    'NESM3', 'NorESM2-LM', 'CESM2'
]

# Process historical surface data
d_sur, b = adaptive_combine_arrays(models, '_hist_sur.npy')
# Initialize arrays for surface trends
trend_sur_hist = np.zeros(np.array(d_sur).shape[0] + 1)
low_hist_sur_arr = np.zeros(np.array(d_sur).shape[0] + 1)
high_hist_sur_arr = np.zeros(np.array(d_sur).shape[0] + 1)

# Calculate surface trends for each model
for i in range((np.array(d_sur).shape[0]) + 1):
    if i != 13:
        # Calculate Theil-Sen slope for surface data (1950-2014)
        slope, intercept, low_hist, high_hist = theilslopes(d_sur[i, 100:], np.arange(1950, 2015, 1))
        trend_sur_hist[i] = slope * 10  # Convert to per-decade trend
        low_hist_sur_arr[i] = low_hist * 10
        high_hist_sur_arr[i] = high_hist * 10

        # Calculate Kendall's tau for significance testing
        tau, p_value = kendalltau(np.arange(1950, 2015, 1), d_sur[i, 100:])
        if p_value > 0.05:
            print(f"Model {models[i]} surface trend not significant (p={p_value:.4f})")
    else:
        # Calculate multi-model ensemble (MME) trend
        slope, intercept, low_hist, high_hist = theilslopes(np.mean(d_sur[:, 100:], axis=0), np.arange(1950, 2015, 1))
        trend_sur_hist[i] = slope * 10
        low_hist_sur_arr[i] = low_hist * 10
        high_hist_sur_arr[i] = high_hist * 10

        tau, p_value = kendalltau(np.arange(1950, 2015, 1), np.mean(d_sur[:, 100:], axis=0))
        if p_value > 0.05:
            print(f"Model {models[i]} surface trend not significant (p={p_value:.4f})")

# Process historical atmospheric data
d, b = adaptive_combine_arrays(models, '_hist.npy')
# Initialize arrays for atmospheric trends
trend_hist = np.zeros(np.array(d).shape[0] + 1)
low_hist_arr = np.zeros(np.array(d).shape[0] + 1)
high_hist_arr = np.zeros(np.array(d).shape[0] + 1)

# Calculate atmospheric trends for each model
for i in range((np.array(d).shape[0]) + 1):
    if i != 13:
        slope, intercept, low_hist, high_hist = theilslopes(d[i, 100:], np.arange(1950, 2015, 1))
        trend_hist[i] = slope * 10
        low_hist_arr[i] = low_hist * 10
        high_hist_arr[i] = high_hist * 10

        tau, p_value = kendalltau(np.arange(1950, 2015, 1), d[i, 100:])
        if p_value > 0.05:
            print(f"Model {models[i]} atmospheric trend not significant (p={p_value:.4f})")
    else:
        # Calculate MME trend for atmospheric data
        slope, intercept, low_hist, high_hist = theilslopes(np.mean(d[:, 100:], axis=0), np.arange(1950, 2015, 1))
        trend_hist[i] = slope * 10
        low_hist_arr[i] = low_hist * 10
        high_hist_arr[i] = high_hist * 10

        tau, p_value = kendalltau(np.arange(1950, 2015, 1), np.mean(d[:, 100:], axis=0))
        if p_value > 0.05:
            print(f"Model {models[i]} atmospheric trend not significant (p={p_value:.4f})")

# Calculate combined trends (surface + atmospheric)
trend_all = np.zeros(np.array(d).shape[0] + 1)
low_all_arr = np.zeros(np.array(d).shape[0] + 1)
high_all_arr = np.zeros(np.array(d).shape[0] + 1)
d = d + d_sur  # Combine surface and atmospheric data
for i in range((np.array(d).shape[0]) + 1):
    if i != 13:
        slope, intercept, low_hist, high_hist = theilslopes(d[i, 100:], np.arange(1950, 2015, 1))
        trend_all[i] = slope * 10
        low_all_arr[i] = low_hist * 10
        high_all_arr[i] = high_hist * 10

        tau, p_value = kendalltau(np.arange(1950, 2015, 1), d[i, 100:])
        if p_value > 0.05:
            print(f"Model {models[i]} combined trend not significant (p={p_value:.4f})")
    else:
        # Calculate MME trend for combined data
        slope, intercept, low_hist, high_hist = theilslopes(np.mean(d[:, 100:], axis=0), np.arange(1950, 2015, 1))
        trend_all[i] = slope * 10
        low_all_arr[i] = low_hist * 10
        high_all_arr[i] = high_hist * 10

        tau, p_value = kendalltau(np.arange(1950, 2015, 1), np.mean(d[:, 100:], axis=0))
        if p_value > 0.05:
            print(f"Model {models[i]} combined trend not significant (p={p_value:.4f})")

# Add MME to model list
models.append('MME')
if len(trend_sur_hist) != len(models) or len(trend_hist) != len(models):
    raise ValueError("Trend data length does not match number of models")

# Create DataFrame for results
df = pd.DataFrame({
    'Model': models,
    'SSP585': trend_sur_hist,
    'Historical': trend_hist,
    'Historical_all': trend_all,
    'SSP585_low': low_hist_sur_arr,
    'SSP585_high': high_hist_sur_arr,
    'Hist_low': low_hist_arr,
    'Hist_high': high_hist_arr,
    'all_low': low_all_arr,
    'all_high': high_all_arr
})

# Sort by SSP585 trend
df = df.sort_values('SSP585', ascending=False)

# Set positions and width for bars
x = np.arange(len(models))
width = 0.4

# Configure plot style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8  # Reduce font size for small plot

# Create figure
fig, ax = plt.subplots(figsize=(6, 4), dpi=500)

# Calculate error ranges
# Error = [lower error, upper error] = [value - lower bound, upper bound - value]
ssp585_err = [
    df['SSP585'] - df['SSP585_low'],
    df['SSP585_high'] - df['SSP585']
]

hist_err = [
    df['Historical'] - df['Hist_low'],
    df['Hist_high'] - df['Historical']
]

all_err = [
    df['Historical_all'] - df['all_low'],
    df['all_high'] - df['Historical_all']
]

# Calculate ratios (Historical/SSP585)
ratios = abs(df['Historical']) / abs(df['SSP585'])

# Plot bars with error bars
rects2 = ax.bar(
    x - width / 2,
    df['Historical'],
    width,
    yerr=hist_err,  # Add error bars
    error_kw={
        'ecolor': '#5299cc',
        'capsize': 1.5,  # Reduce cap size
        'capthick': 0.6,  # Reduce cap thickness
        'elinewidth': 0.6  # Reduce error line width
    },
    label='SW CRE on $A_{atm}$',
    color='#5299cc',
    alpha=0.6
)

# Plot bars with error bars
rects1 = ax.bar(
    x - width / 2,
    df['SSP585'],
    width,
    yerr=ssp585_err,  # Add error bars
    error_kw={
        'ecolor': '#f599a1',
        'capsize': 1.5,  # Reduce cap size
        'capthick': 0.6,  # Reduce cap thickness
        'elinewidth': 0.6  # Reduce error line width
    },
    label='SW CRE on $A_{sfc}$',
    color='#f599a1',
    alpha=0.6
)

rects3 = ax.bar(
    x + width / 2,
    df['Historical_all'],
    width,
    hatch='///',
    yerr=all_err,  # Add error bars
    error_kw={
        'ecolor': '#73c79e',
        'capsize': 1.5,  # Reduce cap size
        'capthick': 0.6,  # Reduce cap thickness
        'elinewidth': 0.6  # Reduce error line width
    },
    label='SW CRE at TOA',
    color='#73c79e',
    alpha=0.6
)

# Add labels
ax.set_ylabel('Trend (W/m²/decade)', fontsize=18, fontweight='bold')
ax.set_xticks(x)

# Set x-axis labels
ax.set_xticklabels(df['Model'], rotation=90, fontsize=8)

# Customize MME label
for label in ax.get_xticklabels():
    if label.get_text() == 'MME':
        label.set_fontsize(18)  # Larger font
        label.set_fontweight('bold')  # Bold

ax.legend(fontsize=15, loc='upper right')  # Reduce legend font size

# Add grid lines
ax.yaxis.grid(True, linestyle='-', alpha=0.2)
ax.xaxis.grid(True, linestyle='-', alpha=0.2)

# Add ratio text below each bar group
# Get y-axis minimum as reference for text position
y_min = min(min(df['SSP585'] - ssp585_err[0]), min(df['Historical'] - hist_err[0]),
            min(df['Historical_all'] - all_err[0]))
text_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # Text offset

# Add ratio text for each model
for i, model in enumerate(df['Model']):
    # Calculate ratio
    ratio = ratios.iloc[i]

    # Add text below each bar group
    ax.text(
        x[i] - width / 2,  # Position below Historical and SSP585 bars
        y_min + text_offset,  # Below y-axis minimum
        f"{ratio:.2f}",  # Format ratio to 2 decimal places
        ha='center',  # Horizontal center
        va='top',  # Vertical top alignment
        fontsize=10,  # Font size
        fontweight='bold',  # Bold
        color='black'  # Text color
    )


# Adjust layout
plt.tight_layout()
plt.show()