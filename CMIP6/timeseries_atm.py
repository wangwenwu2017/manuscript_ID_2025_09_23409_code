import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

def calculate_model_average_trend(data, confidence_level=0.95):
    """
    Calculate multi-model average time series and confidence intervals

    Parameters:
    data: 2D array (number of models × years)
    confidence_level: Confidence level (default 0.95)

    Returns:
    mean_trend: Average time series (length = number of years)
    ci_low: Lower bound of confidence interval
    ci_high: Upper bound of confidence interval
    """
    # Ensure input is a 2D array
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array (number of models × years)")

    n_models, n_years = data.shape

    # Calculate average at each time point
    mean_trend = np.nanmean(data, axis=0)

    # Initialize confidence interval arrays
    ci_low = np.zeros(n_years)
    ci_high = np.zeros(n_years)

    # Calculate confidence interval for each time point
    for t in range(n_years):
        # Get values from all models at this time point
        values = data[:, t]

        # Remove NaN values
        valid_values = values[~np.isnan(values)]
        n_valid = len(valid_values)

        if n_valid == 0:
            ci_low[t] = np.nan
            ci_high[t] = np.nan
        elif n_valid == 1:
            ci_low[t] = valid_values[0]
            ci_high[t] = valid_values[0]
        else:
            # Calculate mean and standard error
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values, ddof=1)  # Sample standard deviation
            se = std_val / np.sqrt(n_valid)  # Standard error

            # Calculate confidence interval
            ci_low[t], ci_high[t] = stats.t.interval(
                confidence_level,
                df=n_valid - 1,
                loc=mean_val,
                scale=se
            )

    return mean_trend, ci_low, ci_high

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

        if not os.path.exists('CMIP6_data/'+file_path):
            continue

        try:
            arr = np.load('CMIP6_data/'+file_path)

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

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Model list
models = [
    'BCC-CSM2-MR', 'CNRM-ESM2-1', 'GFDL-ESM4', 'HadGEM3-GC31-LL',
    'IITM-ESM', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
    'NESM3', 'NorESM2-LM','CESM2'
]

# Alternative model list (commented out)
# models = [
#     'BCC-CSM2-MR', 'CNRM-ESM2-1', 'GFDL-ESM4', 'HadGEM3-GC31-LL',
#     'IITM-ESM', 'INM-CM4-8', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR',
#     'NESM3', 'NorESM2-LM'
# ]

# Create figure
plt.figure(figsize=(6, 3.5), dpi=500)

# Process historical data
d_hist, b = adaptive_combine_arrays(
    models, '_hist.npy'
)
data_hist, data_hist_low, data_hist_high = calculate_model_average_trend(d_hist)

# Plot individual historical model trends
for i in range(d_hist.shape[0]):
    plt.plot(np.arange(1850, 2015, 1), d_hist[i]-np.mean(d_hist[i]), '-', color='#1f77b4', linewidth=0.3, alpha=0.3)

# Process SSP126 scenario data
d_126, b = adaptive_combine_arrays(
    models, '_ssp126.npy'
)
data_ssp126, data_ssp126_low, data_ssp126_high = calculate_model_average_trend(d_126)

# Plot individual SSP126 model trends
for i in range(d_126.shape[0]):
    plt.plot(np.arange(2014,2100,1), np.concatenate(([d_hist[i, -1]], d_126[i]))-np.mean(d_hist[i]), '-', color='#F0A73A', linewidth=0.3, alpha=0.3)

# Process SSP585 scenario data
d_585, b = adaptive_combine_arrays(
    models, '_ssp585.npy'
)
data_ssp585, data_ssp585_low, data_ssp585_high = calculate_model_average_trend(d_585)

# Plot individual SSP585 model trends
for i in range(d_585.shape[0]):
    plt.plot(np.arange(2014,2100,1), np.concatenate(([d_hist[i, -1]], d_585[i]))-np.mean(d_hist[i]), '-', color='#23BAC5', linewidth=0.3, alpha=0.3)

# Plot multi-model average trends
plt.plot(np.arange(1850,2015,1), data_hist-np.mean(data_hist), '-', color='#1f77b4',  linewidth=1.5, label='Historical Simulation')
plt.plot(np.arange(2014,2100,1), np.concatenate(([data_hist[-1]], data_ssp126))-np.mean(data_hist), '-', color='#F0A73A',  linewidth=1.5, label='ssp126 Simulation')
plt.plot(np.arange(2014,2100,1), np.concatenate(([data_hist[-1]], data_ssp585))-np.mean(data_hist), '-', color='#23BAC5',  linewidth=1.5, label='ssp585 Simulation')

# Set x-axis ticks and labels
plt.xticks([1850, 1900, 1950, 2000, 2050, 2100], rotation=45)
plt.xlabel("Year", fontsize=18, fontweight='bold')
plt.ylabel("Anomalies (W/m²)", fontsize=18, fontweight='bold')
plt.xlim(1850, 2100)
# plt.ylim(-1, 4.5)  # Optional y-axis limit

# Add legend and grid
plt.legend(fontsize=16, loc='lower left')
plt.grid(alpha=0.3, linestyle='--')

# Adjust layout and display
plt.tight_layout()
plt.show()