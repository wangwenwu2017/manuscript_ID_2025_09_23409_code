import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from matplotlib.patches import Patch

# Set global parameters
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Define color scheme
variable_colors = {
    'Water vapor': '#1f77b4',  # Water vapor - blue
    'Albedo': '#ff7f0e',  # Albedo - orange
    'Cloud': '#2ca02c'  # Cloud - green
}

# Position group colors
position_colors = {
    'Surface': '#f599a1',  # Surface - purple
    'Atmosphere': '#91d7e9',  # Atmosphere - red
    'TOA': '#fcd590'  # Atmosphere - red
}

# Sky condition styles
sky_styles = {
    'All-sky': {'facecolor': '#ffffff', 'edgecolor': '#333333'},  # All-sky - solid
    'Clear-sky': {'facecolor': '#ffffff', 'edgecolor': '#333333', 'hatch': '///', 'alpha': 0.7}  # Clear-sky - hatched
}


# Data preparation function
def load_and_calculate_stats(file):
    """
    Load file and calculate statistics
    Returns: median, CI_low, CI_high, variable name, group
    """
    data = np.load(file)
    valid_data = data[np.isfinite(data)]

    # Calculate median
    median = np.median(valid_data)

    # Calculate 95% confidence interval (using bootstrap)
    n_boot = 10000
    boot_medians = []
    for _ in range(n_boot):
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        boot_medians.append(np.median(sample))

    ci_low = np.percentile(boot_medians, 2.5)
    ci_high = np.percentile(boot_medians, 97.5)

    # Parse variable and group from filename
    if 'alb' in file:
        variable = 'Albedo'
    elif 'cloud' in file:
        variable = 'Cloud'
    elif 'q' in file:
        variable = 'Water vapor'
    else:
        variable = 'Unknown'

    # Group information
    if 'sur' in file:
        grouping = 'Surface'
    elif 'atm' in file:
        grouping = 'Atmosphere'
    else:
        grouping = 'TOA'
    sky = 'Clear-sky' if 'clr' in file else 'All-sky'

    return {
        'variable': variable,
        'grouping': grouping,
        'sky': sky,
        'median': median,
        'CI_low': ci_low,
        'CI_high': ci_high,
        'dataset': 'SRC' if 'src' in file else 'ATM'  # Add dataset type
    }


# Function to create and save DataFrame
def create_and_save_dataframe():
    # Prepare data
    file_list = [
        'dSW_alb_global_atm_all.npy',
        'dSW_alb_global_atm_clr.npy',
        'dSW_alb_global_sur_all.npy',
        'dSW_alb_global_sur_clr.npy',
        'dSW_alb_global_toa_all.npy',
        'dSW_alb_global_toa_clr.npy',
        'dSW_cloud_global_atm.npy',
        'dSW_cloud_global_sur.npy',
        'dSW_cloud_global_toa.npy',
        'dSW_q_global_atm_all.npy',
        'dSW_q_global_atm_clr.npy',
        'dSW_q_global_sur_all.npy',
        'dSW_q_global_sur_clr.npy',
        'dSW_q_global_toa_all.npy',
        'dSW_q_global_toa_clr.npy'
    ]

    # Collect all data
    data_list = []
    for file in file_list:
        if os.path.exists(file):
            stats = load_and_calculate_stats(file)
            data_list.append(stats)

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Save DataFrame to CSV
    df.to_csv('sw_absorption_trend_data.csv', index=False)

    # Save DataFrame to Pickle (for preserving data types)
    with open('sw_absorption_trend_data.pkl', 'wb') as f:
        pickle.dump(df, f)

    print("DataFrame saved to CSV and Pickle files.")
    return df


# Function to load DataFrame
def load_dataframe():
    # Try to load from Pickle first (preserves data types)
    if os.path.exists('sw_absorption_trend_data.pkl'):
        try:
            with open('sw_absorption_trend_data.pkl', 'rb') as f:
                df = pickle.load(f)
                print("DataFrame loaded from Pickle file.")
                return df
        except:
            print("Error loading Pickle file, trying CSV...")

    # If Pickle fails or doesn't exist, try CSV
    if os.path.exists('sw_absorption_trend_data.csv'):
        try:
            df = pd.read_csv('sw_absorption_trend_data.csv')
            print("DataFrame loaded from CSV file.")
            return df
        except:
            print("Error loading CSV file.")

    # If no file exists, create a new DataFrame
    print("No data file found, creating new DataFrame.")
    return create_and_save_dataframe()


# Main plotting function
def plot_trend_comparison():
    # Load or create DataFrame
    df = load_dataframe()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=500)

    # Set x-axis range
    all_data = np.concatenate([df['CI_low'].values, df['CI_high'].values])
    x_min = np.floor(min(all_data)) - 0.2
    x_max = np.ceil(max(all_data)) + 0.2

    # Define position and sky condition combinations
    positions = ['Surface', 'Atmosphere', 'TOA']
    sky_conditions = ['All-sky', 'Clear-sky']

    # Calculate center positions for each variable group
    group_positions = {}
    for var_idx, var in enumerate(['Water vapor', 'Albedo', 'Cloud']):
        group_positions[var] = var_idx

    # Calculate bar width and offsets to prevent overlap
    bar_width = 0.15
    n_groups = len(positions) * len(sky_conditions)  # 4 groups per variable
    group_spacing = 0.8  # Total space allocated per variable group

    # Define offsets for each combination
    offsets = {
        ('Surface', 'Clear-sky'): -group_spacing / 2 + bar_width * 0.4,
        ('Atmosphere', 'Clear-sky'): -group_spacing / 2 + bar_width * 0.4,
        ('TOA', 'Clear-sky'): -group_spacing / 2 + bar_width * 1.4,
        ('Surface', 'All-sky'): bar_width * 0.5,
        ('Atmosphere', 'All-sky'): bar_width * 0.5,
        ('TOA', 'All-sky'): bar_width * 1.5,
    }

    offsets2 = {
        ('Surface', 'All-sky'): 0 - bar_width * 0.5,
        ('Atmosphere', 'All-sky'): 0 - bar_width * 0.5,
        ('TOA', 'All-sky'): bar_width * 0.5,
    }

    # Plot each variable group
    for pos in positions:
        for var in ['Water vapor', 'Albedo', 'Cloud']:

            for sky in sky_conditions:
                # Get current data
                row = df[(df['grouping'] == pos) &
                         (df['sky'] == sky) &
                         (df['variable'] == var)]

                if not row.empty:
                    if var != 'Cloud':
                        median = row['median'].values[0]
                        ci_low = row['CI_low'].values[0]
                        ci_high = row['CI_high'].values[0]

                        # Calculate position
                        x_pos = group_positions[var]
                        offset = offsets[(pos, sky)]

                        # Draw bar
                        style = sky_styles[sky]
                        if sky == 'All-sky':
                            ax.bar(x_pos + offset, median,
                                   width=bar_width,
                                   color=position_colors[pos],
                                   alpha=0.8,
                                   linewidth=1.5)
                        else:  # Clear-sky
                            ax.bar(x_pos + offset, median,
                                   width=bar_width,
                                   color=position_colors[pos],
                                   linewidth=1.5,
                                   hatch='//',
                                   alpha=0.8)
                        ax.errorbar(x_pos + offset, median,
                                    yerr=[[median - ci_low], [ci_high - median]],
                                    fmt='none',
                                    ecolor='k',
                                    elinewidth=1,
                                    capsize=2.5,
                                    capthick=1)

                        # Add value labels
                        text_x = x_pos + offset
                        text_y = median + (
                            1.12 * median / abs(median) * abs(ci_high - median) if median >= 0 else median / abs(
                                median) * abs(ci_high - median) - 0.11)
                        ax.text(text_x, text_y,
                                f"{median:.2f}",
                                fontsize=11, ha='center')
                    else:
                        median = row['median'].values[0]
                        ci_low = row['CI_low'].values[0]
                        ci_high = row['CI_high'].values[0]

                        # Calculate position
                        x_pos = group_positions[var]
                        offset2 = offsets2[(pos, sky)]

                        # Draw bar
                        style = sky_styles[sky]
                        if sky == 'All-sky':
                            ax.bar(x_pos + offset2, median,
                                   width=bar_width,
                                   color=position_colors[pos],
                                   alpha=0.8,
                                   linewidth=1.5)
                        else:  # Clear-sky
                            ax.bar(x_pos + offset2, median,
                                   width=bar_width,
                                   color=position_colors[pos],
                                   linewidth=1.5,
                                   hatch='//',
                                   alpha=0.8)
                        ax.errorbar(x_pos + offset2, median,
                                    yerr=[[median - ci_low], [ci_high - median]],
                                    fmt='none',
                                    ecolor='k',
                                    elinewidth=1,
                                    capsize=2.5,
                                    capthick=1)

                        # Add value labels
                        text_x = x_pos + offset2
                        text_y = median + (
                            1.12 * median / abs(median) * abs(ci_high - median) if median >= 0 else median / abs(
                                median) * abs(ci_low - median) - 0.08)
                        ax.text(text_x, text_y,
                                f"{median:.2f}",
                                fontsize=11, ha='center')

                    # Add error bars

    # Set X-axis
    ax.set_xticks([i for i in range(len(group_positions))])
    ax.set_xticklabels(['Water Vapor', 'Surface Albedo', 'Cloud'], fontsize=18)
    ax.set_xlim(-0.5, 2.5)

    # Set Y-axis
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel('SW Absorption Trend (W/m²/decade)', fontsize=18)

    # Add zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)

    # Add grid
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')

    # Clean up borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add position legend
    position_legend = [Patch(facecolor=position_colors[p], alpha=0.6, label=p) for p in positions]
    fig.legend(handles=position_legend, loc='lower left',
               bbox_to_anchor=(0.15, 0.08), ncol=1, fontsize=12, frameon=False)

    # Add sky condition legend
    sky_legend = [
        Patch(facecolor='#ffffff', edgecolor='#333333', label='All-sky'),
        Patch(facecolor='#ffffff', edgecolor='#333333', hatch='//', label='Clear-sky')
    ]
    fig.legend(handles=sky_legend, loc='lower left',
               bbox_to_anchor=(0.4, 0.08), ncol=1, fontsize=12, frameon=False)

    # Adjust layout
    plt.tight_layout()  # Make space for legends and title

    # Save and display
    plt.savefig('Shortwave_Absorption_Trend.png', dpi=500, bbox_inches='tight')
    plt.show()


# Run plotting function
plot_trend_comparison()