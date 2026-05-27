"""
Analyze the change in Weighted Degree from the early to late period.
Generates histograms and spatial distribution maps.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # 1. Define paths
    script_dir = Path(__file__).resolve().parent
    nc_path = script_dir.parent / 'pipeline/networks_indexes_combined_output.nc'
    output_dir = script_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {nc_path}")
    try:
        ds = xr.open_dataset(nc_path)
    except FileNotFoundError:
        print(f"Error: Could not find the NetCDF file at {nc_path}")
        return

    if 'weighted_degree_fragmentation' not in ds.data_vars:
        raise ValueError("'weighted_degree_fragmentation' not found in the dataset variables!")

    # 2. Extract Data
    index_data = ds['weighted_degree_fragmentation']
    
    # 3. Flatten valid (non-NaN) values for the distribution plots
    early_flat = index_data.sel(period='early').values.flatten()
    early_flat = early_flat[~np.isnan(early_flat)]
    
    late_flat = index_data.sel(period='late').values.flatten()
    late_flat = late_flat[~np.isnan(late_flat)]

    # 4. Calculate Differences (Late - Early) matching cell-by-cell and month-by-month
    diff_da = index_data.sel(period='late') - index_data.sel(period='early')
    diff_flat = diff_da.values.flatten()
    diff_flat = diff_flat[~np.isnan(diff_flat)]

    # Calculate statistics
    total_valid = len(diff_flat)
    increased = np.sum(diff_flat > 0)
    decreased = np.sum(diff_flat < 0)
    unchanged = np.sum(diff_flat == 0)
    
    pct_inc = (increased / total_valid) * 100
    pct_dec = (decreased / total_valid) * 100
    pct_unc = (unchanged / total_valid) * 100

    print(f"Total analyzed valid points (cells × months): {total_valid}")
    print(f"Increased: {pct_inc:.2f}% | Decreased: {pct_dec:.2f}% | Unchanged: {pct_unc:.2f}%")

    # Set aesthetic style for seaborn
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Overlaid Histograms (Early vs Late) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(early_flat, color='#0369a1', label='Early (1961–1990)', kde=True, stat='density', alpha=0.5, bins=40)
    sns.histplot(late_flat, color='#b91c1c', label='Late (1995–2024)', kde=True, stat='density', alpha=0.5, bins=40)
    plt.title('Weighted Degree Distribution: Early vs. Late Period', fontsize=14, pad=15)
    plt.xlabel('Weighted Degree', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_degree_distribution_comparison.png', dpi=300)
    plt.close()

    # --- Plot 2: Histogram of Differences ---
    plt.figure(figsize=(10, 6))
    sns.histplot(diff_flat, color='#8b5cf6', bins=50, kde=True)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Change in Weighted Degree (Late − Early)', fontsize=14, pad=15)
    plt.xlabel('Δ Weighted Degree', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    stats_text = f"Increased: {pct_inc:.1f}%\nDecreased: {pct_dec:.1f}%\nUnchanged: {pct_unc:.1f}%"
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_degree_change_histogram.png', dpi=300)
    plt.close()

    # --- Plot 3: Spatial Map of Mean Change ---
    mean_diff_spatial = diff_da.mean(dim='month')
    plt.figure(figsize=(10, 8))
    mean_diff_spatial.plot(cmap='coolwarm', center=0, cbar_kwargs={'label': 'Mean Δ Weighted Degree', 'shrink': 0.8})
    plt.title('Spatial Distribution: Mean Change in Weighted Degree\n(Averaged across all months)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_degree_change_spatial_map.png', dpi=300)
    plt.close()

    # --- Plot 4: Per-Month Change Histograms ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    for i, month_val in enumerate(diff_da.month.values):
        month_data = diff_da.sel(month=month_val).values.flatten()
        month_data = month_data[~np.isnan(month_data)]
        
        sns.histplot(month_data, ax=axes[i], color='#0ea5e9', bins=30, kde=True)
        axes[i].axvline(0, color='black', linestyle='--', linewidth=1.2)
        axes[i].set_title(month_names[i], fontsize=12)
        
        if i >= 8: axes[i].set_xlabel('Δ Weighted Degree', fontsize=10)
        else: axes[i].set_xlabel('')
        if i % 4 == 0: axes[i].set_ylabel('Count', fontsize=10)
        else: axes[i].set_ylabel('')

    plt.suptitle('Monthly Change in Weighted Degree (Late − Early)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'weighted_degree_change_per_month_histograms.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()