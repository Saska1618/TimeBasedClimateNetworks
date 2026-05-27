"""
Analyze the change in Percolation Index from the early to late period.
Generates histograms and spatial distribution maps (no month dimension).
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

    if 'percolation_index' not in ds.data_vars:
        raise ValueError("'percolation_index' not found in the dataset variables!")

    # 2. Extract Data
    index_data = ds['percolation_index']
    
    # 3. Flatten valid (non-NaN) values
    flat_data = index_data.values.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]

    # Calculate statistics
    total_valid = len(flat_data)
    increased = np.sum(flat_data > 0)
    decreased = np.sum(flat_data < 0)
    unchanged = np.sum(flat_data == 0)
    
    pct_inc = (increased / total_valid) * 100
    pct_dec = (decreased / total_valid) * 100
    pct_unc = (unchanged / total_valid) * 100

    print(f"Total analyzed valid points (cells): {total_valid}")
    print(f"> 0: {pct_inc:.2f}% | < 0: {pct_dec:.2f}% | = 0: {pct_unc:.2f}%")

    # Set aesthetic style for seaborn
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Histogram of the Index ---
    plt.figure(figsize=(10, 6))
    sns.histplot(flat_data, color='#6366f1', bins=50, kde=True)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Distribution of Percolation Index', fontsize=14, pad=15)
    plt.xlabel('Percolation Index', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    stats_text = f"> 0: {pct_inc:.1f}%\n< 0: {pct_dec:.1f}%\n= 0: {pct_unc:.1f}%"
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(output_dir / 'percolation_index_histogram.png', dpi=300)
    plt.close()

    # --- Plot 2: Spatial Map ---
    plt.figure(figsize=(10, 8))
    index_data.plot(cmap='PiYG', center=0, cbar_kwargs={'label': 'Percolation Index', 'shrink': 0.8})
    plt.title('Spatial Distribution: Percolation Index', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / 'percolation_index_change_spatial_map.png', dpi=300)
    plt.close()

    print(f"\nPlots successfully saved to: {output_dir}")

if __name__ == '__main__':
    main()