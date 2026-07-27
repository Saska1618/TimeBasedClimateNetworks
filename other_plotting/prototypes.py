#!/usr/bin/env python3
"""
Create a 2x5 grid of thresholded prototype heatmaps for 10 selected locations.

This script generates a publication-ready figure showing the thresholded
mean prototype similarity matrices for 10 locations in the Carpathian Basin.
Each heatmap in the 2x5 grid represents one location.

The script depends on the analysis pipeline from the `tbcn_viz` package.
Ensure that the required data (E-OBS NetCDFs) is available and the
`TBCN_EOBS_DIR` environment variable is set correctly if the data is not
in the default location.

Run from the project root directory:
    python other_plotting/prototypes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the 'interface' directory to the Python path to import tbcn_viz
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INTERFACE_DIR = _PROJECT_ROOT / "interface"
if str(_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_DIR))

from tbcn_viz.analysis import run_analysis
from tbcn_viz.analysis.pipeline import snap_to_grid

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Settings ---

# Locations and their coordinates (lat, lon)
# Order is preserved for plotting.
LOCATIONS = [
    ("Cluj-Napoca", (46.75, 23.65)),
    ("Gheorgheni", (46.75, 25.45)),
    ("Brașov", (45.75, 25.55)),
    ("Deva", (45.85, 22.95)),
    ("Pécs", (46.05, 18.25)),
    ("Győr", (47.75, 17.65)),
    ("Oradea", (47.15, 21.85)),
    ("Košice", (48.65, 21.25)),
    ("Kecskemét", (46.85, 19.75)),
    ("Keszthely", (46.75, 17.15)),
]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = "prototype_heatmaps_grid.png"

# --- Main Logic ---

def get_prototype_matrix(lat: float, lon: float) -> np.ndarray | None:
    """
    For a given lat/lon, run the analysis pipeline and return the
    thresholded prototype similarity matrix.
    """
    try:
        snapped_lat, snapped_lon = snap_to_grid(lat, lon)
        result = run_analysis(snapped_lat, snapped_lon)

        if not result or not result.prototypes:
            print(f"Warning: Could not get prototype result for ({lat}, {lon}).")
            return None

        return result.prototypes.similarity_with_threshold
    except Exception as e:
        print(f"Error processing location at ({lat}, {lon}): {e}")
        return None


def main():
    """
    Main function to generate and save the plot.
    """
    print("Generating prototype matrices for all locations...")
    matrices = {}
    for name, (lat, lon) in tqdm(LOCATIONS, desc="Processing locations"):
        matrix = get_prototype_matrix(lat, lon)
        if matrix is not None:
            matrices[name] = matrix

    if not matrices:
        print("No matrices were generated. Exiting.")
        return

    # --- Plotting ---
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.5), constrained_layout=True)
    axes = axes.flatten()

    # Find global max for a consistent color scale across all heatmaps
    vmax = max((m.max() for m in matrices.values() if m.size > 0), default=1.0)
    if vmax == 0:
        vmax = 1.0  # Avoid error if all matrices are zero

    im = None  # To hold the last image for colorbar
    for i, (location_name, _) in enumerate(LOCATIONS):
        ax = axes[i]
        matrix = matrices.get(location_name)

        ax.set_title(location_name, fontsize=12)

        if matrix is None:
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        im = ax.imshow(matrix, cmap='Reds', vmin=0, vmax=vmax, aspect='equal')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(MONTH_LABELS)))
        ax.set_yticks(np.arange(len(MONTH_LABELS)))

        is_first_col = i % 5 == 0
        is_bottom_row = i >= 5

        ax.set_xticklabels(MONTH_LABELS if is_bottom_row else [], fontsize=8, rotation=90)
        ax.set_yticklabels(MONTH_LABELS if is_first_col else [], fontsize=8)

        if is_first_col:
            ax.set_ylabel("Early Period", fontsize=10)
        if is_bottom_row:
            ax.set_xlabel("Late Period", fontsize=10)

        # Thin white grid lines between cells.
        ax.set_xticks(np.arange(-0.5, len(MONTH_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MONTH_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("Thresholded Mean Prototype Similarity", fontsize=16, fontweight='bold')

    # Add a single colorbar for the entire figure
    if im:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
        cbar.set_label("Similarity")

    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()