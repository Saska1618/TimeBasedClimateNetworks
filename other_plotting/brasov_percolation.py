#!/usr/bin/env python3
"""
Create a publication-ready percolation plot for Brașov.

This script runs the analysis pipeline for Brașov, extracts the percolation
results, and generates a high-quality Matplotlib figure comparing the
early (1961–1990) and late (1995–2024) periods.

Run from the project root directory:
    python other_plotting/brasov_percolation.py
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

import matplotlib.pyplot as plt

# --- Settings ---
LOCATION_NAME = "Cluj-Napoca"
LAT, LON = 46.77, 23.62

# Colors matching the interface (blue and orange)
COLOR_EARLY = "#2563eb"
COLOR_LATE = "#ea580c"

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME_STEM = "brasov_percolation"

def main():
    print(f"Running analysis for {LOCATION_NAME} at ({LAT}, {LON})...")
    snapped_lat, snapped_lon = snap_to_grid(LAT, LON)
    result = run_analysis(snapped_lat, snapped_lon)

    if not result or not result.percolation:
        print(f"Error: Could not compute percolation for {LOCATION_NAME}.")
        return

    perc = result.percolation

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Plot early period
    if perc.early:
        ax.plot(
            perc.early.thresholds, 
            perc.early.lcc_fractions, 
            label="Early (1961–1990)", 
            color=COLOR_EARLY, 
            linewidth=2.5,
            marker="o",
            markersize=4,
            zorder=3
        )
        if perc.early.critical_threshold is not None:
            ax.axvline(
                perc.early.critical_threshold, 
                color=COLOR_EARLY, 
                linestyle="--", 
                linewidth=1.5,
                alpha=0.8,
                label=f"Early split: {perc.early.critical_threshold:.3f}",
                zorder=2
            )

    # Plot late period
    if perc.late:
        ax.plot(
            perc.late.thresholds, 
            perc.late.lcc_fractions, 
            label="Late (1995–2024)", 
            color=COLOR_LATE, 
            linewidth=2.5,
            marker="o",
            markersize=4,
            zorder=3
        )
        if perc.late.critical_threshold is not None:
            ax.axvline(
                perc.late.critical_threshold, 
                color=COLOR_LATE, 
                linestyle="--", 
                linewidth=1.5,
                alpha=0.8,
                label=f"Late split: {perc.late.critical_threshold:.3f}",
                zorder=2
            )

    ax.set_title(f"Percolation Comparison — {LOCATION_NAME}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Similarity threshold (edge weight)", fontsize=11)
    ax.set_ylabel("Fraction of nodes in largest component", fontsize=11)

    # Styling
    ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Place legend outside the plot, at the bottom left
    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        framealpha=0.9,
        edgecolor="none",
        ncol=2
    )

    fig.tight_layout()

    # Save output
    output_png = OUTPUT_DIR / f"{OUTPUT_FILENAME_STEM}.png"
    output_pdf = OUTPUT_DIR / f"{OUTPUT_FILENAME_STEM}.pdf"

    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    
    print(f"\nSaved plots to:\n  - {output_png}\n  - {output_pdf}")

if __name__ == "__main__":
    main()