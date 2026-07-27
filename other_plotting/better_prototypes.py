#!/usr/bin/env python3
"""
Create a 2x5 grid of globally thresholded prototype heatmaps for
10 selected locations.

Processing logic
----------------
1. Retrieve the unthresholded prototype similarity matrix for each location.
2. Retrieve the location-specific prototype threshold for each location.
3. Determine the minimum threshold across all locations.
4. Apply this shared global minimum threshold to every unthresholded matrix:

       similarity < global minimum threshold -> 0
       similarity >= global minimum threshold -> original value retained

5. Plot the resulting matrices using:

       0 values                         -> very light blue
       global minimum threshold         -> very light red
       similarity value of 1            -> dark red

The same threshold and colour scale are used for all ten heatmaps.

Run from the project root directory:

    python other_plotting/prototypes.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.patches import Patch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Import tbcn_viz
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INTERFACE_DIR = _PROJECT_ROOT / "interface"

if str(_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_DIR))

from tbcn_viz.analysis import run_analysis
from tbcn_viz.analysis.pipeline import snap_to_grid


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

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

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = "prototype_heatmaps_grid_better.png"

OUTPUT_DPI = 300

# Colour assigned to values removed by the global threshold.
ZERO_COLOUR = "#DCEFF7"

# Start position within the standard Reds colour map.
# Increasing this value makes the lightest retained red darker.
RED_CMAP_START = 0.10


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class LocationPrototype:
    """
    Prototype information required for one location.
    """

    unthresholded_matrix: np.ndarray
    local_threshold: float


# ---------------------------------------------------------------------------
# Retrieve prototype data
# ---------------------------------------------------------------------------

def get_prototype_result(
    lat: float,
    lon: float,
) -> LocationPrototype | None:
    """
    Run the analysis pipeline for one coordinate.

    Returns
    -------
    LocationPrototype or None
        The unthresholded similarity matrix and its location-specific
        threshold, or None if the analysis cannot be completed.
    """
    try:
        snapped_lat, snapped_lon = snap_to_grid(lat, lon)

        result = run_analysis(
            snapped_lat,
            snapped_lon,
        )

        if result is None or result.prototypes is None:
            print(
                "Warning: Could not generate prototype result for "
                f"({lat}, {lon})."
            )
            return None

        matrix = np.asarray(
            result.prototypes.similarity_no_threshold,
            dtype=float,
        )

        threshold = float(
            result.prototypes.threshold
        )

        if matrix.shape != (12, 12):
            print(
                "Warning: Unexpected prototype matrix shape "
                f"{matrix.shape} for ({lat}, {lon})."
            )
            return None

        if not np.isfinite(threshold):
            print(
                "Warning: Non-finite prototype threshold for "
                f"({lat}, {lon})."
            )
            return None

        return LocationPrototype(
            unthresholded_matrix=matrix,
            local_threshold=threshold,
        )

    except Exception as exc:
        print(
            f"Error processing location at ({lat}, {lon}): {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

def create_red_colormap() -> colors.LinearSegmentedColormap:
    """
    Create a truncated red colour map for retained similarity values.

    Masked values are displayed in very light blue.
    """
    base_cmap = plt.get_cmap("Reds")

    red_values = base_cmap(
        np.linspace(
            RED_CMAP_START,
            1.0,
            256,
        )
    )

    cmap = colors.LinearSegmentedColormap.from_list(
        "prototype_reds",
        red_values,
    )

    # Zero values are masked before plotting and receive this colour.
    cmap.set_bad(ZERO_COLOUR)

    return cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Retrieve unthresholded matrices, calculate the shared global threshold,
    apply it to all matrices, and create the 2x5 heatmap grid.
    """
    print("Generating unthresholded prototype matrices...")

    prototype_results: dict[str, LocationPrototype] = {}

    for location_name, (lat, lon) in tqdm(
        LOCATIONS,
        desc="Processing locations",
    ):
        prototype_result = get_prototype_result(
            lat,
            lon,
        )

        if prototype_result is not None:
            prototype_results[location_name] = prototype_result

    if not prototype_results:
        print("No prototype matrices were generated. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Determine the shared global minimum threshold
    # -----------------------------------------------------------------------

    global_min_threshold = min(
        result.local_threshold
        for result in prototype_results.values()
    )

    print("\nLocation-specific thresholds:")

    for location_name, _ in LOCATIONS:
        result = prototype_results.get(location_name)

        if result is not None:
            print(
                f"  {location_name:<14} "
                f"{result.local_threshold:.6f}"
            )

    print(
        "\nGlobal minimum threshold applied to all matrices: "
        f"{global_min_threshold:.6f}"
    )

    # -----------------------------------------------------------------------
    # Apply the global threshold to every unthresholded matrix
    # -----------------------------------------------------------------------

    globally_thresholded_matrices: dict[str, np.ndarray] = {}

    for location_name, result in prototype_results.items():
        matrix = result.unthresholded_matrix.copy()

        # Replace NaN or infinite values with zero.
        matrix[~np.isfinite(matrix)] = 0.0

        # Apply the same global threshold to every unthresholded matrix.
        matrix[matrix < global_min_threshold] = 0.0

        globally_thresholded_matrices[location_name] = matrix

    # -----------------------------------------------------------------------
    # Shared colour normalization
    # -----------------------------------------------------------------------

    red_cmap = create_red_colormap()

    norm_vmin = global_min_threshold

    # Prevent an invalid normalization range if the threshold equals 1.
    if norm_vmin >= 1.0:
        norm_vmin = np.nextafter(1.0, 0.0)

    norm = colors.Normalize(
        vmin=norm_vmin,
        vmax=1.0,
        clip=True,
    )

    # -----------------------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(18, 9.2),
        constrained_layout=False,
    )

    axes = axes.flatten()

    image_handle = None

    for index, (location_name, _) in enumerate(LOCATIONS):
        ax = axes[index]

        matrix = globally_thresholded_matrices.get(
            location_name
        )

        ax.set_title(
            location_name,
            fontsize=12,
        )

        if matrix is None:
            ax.text(
                0.5,
                0.5,
                "Data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            continue

        # Mask zero values so they receive ZERO_COLOUR.
        masked_matrix = np.ma.masked_equal(
            matrix,
            0.0,
        )

        image_handle = ax.imshow(
            masked_matrix,
            cmap=red_cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )

        # Major ticks at cell centres.
        ax.set_xticks(
            np.arange(len(MONTH_LABELS))
        )

        ax.set_yticks(
            np.arange(len(MONTH_LABELS))
        )

        is_first_column = index % 5 == 0
        is_bottom_row = index >= 5

        if is_bottom_row:
            ax.set_xticklabels(
                MONTH_LABELS,
                fontsize=8,
                rotation=90,
            )
        else:
            ax.set_xticklabels([])

        if is_first_column:
            ax.set_yticklabels(
                MONTH_LABELS,
                fontsize=8,
            )
        else:
            ax.set_yticklabels([])

        if is_first_column:
            ax.set_ylabel(
                "Early Period",
                fontsize=10,
            )

        if is_bottom_row:
            ax.set_xlabel(
                "Late Period",
                fontsize=10,
                labelpad=8,
            )

        # Minor ticks define the cell borders.
        ax.set_xticks(
            np.arange(
                -0.5,
                len(MONTH_LABELS),
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                len(MONTH_LABELS),
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=0.5,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        ax.tick_params(
            which="major",
            length=0,
        )

    fig.suptitle(
        "Mean Prototype Similarity Using the Global Minimum Threshold",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    # Reserve a dedicated area on the right for the colourbar.
    # The heatmap grid ends at x=0.88, while the colourbar begins at x=0.91.
    fig.subplots_adjust(
        left=0.055,
        right=0.88,
        top=0.88,
        bottom=0.20,
        wspace=0.28,
        hspace=0.38,
    )

    # -----------------------------------------------------------------------
    # Shared colourbar
    # -----------------------------------------------------------------------

    if image_handle is not None:
        # Dedicated colourbar axis:
        # [left, bottom, width, height] in figure coordinates.
        colourbar_axis = fig.add_axes(
            [0.91, 0.27, 0.015, 0.51]
        )

        colourbar = fig.colorbar(
            image_handle,
            cax=colourbar_axis,
        )

        colourbar.set_label(
            "Similarity",
            fontsize=11,
            labelpad=10,
        )

        colourbar.ax.tick_params(
            labelsize=9,
        )

    # -----------------------------------------------------------------------
    # Legend for below-threshold values
    # -----------------------------------------------------------------------

    zero_patch = Patch(
        facecolor=ZERO_COLOUR,
        edgecolor="0.65",
        linewidth=0.6,
        label=(
            "Below global threshold "
            f"(< {global_min_threshold:.3f})"
        ),
    )

    fig.legend(
        handles=[zero_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        fontsize=10,
    )

    # -----------------------------------------------------------------------
    # Save figure
    # -----------------------------------------------------------------------

    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    fig.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()