#!/usr/bin/env python3
"""
Create 10 × 12 Shannon entropy heatmaps for the time-based climate-network study.

The script reads the Shannon entropy tables from all_locations.txt and creates
three publication-ready heatmaps:
  1. Early period Shannon entropy
  2. Late period Shannon entropy
  3. Difference: Late period minus Early period

Run from the folder containing all_locations.txt:
    python create_shannon_entropy_heatmaps.py

Or specify paths explicitly:
    python create_shannon_entropy_heatmaps.py --input all_locations.txt --output figures

Required packages:
    pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# -----------------------------------------------------------------------------
# Settings you may want to edit
# -----------------------------------------------------------------------------
MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Location order follows the order used in the study-area description.
LOCATION_ORDER = [
    "Győr", "Kecskemét", "Keszthely", "Pécs", "Košice",
    "Oradea", "Cluj-Napoca", "Gheorgheni", "Deva", "Brașov"
]

# Names as they appear as section titles in all_locations.txt -> plot labels.
LOCATION_NAME_MAP = {
    "BRASSÓ": "Brașov",
    "DÉVA": "Deva",
    "GYERGYÓSZENTMIKLÓS": "Gheorgheni",
    "GYŐR": "Győr",
    "KASSA": "Košice",
    "KECSKEMÉT": "Kecskemét",
    "KESZTHELY": "Keszthely",
    "KOLOZSVÁR": "Cluj-Napoca",
    "NAGYVÁRAD": "Oradea",
    "PÉCS": "Pécs",
}

MONTH_NAME_MAP = {
    "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Már": "Mar",
    "Apr": "Apr", "Ápr": "Apr", "May": "May", "Máj": "May",
    "Jun": "Jun", "Jún": "Jun", "Jul": "Jul", "Júl": "Jul",
    "Aug": "Aug", "Sep": "Sep", "Szept": "Sep",
    "Oct": "Oct", "Okt": "Oct", "Nov": "Nov", "Dec": "Dec",
}


# -----------------------------------------------------------------------------
# Parsing functions
# -----------------------------------------------------------------------------
def _to_float(value: str) -> float:
    """Convert numbers with optional plus signs and Unicode minus signs."""
    value = value.strip().replace("+", "").replace("−", "-").replace("–", "-")
    return float(value)


def parse_shannon_entropy_tables(input_txt: Path) -> pd.DataFrame:
    """
    Extract month-level Early, Late and Difference values from all location tables.

    Returns a long-format DataFrame with columns:
        location, month, early, late, difference
    """
    text = input_txt.read_text(encoding="utf-8")

    section_pattern = re.compile(
        r"(?m)^(BRASSÓ|DÉVA|GYERGYÓSZENTMIKLÓS|GYŐR|KASSA|KECSKEMÉT|KESZTHELY|KOLOZSVÁR|NAGYVÁRAD|PÉCS)\s*$"
    )
    row_pattern = re.compile(
        r"^(Jan|Feb|Mar|Már|Apr|Ápr|May|Máj|Jun|Jún|Jul|Júl|Aug|Sep|Szept|Oct|Okt|Nov|Dec)"
        r"\s+([+\-−–]?\d+(?:\.\d+)?)"
        r"\s+([+\-−–]?\d+(?:\.\d+)?)"
        r"\s+([+\-−–]?\d+(?:\.\d+)?)"
    )

    sections = list(section_pattern.finditer(text))
    if not sections:
        raise ValueError("No location sections were found in the input text file.")

    records: List[Dict[str, object]] = []

    for i, match in enumerate(sections):
        raw_location = match.group(1)
        location = LOCATION_NAME_MAP.get(raw_location, raw_location.title())
        start = match.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        section_text = text[start:end]

        month_count = 0
        for line in section_text.splitlines():
            row = row_pattern.match(line.strip())
            if not row:
                continue

            month = MONTH_NAME_MAP[row.group(1)]
            early = _to_float(row.group(2))
            late = _to_float(row.group(3))
            diff_from_file = _to_float(row.group(4))

            # Use Late - Early for numerical consistency. The value in the file is
            # kept only as a validation check.
            difference = late - early
            if abs(difference - diff_from_file) > 0.002:
                print(
                    f"Warning: difference mismatch for {location} {month}: "
                    f"computed {difference:.6f}, file {diff_from_file:.6f}"
                )

            records.append(
                {
                    "location": location,
                    "month": month,
                    "early": early,
                    "late": late,
                    "difference": difference,
                }
            )
            month_count += 1

        if month_count != 12:
            raise ValueError(
                f"Expected 12 monthly rows for {location}, but found {month_count}."
            )

    df = pd.DataFrame.from_records(records)

    # Reorder locations and months.
    df["location"] = pd.Categorical(df["location"], categories=LOCATION_ORDER, ordered=True)
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df = df.sort_values(["location", "month"]).reset_index(drop=True)

    # Basic integrity checks.
    expected_rows = len(LOCATION_ORDER) * len(MONTH_ORDER)
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, but parsed {len(df)} rows.")
    if df[["early", "late", "difference"]].isna().any().any():
        raise ValueError("Missing values were found after parsing the tables.")

    return df


def make_matrix(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return a location × month matrix for a selected value column."""
    return df.pivot(index="location", columns="month", values=value_column).loc[LOCATION_ORDER, MONTH_ORDER]


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------
def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_base: Path,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
    annotate: bool = True,
    dpi: int = 600,
) -> None:
    """Create one annotated heatmap and save it as PNG and PDF."""
    data = matrix.to_numpy(dtype=float)

    if center_zero:
        limit = np.nanmax(np.abs(data))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        imshow_kwargs = {"norm": norm}
    else:
        imshow_kwargs = {"vmin": vmin, "vmax": vmax}

    fig_width = 9.6
    fig_height = 6.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(data, aspect="auto", cmap=cmap, **imshow_kwargs)

    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Location", fontsize=11)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, fontsize=9)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    # Thin white grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        if center_zero:
            threshold = 0.0
        else:
            threshold = (np.nanmin(data) + np.nanmax(data)) / 2.0
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if center_zero:
                    text = f"{value:+.2f}"
                    text_color = "white" if abs(value) > 0.55 * np.nanmax(np.abs(data)) else "black"
                else:
                    text = f"{value:.2f}"
                    text_color = "white" if value > threshold else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=6.5, color=text_color)

    cbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Early, Late and Difference Shannon entropy heatmaps from all_locations.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("all_locations.txt"),
        help="Path to all_locations.txt. Default: all_locations.txt in the current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shannon_entropy_heatmaps"),
        help="Output folder for figures and extracted CSV data.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution of exported PNG files. Default: 600.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable numeric labels inside heatmap cells.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Put all_locations.txt next to this script "
            "or pass its path with --input."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    annotate = not args.no_annotations

    df = parse_shannon_entropy_tables(args.input)
    df.to_csv(args.output / "parsed_shannon_entropy_values.csv", index=False, encoding="utf-8-sig")

    early_matrix = make_matrix(df, "early")
    late_matrix = make_matrix(df, "late")
    difference_matrix = make_matrix(df, "difference")

    # Early and Late use the same scale so they can be compared directly.
    max_entropy = float(max(early_matrix.max().max(), late_matrix.max().max()))

    plot_heatmap(
        early_matrix,
        title="Early Period Shannon Entropy (1961–1990)",
        output_base=args.output / "shannon_entropy_early_heatmap",
        cmap="YlOrRd",
        cbar_label="Shannon entropy",
        vmin=0.0,
        vmax=max_entropy,
        annotate=annotate,
        dpi=args.dpi,
    )

    plot_heatmap(
        late_matrix,
        title="Late Period Shannon Entropy (1995–2024)",
        output_base=args.output / "shannon_entropy_late_heatmap",
        cmap="YlOrRd",
        cbar_label="Shannon entropy",
        vmin=0.0,
        vmax=max_entropy,
        annotate=annotate,
        dpi=args.dpi,
    )

    plot_heatmap(
        difference_matrix,
        title="Change in Shannon Entropy (Late − Early)",
        output_base=args.output / "shannon_entropy_difference_heatmap",
        cmap="RdBu_r",
        cbar_label="Δ Shannon entropy",
        center_zero=True,
        annotate=annotate,
        dpi=args.dpi,
    )

    print(f"Created heatmaps in: {args.output.resolve()}")
    print("Files:")
    for path in sorted(args.output.glob("shannon_entropy_*_heatmap.*")):
        print(f"  - {path}")
    print(f"  - {args.output / 'parsed_shannon_entropy_values.csv'}")


if __name__ == "__main__":
    main()
