"""Per-index display configuration (labels, colormaps, descriptions).

Centralising this lets every page share consistent styling and hover labels
when adding a new visualization for the same index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class IndexSpec:
    name: str
    label: str
    description: str
    colorscale: str = "Viridis"
    diverging: bool = False
    center: Optional[float] = None
    has_month: bool = True
    has_period: bool = True


_SPECS: Dict[str, IndexSpec] = {
    "percolation_index": IndexSpec(
        name="percolation_index",
        label="Percolation Index",
        description=(
            "Difference between the late- and early-period critical percolation "
            "thresholds (late − early). Positive values mean the network "
            "fragments at a higher edge weight in the late period."
        ),
        colorscale="RdBu_r",
        diverging=True,
        center=0.0,
        has_month=False,
        has_period=False,
    ),
    "num_communities": IndexSpec(
        name="num_communities",
        label="Number of Communities",
        description=(
            "Count of distinct Leiden communities containing nodes from the "
            "selected month."
        ),
    ),
    "shannon_entropy": IndexSpec(
        name="shannon_entropy",
        label="Shannon Entropy",
        description="Shannon entropy of the community distribution for the month.",
    ),
    "normalized_entropy": IndexSpec(
        name="normalized_entropy",
        label="Normalized Entropy",
        description=(
            "Shannon entropy normalised to [0, 1] by the number of nodes in the "
            "month — directly comparable across months and grid cells."
        ),
    ),
    "simpson_index": IndexSpec(
        name="simpson_index",
        label="Simpson Index",
        description=(
            "1 − Σ pᵢ² over community shares. Higher values mean a more even "
            "spread across communities."
        ),
    ),
    "unweighted_degree_fragmentation": IndexSpec(
        name="unweighted_degree_fragmentation",
        label="Unweighted Degree Fragmentation",
        description=(
            "Share of unweighted degree connecting to neighbouring (±1) months "
            "vs. all months. Higher values mean stronger local connections."
        ),
    ),
    "weighted_degree_fragmentation": IndexSpec(
        name="weighted_degree_fragmentation",
        label="Weighted Degree Fragmentation",
        description=(
            "Share of weighted degree connecting to neighbouring (±1) months "
            "vs. all months. Higher values mean stronger local connections."
        ),
    ),
}

# Indexes whose 'ratio_late_to_early' period is naturally diverging around 1.
_RATIO_DIVERGING_INDEXES = frozenset(_SPECS.keys()) - {"percolation_index"}


def all_specs() -> Dict[str, IndexSpec]:
    return _SPECS


def get_spec(name: str) -> IndexSpec:
    return _SPECS[name]


def is_ratio_diverging(name: str) -> bool:
    """True when the index, viewed under 'ratio_late_to_early', should diverge around 1."""
    return name in _RATIO_DIVERGING_INDEXES
