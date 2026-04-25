"""High-level orchestrator: run the whole analysis for one (lat, lon) cell.

Caching is at this layer (one ``st.cache_data`` call) so the page can move
the user back and forth without paying the heavy network-build cost twice
for the same coordinate. The cache key is the rounded coordinate plus the
target-degree factor, so two slightly different click coordinates that snap
to the same source cell will share a cache entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from .community import CommunityComparison, detect_for_periods
from .degree import DegreeBundle, collect_degree_distributions
from .network_builder import CellNetworks, build_networks
from .percolation import PercolationResult, run_percolation
from .prototypes import PrototypeResult, compute_prototypes
from .source_data import grid_axes


@dataclass
class LocationAnalysis:
    """Everything the analysis page needs to render."""
    lat: float
    lon: float
    snapped_lat: float
    snapped_lon: float
    networks: CellNetworks
    prototypes: Optional[PrototypeResult]
    percolation: Optional[PercolationResult]
    communities: CommunityComparison
    degrees: DegreeBundle


def snap_to_grid(lat: float, lon: float) -> tuple[float, float]:
    """Return the nearest cell centre on the source grid.

    Used both before kicking off the build (so we can show the user the
    coordinate the analysis will actually run on) and as part of the cache
    key (so jittery click coordinates collapse onto a single entry).
    """
    lats, lons = grid_axes()
    snapped_lat = min(lats, key=lambda v: abs(v - lat))
    snapped_lon = min(lons, key=lambda v: abs(v - lon))
    return float(snapped_lat), float(snapped_lon)


@st.cache_data(show_spinner="Building network and computing analyses…")
def run_analysis(lat: float, lon: float, target_avg_degree_factor: int = 3) -> Optional[LocationAnalysis]:
    """Drive the full pipeline for one cell. ``None`` on missing data."""
    snapped_lat, snapped_lon = snap_to_grid(lat, lon)

    networks = build_networks(snapped_lat, snapped_lon, target_avg_degree_factor=target_avg_degree_factor)
    if networks is None:
        return None

    prototypes = compute_prototypes(
        networks.early, networks.late, networks.full,
        early_threshold=networks.early_threshold,
        late_threshold=networks.late_threshold,
    )
    percolation = run_percolation(networks.early, networks.late)
    communities = detect_for_periods(networks.early, networks.late, resolution=1.0)
    degrees = collect_degree_distributions(networks.early, networks.late)

    return LocationAnalysis(
        lat=lat,
        lon=lon,
        snapped_lat=snapped_lat,
        snapped_lon=snapped_lon,
        networks=networks,
        prototypes=prototypes,
        percolation=percolation,
        communities=communities,
        degrees=degrees,
    )
