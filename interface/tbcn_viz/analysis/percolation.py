"""Percolation curves for the early/late networks of a single cell.

Mirrors ``networks/percolation/percolation_analysis.py`` but operates on the
graphs already in memory and returns numerical arrays (the page itself draws
the Plotly figure via ``figures.percolation_figure``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np


@dataclass
class PercolationCurve:
    thresholds: np.ndarray
    lcc_fractions: np.ndarray
    critical_threshold: Optional[float]


@dataclass
class PercolationResult:
    early: PercolationCurve
    late: PercolationCurve


def _run_one(G: nx.Graph, thresholds: np.ndarray) -> PercolationCurve:
    n = G.number_of_nodes()
    fractions = np.empty(thresholds.shape, dtype=float)

    for idx, threshold in enumerate(thresholds):
        kept_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("weight", 1.0) >= threshold
        ]
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(kept_edges)
        if H.number_of_edges() == 0:
            fractions[idx] = 1.0 / n if n else 0.0
        else:
            largest = max(nx.connected_components(H), key=len)
            fractions[idx] = len(largest) / n

    drops = -np.diff(fractions)
    if drops.size and drops.max() > 0:
        critical = float(thresholds[int(np.argmax(drops))])
    else:
        critical = None
    return PercolationCurve(thresholds=thresholds, lcc_fractions=fractions, critical_threshold=critical)


def run_percolation(early: nx.Graph, late: nx.Graph, num_steps: int = 50) -> Optional[PercolationResult]:
    early_w = [d.get("weight", 1.0) for _, _, d in early.edges(data=True)]
    late_w = [d.get("weight", 1.0) for _, _, d in late.edges(data=True)]
    all_w = early_w + late_w
    if not all_w:
        return None
    thresholds = np.linspace(min(all_w), max(all_w), num_steps)
    return PercolationResult(
        early=_run_one(early, thresholds),
        late=_run_one(late, thresholds),
    )
