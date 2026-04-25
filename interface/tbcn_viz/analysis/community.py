"""Leiden community detection at the default resolution (1.0).

Combines the parts of ``networks/community_detection/leiden`` and
``networks/community_detection/analysis`` that we need for a single cell.
The output bundles:

* the raw partition (community → list of "YYYY-MM" nodes),
* a "community perspective" matrix — what share of each community comes from
  each calendar month,
* a "month perspective" matrix — what share of each month falls into each
  community.

Run for both early and late so the page can show before/after side by side.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx
import numpy as np


_MONTH_NUMS = [f"{m:02d}" for m in range(1, 13)]


@dataclass
class CommunityResult:
    """One Leiden run on one period of one cell."""
    community_to_nodes: Dict[str, List[str]]
    modularity: float
    num_communities: int
    # Heatmap-ready arrays. ``communities`` lists the community labels in
    # the same order they appear as rows/columns in the matrices.
    communities: List[str] = field(default_factory=list)
    community_perspective: np.ndarray = field(default_factory=lambda: np.zeros((0, 12)))
    month_perspective: np.ndarray = field(default_factory=lambda: np.zeros((12, 0)))


@dataclass
class CommunityComparison:
    early: CommunityResult
    late: CommunityResult


# ---------------------------------------------------------------------------
# igraph conversion + Leiden run
# ---------------------------------------------------------------------------

def _convert_to_igraph(G: nx.Graph):
    import igraph as ig  # imported lazily so the rest of the app doesn't depend on it

    nodes = list(G.nodes())
    name_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(directed=G.is_directed())
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes

    edges_data = list(G.edges(data=True))
    g.add_edges([(name_to_idx[u], name_to_idx[v]) for u, v, _ in edges_data])
    if edges_data and "weight" in edges_data[0][2]:
        g.es["weight"] = [d.get("weight", 1.0) for _, _, d in edges_data]
    return g


def _detect(g_ig, resolution: float = 1.0) -> tuple[Dict[str, List[str]], float]:
    import leidenalg

    weights = g_ig.es["weight"] if "weight" in g_ig.edge_attributes() else None
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=42,
    )
    communities: Dict[str, List[str]] = {}
    for idx, cluster in enumerate(partition):
        communities[f"Community_{idx + 1}"] = [g_ig.vs[v]["name"] for v in cluster]
    return communities, float(partition.quality())


# ---------------------------------------------------------------------------
# Perspective matrices
# ---------------------------------------------------------------------------

def _perspective_matrices(
    communities: Dict[str, List[str]],
) -> tuple[List[str], np.ndarray, np.ndarray]:
    """Return community labels, community-perspective and month-perspective matrices.

    * community_perspective: shape (n_comm, 12) — rows sum to 100 (or 0 if empty).
    * month_perspective:     shape (12, n_comm) — rows sum to 100 (or 0 if empty).
    """
    labels = list(communities.keys())
    comm_counts = defaultdict(lambda: defaultdict(int))   # comm -> month -> count
    month_counts = defaultdict(lambda: defaultdict(int))  # month -> comm -> count
    comm_totals: Dict[str, int] = defaultdict(int)
    month_totals: Dict[str, int] = defaultdict(int)

    for comm, nodes in communities.items():
        for node in nodes:
            try:
                month = node.split("-")[1]
            except IndexError:
                continue
            comm_counts[comm][month] += 1
            month_counts[month][comm] += 1
            comm_totals[comm] += 1
            month_totals[month] += 1

    cp = np.zeros((len(labels), 12), dtype=float)
    mp = np.zeros((12, len(labels)), dtype=float)
    for r, comm in enumerate(labels):
        total = comm_totals[comm]
        if total == 0:
            continue
        for c, m in enumerate(_MONTH_NUMS):
            cp[r, c] = comm_counts[comm][m] / total * 100.0
    for r, m in enumerate(_MONTH_NUMS):
        total = month_totals[m]
        if total == 0:
            continue
        for c, comm in enumerate(labels):
            mp[r, c] = month_counts[m][comm] / total * 100.0
    return labels, cp, mp


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_communities(G: nx.Graph, resolution: float = 1.0) -> CommunityResult:
    g_ig = _convert_to_igraph(G)
    communities, modularity = _detect(g_ig, resolution=resolution)
    labels, cp, mp = _perspective_matrices(communities)
    return CommunityResult(
        community_to_nodes=communities,
        modularity=modularity,
        num_communities=len(communities),
        communities=labels,
        community_perspective=cp,
        month_perspective=mp,
    )


def detect_for_periods(early: nx.Graph, late: nx.Graph, resolution: float = 1.0) -> CommunityComparison:
    return CommunityComparison(
        early=detect_communities(early, resolution=resolution),
        late=detect_communities(late, resolution=resolution),
    )
