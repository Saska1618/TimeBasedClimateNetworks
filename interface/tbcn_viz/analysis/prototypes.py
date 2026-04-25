"""Mean-based monthly prototypes for a single grid cell.

Direct port of ``rich_mean_prototype_creation`` + ``rich_mean_prototype_analysis``
adapted to operate on the in-memory graphs produced by
:mod:`tbcn_viz.analysis.network_builder` — no graphmls, no city dependency.

The prototype for ``early_<month>`` is the per-feature mean of every node
in the *early* graph whose calendar month matches ``<month>`` (likewise
for late). The 12 × 12 similarity matrix is then computed by:

1. Projecting each synthetic prototype feature vector through the *full*
   graph's StandardScaler + PCA (the shared 2-D space).
2. Computing Euclidean distance between the early-i and late-j projections.
3. Rescaling that distance against the full graph's [d_min, d_max] and
   flipping to a similarity in [0, 1].

We return *both* the unthresholded matrix (raw similarities) and the
thresholded one (similarities below ``min(early_threshold, late_threshold)``
zeroed). Two heatmaps, same data — same as the notebook code's
``USE_THRESHOLD = True/False`` toggle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from .network_builder import FEATURE_KEYS, fit_full_graph_pca


@dataclass
class PrototypeResult:
    """Both heatmap variants and the underlying mean prototypes."""
    similarity_no_threshold: np.ndarray   # shape (12, 12); rows = early, cols = late
    similarity_with_threshold: np.ndarray  # same shape, sub-threshold cells zeroed
    threshold: float
    early_prototypes: Dict[int, Dict[str, float]]  # month -> {feature -> mean value}
    late_prototypes: Dict[int, Dict[str, float]]


# ---------------------------------------------------------------------------
# Mean prototype
# ---------------------------------------------------------------------------

def _mean_prototype(graph: nx.Graph, node_ids: List[str]) -> Optional[Dict[str, float]]:
    """Per-feature mean across the subset of ``node_ids`` present in ``graph``.

    Returns ``None`` when the subset is empty or every feature is NaN —
    that prototype slot stays unfilled and the corresponding row/column
    in the similarity matrix is left at zero, matching the notebook.
    """
    valid = [n for n in node_ids if n in graph.nodes]
    if not valid:
        return None

    X = np.array(
        [[graph.nodes[n].get(k, np.nan) for k in FEATURE_KEYS] for n in valid],
        dtype=float,
    )

    proto: Dict[str, float] = {}
    for i, key in enumerate(FEATURE_KEYS):
        col = X[:, i]
        if np.isnan(col).all():
            return None
        proto[key] = float(np.nanmean(col))
    return proto


def _project(proto: Dict[str, float], scaler, pca) -> np.ndarray:
    """Project a per-feature mean dict through scaler + PCA → 2-D coords."""
    vec = np.array([[proto[k] for k in FEATURE_KEYS]], dtype=float)
    return pca.transform(scaler.transform(vec))[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_prototypes(
    early_graph: nx.Graph,
    late_graph: nx.Graph,
    full_graph: nx.Graph,
    *,
    early_threshold: float,
    late_threshold: float,
) -> Optional[PrototypeResult]:
    """Build the mean-prototype similarity matrices for one cell.

    The threshold heatmap uses ``min(early_threshold, late_threshold)`` —
    the same rule as ``rich_mean_prototype_analysis.process_city``.
    """
    if full_graph.number_of_nodes() == 0:
        return None

    pca_info = fit_full_graph_pca(full_graph)
    if pca_info.d_max <= pca_info.d_min:
        return None

    early_protos: Dict[int, Dict[str, float]] = {}
    late_protos: Dict[int, Dict[str, float]] = {}

    for month in range(1, 13):
        m_str = f"{month:02d}"
        early_ids = [f"{y}-{m_str}" for y in range(1961, 1991)]
        late_ids = [f"{y}-{m_str}" for y in range(1995, 2025)]
        early_p = _mean_prototype(early_graph, early_ids)
        late_p = _mean_prototype(late_graph, late_ids)
        if early_p is not None:
            early_protos[month] = early_p
        if late_p is not None:
            late_protos[month] = late_p

    raw = np.zeros((12, 12), dtype=float)
    for i, j in product(range(1, 13), range(1, 13)):
        ep = early_protos.get(i)
        lp = late_protos.get(j)
        if ep is None or lp is None:
            continue
        e_score = _project(ep, pca_info.scaler, pca_info.pca)
        l_score = _project(lp, pca_info.scaler, pca_info.pca)
        dist = float(np.linalg.norm(e_score - l_score))
        scaled = (dist - pca_info.d_min) / (pca_info.d_max - pca_info.d_min)
        scaled = float(np.clip(scaled, 0.0, 1.0))
        raw[i - 1, j - 1] = 1.0 - scaled

    threshold = float(min(early_threshold, late_threshold))
    thresholded = raw.copy()
    thresholded[thresholded < threshold] = 0.0

    return PrototypeResult(
        similarity_no_threshold=raw,
        similarity_with_threshold=thresholded,
        threshold=threshold,
        early_prototypes=early_protos,
        late_prototypes=late_protos,
    )
