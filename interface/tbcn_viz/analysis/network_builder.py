"""Build early/late/full networks for a single grid cell.

Direct port of ``create_networks_with_shared_pca`` from
``networks/pipeline/create_networks_calculate_indexes.ipynb`` — same shared
PCA space, same threshold rule (prune to a target average degree). Adapted to
return the three :class:`networkx.Graph` objects in memory instead of writing
graphmls to disk.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import EARLY_PERIOD, FULL_PERIOD, LATE_PERIOD
from .source_data import extract_monthly_features


FEATURE_KEYS = ("mean_tn", "mean_tx", "mean_tg", "rr_sum", "mean_qq", "mean_hu")


@dataclass
class CellNetworks:
    """The three graphs the rest of the analysis pipeline consumes.

    ``early_threshold`` / ``late_threshold`` are the minimum edge weights
    surviving the pruning step for each period — the same values the
    pipeline notebook stores in ``network_rich_thresholds.json``. They feed
    the threshold-aware prototype heatmap (``min(early, late)``).
    """
    early: nx.Graph
    late: nx.Graph
    full: nx.Graph
    early_threshold: float
    late_threshold: float
    full_threshold: float


# ---------------------------------------------------------------------------
# PCA + similarity
# ---------------------------------------------------------------------------

def _build_design_matrix(monthly_nodes: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    months = list(monthly_nodes.keys())
    X = np.array(
        [[monthly_nodes[m][k] for k in FEATURE_KEYS] for m in months],
        dtype=float,
    )
    return X, months


def _fit_shared_pca(monthly_nodes: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str], StandardScaler, PCA]:
    """Fit StandardScaler + 2-component PCA on the full-period nodes.

    Returns the scaled scores (positions in the shared 2D space), the month
    keys in matching order, and the fitted transformers so the early/late
    subsets can be projected into the same space.
    """
    X, months = _build_design_matrix(monthly_nodes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    return scores, months, scaler, pca


def _similarity_from_scores(scores: np.ndarray, d_min: Optional[float] = None, d_max: Optional[float] = None) -> np.ndarray:
    distances = pdist(scores, metric="euclidean")
    if distances.size == 0:
        return np.empty((0, 0))
        
    c_min = distances.min() if d_min is None else d_min
    c_max = distances.max() if d_max is None else d_max
    
    spread = c_max - c_min
    if spread > 0:
        scaled = (distances - c_min) / spread
    else:
        scaled = np.zeros_like(distances)
        
    scaled = np.clip(scaled, 0.0, 1.0)
    return 1.0 - squareform(scaled)


def _project_into(monthly_nodes: Dict[str, Dict[str, float]], scaler: StandardScaler, pca: PCA) -> Tuple[np.ndarray, List[str]]:
    if not monthly_nodes:
        return np.empty((0, 0)), []
    X, months = _build_design_matrix(monthly_nodes)
    scores = pca.transform(scaler.transform(X))
    return scores, months


# ---------------------------------------------------------------------------
# Graph construction + pruning
# ---------------------------------------------------------------------------

def _prune_to_average_degree(G: nx.Graph, target_avg_degree: float, weight_attr: str = "weight") -> nx.Graph:
    """Drop the weakest edges until ``avg_degree`` ≈ ``target_avg_degree``."""
    n = G.number_of_nodes()
    if n == 0:
        return G

    target_edges = int((target_avg_degree * n) / 2)
    to_remove = G.number_of_edges() - target_edges
    if to_remove <= 0:
        return G

    sorted_edges = sorted(G.edges(data=True), key=lambda e: e[2].get(weight_attr, 0.0))
    weakest = sorted_edges[:to_remove]
    G.remove_edges_from([(u, v) for u, v, _ in weakest])
    return G


def _build_period_graph(
    monthly_nodes: Dict[str, Dict[str, float]],
    similarity: np.ndarray,
    months: List[str],
    target_avg_degree: float,
) -> Optional[nx.Graph]:
    if not monthly_nodes:
        return None

    G = nx.Graph()
    for month, attrs in monthly_nodes.items():
        G.add_node(month, **attrs)

    for i, j in itertools.combinations(range(len(months)), 2):
        score = similarity[i, j]
        if score > 0:
            G.add_edge(months[i], months[j], weight=float(score))

    pruned = _prune_to_average_degree(G, target_avg_degree)
    for node, data in pruned.nodes(data=True):
        try:
            year, month = str(node).split("-")
            data["year"] = int(year)
            data["month"] = int(month)
        except ValueError:
            data["year"] = None
            data["month"] = None
    return pruned


def _target_avg_degree(start: str, end: str, factor: int = 3) -> float:
    """Same heuristic as the pipeline notebook: ``years * factor - 1``."""
    return int(end[:4]) - int(start[:4]) + 1


def build_networks(lat: float, lon: float, target_avg_degree_factor: int = 3) -> Optional[CellNetworks]:
    """Build the early / late / full graphs for a single grid cell.

    Returns ``None`` if any period has no usable monthly data (all-NaN cells
    over the sea, for example).
    """
    full_start, full_end = FULL_PERIOD
    full_nodes = extract_monthly_features(lat, lon, full_start, full_end)
    if not full_nodes:
        return None

    full_scores, full_months, scaler, pca = _fit_shared_pca(full_nodes)
    
    full_distances = pdist(full_scores, metric="euclidean")
    if full_distances.size > 0:
        d_min, d_max = float(full_distances.min()), float(full_distances.max())
    else:
        d_min, d_max = 0.0, 1.0
        
    full_similarity = _similarity_from_scores(full_scores, d_min=d_min, d_max=d_max)

    period_specs = [
        ("early", EARLY_PERIOD),
        ("late", LATE_PERIOD),
        ("full", FULL_PERIOD),
    ]

    graphs: Dict[str, Optional[nx.Graph]] = {}
    thresholds: Dict[str, float] = {}
    for label, (start, end) in period_specs:
        if label == "full":
            nodes, similarity, months = full_nodes, full_similarity, full_months
        else:
            nodes = extract_monthly_features(lat, lon, start, end)
            scores, months = _project_into(nodes, scaler, pca)
            similarity = _similarity_from_scores(scores, d_min=d_min, d_max=d_max) if scores.size else np.empty((0, 0))

        years = int(end[:4]) - int(start[:4]) + 1
        target_avg_degree = years * target_avg_degree_factor - 1
        graph = _build_period_graph(nodes, similarity, months, target_avg_degree)
        graphs[label] = graph
        thresholds[label] = _min_edge_weight(graph) if graph is not None else 1.0

    if any(g is None for g in graphs.values()):
        return None
    return CellNetworks(
        early=graphs["early"], late=graphs["late"], full=graphs["full"],
        early_threshold=thresholds["early"],
        late_threshold=thresholds["late"],
        full_threshold=thresholds["full"],
    )


def _min_edge_weight(G: nx.Graph) -> float:
    """Smallest edge weight surviving the prune — i.e. the period's threshold."""
    if G.number_of_edges() == 0:
        return 1.0
    return float(min(d.get("weight", 1.0) for _, _, d in G.edges(data=True)))


@dataclass
class FullGraphPCA:
    """Everything the prototype module needs to project a synthetic prototype
    feature vector into the full graph's PCA space."""
    scores: np.ndarray
    nodes: List[str]
    scaler: StandardScaler
    pca: PCA
    d_min: float
    d_max: float


def fit_full_graph_pca(full_graph: nx.Graph) -> FullGraphPCA:
    """Fit StandardScaler + 2D PCA on the full-period graph.

    The mean-prototype analysis (mirroring
    ``rich_mean_prototype_analysis.py``) needs *both* the fitted
    transformers and the min/max raw-distance bounds in this space — it
    projects a synthetic mean vector and rescales the resulting distance
    against ``[d_min, d_max]``.
    """
    nodes = list(full_graph.nodes())
    X = np.array(
        [[full_graph.nodes[n].get(k, np.nan) for k in FEATURE_KEYS] for n in nodes],
        dtype=float,
    )
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    distances = pdist(scores, metric="euclidean")
    return FullGraphPCA(
        scores=scores,
        nodes=nodes,
        scaler=scaler,
        pca=pca,
        d_min=float(distances.min()) if distances.size else 0.0,
        d_max=float(distances.max()) if distances.size else 1.0,
    )
