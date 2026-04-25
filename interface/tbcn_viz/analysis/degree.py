"""Degree distributions per month (unweighted, weighted, all/close-month).

Mirrors the four scripts under ``networks/degree_distribution`` — the layout
and bin-sharing rules are identical:

* ``unweighted_all`` / ``weighted_all`` use *all* of a node's neighbours.
* ``unweighted_close`` / ``weighted_close`` only count neighbours whose month
  is the previous, current, or next calendar month (wrapping Dec↔Jan).

All four variants are produced for both periods so the page can render any
combination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


VARIANTS = ("unweighted_all", "weighted_all", "unweighted_close", "weighted_close")
PERIODS = ("early", "late")


def _close_months(month: int) -> set[int]:
    prev_m = 12 if month == 1 else month - 1
    next_m = 1 if month == 12 else month + 1
    return {prev_m, month, next_m}


@dataclass
class DegreeBundle:
    """All four variants × 12 months × 2 periods, ready to plot.

    ``data[variant][period][month]`` is the list of node degrees making up the
    histogram for that combination.
    """
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = field(default_factory=dict)

    def variant_range(self, variant: str) -> Tuple[float, float]:
        """Min / max across all months and periods for one variant.

        Used to anchor a single set of bins so the 12 monthly subplots line up.
        """
        values: List[float] = []
        for period in PERIODS:
            for m in range(1, 13):
                values.extend(self.data[variant][period][m])
        if not values:
            return 0.0, 1.0
        return float(min(values)), float(max(values))


def _empty_bundle() -> DegreeBundle:
    bundle = DegreeBundle()
    for variant in VARIANTS:
        bundle.data[variant] = {p: {m: [] for m in range(1, 13)} for p in PERIODS}
    return bundle


def _collect(bundle: DegreeBundle, period: str, G: nx.Graph) -> None:
    for node in G.nodes():
        try:
            month = int(node.split("-")[1])
        except (IndexError, ValueError):
            continue

        unw_all = G.degree(node)
        w_all = G.degree(node, weight="weight")

        close = _close_months(month)
        unw_close = 0
        w_close = 0.0
        for neighbor in G.neighbors(node):
            try:
                n_month = int(neighbor.split("-")[1])
            except (IndexError, ValueError):
                continue
            if n_month in close:
                unw_close += 1
                w_close += G[node][neighbor].get("weight", 1.0)

        bundle.data["unweighted_all"][period][month].append(float(unw_all))
        bundle.data["weighted_all"][period][month].append(float(w_all))
        bundle.data["unweighted_close"][period][month].append(float(unw_close))
        bundle.data["weighted_close"][period][month].append(float(w_close))


def collect_degree_distributions(early: nx.Graph, late: nx.Graph) -> DegreeBundle:
    bundle = _empty_bundle()
    _collect(bundle, "early", early)
    _collect(bundle, "late", late)
    return bundle
