"""Per-location network analysis: build the network for one grid cell on
demand and produce the prototype, percolation, community-detection, and
degree-distribution figures.

The public entry point is :func:`run_analysis`, which is the single function
the Location Analysis page calls. Everything else is internal scaffolding
exposed only for testing.
"""
from __future__ import annotations

from .pipeline import LocationAnalysis, run_analysis

__all__ = ["LocationAnalysis", "run_analysis"]
