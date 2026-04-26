"""Plotly figure builders for the analysis page.

Every function takes the analysis dataclasses defined in this subpackage and
returns a ready-to-display ``plotly.graph_objects.Figure``. Keeping the
plotting code separate from the numerical work means each figure can be
unit-tested or re-themed independently.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import networkx as nx

from .community import CommunityResult
from .degree import DegreeBundle
from .percolation import PercolationResult
from .prototypes import PrototypeResult


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_PERIOD_COLOR = {"early": "#2563eb", "late": "#ea580c"}  # blue / orange — matches the matplotlib originals
_PERIOD_LABEL = {"early": "Early (1961–1990)", "late": "Late (1995–2024)"}


# ---------------------------------------------------------------------------
# Prototype heatmap
# ---------------------------------------------------------------------------

def prototype_heatmap(matrix: np.ndarray, *, title: str) -> go.Figure:
    """One similarity heatmap. Used twice — for the with- and without-threshold variants."""
    finite = np.isfinite(matrix)
    zmax = float(matrix[finite].max()) if finite.any() and matrix[finite].max() > 0 else 1.0
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=MONTH_LABELS,
            y=MONTH_LABELS,
            colorscale="Viridis",
            zmin=0.0,
            zmax=zmax,
            text=[[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in matrix],
            texttemplate="%{text}",
            hovertemplate=(
                "Early month: %{y}<br>"
                "Late month: %{x}<br>"
                "Similarity: %{z:.3f}<extra></extra>"
            ),
            colorbar=dict(title="Similarity", thickness=12),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(title="Late period (1995–2024)", side="bottom"),
        yaxis=dict(title="Early period (1961–1990)", autorange="reversed"),
        height=560,
        margin=dict(l=70, r=30, t=70, b=60),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Percolation comparison
# ---------------------------------------------------------------------------

def percolation_figure(result: PercolationResult, *, title: str = "Percolation comparison") -> go.Figure:
    fig = go.Figure()

    for period_key in ("early", "late"):
        curve = result.early if period_key == "early" else result.late
        color = _PERIOD_COLOR[period_key]
        label = _PERIOD_LABEL[period_key]

        fig.add_trace(go.Scatter(
            x=curve.thresholds,
            y=curve.lcc_fractions,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate="threshold %{x:.3f}<br>LCC %{y:.3f}<extra>" + label + "</extra>",
        ))
        if curve.critical_threshold is not None:
            fig.add_vline(
                x=curve.critical_threshold,
                line=dict(color=color, dash="dash", width=1.5),
                annotation_text=f"{label} split: {curve.critical_threshold:.3f}",
                annotation_position="top right" if period_key == "late" else "top left",
                annotation_font_size=11,
                annotation_font_color=color,
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Similarity threshold (edge weight)",
        yaxis_title="Fraction of nodes in largest component",
        height=480,
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Community heatmaps + scatter
# ---------------------------------------------------------------------------

def community_perspective_heatmap(result: CommunityResult, *, period_label: str) -> go.Figure:
    if result.num_communities == 0:
        return _empty("No communities detected.")
    fig = go.Figure(
        data=go.Heatmap(
            z=result.community_perspective,
            x=MONTH_LABELS,
            y=result.communities,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            text=[[f"{v:.1f}" for v in row] for row in result.community_perspective],
            texttemplate="%{text}",
            hovertemplate=(
                "%{y} · %{x}<br>%{z:.1f}% of community<extra></extra>"
            ),
            colorbar=dict(title="% of community", thickness=12),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Community perspective — {period_label}<br>"
                 f"<span style='font-size:12px;color:#888'>"
                 f"What months make up each community? "
                 f"(rows sum to 100%) · modularity {result.modularity:.3f}</span>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Month",
        yaxis=dict(title="Community", autorange="reversed"),
        height=380 + 18 * max(0, result.num_communities - 4),
        template="plotly_white",
        margin=dict(l=70, r=30, t=80, b=50),
    )
    return fig


def month_perspective_heatmap(result: CommunityResult, *, period_label: str) -> go.Figure:
    if result.num_communities == 0:
        return _empty("No communities detected.")
    fig = go.Figure(
        data=go.Heatmap(
            z=result.month_perspective,
            x=result.communities,
            y=MONTH_LABELS,
            colorscale="Oranges",
            zmin=0,
            zmax=100,
            text=[[f"{v:.1f}" for v in row] for row in result.month_perspective],
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}<br>%{z:.1f}% of month<extra></extra>",
            colorbar=dict(title="% of month", thickness=12),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Month perspective — {period_label}<br>"
                 f"<span style='font-size:12px;color:#888'>"
                 f"Where do months belong? (rows sum to 100%)</span>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Community",
        yaxis=dict(title="Month", autorange="reversed"),
        height=440,
        template="plotly_white",
        margin=dict(l=70, r=30, t=80, b=50),
    )
    return fig


def community_scatter(result: CommunityResult, *, period_label: str) -> go.Figure:
    """Year × month scatter coloured by community — same idea as the
    matplotlib version under ``networks/community_detection/leiden``.
    """
    if result.num_communities == 0:
        return _empty("No communities detected.")
    palette = _qualitative_palette(result.num_communities)

    fig = go.Figure()
    all_years: List[int] = []
    for idx, (comm, nodes) in enumerate(result.community_to_nodes.items()):
        years, months = [], []
        for node in nodes:
            try:
                y, m = node.split("-")
                years.append(int(y))
                months.append(int(m))
            except ValueError:
                continue
        all_years.extend(years)
        fig.add_trace(go.Scatter(
            x=years,
            y=months,
            mode="markers",
            name=comm,
            marker=dict(color=palette[idx % len(palette)], size=9, line=dict(width=1, color="white")),
            hovertemplate="%{x}-%{y:02d}<extra>" + comm + "</extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Temporal community structure — {period_label}<br>"
                 f"<span style='font-size:12px;color:#888'>"
                 f"{result.num_communities} communities · modularity {result.modularity:.3f}</span>",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Year", range=([min(all_years) - 1, max(all_years) + 1] if all_years else None)),
        yaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=MONTH_LABELS,
            range=[0.5, 12.5],
        ),
        height=460,
        legend=dict(title="Community"),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Degree distributions
# ---------------------------------------------------------------------------

def monthly_degree_grid(
    bundle: DegreeBundle,
    *,
    variant: str,
    title: str,
    x_label: str,
) -> go.Figure:
    """4×3 grid of monthly histograms (early vs late) for one variant.

    Mirrors the seaborn version under
    ``networks/degree_distribution/{unweighted,weighted}/all_months/`` —
    overlaid density histograms with a KDE line per period.
    """
    lo, hi = bundle.variant_range(variant)
    bin_size = (hi - lo) / 30 if hi > lo else 1.0

    # Y range: compute it ourselves rather than letting Plotly auto-range
    # in shared mode. The auto-range follows the *tallest* KDE/histogram
    # peak across the 12 panels — fine when distributions are similar,
    # but a single narrow month flattens every other panel into a thin
    # line at the bottom. We pick the max actual density across the
    # panels we're about to draw and add a small headroom.
    y_max = max(
        (
            _panel_density_peak(bundle.data[variant][period][month], lo, hi, bin_size)
            for month in range(1, 13)
            for period in ("early", "late")
        ),
        default=1.0,
    )
    if y_max <= 0:
        y_max = 1.0
    y_range = [0.0, y_max * 1.10]

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=MONTH_LABELS,
        shared_xaxes="all",
        shared_yaxes="all",
        horizontal_spacing=0.05,
        vertical_spacing=0.07,
    )

    for month in range(1, 13):
        row = (month - 1) // 3 + 1
        col = (month - 1) % 3 + 1
        for period_key in ("early", "late"):
            values = bundle.data[variant][period_key][month]
            _add_hist_with_kde(
                fig,
                values=values,
                period_key=period_key,
                bin_lo=lo, bin_hi=hi, bin_size=bin_size,
                row=row, col=col,
                show_legend=(month == 1),
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        barmode="overlay",
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    # Lock both axes explicitly so every panel renders on the *same*
    # window — no auto-trim, no Plotly fudging.
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=y_range)
    fig.update_xaxes(title_text=x_label, row=4)
    fig.update_yaxes(title_text="Density", col=1)
    return fig


def per_month_degree_panel(
    bundle: DegreeBundle,
    *,
    month: int,
    title_prefix: str = "",
) -> go.Figure:
    """2×2 layout for one month, replicating ``per_month/analyze_specific_months.py``."""
    titles = [
        "Unweighted · all months", "Weighted · all months",
        "Unweighted · close months", "Weighted · close months",
    ]
    variants = ["unweighted_all", "weighted_all", "unweighted_close", "weighted_close"]
    cells = [(1, 1), (1, 2), (2, 1), (2, 2)]

    # Single global bin range across all four variants, matching the
    # seaborn version: it concatenates every variant × period for the
    # month before deriving ``common_bins``. With ``shared_*='all'``
    # below, every panel ends up on identical X / Y axes.
    all_values: List[float] = []
    for variant in variants:
        for period_key in ("early", "late"):
            all_values.extend(bundle.data[variant][period_key][month])
    if all_values:
        lo = float(min(all_values))
        hi = float(max(all_values))
    else:
        lo, hi = 0.0, 1.0
    bin_size = (hi - lo) / 30 if hi > lo else 1.0

    # Pre-compute the global Y max so the four shared-axis panels render
    # on the same window without one variant's narrow peak crushing the
    # others into a flat line at the bottom.
    y_max = max(
        (
            _panel_density_peak(bundle.data[variant][period][month], lo, hi, bin_size)
            for variant in variants
            for period in ("early", "late")
        ),
        default=1.0,
    )
    if y_max <= 0:
        y_max = 1.0
    y_range = [0.0, y_max * 1.10]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        shared_xaxes="all",
        shared_yaxes="all",
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    for variant, (row, col) in zip(variants, cells):
        for period_key in ("early", "late"):
            values = bundle.data[variant][period_key][month]
            _add_hist_with_kde(
                fig,
                values=values,
                period_key=period_key,
                bin_lo=lo, bin_hi=hi, bin_size=bin_size,
                row=row, col=col,
                show_legend=(row == 1 and col == 1),
            )

    month_label = MONTH_LABELS[month - 1]
    fig.update_layout(
        title=dict(
            text=f"{title_prefix}Degree distributions · {month_label}",
            x=0.5, xanchor="center",
        ),
        barmode="overlay",
        height=620,
        template="plotly_white",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=y_range)
    fig.update_xaxes(title_text="Degree", row=2)
    fig.update_yaxes(title_text="Density", col=1)
    return fig


# ---------------------------------------------------------------------------
# Network graphs
# ---------------------------------------------------------------------------

def network_graph_figure(G: nx.Graph, *, title: str) -> go.Figure:
    """Interactive network visualization using Plotly and NetworkX spring layout."""
    if G is None or G.number_of_nodes() == 0:
        return _empty("Network is empty.")

    # Calculate layout
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.15)

    fig = go.Figure()

    # Edges trace
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        line=dict(width=0.2, color='#888'),
        opacity=0.3,
        hoverinfo='skip',
        mode='lines',
        showlegend=False
    ))

    # Nodes grouped by month
    palette = [
        "#1f77b4",         # 1: Jan (blue)
        "darkblue",        # 2: Feb (darkblue)
        "lightgreen",      # 3: Mar (lightgreen)
        "green",           # 4: Apr (green)
        "darkgreen",       # 5: May (darkgreen)
        "#ff7f7f",         # 6: Jun (lightred - hex for valid CSS)
        "red",             # 7: Jul (red)
        "darkred",         # 8: Aug (darkred)
        "#ffeb73",         # 9: Sep (lightyellow - darker hex for visibility on white background)
        "gold",            # 10: Oct (yellow - gold reads better than #ffff00)
        "darkgoldenrod",   # 11: Nov (darkyellow)
        "lightblue",       # 12: Dec (lightblue)
    ]
    
    month_nodes = {m: {"x": [], "y": [], "text": []} for m in range(1, 13)}
    other_nodes = {"x": [], "y": [], "text": []}
    
    for node in G.nodes():
        x, y = pos[node]
        data = G.nodes[node]
        month = data.get("month")
        
        deg = G.degree(node)
        
        # Build hover text with feature info if available
        features = ["mean_tn", "mean_tx", "mean_tg", "rr_sum", "mean_qq", "mean_hu"]
        feat_str = ""
        for f in features:
            if f in data:
                feat_str += f"<br>{f}: {data[f]:.2f}"
                
        details = f"<b>{node}</b><br>Degree: {deg}{feat_str}"
        
        if month is not None and 1 <= month <= 12:
            month_nodes[month]["x"].append(x)
            month_nodes[month]["y"].append(y)
            month_nodes[month]["text"].append(details)
        else:
            other_nodes["x"].append(x)
            other_nodes["y"].append(y)
            other_nodes["text"].append(details)

    for m in range(1, 13):
        if not month_nodes[m]["x"]:
            continue
        fig.add_trace(go.Scattergl(
            x=month_nodes[m]["x"],
            y=month_nodes[m]["y"],
            mode='markers',
            name=MONTH_LABELS[m - 1],
            text=month_nodes[m]["text"],
            hoverinfo='text',
            marker=dict(color=palette[m - 1], size=7, line=dict(width=0.5, color='white'))
        ))

    if other_nodes["x"]:
        fig.add_trace(go.Scattergl(
            x=other_nodes["x"],
            y=other_nodes["y"],
            mode='markers',
            name="Other",
            text=other_nodes["text"],
            hoverinfo='text',
            marker=dict(color='#888', size=7, line=dict(width=0.5, color='white'))
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(title="Month", orientation="v", y=1, x=1.02),
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        height=600
    )
    
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, showarrow=False,
        xref="paper", yref="paper", font=dict(size=14, color="#777"),
    )
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _add_hist_with_kde(
    fig: go.Figure,
    *,
    values: Sequence[float],
    period_key: str,
    bin_lo: float,
    bin_hi: float,
    bin_size: float,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    """Add a density histogram + KDE curve for ``values`` to ``(row, col)``.

    Two traces share the same legend group so the user toggles them
    together. The KDE silently degrades (skipped) when the input has too
    little variance to fit ``gaussian_kde``.
    """
    color = _PERIOD_COLOR[period_key]
    label = _PERIOD_LABEL[period_key]

    fig.add_trace(
        go.Histogram(
            x=list(values),
            name=label,
            marker_color=color,
            opacity=0.45,
            histnorm="probability density",
            xbins=dict(start=bin_lo, end=bin_hi, size=bin_size),
            showlegend=show_legend,
            legendgroup=period_key,
        ),
        row=row, col=col,
    )

    kde_x, kde_y = _kde_curve(values, bin_lo, bin_hi)
    if kde_x is None:
        return
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
            legendgroup=period_key,
            showlegend=False,  # share the legend slot with the histogram
            hoverinfo="skip",
        ),
        row=row, col=col,
    )


def _panel_density_peak(
    values: Sequence[float],
    bin_lo: float,
    bin_hi: float,
    bin_size: float,
) -> float:
    """Tallest density value rendered for ``values`` — bar peak or KDE peak.

    The shared-axis subplot grid uses ``max`` over every panel's peak as
    the Y range, so we can't rely on Plotly's auto-range (it tends to
    follow the single tallest spike and flatten the rest). Both
    contributions are checked because either side may dominate depending
    on the data: a sparse distribution gives a sharp KDE peak, a dense
    one a tall histogram bar.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or bin_hi <= bin_lo or bin_size <= 0:
        return 0.0

    # Histogram peak. ``np.histogram`` with ``density=True`` matches
    # Plotly's ``histnorm='probability density'`` output bar-for-bar
    # when the bin edges line up.
    edges = np.arange(bin_lo, bin_hi + bin_size * 0.5, bin_size)
    if edges.size < 2:
        edges = np.array([bin_lo, bin_hi])
    hist, _ = np.histogram(arr, bins=edges, density=True)
    hist_peak = float(hist.max()) if hist.size else 0.0

    # KDE peak (scipy's gaussian_kde, evaluated at 200 points).
    _, kde_y = _kde_curve(values, bin_lo, bin_hi)
    kde_peak = float(kde_y.max()) if kde_y is not None else 0.0

    return max(hist_peak, kde_peak)


def _kde_curve(
    values: Sequence[float],
    lo: float,
    hi: float,
    *,
    n: int = 200,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """KDE of ``values`` evaluated on ``n`` points spanning [lo, hi].

    Returns ``(None, None)`` when KDE can't be fit (too few samples,
    constant input, etc.) so the caller skips the overlay rather than
    crashing the whole figure.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or hi <= lo:
        return None, None
    if np.allclose(arr, arr[0]):
        return None, None
    try:
        kde = gaussian_kde(arr)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    xs = np.linspace(lo, hi, n)
    return xs, kde(xs)


def _qualitative_palette(n: int) -> List[str]:
    """Plotly-style qualitative palette that survives more than 10 categories."""
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#6a3d9a", "#b15928", "#a6cee3", "#fdbf6f", "#fb9a99",
        "#ffff99", "#cab2d6", "#33a02c", "#fb462f", "#1b9e77",
    ]
    if n <= len(base):
        return base[:n]
    # Repeat with slight rotation to keep neighbouring communities distinguishable.
    return (base * ((n // len(base)) + 1))[:n]
