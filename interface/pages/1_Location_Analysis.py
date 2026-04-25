"""Per-location analysis page.

Picks up the (lat, lon) the user submitted on the overview page, rebuilds
the network for that grid cell, and presents:

* The rich monthly prototype heatmap.
* The percolation comparison curve (early vs late).
* Leiden community detection (default resolution = 1.0) — three plots per
  period: temporal scatter, community-perspective heatmap, month-perspective
  heatmap.
* Degree-distribution histograms (unweighted, weighted, per-month 2×2).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the package importable when the page is reached via Streamlit's
# multi-page navigator (which runs each page as a fresh script).
_INTERFACE_ROOT = Path(__file__).resolve().parents[1]
if str(_INTERFACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tbcn_viz.analysis import run_analysis  # noqa: E402
from tbcn_viz.analysis.figures import (  # noqa: E402
    MONTH_LABELS,
    community_perspective_heatmap,
    community_scatter,
    month_perspective_heatmap,
    monthly_degree_grid,
    per_month_degree_panel,
    percolation_figure,
    prototype_heatmap,
)
from tbcn_viz.analyze_control import consume_selection  # noqa: E402
from tbcn_viz.config import DEFAULT_PAGE_CONFIG  # noqa: E402


st.set_page_config(**{**DEFAULT_PAGE_CONFIG, "page_title": "TBCN — Location Analysis"})

_CSS = """
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stToolbarActions"],
        [data-testid="stMainMenu"],
        [data-testid="stDecoration"],
        .stDeployButton,
        .stAppDeployButton {
            display: none !important;
        }
        .block-container { padding-top: 0.8rem; max-width: 1500px; }
        h1 { text-align: center; margin: 0 0 0.5rem 0 !important; font-size: 1.6rem !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        #tbcn-panel-toggle { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
"""

st.markdown(
    f"""
    <style>
    /* {time.time()} */
    {_CSS}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.page_link("app.py", label="Carpathian Basin Overview")
    st.page_link("pages/1_Location_Analysis.py", label="Location Analysis")


# ---------------------------------------------------------------------------
# Resolve the selection
# ---------------------------------------------------------------------------

selection = consume_selection()
if selection is None:
    st.title("Location Analysis")
    st.info(
        "Pick a point on the **Carpathian Basin Overview** page first — "
        "click any cell on the map or type coordinates into the *Analyze a "
        "location* panel, then press **Analyze**."
    )
    st.stop()


lat = selection["lat"]
lon = selection["lon"]
snapped_lat = selection["snapped_lat"]
snapped_lon = selection["snapped_lon"]

st.title("Location Analysis")
header_cols = st.columns([3, 1])
with header_cols[0]:
    st.markdown(
        f"**Requested point:** {lat:.4f}°N, {lon:.4f}°E  ·  "
        f"**Analysed grid cell:** {snapped_lat:.4f}°N, {snapped_lon:.4f}°E"
    )
with header_cols[1]:
    if st.button("← Back to map", use_container_width=True):
        st.switch_page("app.py")


# ---------------------------------------------------------------------------
# Run the pipeline (cached)
# ---------------------------------------------------------------------------

result = run_analysis(snapped_lat, snapped_lon)
if result is None:
    from tbcn_viz.analysis.source_data import grid_axes
    src_lats, src_lons = grid_axes()
    in_bounds = (min(src_lats) <= snapped_lat <= max(src_lats)
                 and min(src_lons) <= snapped_lon <= max(src_lons))
    if not in_bounds:
        st.error(
            f"This cell is outside the source-data extent "
            f"({min(src_lats):.2f}–{max(src_lats):.2f}°N, "
            f"{min(src_lons):.2f}–{max(src_lons):.2f}°E). "
            "Set the `TBCN_EOBS_DIR` environment variable to the folder "
            "containing the clipped E-OBS NetCDFs, or pick a cell inside "
            "the available extent."
        )
    else:
        st.error(
            "No usable monthly data at this grid cell — every month of the "
            "shared period is masked (NaN). The source files cover this "
            "extent but the cell itself is missing data."
        )
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_proto, tab_perc, tab_comm, tab_deg = st.tabs([
    "Prototypes",
    "Percolation",
    "Community detection",
    "Degree distributions",
])

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ----- Prototypes ----------------------------------------------------------

with tab_proto:
    st.markdown(
        "Each cell is the similarity between the *mean* prototype for the "
        "early-period row and the *mean* prototype for the late-period "
        "column in the full graph's PCA space — same logic as "
        "`rich_mean_prototype_creation` + `rich_mean_prototype_analysis`. "
        "The threshold heatmap zeros out cells below "
        "`min(early_threshold, late_threshold)` from the network's pruning step."
    )
    if result.prototypes is None:
        st.warning("No prototype matrix could be computed for this cell.")
    else:
        col_no, col_thr = st.columns(2)
        with col_no:
            st.plotly_chart(
                prototype_heatmap(
                    result.prototypes.similarity_no_threshold,
                    title="Without threshold",
                ),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
        with col_thr:
            st.plotly_chart(
                prototype_heatmap(
                    result.prototypes.similarity_with_threshold,
                    title=f"With threshold (cut-off {result.prototypes.threshold:.3f})",
                ),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
        with st.expander("Mean prototype feature values"):
            cols = st.columns(2)
            for col, period_label, protos in (
                (cols[0], "Early period (1961–1990)", result.prototypes.early_prototypes),
                (cols[1], "Late period (1995–2024)", result.prototypes.late_prototypes),
            ):
                # Rows = months, columns = features.
                # ``orient='index'`` makes the outer keys (month numbers)
                # the rows; without it pandas would put features on the
                # rows because each inner dict becomes a column by default.
                df = pd.DataFrame.from_dict(
                    {MONTH_LABELS[m - 1]: v for m, v in sorted(protos.items())},
                    orient="index",
                )
                col.markdown(f"**{period_label}**")
                col.dataframe(df.style.format("{:.3f}"), use_container_width=True)


# ----- Percolation ---------------------------------------------------------

with tab_perc:
    st.markdown(
        "Edges are removed in order of increasing weight; the curve tracks "
        "the share of nodes that remain in the largest connected component. "
        "The dashed lines mark the threshold at which each network "
        "fragments most steeply (the *critical threshold*)."
    )
    if result.percolation is None:
        st.warning("Percolation analysis needs at least one weighted edge in each period.")
    else:
        st.plotly_chart(percolation_figure(result.percolation), use_container_width=True, config=PLOTLY_CONFIG)
        m1, m2 = st.columns(2)
        m1.metric(
            "Early critical threshold",
            f"{result.percolation.early.critical_threshold:.3f}"
            if result.percolation.early.critical_threshold is not None else "n/a",
        )
        m2.metric(
            "Late critical threshold",
            f"{result.percolation.late.critical_threshold:.3f}"
            if result.percolation.late.critical_threshold is not None else "n/a",
        )


# ----- Community detection -------------------------------------------------

with tab_comm:
    st.markdown(
        "Leiden algorithm with the default resolution (1.0). The temporal "
        "scatter shows which year/month nodes ended up in which community. "
        "The two heatmaps are the *community* and *month* perspectives "
        "described in `networks/community_detection/analysis`."
    )

    period_labels = {"early": "Early (1961–1990)", "late": "Late (1995–2024)"}
    for key, period_result in (("early", result.communities.early), ("late", result.communities.late)):
        st.markdown(f"### {period_labels[key]}  ·  {period_result.num_communities} communities  ·  modularity {period_result.modularity:.3f}")
        st.plotly_chart(
            community_scatter(period_result, period_label=period_labels[key]),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(
            community_perspective_heatmap(period_result, period_label=period_labels[key]),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        col2.plotly_chart(
            month_perspective_heatmap(period_result, period_label=period_labels[key]),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        st.markdown("---")


# ----- Degree distributions ------------------------------------------------

with tab_deg:
    st.markdown(
        "Density histograms of the per-node degree, by calendar month, "
        "comparing the early (blue) and late (orange) periods. The "
        "*close-month* variants only count neighbours whose month is the "
        "previous, current, or next calendar month (Dec ↔ Jan wraps)."
    )

    sub_unw, sub_w, sub_per = st.tabs(["Unweighted", "Weighted", "Per-month"])

    # Map the user-friendly scope label to the variant suffix so the radio
    # widget controls *which* of the four bundles we draw without
    # duplicating the plot call.
    SCOPE_OPTIONS = {"All months": "all", "Close months only": "close"}

    def _scope_radio(key: str) -> str:
        choice = st.radio(
            "Neighbour scope",
            options=list(SCOPE_OPTIONS.keys()),
            horizontal=True,
            key=key,
        )
        return SCOPE_OPTIONS[choice]

    with sub_unw:
        scope = _scope_radio("deg_scope_unw")
        suffix = "Close months only" if scope == "close" else "all months"
        x_label = "Node degree (close-month neighbours)" if scope == "close" else "Node degree"
        st.plotly_chart(
            monthly_degree_grid(
                result.degrees,
                variant=f"unweighted_{scope}",
                title=f"Monthly unweighted degree distributions · {suffix}",
                x_label=x_label,
            ),
            use_container_width=True, config=PLOTLY_CONFIG,
        )

    with sub_w:
        scope = _scope_radio("deg_scope_w")
        suffix = "Close months only" if scope == "close" else "all months"
        x_label = (
            "Weighted degree (close-month neighbours)" if scope == "close"
            else "Weighted node degree (strength)"
        )
        st.plotly_chart(
            monthly_degree_grid(
                result.degrees,
                variant=f"weighted_{scope}",
                title=f"Monthly weighted degree distributions · {suffix}",
                x_label=x_label,
            ),
            use_container_width=True, config=PLOTLY_CONFIG,
        )

    with sub_per:
        month_value = st.select_slider(
            "Month",
            options=list(range(1, 13)),
            value=6,
            format_func=lambda m: MONTH_LABELS[m - 1],
        )
        st.plotly_chart(
            per_month_degree_panel(result.degrees, month=int(month_value)),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
