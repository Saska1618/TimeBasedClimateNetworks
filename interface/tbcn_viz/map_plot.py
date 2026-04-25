"""Build the interactive Plotly heatmap of an index over the Carpathian basin."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from .controls import MONTH_NAMES, PERIOD_LABELS, Selection
from .index_metadata import IndexSpec, is_ratio_diverging


def select_data(ds: xr.Dataset, sel: Selection) -> Tuple[xr.DataArray, str]:
    """Return the 2D (lat, lon) slice for the user's selection plus a subtitle string."""
    da = ds[sel.spec.name]
    parts = []

    if sel.spec.has_month:
        if sel.month is None:
            da = da.mean(dim="month", skipna=True)
            parts.append("mean of all months")
        else:
            da = da.sel(month=sel.month)
            parts.append(MONTH_NAMES[sel.month - 1])

    if sel.spec.has_period and sel.period is not None:
        da = da.sel(period=sel.period)
        parts.append(PERIOD_LABELS.get(sel.period, sel.period))

    return da, " · ".join(parts)


def build_heatmap(
    da: xr.DataArray,
    sel: Selection,
    subtitle: str,
    *,
    height: int = 500,
) -> go.Figure:
    """Return a Plotly figure with hover-enabled cells (lat, lon, value).

    Applies an equirectangular projection: the x-axis is compressed by
    ``cos(mean_lat)`` so that 1° of longitude takes the right share of pixels
    relative to 1° of latitude — without that the map looks stretched
    horizontally at mid-latitudes.
    """
    spec = sel.spec
    diverging, center = _resolve_diverging(spec, sel.period)
    zmin, zmax = _color_range(da.values, diverging=diverging, center=center)
    colorscale = _resolve_colorscale(spec, sel.period)
    lon_per_lat = _equirectangular_ratio(da)

    hovertemplate = (
        "<b>Latitude:</b> %{y:.4f}°N<br>"
        "<b>Longitude:</b> %{x:.4f}°E<br>"
        f"<b>{spec.label}:</b> %{{z:.4f}}"
        "<extra></extra>"
    )

    heatmap = go.Heatmap(
        z=da.values,
        x=da.lon.values,
        y=da.lat.values,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=center if diverging else None,
        colorbar=dict(title=spec.label, thickness=12, len=0.92, x=1.02),
        hovertemplate=hovertemplate,
        hoverongaps=False,
        connectgaps=False,
    )

    title = spec.label
    if subtitle:
        title += (
            f"<br><span style='font-size:12px;color:#888'>{subtitle}</span>"
        )

    fig = go.Figure(data=heatmap)
    # The overlay reuses the heatmap's hovertemplate so the user sees an
    # identical tooltip whether the cursor is over an in-data cell
    # (heatmap label fired) or the always-on-top scatter overlay.
    _add_click_overlay(fig, da, hovertemplate=hovertemplate)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        xaxis=dict(
            title="Longitude (°E)",
            scaleanchor="y",
            scaleratio=lon_per_lat,
            constrain="domain",
            constraintoward="center",
        ),
        yaxis=dict(title="Latitude (°N)", constrain="domain", constraintoward="middle"),
        margin=dict(l=20, r=20, t=55, b=40),
        height=height,
        template="plotly_white",
        # ``clickmode='event+select'`` is what makes a *single click* on
        # a scatter marker add it to the trace's ``selectedpoints``,
        # which in turn is what fires ``plotly_selected`` — the event
        # Streamlit's ``on_select='rerun'`` listens for. Without this,
        # only box / lasso selection trigger reruns. Streamlit does not
        # set this automatically.
        clickmode="event+select",
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_diverging(spec: IndexSpec, period: str | None) -> tuple[bool, float]:
    """Decide whether a given (index, period) pair is a diverging colormap.

    Ratios are diverging around 1; signed indexes (percolation) around 0.
    """
    if spec.diverging:
        return True, spec.center if spec.center is not None else 0.0
    if period == "ratio_late_to_early" and is_ratio_diverging(spec.name):
        return True, 1.0
    return False, 0.0


def _resolve_colorscale(spec: IndexSpec, period: str | None) -> str:
    if not spec.diverging and period == "ratio_late_to_early" and is_ratio_diverging(spec.name):
        # Override to a diverging colorscale for ratio views.
        return "RdBu_r"
    return spec.colorscale


def _color_range(values: np.ndarray, *, diverging: bool, center: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if diverging:
        bound = float(np.max(np.abs(finite - center)))
        if bound == 0:
            bound = 1.0
        return center - bound, center + bound
    return float(np.min(finite)), float(np.max(finite))


def _add_click_overlay(fig: go.Figure, da: xr.DataArray, *, hovertemplate: str) -> None:
    """Tile the heatmap with an invisible scatter so single clicks register.

    Three Plotly gotchas, all of which had to be worked around together:

    1. ``Heatmap`` traces don't emit selection events for single clicks
       (only for lasso / box; Streamlit issues #8760, #8933). The
       documented workaround is a transparent scatter overlay; a click
       on a scatter marker DOES emit ``plotly_selected``.

    2. ``hoverinfo='skip'`` would let the heatmap's hover label show
       through the overlay — but it also kills click events
       (Plotly community thread on hoverinfo='skip'). ``hoverinfo='none'``
       suppresses the label only, but the overlay still intercepts the
       hover, so the heatmap's tooltip never gets a chance to appear.
       Solution: give the overlay its *own* hovertemplate, identical to
       the heatmap's. Hover labels work, clicks work, and the tooltip
       text is the same whichever trace produced it.

    3. ``rgba(0, 0, 0, 0)`` (alpha 0) is unreliable for hit-testing in
       scattergl (plotly.js #3413). ``rgba(0, 0, 0, 0.01)`` is visually
       invisible but engages the hit-test.

    The hovertemplate references ``%{customdata[0]}`` because scatter
    traces don't carry a ``z`` attribute — we feed the heatmap's
    underlying values via ``customdata`` so the same template works.
    """
    lons = da.lon.values
    lats = da.lat.values
    if lons.size == 0 or lats.size == 0:
        return

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    z_flat = da.values.flatten().reshape(-1, 1)

    # Scatter has no `z`, so swap `%{z...}` → `%{customdata[0]...}`
    # without touching the rest of the template (labels, formatting).
    overlay_template = hovertemplate.replace("%{z", "%{customdata[0]")

    fig.add_trace(go.Scattergl(
        x=lon_grid.flatten(),
        y=lat_grid.flatten(),
        customdata=z_flat,
        mode="markers",
        marker=dict(
            size=18,
            symbol="square",
            color="rgba(0, 0, 0, 0.01)",   # invisible but hit-testable
            line=dict(width=0),
        ),
        hovertemplate=overlay_template,
        showlegend=False,
        name="_click_overlay",
    ))


def _equirectangular_ratio(da: xr.DataArray) -> float:
    """Pixel-per-unit ratio for the x-axis (longitude) anchored to the y-axis (latitude).

    At latitude φ, 1° of longitude spans ``cos(φ)`` times the ground distance of
    1° of latitude. We use the mean latitude of the data slice as the projection
    parallel — accurate enough for a small region like the Carpathian basin.
    """
    lat_center = float(da.lat.mean())
    return max(0.05, math.cos(math.radians(lat_center)))
