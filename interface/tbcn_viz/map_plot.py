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
    opacity: float = 0.8,
) -> go.Figure:
    """Return a Plotly figure with hover-enabled cells (lat, lon, value).

    Uses a Choroplethmapbox with an OpenStreetMap underlay.
    """
    spec = sel.spec
    diverging, center = _resolve_diverging(spec, sel.period)
    zmin, zmax = _color_range(da.values, diverging=diverging, center=center)
    colorscale = _resolve_colorscale(spec, sel.period)

    hovertemplate = (
        "<b>Latitude:</b> %{customdata[1]:.4f}°N<br>"
        "<b>Longitude:</b> %{customdata[0]:.4f}°E<br>"
        f"<b>{spec.label}:</b> %{{z:.4f}}"
        "<extra></extra>"
    )

    geojson, ids, z_flat, custom_lons, custom_lats = _create_grid_geojson(da)

    if not z_flat:
        # Fallback to an empty map if there's no finite data
        fig = go.Figure(go.Scattermapbox())
    else:
        choropleth = go.Choroplethmapbox(
            geojson=geojson,
            locations=ids,
            z=z_flat,
            customdata=np.column_stack((custom_lons, custom_lats)),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=center if diverging else None,
            colorbar=dict(title=spec.label, thickness=12, len=0.92, x=1.02),
            hovertemplate=hovertemplate,
            marker_opacity=opacity,
            marker_line_width=0,
        )
        fig = go.Figure(data=choropleth)

    title = spec.label
    if subtitle:
        title += (
            f"<br><span style='font-size:12px;color:#888'>{subtitle}</span>"
        )

    lat_center = float(da.lat.mean())
    lon_center = float(da.lon.mean())

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=5
        ),
        margin=dict(l=20, r=20, t=55, b=40),
        height=height,
        # ``clickmode='event+select'`` is what makes a *single click* on
        # a marker add it to the trace's ``selectedpoints``,
        # which in turn is what fires ``plotly_selected``.
        clickmode="event+select",
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_grid_geojson(da: xr.DataArray) -> tuple[dict, list[str], list[float], list[float], list[float]]:
    lons = da.lon.values
    lats = da.lat.values
    vals = da.values

    dlon = lons[1] - lons[0] if len(lons) > 1 else 0.1
    dlat = lats[1] - lats[0] if len(lats) > 1 else 0.1

    features = []
    ids = []
    z_flat = []
    custom_lons = []
    custom_lats = []

    for j, lat in enumerate(lats):
        lat0, lat1 = lat - dlat/2, lat + dlat/2
        for i, lon in enumerate(lons):
            val = vals[j, i]
            if np.isnan(val):
                continue
            
            lon0, lon1 = lon - dlon/2, lon + dlon/2
            feature_id = f"{j}_{i}"
            features.append({
                "type": "Feature",
                "id": feature_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]
                }
            })
            ids.append(feature_id)
            z_flat.append(float(val))
            custom_lons.append(float(lon))
            custom_lats.append(float(lat))
            
    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, ids, z_flat, custom_lons, custom_lats


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