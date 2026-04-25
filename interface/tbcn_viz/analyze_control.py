"""Sidebar widget that lets the user pick a location to analyse in depth.

The user types a latitude / longitude into the text inputs and presses
**Analyze**. The selection is stored in ``st.session_state`` and the page
hands off to ``pages/1_Location_Analysis.py``. Validation happens on submit
so the inputs accept arbitrary keystrokes (clearing, partial typing) without
fighting the user.

Persistence: Streamlit removes widget-bound session keys after a few reruns
when the widget isn't rendered, so values would otherwise be lost when the
user navigates to the analysis page and back. The fix is to mirror each
input into a separate, *non-widget* key (``_persist_*``). On every render
we copy persist→widget on the way in and widget→persist on the way out;
the persist keys outlive any page transition.
"""
from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import xarray as xr

from .analysis.pipeline import snap_to_grid
from .analysis.source_data import grid_axes


_SESSION_KEY = "tbcn_analyze_location"

# Widget keys (bound to the inputs).
_LAT_WIDGET_KEY = "_analyze_lat_text"
_LON_WIDGET_KEY = "_analyze_lon_text"

# Shadow keys — never passed as ``key=`` to any widget, so Streamlit will
# never garbage-collect them when the user navigates away.
_LAT_PERSIST_KEY = "_analyze_lat_persist"
_LON_PERSIST_KEY = "_analyze_lon_persist"


def render_analyze_section(container: DeltaGenerator, ds: xr.Dataset) -> None:
    """Render the section. Validates inputs only on submit."""
    src_lats, src_lons = grid_axes()
    lat_min, lat_max = float(min(src_lats)), float(max(src_lats))
    lon_min, lon_max = float(min(src_lons)), float(max(src_lons))
    mid_lat = (lat_min + lat_max) / 2
    mid_lon = (lon_min + lon_max) / 2

    # Initialise the persistent shadow once per session — preferring the
    # most recently analysed coordinate so the inputs reflect "what
    # you last looked at" when the user comes back to the map.
    if _LAT_PERSIST_KEY not in st.session_state:
        last = st.session_state.get(_SESSION_KEY)
        st.session_state[_LAT_PERSIST_KEY] = (
            f"{last['lat']:.4f}" if last else f"{mid_lat:.4f}"
        )
    if _LON_PERSIST_KEY not in st.session_state:
        last = st.session_state.get(_SESSION_KEY)
        st.session_state[_LON_PERSIST_KEY] = (
            f"{last['lon']:.4f}" if last else f"{mid_lon:.4f}"
        )

    # Sync persist → widget on the way in. Required when the widget keys
    # have been cleaned up (e.g. after navigating to the analysis page).
    if _LAT_WIDGET_KEY not in st.session_state:
        st.session_state[_LAT_WIDGET_KEY] = st.session_state[_LAT_PERSIST_KEY]
    if _LON_WIDGET_KEY not in st.session_state:
        st.session_state[_LON_WIDGET_KEY] = st.session_state[_LON_PERSIST_KEY]

    container.markdown("### Analyze a location")
    container.caption(
        f"Type any coordinate inside the source-data extent "
        f"({lat_min:.2f}–{lat_max:.2f}°N, {lon_min:.2f}–{lon_max:.2f}°E). "
        "The point is snapped to the nearest grid cell on submit."
    )

    col_lat, col_lon = container.columns(2, gap="small")
    col_lat.text_input(
        "Latitude (°N)",
        key=_LAT_WIDGET_KEY,
        placeholder=f"e.g. {mid_lat:.2f}",
    )
    col_lon.text_input(
        "Longitude (°E)",
        key=_LON_WIDGET_KEY,
        placeholder=f"e.g. {mid_lon:.2f}",
    )

    # Sync widget → persist on the way out. After this line, even if the
    # user navigates away and Streamlit drops the widget key, the persist
    # key still holds the most recently typed value.
    st.session_state[_LAT_PERSIST_KEY] = st.session_state[_LAT_WIDGET_KEY]
    st.session_state[_LON_PERSIST_KEY] = st.session_state[_LON_WIDGET_KEY]

    lat_value, lat_err = _parse_coord(
        st.session_state[_LAT_WIDGET_KEY], lat_min, lat_max, "Latitude"
    )
    lon_value, lon_err = _parse_coord(
        st.session_state[_LON_WIDGET_KEY], lon_min, lon_max, "Longitude"
    )

    if lat_err or lon_err:
        # Live-validation hint in the caption — does NOT block typing.
        container.caption(f"⚠️  {lat_err or lon_err}")
        snapped_lat = snapped_lon = None
        can_submit = False
    else:
        snapped_lat, snapped_lon = snap_to_grid(lat_value, lon_value)
        container.caption(f"Snapped grid cell: {snapped_lat:.4f}°N, {snapped_lon:.4f}°E")
        can_submit = True

    if container.button("Analyze", type="primary", use_container_width=True, disabled=not can_submit):
        _queue_and_switch(lat_value, lon_value, snapped_lat, snapped_lon)

    has_prior = _SESSION_KEY in st.session_state
    revisit_label = "Open Location Analysis" if has_prior else "Open with current coordinates"
    revisit_disabled = not (has_prior or can_submit)
    if container.button(revisit_label, use_container_width=True, disabled=revisit_disabled):
        if has_prior:
            try:
                st.switch_page("pages/1_Location_Analysis.py")
            except Exception:
                container.info("Use the page navigator to open Location Analysis.")
        else:
            _queue_and_switch(lat_value, lon_value, snapped_lat, snapped_lon)


def consume_selection() -> Optional[dict]:
    """Return ``{lat, lon, snapped_lat, snapped_lon}`` if one is queued, else None."""
    return st.session_state.get(_SESSION_KEY)


# ---------------------------------------------------------------------------
# Map click → analysis page navigation
# ---------------------------------------------------------------------------

_LAST_CLICK_KEY = "_tbcn_last_map_click"


def handle_map_click(chart_state) -> None:
    """If the heatmap was just clicked, queue the cell and switch pages.

    ``chart_state`` is whatever ``st.plotly_chart(..., on_select='rerun')``
    returned. We extract the most recent point from
    ``state.selection.points`` and switch only on a *new* coordinate —
    Streamlit replays the last selection on every rerun, so without
    deduplication the click would re-fire on unrelated interactions.

    Coordinate semantics: the heatmap is built with ``x=lon, y=lat``, so
    the click's ``y`` is the latitude and its ``x`` is the longitude.
    """
    click_lat, click_lon = _extract_click_point(chart_state)
    if click_lat is None or click_lon is None:
        return

    click_key = (round(click_lat, 6), round(click_lon, 6))
    if st.session_state.get(_LAST_CLICK_KEY) == click_key:
        return  # already handled in a previous rerun

    snapped_lat, snapped_lon = snap_to_grid(click_lat, click_lon)

    # Mirror the click into the persist keys only — Streamlit forbids
    # writing to a session_state key bound to a widget *after* that
    # widget has been instantiated, and ``render_analyze_section``
    # already ran (and instantiated the text inputs) earlier in this
    # script run. The widget keys would raise
    # ``StreamlitAPIException`` here.
    #
    # Persist keys are pure session state, never passed as ``key=`` to
    # any widget, so they're free to update at any point. On the next
    # render of this page, ``render_analyze_section`` copies persist →
    # widget *before* creating the inputs, which is the only legal
    # moment to do it. By then the user has navigated to the analysis
    # page and back, so the widget keys have been garbage-collected
    # by Streamlit and the sync writes them fresh.
    st.session_state[_LAT_PERSIST_KEY] = f"{click_lat:.4f}"
    st.session_state[_LON_PERSIST_KEY] = f"{click_lon:.4f}"

    st.session_state[_LAST_CLICK_KEY] = click_key
    _queue_and_switch(click_lat, click_lon, snapped_lat, snapped_lon)


def _extract_click_point(chart_state) -> Tuple[Optional[float], Optional[float]]:
    """Pull the latest ``(lat, lon)`` out of Streamlit's plotly selection state.

    Robust to both the dict-like and AttrDict-like shapes Streamlit has
    used across versions: tries ``state["selection"]["points"]`` first,
    falls back to attribute access. Returns ``(None, None)`` on anything
    we don't recognise so the caller can simply early-return.
    """
    if chart_state is None:
        return None, None

    selection = _get(chart_state, "selection")
    if selection is None:
        return None, None
    points = _get(selection, "points") or []
    if not points:
        return None, None
    last = points[-1]

    try:
        y_val = _get(last, "y")
        x_val = _get(last, "x")
        if y_val is not None and x_val is not None:
            lat = float(y_val)
            lon = float(x_val)
        else:
            customdata = _get(last, "customdata")
            if customdata:
                try:
                    lon_val = customdata[0]
                    lat_val = customdata[1]
                except (KeyError, TypeError):
                    lon_val = _get(customdata, "0")
                    lat_val = _get(customdata, "1")
                if lon_val is not None and lat_val is not None:
                    lat = float(lat_val)
                    lon = float(lon_val)
                else:
                    return None, None
            else:
                return None, None
    except (TypeError, ValueError, IndexError):
        return None, None
    return lat, lon


def _get(obj, key):
    """``obj[key]`` for dicts, ``obj.key`` for attribute-style objects."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except TypeError:
            pass
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_coord(text: str, lo: float, hi: float, name: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse + bounds-check a free-form coordinate string.

    Returns ``(value, None)`` on success and ``(None, error)`` on any
    failure — empty input, non-numeric, or out-of-range.
    """
    text = (text or "").strip().replace(",", ".")  # tolerate European commas
    if not text:
        return None, f"{name} is empty."
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None, f"{name} is not a number."
    if not (lo <= value <= hi):
        return None, f"{name} must be between {lo:.2f} and {hi:.2f}."
    return value, None


def _queue_and_switch(
    lat: float,
    lon: float,
    snapped_lat: float,
    snapped_lon: float,
) -> None:
    """Store the coordinate as the analysis selection and navigate."""
    st.session_state[_SESSION_KEY] = {
        "lat": float(lat),
        "lon": float(lon),
        "snapped_lat": float(snapped_lat),
        "snapped_lon": float(snapped_lon),
    }
    try:
        st.switch_page("pages/1_Location_Analysis.py")
    except Exception:
        st.info("Use the page navigator to open Location Analysis.")
