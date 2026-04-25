"""Carpathian Basin Overview — TBCN visualization interface (entry point).

Run with:
    streamlit run interface/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable regardless of the directory streamlit is invoked from.
_INTERFACE_ROOT = Path(__file__).resolve().parent
if str(_INTERFACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERFACE_ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from tbcn_viz.analyze_control import handle_map_click, render_analyze_section  # noqa: E402
from tbcn_viz.config import DEFAULT_PAGE_CONFIG  # noqa: E402
from tbcn_viz.controls import render_controls  # noqa: E402
from tbcn_viz.data_loader import load_dataset  # noqa: E402
from tbcn_viz.map_plot import build_heatmap, select_data  # noqa: E402


# ---------------------------------------------------------------------------
# Page configuration & global styling
# ---------------------------------------------------------------------------

st.set_page_config(**DEFAULT_PAGE_CONFIG)

# CSS handles four things:
#   1. Hide the Streamlit top bar (Deploy button etc.) — it was overlapping the title.
#   2. Move the native sidebar to the right edge of the page.
#   3. Pin the sidebar's expand/collapse toggle to the top-right corner.
#   4. Tighten paddings so title + map + stats fit a single viewport.
st.markdown(
    """
    <style>
        /* 1. Keep the header element AND the toolbar wrapper in the DOM —
              on some Streamlit versions the sidebar's expand button is nested
              inside the toolbar, so blanket-hiding either kills the toggle.
              Make the header invisible visually, and hide ONLY the specific
              toolbar items (Deploy, menu, decoration line). */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
            overflow: visible !important;
        }
        [data-testid="stToolbarActions"],
        [data-testid="stMainMenu"],
        [data-testid="stDecoration"],
        .stDeployButton,
        .stAppDeployButton {
            display: none !important;
        }

        /* 2. Compact main-content padding so we get every available pixel. */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.4rem !important;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 1700px;
        }
        h1 {
            text-align: center;
            margin: 0 0 0.3rem 0 !important;
            font-size: 1.6rem !important;
        }

        /* 3. Flip the native sidebar to the right side of the page. */
        section[data-testid="stSidebar"] {
            order: 2 !important;
            right: 0 !important;
            left: auto !important;
        }

        /* 4. Hide every native sidebar toggle Streamlit might draw, but keep
              them in the layout (1x1, opacity 0) so JS can still dispatch
              events to them — that's how we force Streamlit's React state to
              "expanded" when it starts up collapsed. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarToggleControl"],
        [data-testid="stSidebarHeader"] button,
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"],
        button[aria-label*="ollapse" i] {
            opacity: 0 !important;
            pointer-events: none !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            z-index: -1 !important;
        }

        /* 4b. Hide the auto-injected multi-page navigator. With a `pages/`
              folder, Streamlit puts a navigator at the top of the sidebar —
              we don't want it: navigation happens via the explicit
              `Analyze` button and the `← Back to map` button on the
              analysis page. Hiding it also prevents its container from
              shifting the JS-injected toggle button's anchor. */
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"] {
            display: none !important;
        }

        /* When our toggle adds this class to <body>, the right-side panel
           disappears and the main content reflows to fill the page. */
        body.tbcn-panel-hidden section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* 5. Tighter widget spacing in the side panel. */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
            gap: 0.4rem;
        }

        /* 6. Compact stats row below the map. */
        [data-testid="stMetricValue"] { font-size: 1.05rem; }
        [data-testid="stMetricLabel"] { font-size: 0.74rem; opacity: 0.7; }
        [data-testid="stMetric"] {
            padding: 0.25rem 0.6rem;
            background: rgba(127, 127, 127, 0.06);
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ds = load_dataset()


# ---------------------------------------------------------------------------
# Right-side panel — visualization controls
# ---------------------------------------------------------------------------

with st.sidebar:
    selection = render_controls(st.sidebar, ds)
    map_opacity = st.slider("Map Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    st.markdown("---")
    render_analyze_section(st.sidebar, ds)


# ---------------------------------------------------------------------------
# Main area — title, map, stats (all centered, no scroll)
# ---------------------------------------------------------------------------

st.title("Carpathian Basin Overview")

data_slice, subtitle = select_data(ds, selection)
fig = build_heatmap(data_slice, selection, subtitle, height=500, opacity=map_opacity)

chart_state = st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    on_select="rerun",
    selection_mode="points",
    key="overview_map",
)

# Click-to-analyse: any new click on the heatmap routes the user to the
# analysis page for the snapped grid cell. Streamlit's selection state
# persists across reruns, so the helper deduplicates against the last
# coordinate we already handled — without that guard, every unrelated
# rerun (e.g. a sidebar tweak) would re-fire the navigation.
handle_map_click(chart_state)

finite = data_slice.values[np.isfinite(data_slice.values)]
if finite.size == 0:
    st.info("No data in the current selection.")
else:
    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Min", f"{float(finite.min()):.3f}")
    m2.metric("Max", f"{float(finite.max()):.3f}")
    m3.metric("Mean", f"{float(finite.mean()):.3f}")
    m4.metric("Cells", f"{int(finite.size):d}")


# ---------------------------------------------------------------------------
# Custom panel-toggle button.
#
# Why this exists: Streamlit's native sidebar-toggle button lives inside a
# flex slot whose ancestor uses `transform`/`will-change`, which makes
# `position: fixed` resolve relative to that ancestor instead of the viewport
# — leaving the icon at the inner edge of the right-side panel rather than
# at the page's right edge. Working *with* Streamlit's React state via click
# delegation also turned out to be brittle across versions.
#
# Reliable fix: hide all of Streamlit's own toggles via CSS, inject our own
# button as a direct child of <body> (no transformed ancestors → fixed
# positioning works), and have it toggle a `tbcn-panel-hidden` class on
# <body>. A CSS rule then sets the sidebar to `display: none` whenever the
# class is present. No event reverse-engineering, no React-state guessing —
# the panel hides and reappears purely through CSS.
# ---------------------------------------------------------------------------

components.html(
    """
    <script>
    (function () {
        const BTN_ID = 'tbcn-panel-toggle';
        const HIDDEN_CLASS = 'tbcn-panel-hidden';
        const STORAGE_KEY = 'tbcn-panel-hidden';
        const win = window.parent;
        const doc = () => win.document;

        // ----- Persisted user preference (survives reloads/reruns) -------
        function getStored() {
            try { return win.localStorage.getItem(STORAGE_KEY) === 'true'; }
            catch (e) { return false; }
        }
        function setStored(val) {
            try { win.localStorage.setItem(STORAGE_KEY, val ? 'true' : 'false'); }
            catch (e) {}
        }

        // ----- The custom toggle button ----------------------------------
        // The button is injected into the *parent* document's <body>, so it
        // survives page navigation. The JS interval, on the other hand,
        // lives inside an iframe that Streamlit re-creates on every page
        // change — meaning any click handler we attached references closures
        // from the previous iframe. After a navigation those closures point
        // at functions in a destroyed scope and clicks become no-ops.
        //
        // The fix: track a per-IIFE ``initialized`` flag. On the first tick
        // of a freshly-mounted iframe, delete any stale button and recreate
        // it. The button is then bound to handlers in the *current* iframe.
        let initialized = false;
        function ensureCustomBtn() {
            const d = doc();
            if (initialized && d.getElementById(BTN_ID)) return;

            const stale = d.getElementById(BTN_ID);
            if (stale) stale.remove();

            const btn = d.createElement('button');
            btn.id = BTN_ID;
            btn.type = 'button';
            btn.title = 'Toggle controls panel';
            btn.innerHTML = '\\u2630';  // ☰
            btn.style.cssText = [
                'position: fixed',
                'top: 0.4rem',
                'right: 0.5rem',
                'z-index: 999999',
                'background: var(--secondary-background-color, #f0f2f6)',
                'color: var(--text-color, #333)',
                'border: 1px solid rgba(127,127,127,0.25)',
                'border-radius: 6px',
                'padding: 4px 10px',
                'font-size: 16px',
                'line-height: 1.2',
                'cursor: pointer',
                'box-shadow: 0 1px 4px rgba(0,0,0,0.08)'
            ].join(';') + ';';
            btn.addEventListener('mouseenter', () => { btn.style.filter = 'brightness(0.96)'; });
            btn.addEventListener('mouseleave', () => { btn.style.filter = ''; });
            btn.addEventListener('click', () => {
                const willHide = !d.body.classList.contains(HIDDEN_CLASS);
                d.body.classList.toggle(HIDDEN_CLASS, willHide);
                setStored(willHide);
                // No need to nudge Streamlit's React state: the CSS that
                // hides every native toggle keeps Streamlit permanently
                // in its initial "expanded" state, so visibility is
                // governed entirely by the body class above.
            });
            d.body.appendChild(btn);

            // Restore prior preference.
            if (getStored()) d.body.classList.add(HIDDEN_CLASS);

            initialized = true;
        }

        // ----- Tick: keep our custom button alive across reruns ----------
        // Streamlit may re-render the page (and tear down portions of the
        // DOM) on every interaction. The tick reinstates our button if a
        // rerun removed it — that's all it does. The previous version
        // also force-clicked Streamlit's collapse/expand button every
        // 200ms to keep its React state "expanded", but the testid we
        // need to click drifted across versions: in Streamlit ≥1.50,
        // the visible-when-expanded ``stSidebarCollapseButton`` matched
        // our selector, so we were *collapsing* the sidebar twelve times
        // a second — closing dropdowns and aborting slider drags. The
        // CSS already pins Streamlit's state to "expanded" by hiding
        // every native toggle, so the synthetic click is unnecessary.
        function tick() {
            try {
                ensureCustomBtn();
            } catch (e) { /* same-origin guard */ }
        }

        tick();
        setInterval(tick, 200);
    })();
    </script>
    """,
    height=0,
)
