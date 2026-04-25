"""Inline controls for the Carpathian Overview page.

`render_controls` writes its widgets into whatever Streamlit container is passed
in (a column, an expander, or `st` itself), and returns a ``Selection`` that the
rest of the app consumes — the page never has to know which widget produced
which value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from streamlit.delta_generator import DeltaGenerator
import xarray as xr

from .index_metadata import IndexSpec, all_specs, get_spec


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PERIOD_LABELS = {
    "early": "Early (1961–1990)",
    "late": "Late (1995–2024)",
    "ratio_late_to_early": "Ratio (late / early)",
}

_DEFAULT_INDEX = "normalized_entropy"
_DEFAULT_PERIOD = "ratio_late_to_early"
_DEFAULT_MONTH = 6  # June


@dataclass(frozen=True)
class Selection:
    """User's choices, normalised for the rest of the app to consume."""
    spec: IndexSpec
    period: Optional[str]   # None when the index has no period dimension
    month: Optional[int]    # None when 'mean of all months' or no month dim


def render_controls(container: DeltaGenerator, ds: xr.Dataset) -> Selection:
    """Render the index/period/month widgets into ``container`` and return a Selection."""
    container.markdown("### Visualization controls")

    spec = _index_picker(container)
    container.caption(spec.description)

    period = _period_picker(container, ds) if spec.has_period else None
    month = _month_picker(container) if spec.has_month else None

    return Selection(spec=spec, period=period, month=month)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _index_picker(container: DeltaGenerator) -> IndexSpec:
    specs = all_specs()
    names = list(specs.keys())
    default_idx = names.index(_DEFAULT_INDEX) if _DEFAULT_INDEX in names else 0
    selected = container.selectbox(
        "Index",
        options=names,
        index=default_idx,
        format_func=lambda n: specs[n].label,
    )
    return get_spec(selected)


def _period_picker(container: DeltaGenerator, ds: xr.Dataset) -> str:
    """Horizontal radio — three options, one click."""
    periods = [str(v) for v in ds.period.values]
    default_idx = periods.index(_DEFAULT_PERIOD) if _DEFAULT_PERIOD in periods else 0
    return container.radio(
        "Period",
        options=periods,
        index=default_idx,
        format_func=lambda p: PERIOD_LABELS.get(p, p),
        horizontal=False,  # vertical fits better in a narrow side column
    )


def _month_picker(container: DeltaGenerator) -> Optional[int]:
    """Slider over months 1–12, with an opt-in checkbox to average over all months.

    The slider stays visible (just disabled) when the checkbox is on, so the
    layout doesn't reflow when the user toggles between modes.

    ``st.slider`` (numeric, step=1) is used instead of ``st.select_slider``
    because the latter snaps the thumb between discrete option stops as
    the user drags, which feels jumpy. ``st.slider`` tracks the cursor
    smoothly during the drag and the resolved month name is shown as a
    caption below — same visual outcome, better drag UX.
    """
    aggregate = container.checkbox("Average over all months", value=False)
    month = container.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=_DEFAULT_MONTH,
        step=1,
        disabled=aggregate,
    )
    container.caption(MONTH_NAMES[month - 1])
    return None if aggregate else int(month)
