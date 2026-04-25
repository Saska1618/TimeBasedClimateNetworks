"""Cached loaders for the six clipped E-OBS NetCDFs the network builder needs.

Each file is opened once and held in memory for the lifetime of the Streamlit
process (``st.cache_resource``). Extracting the monthly time series for a
single grid cell is then a fast in-memory ``sel`` per variable.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st
import xarray as xr

from ..config import EOBS_FILES


@st.cache_resource(show_spinner="Loading E-OBS source datasets…")
def load_eobs_datasets() -> Dict[str, xr.Dataset]:
    """Open all six clipped E-OBS NetCDFs and return them keyed by variable.

    The directory is resolved by ``config._resolve_eobs_dir`` — set the
    ``TBCN_EOBS_DIR`` environment variable to override.
    """
    out: Dict[str, xr.Dataset] = {}
    for var, path in EOBS_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"E-OBS file missing: {path}\n"
                "Set the TBCN_EOBS_DIR environment variable to the folder "
                "containing the six 'CB_clipped_nc' files, or place them in "
                "data/nc_files/clipped/ inside the repository."
            )
        out[var] = xr.open_dataset(path)
    return out


def grid_axes() -> tuple[list[float], list[float]]:
    """Return the (latitudes, longitudes) the source data is defined on.

    Used to snap a user-entered or map-clicked coordinate to the nearest
    cell — the network builder otherwise relies on xarray's ``method='nearest'``
    every call, but having the axes available lets the UI display the snapped
    coordinate before the (slow) build kicks off.
    """
    ds = load_eobs_datasets()["tg"]
    lats = ds["latitude"].values.tolist()
    lons = ds["longitude"].values.tolist()
    return lats, lons


def extract_monthly_features(
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> Dict[str, Dict[str, float]]:
    """Build the {month_str -> features} dict the network builder consumes.

    Mirrors ``get_rich_monthly_nodes`` from the pipeline notebook, but reads
    from the cached datasets instead of re-opening files. Months containing
    NaN for any of the six features are skipped (same rule as the notebook).
    """
    datasets = load_eobs_datasets()
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    series = {}
    for var, ds in datasets.items():
        ds_slice = ds.sel(time=slice(start, end))
        ts = ds_slice[var].sel(latitude=lat, longitude=lon, method="nearest")
        df = ts.to_dataframe().reset_index()[["time", var]]
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        series[var] = df

    merged = (
        series["tg"]
        .join(series["tn"])
        .join(series["tx"])
        .join(series["rr"])
        .join(series["qq"])
        .join(series["hu"])
    )

    monthly_nodes: Dict[str, Dict[str, float]] = {}
    for name, group in merged.groupby(pd.Grouper(freq="ME")):
        if name < start_date or name.to_period("M") > end_date.to_period("M"):
            continue

        record = {
            "mean_tn": group["tn"].mean(),
            "mean_tx": group["tx"].mean(),
            "mean_tg": group["tg"].mean(),
            "rr_sum": group["rr"].sum(),
            "mean_qq": group["qq"].mean(),
            "mean_hu": group["hu"].mean(),
        }
        if any(pd.isna(v) for v in record.values()):
            continue

        monthly_nodes[name.strftime("%Y-%m")] = {k: float(v) for k, v in record.items()}

    return monthly_nodes
