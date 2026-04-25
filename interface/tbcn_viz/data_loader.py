"""Load and access the TBCN network-index dataset."""
from __future__ import annotations

import streamlit as st
import xarray as xr

from .config import DATASET_PATH


@st.cache_resource(show_spinner="Loading network index dataset…")
def load_dataset() -> xr.Dataset:
    """Open the combined NetCDF dataset with latitude sorted ascending.

    Sorting matters because Plotly heatmaps place lower y-values at the bottom;
    sorting ascending puts the high-latitude (north) cells at the top of the
    figure, which matches the user's geographic intuition.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Generate it via networks/pipeline/create_networks_calculate_indexes.ipynb."
        )

    ds = xr.open_dataset(DATASET_PATH)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sortby("lat")
    return ds
