"""Static configuration for the visualization interface."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


PACKAGE_ROOT = Path(__file__).resolve().parent
INTERFACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = INTERFACE_ROOT.parent

DATASET_PATH = (
    PROJECT_ROOT / "networks" / "pipeline" / "networks_indexes_combined_output.nc"
)

# Source E-OBS NetCDFs used to rebuild a network for an arbitrary cell on
# demand. The notebook references these as
# ``/Volumes/T7/Data/Carpathian_basin/CB_clipped_nc/...`` (Mac), so we look
# them up in this order:
#   1. ``TBCN_EOBS_DIR`` environment variable, if set.
#   2. A short list of platform-typical candidates.
#   3. ``data/nc_files/clipped/`` inside the repo (legacy fallback).
# The first candidate that contains a ``tg_ens_mean_*_clipped.nc`` wins.
_EOBS_FILE_NAMES: Dict[str, str] = {
    "hu": "hu_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
    "qq": "qq_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
    "rr": "rr_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
    "tg": "tg_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
    "tn": "tn_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
    "tx": "tx_ens_mean_0.1deg_reg_v31.0e_clipped.nc",
}

_EOBS_CANDIDATE_DIRS = [
    Path(r"D:/Data/Carpathian_basin/CB_clipped_nc"),                # user's Windows location
    Path("/Volumes/T7/Data/Carpathian_basin/CB_clipped_nc"),         # original Mac location
    PROJECT_ROOT / "data" / "nc_files" / "clipped",                  # repo fallback
]


def _resolve_eobs_dir() -> Path:
    """Pick the first candidate that contains the canonical 'tg' file.

    ``TBCN_EOBS_DIR`` overrides everything when set. Returning the *first*
    candidate when no resolution succeeds keeps error messages downstream
    informative — they can show the path that was tried.
    """
    env = os.environ.get("TBCN_EOBS_DIR")
    if env:
        return Path(env)

    canonical = _EOBS_FILE_NAMES["tg"]
    for cand in _EOBS_CANDIDATE_DIRS:
        if (cand / canonical).exists():
            return cand
    return _EOBS_CANDIDATE_DIRS[0]


EOBS_DIR = _resolve_eobs_dir()
EOBS_FILES = {var: EOBS_DIR / fname for var, fname in _EOBS_FILE_NAMES.items()}

# Period definitions used by the network pipeline (year-month bounds, inclusive).
EARLY_PERIOD = ("1961-01", "1990-12")
LATE_PERIOD = ("1995-01", "2024-12")
FULL_PERIOD = ("1961-01", "2024-12")

# Carpathian basin nominal bounds — used purely for axis hints.
CARPATHIAN_BOUNDS = {
    "lat_min": 45.0,
    "lat_max": 49.0,
    "lon_min": 15.0,
    "lon_max": 26.5,
}

DEFAULT_PAGE_CONFIG = {
    "page_title": "TBCN — Carpathian Basin Visualizer",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
