# TBCN visualization interface

A Streamlit app for exploring the network indexes computed by
`networks/pipeline/create_networks_calculate_indexes.ipynb`.

## Quick start

From the repository root:

```bash
pip install -r interface/requirements.txt
streamlit run interface/app.py
```

The app reads `networks/pipeline/networks_indexes_combined_output.nc`. Generate
that file via the notebook before launching.

## Current page

`app.py` is the **Carpathian Basin Overview**. The whole page is laid out as
title + two-column body (map on the left, controls and statistics on the right)
so everything fits in a single viewport without scrolling. The Streamlit sidebar
is collapsed by default.

Hovering over any cell of the heatmap shows its latitude, longitude and value.
The Plotly toolbar (top-right of the chart) lets you pan, zoom and download a
PNG.

## Layout

```
interface/
├── app.py                  # Carpathian Overview (entry point)
├── pages/                  # Future Streamlit pages go here
└── tbcn_viz/               # Reusable building blocks
    ├── config.py           # Paths, page config
    ├── data_loader.py      # Cached NetCDF loader
    ├── index_metadata.py   # Per-index labels, colormaps, descriptions
    ├── controls.py         # `render_controls(container, ds) -> Selection`
    └── map_plot.py         # Plotly heatmap factory + slice helper
```

## Adding a new visualization

1. **New page** — drop a `pages/N_My_Page.py` file. It can re-use
   `load_dataset`, `render_controls`, `build_heatmap`, etc. Once a file is in
   `pages/`, Streamlit's page navigator (in the sidebar) appears automatically.
2. **New index** — add an `IndexSpec` entry to
   `tbcn_viz/index_metadata.py`; the controls and plots will pick it up
   automatically.
