"""Render the three figures.

    uv run python scripts/make_figures.py data/processed/aerpaw_CC1_blocks5MHz.npz \
        results/dev_window_forecast.json --out results/figures
"""

from __future__ import annotations

import argparse

from spectrum_occupancy_forecasting.viz.figures import (
    figure1_characterisation,
    figure2_methods,
    figure3_robustness,
    figure4_transfer,
)

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("results_json")
ap.add_argument("--out", default="results/figures")
ap.add_argument("--region", choices=["dev", "test", "lw1", "transfer"], default="dev")
a = ap.parse_args()
if a.region == "transfer":
    figure4_transfer(a.results_json, a.out)
else:
    if a.region == "dev":  # Fig 1 uses the development region only, never the test region
        figure1_characterisation(a.npz, a.out)
    figure2_methods(a.results_json, a.out, region=a.region)
    figure3_robustness(a.results_json, a.out, region=a.region)
print("figures written to", a.out)
