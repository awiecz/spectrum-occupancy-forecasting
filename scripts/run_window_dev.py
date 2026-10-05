"""Window-occupancy forecasting on a real AERPAW block spectrogram. DEVELOPMENT REGION ONLY.

    uv run python scripts/run_window_dev.py data/processed/aerpaw_CC1_blocks5MHz.npz

See ``spectrum_occupancy_forecasting.experiments.window_forecast`` for the protocol. Evaluation =
rolling-origin 1-day folds inside the development region (first 75 % of the week); the final 25 %
(test) is never read here. Noise floors are refit per fold from its own training period.

Choices made before seeing these scores: embargo = max look-back + W; X = 3 dB; fold length
1 day; reference baselines = train climatology and the 2 h recent fraction. The move from
single-sweep to window targets, and from one dev day to rolling folds, was made after seeing
earlier dev scores (disclosed in docs/technical/dev_results.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.evaluation.splits import development_region_end
from spectrum_occupancy_forecasting.experiments.window_forecast import (
    CHANNELS,
    DAY,
    REF,
    THETAS,
    evaluate_folds,
    print_report,
    run_fold,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--windows", type=float, nargs="+", default=[300.0, 900.0, 3600.0])
    ap.add_argument("--x-db", type=float, default=3.0)
    ap.add_argument("--robust-window", type=float, default=900.0)
    ap.add_argument("--out", default="results/dev_window_forecast.json")
    a = ap.parse_args()

    raw = band_table_from_blocks(a.npz, CHANNELS, bw_mhz=20.0, session=0)
    t0, t_last = raw["t_start_s"].min(), raw["t_start_s"].max()
    region_end = development_region_end(t0, t_last)  # test region lies beyond this time
    raw = raw[raw["t_start_s"] <= region_end].reset_index(drop=True)  # test data never loaded
    folds_def = [
        (t0 + 2 * DAY, t0 + 3 * DAY),
        (t0 + 3 * DAY, t0 + 4 * DAY),
        (t0 + 4 * DAY, region_end),
    ]
    print(
        f"development region ends {(region_end - t0) / DAY:.2f} d after start; "
        f"test region ({(t_last - region_end) / DAY:.2f} d) removed before any computation"
    )
    results = {
        "x_db": a.x_db,
        "channels": list(CHANNELS),
        "thetas": THETAS,
        "reference": REF,
        "folds": [[lo - t0, hi - t0] for lo, hi in folds_def],
        "windows": {},
    }
    for W in a.windows:
        rob = W == a.robust_window
        folds = [run_fold(raw, W, a.x_db, lo, hi, rob) for lo, hi in folds_def]
        res = evaluate_folds(folds, rob)
        res["n_dev"] = res.pop("n_eval")
        print_report({**res, "n_eval": res["n_dev"]}, W)
        results["windows"][f"{W:.0f}s"] = res
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
