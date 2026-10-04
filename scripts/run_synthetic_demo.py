"""Download-free demonstration. Synthetic output is never evidence for the real-data study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments import window_forecast as wf
from spectrum_occupancy_forecasting.viz.figures import figure2_methods


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("reproductions/synthetic_demo"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = a.out_dir / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(blocks, n_days=3.1, dt_s=60.0, seed=3)
    raw = band_table_from_blocks(blocks, wf.CHANNELS)
    t0 = float(raw["t_start_s"].min())
    fold = wf.run_fold(raw, 900.0, 3.0, t0 + 2 * wf.DAY, t0 + 3 * wf.DAY, False)
    report = wf.evaluate_folds([fold], False, n_boot=100)
    result = {
        "synthetic": True,
        "purpose": "Workflow demonstration only; not evidence for the real-data study",
        "seed": 3,
        "x_db": 3.0,
        "reference": wf.REF,
        "windows": {"900s": report},
    }
    out = a.out_dir / "synthetic_window_forecast.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    figure2_methods(out, a.out_dir / "figures", region="synthetic")
    print(f"SYNTHETIC DEMO ONLY: results in {out}; figures in {a.out_dir / 'figures'}")


if __name__ == "__main__":
    main()
