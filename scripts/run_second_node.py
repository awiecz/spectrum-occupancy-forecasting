"""Confirmation run of the frozen design on a second AERPAW node. Runs once; see docs/lw1_protocol.md.

    uv run python scripts/run_second_node.py data/processed/aerpaw_LW1_blocks5MHz.npz

Writes results/node_<SITE>_window_forecast.json and a one-shot lock results/NODE_<SITE>_RUN_LOCK.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.experiments import window_forecast as wf
from spectrum_occupancy_forecasting.experiments.protocol import (
    acquire_lock,
    complete_lock,
    fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = [ROOT / "src", ROOT / "scripts" / "run_second_node.py",
                ROOT / "configs" / "frozen_experiment.yaml", ROOT / "docs" / "lw1_protocol.md"]
X_DB = 3.0
WINDOWS = (300.0, 900.0, 3600.0)
PRIMARY_W = 900.0
N_BOOT = 1000
MIN_PREVALENCE = 0.05  # below this the target channel is declared degenerate


def select_channel(raw, t0, t_last):
    """Rule fixed in the protocol: highest busy prevalence at +3 dB on the first 60 % of the session."""
    cut = t0 + 0.6 * (t_last - t0)
    table = {}
    for c in wf.CHANNELS:
        x = raw.loc[(raw["freq_mhz"] == c) & (raw["t_start_s"] < cut), "mean_dbm"].to_numpy()
        floor = float(np.percentile(x, 5))
        table[c] = {"noise_floor_dbm": floor, "prevalence_x3db": float(np.mean(x > floor + X_DB))}
    best = max(table, key=lambda c: table[c]["prevalence_x3db"])
    return best, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lock", default=None)
    ap.add_argument("--acknowledge-rerun", action="store_true",
                    help="burns the confirmation run: counted in the lock, must be disclosed")
    a = ap.parse_args()

    site = Path(a.npz).stem.split("_")[1]
    out = Path(a.out) if a.out else ROOT / "results" / f"node_{site}_window_forecast.json"
    lock_path = Path(a.lock) if a.lock else ROOT / "results" / f"NODE_{site}_RUN_LOCK.json"
    fp = fingerprint(FROZEN_FILES, root=ROOT)
    lock = acquire_lock(lock_path, fp, a.acknowledge_rerun)
    print(f"site {site}\nfingerprint {fp}\nlock written ({lock['status']})", flush=True)

    raw = band_table_from_blocks(a.npz, wf.CHANNELS, bw_mhz=20.0, session=0)
    t0, t_last = float(raw["t_start_s"].min()), float(raw["t_start_s"].max())
    span_d = (t_last - t0) / wf.DAY
    target, table = select_channel(raw, t0, t_last)
    best = table[target]["prevalence_x3db"]
    print(f"main session {span_d:.2f} d; channel selection (first 60 %): "
          + ", ".join(f"{c:g}: prev {v['prevalence_x3db']:.3f}" for c, v in table.items())
          + f" -> target {target:g}", flush=True)
    results = {"site": site, "fingerprint": fp, "x_db": X_DB, "span_days": span_d,
               "channel_selection": {str(k): v for k, v in table.items()},
               "target_mhz": target, "degenerate": best < MIN_PREVALENCE, "windows": {}}
    if best < MIN_PREVALENCE:
        print(f"target degenerate (best prevalence {best:.3f} < {MIN_PREVALENCE}): descriptive only")
    else:
        starts = [t0 + k * wf.DAY for k in range(2, int(span_d) + 1)]
        folds_def = []
        for s0 in starts:
            hi = min(s0 + wf.DAY, t_last)
            if hi - s0 >= 0.5 * wf.DAY:
                folds_def.append((s0, hi))
        results["folds_days"] = [[(lo - t0) / wf.DAY, (hi - t0) / wf.DAY] for lo, hi in folds_def]
        for W in WINDOWS:
            rob = W == PRIMARY_W
            folds = [wf.run_fold(raw, W, X_DB, lo, hi, rob, wf.CHANNELS, target)
                     for lo, hi in folds_def]
            res = wf.evaluate_folds(folds, rob, n_boot=N_BOOT)
            wf.print_report(res, W)
            results["windows"][f"{W:.0f}s"] = res

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    done = complete_lock(lock_path, out)
    print(f"\nwrote {out}\nlock completed; results sha256 {done['results_sha256']}")


if __name__ == "__main__":
    main()
