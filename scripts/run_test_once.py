"""Evaluate the frozen design on the HELD-OUT TEST region. Runs once; see docs/test_protocol.md.

    uv run python scripts/run_test_once.py data/processed/aerpaw_CC1_blocks5MHz.npz

Before any test data is loaded it (1) checks the frozen config matches the code constants,
(2) fingerprints code + config + protocol, and (3) writes results/TEST_RUN_LOCK.json. A second
run is refused unless ``--acknowledge-rerun`` is given (which is counted and must be disclosed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.evaluation.splits import development_region_end
from spectrum_occupancy_forecasting.experiments import window_forecast as wf
from spectrum_occupancy_forecasting.experiments.protocol import (
    acquire_lock,
    complete_lock,
    fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "frozen_experiment.yaml"
FROZEN_FILES = [ROOT / "src", ROOT / "scripts" / "run_test_once.py", CONFIG,
                ROOT / "docs" / "test_protocol.md"]
X_DB = 3.0
WINDOWS = (300.0, 900.0, 3600.0)
PRIMARY_W = 900.0
N_BOOT = 1000


def check_config_matches_code(cfg: dict) -> None:
    """The config is documentation that must not drift from the executed constants."""
    assert tuple(cfg["target"]["channels_for_features_mhz"]) == wf.CHANNELS
    assert cfg["target"]["channel_mhz"] == wf.TARGET_MHZ
    assert cfg["target"]["x_db"] == X_DB
    assert tuple(cfg["target"]["windows_s"]) == WINDOWS
    assert cfg["target"]["primary_window_s"] == PRIMARY_W
    assert cfg["features"]["tau_s"] == wf.SPEC.tau_s
    assert tuple(cfg["features"]["windows_s"]) == wf.SPEC.windows_s
    assert cfg["features"]["cross_channel"] == wf.SPEC.cross_channel
    assert tuple(cfg["metrics"]["event_thetas"]) == wf.THETAS
    assert cfg["metrics"]["reference_model"] == wf.REF
    assert cfg["metrics"]["bootstrap_resamples"] == N_BOOT
    assert cfg["metrics"]["bootstrap_block_s"] == 7200
    assert cfg["robustness"]["random_mask_seeds"] == wf.N_MASK_SEEDS
    assert cfg["robustness"]["window_s"] == PRIMARY_W
    assert cfg["dataset"]["session"] == 0 and cfg["dataset"]["block_mhz"] == 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="results/test_window_forecast.json")
    ap.add_argument("--lock", default="results/TEST_RUN_LOCK.json")
    ap.add_argument("--acknowledge-rerun", action="store_true",
                    help="burns the test set: counted in the lock, must be disclosed")
    a = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    check_config_matches_code(cfg)
    fp = fingerprint(FROZEN_FILES)
    lock = acquire_lock(ROOT / a.lock, fp, a.acknowledge_rerun)
    print(f"fingerprint {fp}\nlock written ({lock['status']}); loading data", flush=True)

    raw = band_table_from_blocks(a.npz, wf.CHANNELS, bw_mhz=20.0, session=cfg["dataset"]["session"])
    t0, t_last = raw["t_start_s"].min(), raw["t_start_s"].max()
    test_start = development_region_end(t0, t_last)
    print(f"test region: {(test_start - t0) / wf.DAY:.2f} d -> {(t_last - t0) / wf.DAY:.2f} d "
          f"after session start", flush=True)

    results = {"fingerprint": fp, "x_db": X_DB, "channels": list(wf.CHANNELS),
               "thetas": wf.THETAS, "reference": wf.REF,
               "test_start_s_after_session_start": test_start - t0,
               "test_end_s_after_session_start": t_last - t0, "windows": {}}
    for W in WINDOWS:
        rob = W == PRIMARY_W
        folds = [wf.run_fold(raw, W, X_DB, test_start, t_last, rob)]
        res = wf.evaluate_folds(folds, rob, n_boot=N_BOOT)
        res["thresholds_dbm"] = folds[0]["thresholds_dbm"]
        wf.print_report(res, W)
        results["windows"][f"{W:.0f}s"] = res

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    done = complete_lock(ROOT / a.lock, out)
    print(f"\nwrote {out}\nlock completed; results sha256 {done['results_sha256']}")


if __name__ == "__main__":
    main()
