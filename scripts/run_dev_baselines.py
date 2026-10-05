"""SUPERSEDED (kept for the record): single-sweep binary occupancy baselines on the development split.

This was the first experiment. Its negative result (persistence worse than climatology; labels are
noisy snapshots) motivated the window-fraction targets used everywhere else; see docs/technical/dev_results.md
and scripts/run_window_dev.py. It does not touch the test split. Output defaults to
results/dev_baselines.json (not tracked).

Development-split baselines on a real AERPAW block spectrogram.

    uv run python scripts/run_dev_baselines.py data/processed/aerpaw_CC1_blocks5MHz.npz

Target: busy/idle of one 20 MHz channel band at the next sweep, where *busy* means the channel's
band-mean power exceeds a noise floor + X dB. The floor is the 5th percentile of the TRAINING
period's band mean (per channel). Models are fitted on train and scored on dev only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks, label_band_table
from spectrum_occupancy_forecasting.evaluation.metrics import binary_report, brier
from spectrum_occupancy_forecasting.evaluation.splits import assert_no_leakage, chronological_split
from spectrum_occupancy_forecasting.features.age_aware import FeatureSpec, build_features
from spectrum_occupancy_forecasting.models.baselines import (
    AgeConditionedPersistence,
    DecayedHistory,
    GlobalRate,
    HourOfWeekRate,
    Persistence,
    gradient_boosting,
    logistic_regression,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask

CHANNELS = (2412.0, 2437.0, 2462.0)
TZ = "America/New_York"  # NC State University (Raleigh)


def make_models():
    return [
        GlobalRate(),
        HourOfWeekRate(tz=TZ, period="day", min_count=100),
        Persistence(),
        DecayedHistory(prior_weight=1.0),
        AgeConditionedPersistence(),
        logistic_regression(),
        gradient_boosting(max_iter=200, learning_rate=0.05),
    ]


def sweep_level_mask(t_s: np.ndarray, deg: Degradation, seed: int) -> np.ndarray:
    """Visibility decided per sweep (all channels of a sweep are seen or lost together)."""
    uniq, inv = np.unique(t_s, return_inverse=True)
    return visibility_mask(uniq, deg, seed)[inv]


def score(y, p, rate):
    rep = binary_report(y, p, threshold=np.nextafter(rate, 1.0))  # busy iff p > training rate
    rep["auc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 and np.ptp(p) > 0 else 0.5
    rep["bss"] = 1 - brier(y, p) / brier(y, np.full(len(y), rate))  # vs training climatology
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--target", type=float, default=2437.0)
    ap.add_argument("--x-db", type=float, nargs="+", default=[3.0, 6.0])
    ap.add_argument("--out", default="results/dev_baselines.json")
    a = ap.parse_args()

    spec = FeatureSpec(tau_s=900.0, windows_s=(300.0, 1800.0, 7200.0))
    emb = spec.max_lookback_s
    raw = band_table_from_blocks(a.npz, CHANNELS, bw_mhz=20.0, session=0)
    tgt_t = raw.loc[raw["freq_mhz"] == a.target, "t_start_s"].to_numpy()
    tr, dv, te = chronological_split(tgt_t, embargo_s=emb)
    assert_no_leakage(tgt_t, tr, dv, te, emb)
    t_train_end = tgt_t[tr].max()
    print(f"sweeps {len(tgt_t)}  train {len(tr)}  dev {len(dv)}  test {len(te)} (held out)  "
          f"embargo {emb:.0f}s", flush=True)

    results = {
        "target_mhz": a.target,
        "n": {"train": len(tr), "dev": len(dv), "test_held_out": len(te)},
        "embargo_s": emb,
        "scenarios": {},
    }
    for x_db in a.x_db:
        # Noise floor per channel from the TRAIN period only.
        thr = {}
        for c in CHANNELS:
            m = (raw["freq_mhz"] == c) & (raw["t_start_s"] <= t_train_end)
            thr[c] = float(np.percentile(raw.loc[m, "mean_dbm"], 5)) + x_db
        d = label_band_table(raw, thr, stat="mean_dbm").reset_index(drop=True)
        is_t = (d["freq_mhz"] == a.target).to_numpy()
        X = build_features(d, is_t, spec=spec)
        y = d.loc[X.index, "occ_any"].to_numpy()
        order = np.argsort(X["t_start_s"].to_numpy(), kind="stable")  # align with tgt_t order
        X, y = X.iloc[order], y[order]
        assert np.allclose(X["t_start_s"].to_numpy(), tgt_t)
        rate = float(y[tr].mean())
        thr_txt = ", ".join(f"{k:g}: {v:.1f}" for k, v in thr.items())
        print(f"\n=== X=+{x_db:g} dB  thresholds {{{thr_txt}}}  "
              f"prevalence train {rate:.3f} dev {y[dv].mean():.3f}", flush=True)
        models = make_models()
        for m in models:
            m.fit(X.iloc[tr], y[tr])
        rows = {}
        for m in models:
            rows[m.name] = score(y[dv], m.predict_proba(X.iloc[dv]), rate)
        tab = pd.DataFrame(rows).T[["brier", "bss", "auc", "balanced_accuracy", "recall",
                                     "false_idle_rate", "false_busy_rate", "ece"]]
        print(tab.round(3).to_string(), flush=True)
        sc = {"threshold_dbm": thr, "prevalence_train": rate, "prevalence_dev": float(y[dv].mean()),
              "dev_metrics": rows, "robustness_dev": {}}

        if x_db == a.x_db[0]:  # robustness preview (clean-trained models, degraded inputs)
            t0 = time.time()
            visible_stream = d["t_start_s"].to_numpy()
            scen = {"clean": Degradation()}
            for p in (0.5, 0.9, 0.99):
                scen[f"dropout_{p:g}"] = Degradation(dropout_p=p)
            for L in (900.0, 3600.0):
                scen[f"outage_{int(L)}s_25pct"] = Degradation(outage_len_s=L, outage_frac=0.25)
            for dl in (60.0, 600.0, 3600.0):
                scen[f"delay_{int(dl)}s"] = Degradation(delay_s=dl)
            for name, deg in scen.items():
                vis = sweep_level_mask(visible_stream, deg, seed=1)
                Xd = build_features(d, is_t, vis, delay_s=deg.delay_s, spec=spec)
                Xd = Xd.iloc[order]
                sc["robustness_dev"][name] = {
                    m.name: score(y[dv], m.predict_proba(Xd.iloc[dv]), rate) for m in models}
            print(f"\nrobustness preview (dev, clean-trained; BSS / false-idle) "
                  f"[{time.time() - t0:.0f}s]")
            hdr = f"{'scenario':<22}" + "".join(f"{m.name[:14]:>16}" for m in models)
            print(hdr)
            for name in scen:
                cells = "".join(
                    f"{sc['robustness_dev'][name][m.name]['bss']:>+8.3f}/{sc['robustness_dev'][name][m.name]['false_idle_rate']:<6.2f}"
                    for m in models)
                print(f"{name:<22}{cells}")
        results["scenarios"][f"x_{x_db:g}db"] = sc

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
