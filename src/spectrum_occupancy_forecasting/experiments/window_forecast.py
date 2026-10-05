"""Window-occupancy forecasting experiment (shared by the dev and the one-shot test scripts).

Target: busy fraction of one 20 MHz channel band (2.4 GHz channel 6) over the NEXT window [o, o+W),
forecast at origin ``o`` from observations that ended before ``o`` (minus any delay).
Busy = band-mean power > train-period noise floor + X dB. Labels always come from the intact
stream; degradations touch predictor inputs only. ``run_fold`` fits on data before
``eval_lo - embargo`` and scores non-overlapping windows in ``[eval_lo, eval_hi]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.data.aerpaw import label_band_table
from spectrum_occupancy_forecasting.evaluation.bootstrap import block_bootstrap_ci, block_ids
from spectrum_occupancy_forecasting.evaluation.metrics import event_tradeoff, fraction_report
from spectrum_occupancy_forecasting.features.age_aware import FeatureSpec, build_features
from spectrum_occupancy_forecasting.features.windows import origin_grid, window_fraction
from spectrum_occupancy_forecasting.models.regressors import (
    ClimatologyFraction,
    DecayedFraction,
    HourOfDayFraction,
    RecentFraction,
    gbt_regressor,
    residual_gbt,
    residual_ridge,
    ridge_regressor,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, sweep_visibility_mask

CHANNELS = (2412.0, 2437.0, 2462.0)
TARGET_MHZ = 2437.0
TZ = "America/New_York"
SPEC = FeatureSpec(tau_s=900.0, windows_s=(300.0, 1800.0, 7200.0))
THETAS = (0.1, 0.2, 0.3)  # fixed "high-activity window" cut-offs (busy fraction)
REF = "recent_7200s"  # strong naive reference (2 h moving busy fraction)
DAY = 86400.0
N_MASK_SEEDS = 5  # random degradation masks averaged per stochastic robustness scenario


def base_models():
    return [
        ClimatologyFraction(),
        HourOfDayFraction(tz=TZ),
        RecentFraction(300),
        RecentFraction(1800),
        RecentFraction(7200),
        DecayedFraction(),
        ridge_regressor(TZ),
        gbt_regressor(TZ),
        residual_ridge(TZ),
        residual_gbt(TZ),
    ]


def scenarios():
    s = {"clean": Degradation()}
    for p in (0.5, 0.9, 0.99):
        s[f"dropout_{p:g}"] = Degradation(dropout_p=p)
    for length in (900.0, 3600.0):
        s[f"outage_{int(length)}s_x25pct"] = Degradation(outage_len_s=length, outage_frac=0.25)
    for dl in (60.0, 600.0, 1800.0, 3600.0, 7200.0):
        s[f"delay_{int(dl)}s"] = Degradation(delay_s=dl)
    return s


def feats(d, tg, deg, seed):
    vis = sweep_visibility_mask(d["t_start_s"].to_numpy(), deg, seed)
    return build_features(
        d, targets=tg, visible=vis, delay_s=deg.delay_s, spec=SPEC, power_col="mean_dbm"
    )


def run_fold(raw, W, x_db, eval_lo, eval_hi, robustness, channels=CHANNELS,
             target_mhz=TARGET_MHZ, nominal_dt_s=None):
    """Fit on data before ``eval_lo - embargo``; return eval labels and per-model predictions."""
    t0 = raw["t_start_s"].min()
    emb = SPEC.max_lookback_s + W
    train_hi = eval_lo - emb  # last training origin; its window ends before eval_lo - lookback
    thr = {}
    for c in channels:  # noise floor from the training period only
        m = (raw["freq_mhz"] == c) & (raw["t_start_s"] < train_hi + W)
        thr[c] = float(np.percentile(raw.loc[m, "mean_dbm"], 5)) + x_db
    d = label_band_table(raw, thr, stat="mean_dbm").reset_index(drop=True)
    ch = d[d["freq_mhz"] == target_mhz]
    # expected sweep spacing for the window-coverage filter: the node's own median period
    # (19 s on CC1, ~30 s on LW1); a hard-coded value would discard every LW1 window
    dt = nominal_dt_s or float(np.median(np.diff(np.sort(ch["t_start_s"].to_numpy()))))

    def targets(o):
        return pd.DataFrame({"site": d["site"].iloc[0], "freq_mhz": target_mhz, "t_start_s": o})

    fine = origin_grid(t0, train_hi, max(W / 5.0, 60.0), W)
    y_tr, _ = window_fraction(ch["t_start_s"], ch["occ_any"], fine, W, nominal_dt_s=dt)
    o_ev = origin_grid(eval_lo, eval_hi, W, W)
    y_ev, _ = window_fraction(ch["t_start_s"], ch["occ_any"], o_ev, W, nominal_dt_s=dt)
    fine, y_tr = fine[np.isfinite(y_tr)], y_tr[np.isfinite(y_tr)]
    o_ev, y_ev = o_ev[np.isfinite(y_ev)], y_ev[np.isfinite(y_ev)]
    assert fine.max() + W <= eval_lo - SPEC.max_lookback_s + 1e-6, "training window reaches eval"
    X_tr = feats(d, targets(fine), Degradation(), 0)
    X_ev = feats(d, targets(o_ev), Degradation(), 0)

    models = base_models()
    for m in models:
        m.fit(X_tr, y_tr)
    clim = float(y_tr.mean())
    out = {"t": o_ev, "y": y_ev, "clim": clim, "n_train": len(y_tr),
           "thresholds_dbm": thr, "preds": {"clean": {m.name: m.predict(X_ev) for m in models}}}
    if robustness:
        aug = [Degradation(dropout_p=0.5), Degradation(dropout_p=0.9), Degradation(delay_s=600.0),
               Degradation(delay_s=3600.0), Degradation(outage_len_s=3600.0, outage_frac=0.25)]
        Xa = pd.concat([X_tr] + [feats(d, targets(fine), g, 10 + i) for i, g in enumerate(aug)],
                       ignore_index=True)
        ya = np.concatenate([y_tr] * (1 + len(aug)))
        aug_models = [ridge_regressor(TZ), gbt_regressor(TZ)]
        for m in aug_models:
            m.name += "_aug"
            m.fit(Xa, ya)
        every = models + aug_models
        out["preds_seeds"] = {}
        for name, deg in scenarios().items():
            stochastic = deg.dropout_p > 0 or deg.outage_frac > 0
            runs = []
            for sd in range(1, N_MASK_SEEDS + 1) if stochastic else [1]:
                Xd = feats(d, targets(o_ev), deg, seed=sd)
                runs.append({m.name: m.predict(Xd) for m in every})
            out["preds_seeds"][name] = runs
        out["preds"]["clean"] = out["preds_seeds"]["clean"][0]
    return out


def pooled(folds, scen, name, theta):
    """Pool one scenario/model over folds: MSE skills, within-fold R^2, calibration, events."""
    y = np.concatenate([f["y"] for f in folds])
    p = np.concatenate([f["preds"][scen][name] for f in folds])
    clim = np.concatenate([np.full(len(f["y"]), f["clim"]) for f in folds])
    ref = np.concatenate([f["preds"]["clean"][REF] for f in folds])
    ymean = np.concatenate([np.full(len(f["y"]), f["y"].mean()) for f in folds])
    rep = fraction_report(y, p, clim, theta)
    sse = np.sum((p - y) ** 2)
    rep["mss_vs_ref"] = 1 - sse / np.sum((ref - y) ** 2)
    rep["r2_within_fold"] = 1 - sse / np.sum((y - ymean) ** 2)
    return rep, y, p, clim


def _aggregate(reps):
    """Mean of each numeric metric over random-mask seeds, plus SD / extremes for key metrics."""
    out = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
    out["n_seeds"] = len(reps)
    for k in ("rmse", "mss", "mss_vs_ref"):
        vals = [r[k] for r in reps]
        out[f"{k}_sd"] = float(np.std(vals))
        out[f"{k}_min"], out[f"{k}_max"] = float(np.min(vals)), float(np.max(vals))
    return out


def evaluate_folds(folds, robustness: bool, n_boot: int = 300):
    """Summarise folds into the result dict (clean metrics with CIs/events, robustness table)."""
    names = list(folds[0]["preds"]["clean"])
    res = {"n_eval": [len(f["y"]) for f in folds], "n_train": [f["n_train"] for f in folds],
           "fold_mean_fraction": [float(f["y"].mean()) for f in folds],
           "fold_train_climatology": [f["clim"] for f in folds], "clean": {}, "robustness": {}}
    t_all = np.concatenate([f["t"] for f in folds])
    rows = {}
    for n in names:
        rep, y, p, clim = pooled(folds, "clean", n, 0.2)

        def mss(yy, pc):  # pc = [prediction, climatology] columns, resampled together
            return 1 - np.mean((pc[:, 0] - yy) ** 2) / np.mean((yy - pc[:, 1]) ** 2)

        ci = block_bootstrap_ci(
            y, np.column_stack([p, clim]), block_ids(t_all, 7200.0), mss, n_boot=n_boot, seed=0
        )
        rep["mss_ci95"] = [ci[1], ci[2]]
        ref = np.concatenate([f["preds"]["clean"][REF] for f in folds])

        def vs_ref(yy, pr):  # skill of the model relative to the 2 h reference, resampled jointly
            return 1 - np.sum((pr[:, 0] - yy) ** 2) / np.sum((pr[:, 1] - yy) ** 2)

        cr = block_bootstrap_ci(
            y, np.column_stack([p, ref]), block_ids(t_all, 7200.0), vs_ref, n_boot=n_boot, seed=1
        )
        rep["mss_vs_ref_ci95"] = [cr[1], cr[2]]
        rep["events"] = {str(th): pooled(folds, "clean", n, th)[0] for th in THETAS}
        rep["tradeoff"] = {str(th): event_tradeoff(y, p, th) for th in THETAS}
        rows[n] = rep
    res["clean"] = rows
    if robustness:
        every = list(folds[0]["preds"]["clean"])
        for sc, runs in folds[0]["preds_seeds"].items():
            per_seed = []
            for s_idx in range(len(runs)):
                tmp = [{**f, "preds": {**f["preds"], sc: f["preds_seeds"][sc][s_idx]}}
                       for f in folds]
                per_seed.append({n: pooled(tmp, sc, n, 0.2)[0] for n in every})
            res["robustness"][sc] = {n: _aggregate([ps[n] for ps in per_seed]) for n in every}
    return res


def print_report(res, W: float) -> None:
    rows = res["clean"]
    names = list(rows)
    print(f"\n=== W={W:.0f}s | eval windows {res['n_eval']}, train {res['n_train']} | "
          f"eval mean fraction {np.round(res['fold_mean_fraction'], 3).tolist()} vs train "
          f"climatology {np.round(res['fold_train_climatology'], 3).tolist()}", flush=True)
    tab = pd.DataFrame(rows).T[["rmse", "mss", "mss_vs_ref", "r2_within_fold", "bias",
                                 "calib_slope"]]
    tab["mss_ci95"] = [f"[{rows[k]['mss_ci95'][0]:+.2f},{rows[k]['mss_ci95'][1]:+.2f}]"
                       for k in tab.index]
    print(tab.round(3).to_string())
    print("event false-idle / prevalence at theta (busy windows forecast below theta):")
    for n in ("recent_7200s", "decayed", "ridge", "hgb", "resid_ridge", "resid_hgb"):
        cells = "  ".join(
            f"th={th}: {rows[n]['events'][str(th)]['event_false_idle']:.2f}"
            f"/{rows[n]['events'][str(th)]['event_prevalence']:.2f}" for th in THETAS)
        print(f"  {n:<14}{cells}")
    print("equal-false-busy comparison at theta=0.2: AUC | false-idle at 10% / 20% false-busy")
    for n in names:
        tr_ = rows[n]["tradeoff"]["0.2"]
        print(f"  {n:<14}auc {tr_.get('auc', float('nan')):.3f} | "
              f"{tr_.get('false_idle_at_fb10', float('nan')):.2f} / "
              f"{tr_.get('false_idle_at_fb20', float('nan')):.2f}")
    if res["robustness"]:
        every = list(next(iter(res["robustness"].values())))
        print(f"\nrobustness (W={W:.0f}s): MSE skill vs climatology / vs {REF} (clean) -- "
              f"mean of {N_MASK_SEEDS} random masks for dropout/outage; clean-trained except *_aug")
        print(f"{'scenario':<20}" + "".join(f"{n[:12]:>16}" for n in every))
        for sc, byname in res["robustness"].items():
            cells = "".join(f"{byname[n]['mss']:>+7.2f}/{byname[n]['mss_vs_ref']:>+6.2f} "
                            for n in every)
            print(f"{sc:<20}{cells}")


# ---------------------------------------------------------------------------------------------
# Cross-node transfer (see docs/technical/transfer_protocol.md)
# ---------------------------------------------------------------------------------------------
def _floors(raw, channels, x_db, limit_t):
    """Per-channel busy threshold (dBm): 5th percentile of band-mean power before limit_t, + X."""
    thr = {}
    for c in channels:
        m = (raw["freq_mhz"] == c) & (raw["t_start_s"] < limit_t)
        thr[c] = float(np.percentile(raw.loc[m, "mean_dbm"], 5)) + x_db
    return thr


def _median_dt(ch):
    return float(np.median(np.diff(np.sort(ch["t_start_s"].to_numpy()))))


def _node_windows(raw, thr, W, train_hi, channels, target_mhz):
    """Training windows (features, busy fractions) of one node using thresholds ``thr``."""
    d = label_band_table(raw, thr, stat="mean_dbm").reset_index(drop=True)
    ch = d[d["freq_mhz"] == target_mhz]
    dt = _median_dt(ch)
    t0 = raw["t_start_s"].min()
    fine = origin_grid(t0, train_hi, max(W / 5.0, 60.0), W)
    y, _ = window_fraction(ch["t_start_s"], ch["occ_any"], fine, W, nominal_dt_s=dt)
    fine, y = fine[np.isfinite(y)], y[np.isfinite(y)]
    assert fine.max() + W <= train_hi + W + 1e-6
    tg = pd.DataFrame({"site": d["site"].iloc[0], "freq_mhz": target_mhz, "t_start_s": fine})
    return feats(d, tg, Degradation(), 0), y


def run_transfer_fold(nodes, train_nodes, target, W, x_db, eval_lo, eval_hi,
                      channels=CHANNELS, target_mhz=TARGET_MHZ, abs_thresholds_from=None):
    """Fit on ``train_nodes`` (all cut at ``eval_lo - embargo``), score windows of ``target``.

    ``nodes`` maps node name -> band table (single site each). Each node's noise floors come from
    its *own* pre-cutoff data; with ``abs_thresholds_from`` the target is instead labelled with that
    node's absolute dBm thresholds (no target-node calibration). The in-node arm
    (``train_nodes=[target]``) is numerically identical to :func:`run_fold`.
    """
    emb = SPEC.max_lookback_s + W
    train_hi = eval_lo - emb  # last training origin; its window ends before eval_lo - lookback
    limit = train_hi + W
    parts = []
    for n in train_nodes:
        parts.append(_node_windows(nodes[n], _floors(nodes[n], channels, x_db, limit), W,
                                   train_hi, channels, target_mhz))
    X_tr = pd.concat([x for x, _ in parts], ignore_index=True)
    y_tr = np.concatenate([y for _, y in parts])

    raw_t = nodes[target]
    own = _floors(raw_t, channels, x_db, limit)
    thr_eval = _floors(nodes[abs_thresholds_from], channels, x_db, limit) if abs_thresholds_from \
        else own
    d = label_band_table(raw_t, thr_eval, stat="mean_dbm").reset_index(drop=True)
    ch = d[d["freq_mhz"] == target_mhz]
    dt = _median_dt(ch)
    o_ev = origin_grid(eval_lo, eval_hi, W, W)
    y_ev, _ = window_fraction(ch["t_start_s"], ch["occ_any"], o_ev, W, nominal_dt_s=dt)
    o_ev, y_ev = o_ev[np.isfinite(y_ev)], y_ev[np.isfinite(y_ev)]
    tg = pd.DataFrame({"site": d["site"].iloc[0], "freq_mhz": target_mhz, "t_start_s": o_ev})
    X_ev = feats(d, tg, Degradation(), 0)

    models = base_models()
    for m in models:
        m.fit(X_tr, y_tr)
    preds = {m.name: m.predict(X_ev) for m in models}
    return {"t": o_ev, "y": y_ev, "clim": float(y_tr.mean()), "n_train": len(y_tr),
            "thresholds_dbm": thr_eval, "preds": {"clean": preds}}


def paired_mse_ratio(folds_a, folds_b, name, n_boot=1000, seed=0):
    """MSE(a) / MSE(b) for model ``name`` over identical windows, with a paired block-bootstrap CI.

    Returns ``(ratio, lo, hi)``. ``folds_a`` and ``folds_b`` must score the same windows.
    """
    t = np.concatenate([f["t"] for f in folds_a])
    y = np.concatenate([f["y"] for f in folds_a])
    yb = np.concatenate([f["y"] for f in folds_b])
    if not (np.array_equal(y, yb) and np.array_equal(t, np.concatenate([f["t"] for f in folds_b]))):
        raise ValueError("arms do not score identical windows")
    pa = np.concatenate([f["preds"]["clean"][name] for f in folds_a])
    pb = np.concatenate([f["preds"]["clean"][name] for f in folds_b])

    def ratio(yy, pp):
        return np.sum((pp[:, 0] - yy) ** 2) / np.sum((pp[:, 1] - yy) ** 2)

    return block_bootstrap_ci(y, np.column_stack([pa, pb]), block_ids(t, 7200.0), ratio,
                              n_boot=n_boot, seed=seed)
