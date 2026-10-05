"""Power-first step of the sparse-regime study, computed from saved predictions only.

No model is fitted and no recording is read. The saved exploratory predictions have already
been inspected, so the effect sizes seen here informed the choice of scenarios; this step only
fixes how large a gain would have to be before it could be claimed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.evaluation.power import Z, block_indices, relative_se

LEARNED = ("ridge", "hgb", "resid_ridge", "resid_hgb", "ridge_aug", "hgb_aug")
SIMPLE = ("fixed_history", "selected_history", "climatology", "oracle_shrunk")
REGIONS = {"all_folds": None, "later_folds": 1}
MIN_DELTA = 0.20
DEMOTE_ABOVE = 0.50
STEP = 0.05


def loss(error, kind):
    return error**2 if kind == "squared" else np.abs(error)


def seed_averaged(frame):
    """One row per origin: truth, training mean, and per-method predictions averaged over seeds.

    Losses are averaged over masks by the caller; predictions are averaged here only to build the
    shrinkage proxy and are never scored.
    """
    wide = frame.pivot_table(
        index="origin_s", columns="method", values="prediction", aggfunc="mean"
    )
    meta = frame.groupby("origin_s").agg(y=("y", "mean"), mean=("training_mean", "mean"))
    return wide.join(meta)


def oracle_shrinkage(y, history, mean):
    """Least-squares weight of history over the training mean; chosen on the scored windows.

    This favours the baseline (the weight is fitted in-sample), so it is a stringent proxy.
    """
    centred = history - mean
    denominator = float((centred**2).sum())
    weight = 0.0 if denominator == 0 else float((centred * (y - mean)).sum() / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def cell_losses(frame, kind):
    """Per-origin loss (averaged over mask seeds) for every method plus the shrinkage proxy."""
    errors = frame.assign(error=frame["y"] - frame["prediction"])
    errors["loss"] = loss(errors["error"].to_numpy(), kind)
    losses = errors.pivot_table(index="origin_s", columns="method", values="loss", aggfunc="mean")
    mean_prediction = seed_averaged(frame)
    weight = oracle_shrinkage(
        mean_prediction["y"], mean_prediction["fixed_history"], mean_prediction["mean"]
    )
    shrunk = mean_prediction["mean"] + weight * (
        mean_prediction["fixed_history"] - mean_prediction["mean"]
    )
    losses["oracle_shrunk"] = loss(mean_prediction["y"] - shrunk, kind)
    return losses.sort_index(), weight


def baseline_cv(baseline_loss, length, replicates, rng):
    """Block-bootstrap standard error of the baseline's mean loss, relative to that mean."""
    idx = block_indices(len(baseline_loss), min(length, len(baseline_loss)), replicates, rng)
    means = baseline_loss[idx].mean(axis=1)
    return float(means.std(ddof=1) / baseline_loss.mean())


def analyse_cell(frame, kind, replicates, rng, block_windows=32):
    losses, weight = cell_losses(frame, kind)
    simple = {name: float(losses[name].mean()) for name in SIMPLE}
    best = min(simple, key=simple.get)
    baseline = losses[best].to_numpy()
    cv = baseline_cv(baseline, block_windows, replicates, rng)
    candidates = [name for name in LEARNED if name in losses]
    difference = baseline[:, None] - losses[candidates].to_numpy()
    se = relative_se(difference, baseline, block_windows, replicates, rng)
    gain = difference.mean(axis=0) / baseline.mean()
    rows = [
        dict(
            candidate=name,
            candidate_loss=float(losses[name].mean()),
            gain=float(gain[i]),
            se=float(se[i]),
            z=float(gain[i] / se[i]) if se[i] > 0 else float("nan"),
            mde_80=float(Z * se[i]),
        )
        for i, name in enumerate(candidates)
    ]
    return dict(
        n_windows=len(baseline),
        n_blocks=len(baseline) / block_windows,
        shrinkage_weight=weight,
        baseline_cv=cv,
        best_simple=best,
        best_simple_loss=simple[best],
        **{f"{name}_loss": value for name, value in simple.items()},
    ), rows


def power_tables(predictions, scenarios, replicates, seed, noise_ratios, block_windows=32):
    """Cell-level evidence and per-cell minimum detectable gain across learned families."""
    rng = np.random.default_rng(seed)
    primary = predictions[predictions["task"] == "primary"]
    baselines, evidence = [], []
    for node, node_frame in primary.groupby("node"):
        for region, first_fold in REGIONS.items():
            if node == "CC1" and region == "later_folds":
                continue
            region_frame = (
                node_frame if first_fold is None else node_frame[node_frame["fold"] >= first_fold]
            )
            for scenario in scenarios:
                frame = region_frame[region_frame["scenario"] == scenario]
                if frame.empty:
                    continue
                for kind in ("squared", "absolute"):
                    info, rows = analyse_cell(frame, kind, replicates, rng, block_windows)
                    key = dict(node=node, region=region, scenario=scenario, loss=kind)
                    baselines.append({**key, **info})
                    evidence += [{**key, "best_simple": info["best_simple"], **row} for row in rows]
    baselines = pd.DataFrame(baselines)
    evidence = pd.DataFrame(evidence)
    keys = ["node", "region", "scenario", "loss"]
    spread = (
        evidence.groupby(keys)
        .agg(
            se_optimistic=("se", lambda s: float(np.quantile(s, 0.1))),
            se_typical=("se", "median"),
            se_aggressive=("se", lambda s: float(np.quantile(s, 0.9))),
            best_gain=("gain", "max"),
            best_z=("z", "max"),
            families=("candidate", "size"),
        )
        .reset_index()
    )
    for level in ("optimistic", "typical", "aggressive"):
        spread[f"mde_{level}"] = Z * spread[f"se_{level}"]
    cv = baselines.set_index(keys)["baseline_cv"]
    spread = spread.join(cv, on=keys)
    for level in ("optimistic", "typical", "aggressive"):
        ratio = spread["node"].map(lambda node, level=level: noise_ratios[node][level])
        spread[f"scaled_mde_{level}"] = Z * ratio * spread["baseline_cv"]
    return baselines, evidence, spread


def margin(mde_typical):
    """delta = max(20%, typical minimum detectable gain rounded up to the next 5%)."""
    return max(MIN_DELTA, math.ceil(round(mde_typical / STEP, 9)) * STEP)


PRIMARY_CELLS = (("CC1", "all_folds"), ("LW1", "later_folds"))


def decide(spread, primary_scenario, loss_kind="squared", proxy="scaled"):
    """Apply the declared margin and demotion rules to the primary cells.

    ``proxy="unmatched"`` is the original rule (noise of the unmatched exploratory families);
    ``proxy="scaled"`` is the amended rule (baseline noise times the clean-input candidate-to-
    baseline noise ratio, which uses no sparse-regime learned results).
    """
    column = "mde_typical" if proxy == "unmatched" else "scaled_mde_typical"
    cells = spread[(spread["scenario"] == primary_scenario) & (spread["loss"] == loss_kind)]
    out = []
    for row in cells.itertuples():
        value = float(getattr(row, column))
        out.append(
            dict(
                node=row.node,
                region=row.region,
                primary=(row.node, row.region) in PRIMARY_CELLS,
                mde_typical=value,
                delta=margin(value),
                demoted=bool(value > DEMOTE_ABOVE),
            )
        )
    abort = all(c["demoted"] for c in out if c["primary"])
    return out, abort
