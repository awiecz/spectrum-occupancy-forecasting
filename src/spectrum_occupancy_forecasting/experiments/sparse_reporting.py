"""Summaries and the mechanical outcome label for the sparse-regime study.

Every number derives from saved prediction files. The outcome label is computed by
``decide`` from the declared rule in the configuration; nobody chooses it afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.evaluation.power import block_indices
from spectrum_occupancy_forecasting.evaluation.quantile_metrics import (
    coverage,
    interval_score,
    mean_pinball,
)

LEARNED = ("selected_learned", "ridge", "hgb", "resid_hgb")
COMPARATORS = (
    "hstar",
    "fixed_history",
    "selected_history",
    "shrunk_history",
    "climatology",
)
SIMPLE = ("fixed_history", "selected_history", "shrunk_history", "climatology")
WINDOWS_PER_DAY = 96


def regions(node, overlap_fold):
    """Region name -> predicate on the fold number. LW1's first fold overlaps old selection."""
    if node == "LW1":
        return {
            "all_folds": lambda fold: fold >= 0,
            "later_folds": lambda fold: fold > overlap_fold,
            "first_fold": lambda fold: fold == overlap_fold,
        }
    return {"all_folds": lambda fold: fold >= 0}


def window_losses(frame, kind):
    """Per-origin loss for each method, averaged over mask seeds (losses, not predictions)."""
    error = frame["y"] - frame["prediction"]
    loss = error**2 if kind == "squared" else error.abs()
    table = frame.assign(loss=loss).pivot_table(
        index="origin_s", columns="method", values="loss", aggfunc="mean"
    )
    return table.sort_index()


def bootstrap_gain(difference, baseline, length, replicates, rng):
    """Gain = sum(difference)/sum(baseline) with a one-sided 95% lower bound by block bootstrap."""
    n = len(baseline)
    gain = float(np.sum(difference) / np.sum(baseline))
    idx = block_indices(n, min(length, n), replicates, rng)
    ratio = difference[idx].mean(axis=1) / baseline[idx].mean(axis=1)
    return gain, float(ratio.std(ddof=1)), float(np.quantile(ratio, 0.05))


def block_length(n, cfg):
    primary = cfg["bootstrap"]["block_windows"]
    return (
        primary
        if n / primary >= cfg["bootstrap"]["minimum_blocks"]
        else cfg["bootstrap"]["later_block_windows"]
    )


def cell_table(predictions, cfg, replicates, rng):
    """Gain of every learned strategy over every comparator in every node/region/scenario."""
    rows = []
    overlap = cfg["lw1_overlap_fold"]
    for (node, scenario), frame in predictions.groupby(["node", "scenario"]):
        for region, keep in regions(node, overlap).items():
            part = frame[frame["fold"].map(keep)]
            if part.empty:
                continue
            for kind in ("squared", "absolute"):
                losses = window_losses(part, kind)
                length = block_length(len(losses), cfg)
                for learned in LEARNED:
                    for comparator in COMPARATORS:
                        if learned not in losses or comparator not in losses:
                            continue
                        base = losses[comparator].to_numpy()
                        difference = base - losses[learned].to_numpy()
                        gain, se, lower = bootstrap_gain(difference, base, length, replicates, rng)
                        rows.append(
                            dict(
                                node=node,
                                region=region,
                                scenario=scenario,
                                loss=kind,
                                learned=learned,
                                comparator=comparator,
                                n_windows=len(losses),
                                n_blocks=len(losses) / length,
                                block_windows=length,
                                comparator_loss=float(base.mean()),
                                learned_loss=float(losses[learned].mean()),
                                gain=gain,
                                se=se,
                                lower95=lower,
                            )
                        )
    return pd.DataFrame(rows)


def method_losses(predictions, cfg):
    """Mean squared error of every strategy in every node/region/scenario."""
    rows = []
    overlap = cfg["lw1_overlap_fold"]
    for (node, scenario), frame in predictions.groupby(["node", "scenario"]):
        for region, keep in regions(node, overlap).items():
            part = frame[frame["fold"].map(keep)]
            if part.empty:
                continue
            losses = window_losses(part, "squared")
            absolute = window_losses(part, "absolute")
            for method in losses:
                rows.append(
                    dict(
                        node=node,
                        region=region,
                        scenario=scenario,
                        method=method,
                        n_windows=len(losses),
                        mse=float(losses[method].mean()),
                        mae=float(absolute[method].mean()),
                    )
                )
    return pd.DataFrame(rows)


def robustness(predictions, cfg, node, region, scenario, learned="selected_learned"):
    """Tail, day and block descriptors for the learned strategy against the chosen baseline."""
    keep = regions(node, cfg["lw1_overlap_fold"])[region]
    frame = predictions[
        (predictions["node"] == node)
        & (predictions["scenario"] == scenario)
        & predictions["fold"].map(keep)
    ]
    squared = window_losses(frame, "squared")
    absolute = window_losses(frame, "absolute")
    base, cand = squared["hstar"].to_numpy(), squared[learned].to_numpy()
    difference = base - cand
    keep_mask = base <= np.quantile(base, 1 - cfg["decision"]["trim_fraction"])
    trimmed = float(difference[keep_mask].sum() / base[keep_mask].sum())
    days = (squared.index.to_numpy() - squared.index.min()) // 86400
    leave_out = [
        float(difference[days != d].sum() / base[days != d].sum()) for d in np.unique(days)
    ]
    length = block_length(len(base), cfg)
    blocks = np.arange(len(base)) // length
    totals = np.array([difference[blocks == b].sum() for b in np.unique(blocks)])
    improvement = difference.sum()
    share = float(totals.max() / improvement) if improvement > 0 else float("nan")
    mae_gain = float(1 - absolute[learned].sum() / absolute["hstar"].sum())
    return dict(
        node=node,
        region=region,
        scenario=scenario,
        squared_gain=float(difference.sum() / base.sum()),
        absolute_gain=mae_gain,
        trimmed_gain=trimmed,
        leave_one_day_out_all_positive=bool(min(leave_out) > 0),
        leave_one_day_out_min=float(min(leave_out)),
        best_block_share=share,
    )


def decide(cells, descriptors, delta, cfg):
    """Apply the declared rule to the primary scenario and return the label with every test."""
    primary = cfg["primary_scenario"]
    rule = cfg["decision"]

    def pick(node, region):
        row = cells[
            (cells["node"] == node)
            & (cells["region"] == region)
            & (cells["scenario"] == primary)
            & (cells["loss"] == "squared")
            & (cells["learned"] == "selected_learned")
            & (cells["comparator"] == "hstar")
        ]
        return None if row.empty else row.iloc[0]

    def described(node, region):
        match = [d for d in descriptors if d["node"] == node and d["region"] == region]
        return match[0] if match else None

    cc1, lw_later, lw_all = (
        pick("CC1", "all_folds"),
        pick("LW1", "later_folds"),
        pick("LW1", "all_folds"),
    )
    d_cc1, d_lw = described("CC1", "all_folds"), described("LW1", "later_folds")
    if any(v is None for v in (cc1, lw_later, lw_all, d_cc1, d_lw)):
        return dict(label="not_evaluable", reason="a primary cell is missing", tests={})
    tests = {
        "cc1_lower_bound_at_least_delta": bool(cc1["lower95"] >= delta),
        "lw1_later_lower_bound_at_least_delta": bool(lw_later["lower95"] >= delta),
        "lw1_all_folds_same_sign_and_half_delta": bool(
            lw_all["gain"] >= rule["secondary_point_fraction"] * delta
        ),
        "absolute_error_gain_positive_both": bool(
            d_cc1["absolute_gain"] > 0 and d_lw["absolute_gain"] > 0
        ),
        "best_block_share_within_limit_both": bool(
            d_cc1["best_block_share"] <= rule["best_block_share_max"]
            and d_lw["best_block_share"] <= rule["best_block_share_max"]
        ),
    }
    if all(tests.values()):
        label = "large_effect_supported"
    elif (
        cc1["lower95"] <= 0
        or lw_later["lower95"] <= 0
        or np.sign(cc1["gain"]) != np.sign(lw_later["gain"])
    ):
        label = "honest_negative"
    else:
        label = "directional_only"
    return dict(
        label=label,
        delta=delta,
        tests=tests,
        cc1=dict(
            gain=float(cc1["gain"]), lower95=float(cc1["lower95"]), n_blocks=float(cc1["n_blocks"])
        ),
        lw1_later=dict(
            gain=float(lw_later["gain"]),
            lower95=float(lw_later["lower95"]),
            n_blocks=float(lw_later["n_blocks"]),
        ),
        lw1_all=dict(gain=float(lw_all["gain"]), lower95=float(lw_all["lower95"])),
    )


def interval_table(intervals, cfg):
    """Pinball, interval score, coverage and width for each interval method."""
    levels = list(cfg["quantiles"])
    columns = [f"q{tau:g}" for tau in levels]
    low, high = f"q{levels[1]:g}", f"q{levels[-2]:g}"
    alpha = cfg["interval_alpha"]
    if not np.isclose(levels[1], alpha / 2) or not np.isclose(levels[-2], 1 - alpha / 2):
        raise ValueError("quantile levels must contain the declared central interval")
    rows = []
    overlap = cfg["lw1_overlap_fold"]
    for (node, scenario), frame in intervals.groupby(["node", "scenario"]):
        for region, keep in regions(node, overlap).items():
            part = frame[frame["fold"].map(keep)]
            if part.empty:
                continue
            for method, group in part.groupby("method"):
                y = group["y"].to_numpy()
                q = group[columns].to_numpy()
                rows.append(
                    dict(
                        node=node,
                        region=region,
                        scenario=scenario,
                        method=method,
                        n_forecasts=len(group),
                        mean_pinball=mean_pinball(y, q, levels),
                        interval_score_80=interval_score(y, group[low], group[high], alpha),
                        coverage_80=coverage(y, group[low], group[high]),
                        mean_width_80=float((group[high] - group[low]).mean()),
                        nominal_coverage=1 - alpha,
                    )
                )
    return pd.DataFrame(rows)
