"""Rebuild v2 metrics and temporal uncertainty from archived per-origin predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.evaluation.metrics import event_tradeoff

KEYS = ["node", "task", "variant", "lead_s", "offset_db", "scenario", "method"]
PAIR = ["fold", "origin_s", "seed"]


def paired_losses(frame, reference):
    columns = PAIR + ["y", "prediction"]
    joined = frame[columns].merge(
        reference[columns], on=PAIR, suffixes=("", "_ref"), validate="one_to_one", how="inner"
    )
    if len(joined) != len(frame) or not np.array_equal(joined["y"], joined["y_ref"]):
        raise ValueError("paired comparisons require identical origins, seeds and labels")
    joined["loss"] = (joined["prediction"] - joined["y"]) ** 2
    joined["ref_loss"] = (joined["prediction_ref"] - joined["y"]) ** 2
    # Average LOSSES over masks, not predictions; masks are not independent time samples.
    return joined.groupby(["fold", "origin_s"], as_index=False)[["loss", "ref_loss"]].mean()


def temporal_blocks(frame, seconds, cadence):
    """Block groups never cross a fold or a recording gap."""
    out = []
    for fold, group in frame.groupby("fold", sort=True):
        group = group.sort_values("origin_s").copy()
        segment = (group["origin_s"].diff().fillna(0) > 1.5 * cadence).cumsum()
        for seg, part in group.groupby(segment):
            ids = np.floor((part["origin_s"] - part["origin_s"].min()) / seconds).astype(int)
            for block, rows in part.groupby(ids):
                out.append(
                    {
                        "fold": fold,
                        "segment": seg,
                        "block": block,
                        "n": len(rows),
                        "loss": rows["loss"].sum(),
                        "ref_loss": rows["ref_loss"].sum(),
                    }
                )
    return pd.DataFrame(out)


def paired_intervals(losses, block_s, cfg):
    blocks = temporal_blocks(losses, block_s, cfg["window_s"])
    count = len(blocks)
    result = {
        "block_s": block_s,
        "n_blocks": count,
        "weak_evidence": count < cfg["weak_evidence_blocks"],
    }
    if count < cfg["minimum_interval_blocks"]:
        return result | {
            "status": "insufficient_blocks",
            "mss_lo": None,
            "mss_hi": None,
            "difference_lo": None,
            "difference_hi": None,
        }
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights = rng.multinomial(count, np.full(count, 1.0 / count), size=cfg["bootstrap_resamples"])
    n = weights @ blocks["n"].to_numpy()
    error = weights @ blocks["loss"].to_numpy()
    reference = weights @ blocks["ref_loss"].to_numpy()
    difference = (error - reference) / n
    skill = np.divide(error, reference, out=np.full(len(error), np.nan), where=reference > 0)
    skill = 1 - skill

    def interval(values):
        valid = values[np.isfinite(values)]
        return np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]

    lo, hi = interval(skill)
    dlo, dhi = interval(difference)
    return result | {
        "status": "weak_evidence" if result["weak_evidence"] else "estimated",
        "mss_lo": lo,
        "mss_hi": hi,
        "difference_lo": dlo,
        "difference_hi": dhi,
    }


def dependence(frame, value, cadence):
    """Demeaned autocorrelation within contiguous segments; never bridge gaps or folds."""
    result = []
    for lag in (1, 2, 4, 8):
        estimates, counts = [], []
        for _, group in frame.groupby("fold"):
            group = group.sort_values("origin_s")
            segments = (group["origin_s"].diff().fillna(0) > 1.5 * cadence).cumsum()
            for _, part in group.groupby(segments):
                z = part[value].to_numpy()
                if len(z) <= lag or np.var(z) == 0:
                    continue
                z = z - z.mean()
                estimates.append(float(np.mean(z[:-lag] * z[lag:]) / np.mean(z**2)))
                counts.append(len(z) - lag)
        result.append(
            {
                "lag_s": lag * cadence,
                "n_pairs": sum(counts),
                "autocorrelation": float(np.average(estimates, weights=counts)) if counts else None,
            }
        )
    return result


def point_metrics(frame, ref, fixed, clean):
    mse = float(np.mean((frame["prediction"] - frame["y"]) ** 2))
    mae = float(np.mean(np.abs(frame["prediction"] - frame["y"])))
    mean_y = frame.groupby("fold")["y"].transform("mean")
    variance = float(np.mean((frame["y"] - mean_y) ** 2))
    losses = paired_losses(frame, ref)
    fixed_losses = paired_losses(frame, fixed)
    clean_reference = clean[PAIR[:-1] + ["y", "prediction"]].copy()
    clean_reference = frame[PAIR].merge(clean_reference, on=PAIR[:-1], validate="many_to_one")
    clean_losses = paired_losses(frame, clean_reference)

    def skill(reference):
        denominator = float(reference["ref_loss"].mean())
        return 1 - mse / denominator if denominator > 0 else None

    unique = frame.drop_duplicates(["fold", "origin_s"])
    per_seed = (
        frame.assign(loss=(frame["prediction"] - frame["y"]) ** 2).groupby("seed")["loss"].mean()
    )
    return {
        "n_windows": len(unique),
        "n_seeds": len(per_seed),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": mae,
        "mse_mask_sd": float(per_seed.std(ddof=0)),
        "r2_within_fold": 1 - mse / variance if variance > 0 else None,
        "mean_y": float(unique["y"].mean()),
        "busy_sweep_share": float(np.average(unique["y"], weights=unique["target_count"])),
        "n_target_observations": int(unique["target_count"].sum()),
        "n_events": int((unique["y"] >= 0.2).sum()),
        "mss_vs_selected": skill(losses),
        "mss_vs_fixed": skill(fixed_losses),
        "mss_vs_clean_selected": skill(clean_losses),
        "paired_mse_difference": float((losses["loss"] - losses["ref_loss"]).mean()),
    }, losses


def event_results(predictions, selections, cfg):
    rows = []
    for record in selections:
        if record["task"] != "primary":
            continue
        fold = predictions[
            (predictions["node"] == record["node"])
            & (predictions["task"] == record["task"])
            & (predictions["fold"] == record["fold"])
        ]
        for threshold in record["operating_thresholds"]:
            method = fold[fold["method"] == threshold["method"]]
            for (scenario, seed), part in method.groupby(["scenario", "seed"]):
                event = part["y"].to_numpy() >= cfg["event_cutoff"]
                row = {
                    "node": record["node"],
                    "fold": record["fold"],
                    "method": threshold["method"],
                    "scenario": scenario,
                    "seed": seed,
                    **threshold,
                    "n_evaluation_events": int(event.sum()),
                    "n_evaluation_nonevents": int((~event).sum()),
                }
                if threshold["threshold"] is not None:
                    busy = part["prediction"].to_numpy() >= threshold["threshold"]
                    row["realized_false_busy"] = (
                        float(busy[~event].mean()) if (~event).any() else None
                    )
                    row["realized_false_idle"] = (
                        float((~busy[event]).mean()) if event.any() else None
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(predictions, selections, cfg, out):
    """No model fitting: metrics, uncertainty, events and diagnostics derive from saved rows."""
    if predictions.empty:
        raise ValueError("no prediction rows to summarize")
    summaries, daily, bootstrap, acf, oracle = [], [], [], [], []
    by_task = {key: group for key, group in predictions.groupby(["node", "task"])}
    for key, frame in predictions.groupby(KEYS, sort=True):
        meta = dict(zip(KEYS, key, strict=True))
        all_rows = by_task[(meta["node"], meta["task"])]
        scenario = all_rows[all_rows["scenario"] == meta["scenario"]]
        ref = scenario[scenario["method"] == "selected_history"]
        fixed = scenario[scenario["method"] == "fixed_history"]
        clean = all_rows[
            (all_rows["scenario"] == "clean") & (all_rows["method"] == "selected_history")
        ]
        point, loss = point_metrics(frame, ref, fixed, clean)
        summaries.append(meta | point)
        for f, part in frame.groupby("fold"):
            metric, _ = point_metrics(
                part, ref[ref["fold"] == f], fixed[fixed["fold"] == f], clean[clean["fold"] == f]
            )
            daily.append(meta | {"fold": f} | metric)
        for block in cfg["bootstrap_blocks_s"]:
            bootstrap.append(meta | paired_intervals(loss, block, cfg))
        err = (
            frame.assign(error=frame["prediction"] - frame["y"])
            .groupby(["fold", "origin_s"], as_index=False)["error"]
            .mean()
        )
        loss["difference"] = loss["loss"] - loss["ref_loss"]
        for quantity, data in (("error", err), ("difference", loss)):
            acf.extend(
                meta | {"quantity": quantity} | r
                for r in dependence(data, quantity, cfg["window_s"])
            )
        if meta["task"] == "primary":
            for seed, part in frame.groupby("seed"):
                oracle.append(
                    meta
                    | {"seed": seed, "interpretation": "evaluation-label oracle"}
                    | event_tradeoff(part["y"], part["prediction"], cfg["event_cutoff"])
                )
    out = Path(out)
    outputs = {
        "summary": pd.DataFrame(summaries),
        "daily": pd.DataFrame(daily),
        "bootstrap": pd.DataFrame(bootstrap),
        "dependence": pd.DataFrame(acf),
        "events": event_results(predictions, selections, cfg),
        "operating_thresholds": pd.DataFrame(
            [
                {
                    "node": record["node"],
                    "fold": record["fold"],
                    "task": record["task"],
                    **threshold,
                }
                for record in selections
                for threshold in record.get("operating_thresholds", [])
            ]
        ),
        "oracle": pd.DataFrame(oracle),
    }
    for name, table in outputs.items():
        table.to_csv(out / f"{name}.csv", index=False)
    # pandas converts non-finite values to JSON null, rather than emitting non-standard NaN.
    report = {
        "interpretation": "exploratory reuse; fresh confirmation pending",
        "metrics": json.loads(outputs["summary"].to_json(orient="records", double_precision=15)),
        "main_bootstrap_block_s": cfg["main_bootstrap_block_s"],
        "mask_uncertainty": "losses averaged over masks before temporal resampling",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return outputs
