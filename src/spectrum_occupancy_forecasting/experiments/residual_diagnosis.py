"""Saved-prediction diagnostics; retrospective choices are never operational forecasts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adaptive_models import candidates
from .adaptive_reporting import verify_records
from .fine_spectrum import sha, write_json


def source_hashes(source):
    source = Path(source)
    names = [
        "predictions.parquet",
        "validation_predictions.parquet",
        "feedback.parquet",
        "feature_snapshots.parquet",
        "selections.json",
        "candidate_scores.json",
        "configuration.json",
        "failures.json",
        "manifest.json",
        *[
            f"{name}_candidate_forecasts.parquet"
            for name in ("CC1_first", "LW1_first", "LW1_later")
        ],
    ]
    return {name: sha(source / name) for name in names}


def skill(mse, reference):
    return 1 - mse / reference if reference > 0 else None


def residual_pairs(labels, anchors, feedback):
    """Pair within scored contiguous segments, using released immutable anchors only."""
    labels = labels.sort_values(["fold", "origin_s"]).copy()
    usable = (
        feedback[feedback.update_reason == "updated"]
        .merge(anchors[["origin_s", "fixed_history"]], on="origin_s", validate="one_to_one")
        .sort_values("release_s")
    )
    usable["known_residual"] = usable.feedback_y - usable.fixed_history
    rows = []
    for fold, group in labels.groupby("fold"):
        segments = (group.origin_s.diff().fillna(900) != 900).cumsum()
        for segment, part in group.groupby(segments):
            # Feedback targets must themselves originate in this eligible segment.
            available = usable[usable.origin_s.isin(part.origin_s)]
            for row in part.itertuples():
                earlier = available[available.release_s <= row.origin_s]
                if earlier.empty:
                    continue
                last = earlier.iloc[-1]
                rows.append(
                    {
                        "fold": int(fold),
                        "segment": int(segment),
                        "origin_s": row.origin_s,
                        "feedback_origin_s": float(last.origin_s),
                        "release_s": float(last.release_s),
                        "known_residual": float(last.known_residual),
                        "future_residual": float(row.y - row.fixed_history),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "fold",
            "segment",
            "origin_s",
            "feedback_origin_s",
            "release_s",
            "known_residual",
            "future_residual",
        ],
    )


def diagnose(source, out, cfg, specification_hash):
    source, out = Path(source), Path(out)
    old_cfg = json.loads((source / "configuration.json").read_text())
    checks = verify_records(source, old_cfg)
    pred = pd.read_parquet(source / "predictions.parquet")
    feedback = pd.read_parquet(source / "feedback.parquet")
    grid = {c.id: c for c in candidates(old_cfg)}
    headroom, selection_losses, controllers, corrections, persistence, pairs_all = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for recording in ("CC1_first", "LW1_first", "LW1_later"):
        clean = pred[(pred.recording == recording) & (pred.scenario == "clean")]
        labels = clean[(clean.strategy == "bias") & (clean.method == "adaptive")]
        labels = labels[["origin_s", "fold", "y", "fixed_history", "selected_history"]]
        if labels.origin_s.duplicated().any():
            raise ValueError("duplicate scored origins")
        all_candidates = pd.read_parquet(source / f"{recording}_candidate_forecasts.parquet")
        anchors = all_candidates[all_candidates.candidate == next(iter(grid))]
        frame = all_candidates.merge(
            labels[["origin_s", "fold", "y"]], on="origin_s", validate="many_to_one"
        )
        for identifier, group in frame[frame.candidate.isin(grid)].groupby("candidate"):
            if set(group.origin_s) != set(labels.origin_s):
                raise ValueError("candidate scored population mismatch")
            for fold, part in [("pooled", group), *list(group.groupby("fold"))]:
                mse = float(np.mean((part.prediction - part.y) ** 2))
                fixed = float(np.mean((part.fixed_history - part.y) ** 2))
                selected = float(np.mean((part.selected_history - part.y) ** 2))
                headroom.append(
                    {
                        "recording": recording,
                        "candidate": identifier,
                        "family": grid[identifier].family,
                        "fold": str(fold),
                        "n": len(part),
                        "mse": mse,
                        "skill_fixed": skill(mse, fixed),
                        "skill_selected": skill(mse, selected),
                    }
                )
        for strategy, cohort in clean.groupby("strategy"):
            active = cohort[cohort.method == "adaptive"].copy()
            families = {
                "bias": "bias",
                "aggregation": "aggregation",
                "ridge_history": "ridge",
                "ridge_fine": "ridge",
            }
            allowed = [
                c.id
                for c in grid.values()
                if c.family == "history"
                or (
                    c.family == families[strategy]
                    and (
                        not strategy.startswith("ridge")
                        or c.features == ("history_fine" if strategy == "ridge_fine" else "history")
                    )
                )
            ]
            for fold, part in active.groupby("fold"):
                pool = frame[(frame.fold == fold) & frame.candidate.isin(allowed)]
                losses = (
                    pool.assign(loss=(pool.prediction - pool.y) ** 2)
                    .groupby("candidate")
                    .loss.mean()
                )
                best = min(losses.index, key=lambda c: (losses[c], grid[c].tie))
                selected_mse = float(np.mean((part.prediction - part.y) ** 2))
                fixed_mse = float(np.mean((part.fixed_history - part.y) ** 2))
                selected_ref_mse = float(np.mean((part.selected_history - part.y) ** 2))
                rejected = sum(
                    losses[c] < min(fixed_mse, selected_ref_mse)
                    and c != part.candidate.iloc[0]
                    and grid[c].family != "history"
                    for c in losses.index
                )
                selection_losses.append(
                    {
                        "recording": recording,
                        "strategy": strategy,
                        "fold": int(fold),
                        "selected": part.candidate.iloc[0],
                        "retrospective_best": best,
                        "selected_mse": selected_mse,
                        "best_mse": float(losses[best]),
                        "selection_excess_mse": selected_mse - float(losses[best]),
                        "selected_harmful": selected_mse > min(fixed_mse, selected_ref_mse),
                        "controller_selected": grid[part.candidate.iloc[0]].family == "history",
                        "rejected_useful_candidates": int(rejected),
                    }
                )
        for (strategy, scenario, seed), cohort in pred[pred.recording == recording].groupby(
            ["strategy", "scenario", "seed"]
        ):
            active = cohort[cohort.method == "adaptive"].copy()
            active["reason"] = active.fallback_reason.fillna("").replace("", "correction_active")
            for reason, part in active.groupby("reason"):
                loss = (part.prediction - part.y) ** 2
                counter = np.where(
                    reason != "correction_active", part.fixed_history, part.prediction
                )
                controllers.append(
                    {
                        "recording": recording,
                        "strategy": strategy,
                        "scenario": scenario,
                        "seed": int(seed),
                        "reason": reason,
                        "n": len(part),
                        "issued_mse": float(loss.mean()),
                        "fixed_fallback_mse": float(np.mean((counter - part.y) ** 2)),
                        "paired_difference": float(np.mean((counter - part.y) ** 2 - loss)),
                    }
                )
            part = active[active.reason == "correction_active"].copy()
            part = part[np.isfinite(part.raw_correction)]
            if len(part):
                strengths = part.candidate.map(lambda c: grid[c].strength).to_numpy()
                raw_forecast = np.clip(part.fixed_history + strengths * part.raw_correction, 0, 1)
                corrections.append(
                    {
                        "recording": recording,
                        "strategy": strategy,
                        "scenario": scenario,
                        "seed": int(seed),
                        "n": len(part),
                        "mean_error": float((part.prediction - part.y).mean()),
                        "mean_correction": float(part.correction.mean()),
                        "mean_absolute_correction": float(part.correction.abs().mean()),
                        "saturation_share": float((part.raw_correction.abs() >= 0.1).mean()),
                        "issued_mse": float(np.mean((part.prediction - part.y) ** 2)),
                        "unclipped_correction_mse": float(np.mean((raw_forecast - part.y) ** 2)),
                    }
                )
        fb = feedback[(feedback.recording == recording) & (feedback.scenario == "clean")]
        pairs = residual_pairs(labels, anchors, fb)
        pairs["recording"] = recording
        pairs_all.append(pairs)
        valid = len(pairs) >= cfg["gate_minimum_pairs"] and all(
            pairs[c].var(ddof=0) > 0 for c in ("known_residual", "future_residual")
        )
        correlation = float(pairs.known_residual.corr(pairs.future_residual)) if valid else None
        persistence.append(
            {
                "recording": recording,
                "n_pairs": len(pairs),
                "correlation": correlation,
                "available": bool(valid),
                "passes": bool(valid and correlation >= cfg["gate_correlation"]),
            }
        )
    table = pd.DataFrame(headroom)
    eligible = []
    measurements = []
    for identifier, group in table[table.family != "history"].groupby("candidate"):
        checks_node = []
        for recording in ("CC1_first", "LW1_first"):
            part = group[group.recording == recording]
            pooled = part[part.fold == "pooled"].iloc[0]
            daily = part[part.fold != "pooled"]
            fraction = float(((daily.skill_fixed > 0) & (daily.skill_selected > 0)).mean())
            passes = bool(
                pooled.skill_fixed >= cfg["gate_headroom_skill"]
                and pooled.skill_selected > 0
                and fraction >= 0.5
            )
            checks_node.append(passes)
            measurements.append(
                {
                    "candidate": identifier,
                    "recording": recording,
                    "skill_fixed": pooled.skill_fixed,
                    "skill_selected": pooled.skill_selected,
                    "positive_fold_fraction": fraction,
                    "passes": passes,
                }
            )
        if all(checks_node):
            eligible.append(identifier)
    corr_pass = all(p["passes"] for p in persistence if p["recording"] != "LW1_later")
    gate = {
        "passes": bool(eligible or corr_pass),
        "policy": "either",
        "headroom_pass": bool(eligible),
        "qualifying_existing_candidates": eligible,
        "persistence_pass": corr_pass,
        "persistence": persistence,
        "specification_hash": specification_hash,
        "source_hashes": source_hashes(source),
        "all_results_exploratory": True,
        "CC2_accessed": False,
    }
    tables = {
        "candidate_headroom": table,
        "selection_losses": pd.DataFrame(selection_losses),
        "controller_contribution": pd.DataFrame(controllers),
        "correction_behavior": pd.DataFrame(corrections),
        "residual_pairs": pd.concat(pairs_all, ignore_index=True),
        "gate_candidate_checks": pd.DataFrame(measurements),
    }
    winners = []
    for keys, group in table.groupby(["recording", "fold"]):
        best = group.sort_values(["mse", "candidate"]).iloc[0]
        winners.append(
            {
                "recording": keys[0],
                "fold": keys[1],
                "retrospective_best": best.candidate,
                "mse": best.mse,
            }
        )
    # A global retrospective choice minimizes the mean of recording-normalized losses.
    pooled = table[table.fold == "pooled"]
    scores = (
        pooled.assign(normalized_loss=1 - pooled.skill_fixed)
        .groupby("candidate")
        .normalized_loss.mean()
    )
    gate["retrospective_global_candidate"] = min(
        scores.index, key=lambda c: (scores[c], grid[c].tie)
    )
    tables["retrospective_winners"] = pd.DataFrame(winners)
    for name, data in tables.items():
        data.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "gate.json", gate)
    write_json(out / "source_verification.json", checks)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, data in zip(axes, pairs_all, strict=True):
        ax.scatter(data.known_residual, data.future_residual, s=9, alpha=0.4)
        name = data.recording.iloc[0] if len(data) else "Unavailable"
        metric = next((p for p in persistence if p["recording"] == name), {})
        value = metric.get("correlation")
        ax.set(
            title=f"{name}: n={len(data)}, r={value:.2f}" if value is not None else name,
            xlabel="Previously delivered residual",
            ylabel="Subsequent scoring residual",
        )
    fig.suptitle("Exploratory +3 dB labels; residuals in busy-fraction units")
    fig.savefig(out / "residual_persistence.png", dpi=150)
    plt.close(fig)
    return gate
