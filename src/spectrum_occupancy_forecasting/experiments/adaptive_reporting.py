"""Prediction-only metrics, advancement decisions and figures for adaptive replay."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adaptive_models import candidates
from .adaptive_replay import STRATEGIES
from .fine_spectrum import write_json
from .followup_models import operating_threshold
from .followup_reporting import dependence, paired_intervals, point_metrics


def verify_records(source, cfg):
    """Recompute scores and conservative selections without fitting or replaying models."""
    validation = pd.read_parquet(source / "validation_predictions.parquet")
    scores = json.loads((source / "candidate_scores.json").read_text())
    selections = json.loads((source / "selections.json").read_text())
    grid = {candidate.id: candidate for candidate in candidates(cfg)}
    checked = 0
    for selection in selections:
        if selection["recording"] == "LW1_later" or selection["failure"]:
            continue
        frame = validation[
            (validation.recording == selection["recording"])
            & (validation.fold == selection["fold"])
        ].copy()
        frame["loss"] = (frame.prediction - frame.y) ** 2
        pooled = frame.groupby("candidate").loss.mean()
        blocked = frame.groupby(["candidate", "block"]).loss.mean().unstack()
        saved = [
            row
            for row in scores
            if row["recording"] == selection["recording"] and row["fold"] == selection["fold"]
        ]
        if len(saved) != 145:
            raise ValueError("missing declared validation candidates")
        for row in saved:
            identifier = row["candidate"]
            if not np.isclose(pooled[identifier], row["pooled_mse"], rtol=1e-12, atol=1e-15):
                raise ValueError("candidate MSE cannot be reproduced")
            for block in (0, 1):
                if not np.isclose(
                    blocked.loc[identifier, block],
                    row[f"block_{block}_mse"],
                    rtol=1e-12,
                    atol=1e-15,
                ):
                    raise ValueError("validation block score cannot be reproduced")
            eligible = grid[identifier].family != "history"
            for reference in ("recent_7200s", selection["selected_history"]):
                eligible &= pooled[identifier] <= (1 - cfg["validation_margin"]) * pooled[reference]
                eligible &= bool((blocked.loc[identifier] < blocked.loc[reference]).all())
            if bool(eligible) != row["selection_eligible"]:
                raise ValueError("conservative selection eligibility changed")
            checked += 1
        for strategy, predicate in STRATEGIES.items():
            eligible = [
                grid[row["candidate"]]
                for row in saved
                if row["selection_eligible"] and predicate(grid[row["candidate"]])
            ]
            expected = (
                min(eligible, key=lambda c: (pooled[c.id], c.tie)).id
                if eligible
                else candidates(cfg)[0].id
            )
            if expected != selection["chosen"][strategy]:
                raise ValueError("saved candidate selection is not reproducible")
    return {"candidate_scores_reproduced": checked, "selection_rules_verified": True}


def regenerate(source, out, cfg):
    source, out = Path(source), Path(out)
    predictions = pd.read_parquet(source / "predictions.parquet")
    checks = verify_records(source, cfg)
    if not predictions.prediction.between(0, 1).all():
        raise ValueError("forecast bounds violated")
    write_json(out / "prediction_audit.json", checks)
    rows, daily, intervals, correlations = [], [], [], []
    for (recording, strategy, scenario), cohort in predictions.groupby(
        ["recording", "strategy", "scenario"]
    ):
        ref = cohort[cohort.method == "selected_history"]
        fixed = cohort[cohort.method == "fixed_history"]
        clean = predictions[
            (predictions.recording == recording)
            & (predictions.strategy == strategy)
            & (predictions.scenario == "clean")
            & (predictions.method == "selected_history")
        ]
        for method, frame in cohort.groupby("method"):
            meta = {
                "recording": recording,
                "strategy": strategy,
                "scenario": scenario,
                "method": method,
            }
            metrics, losses = point_metrics(frame, ref, fixed, clean)
            rows.append(
                meta
                | metrics
                | {
                    "fallback_share": float((frame.fallback_reason.fillna("") != "").mean()),
                    "mean_correction": float(frame.correction.fillna(0).mean()),
                    "mean_absolute_correction": float(frame.correction.fillna(0).abs().mean()),
                }
            )
            for fold, part in frame.groupby("fold"):
                metric, _ = point_metrics(
                    part,
                    ref[ref.fold == fold],
                    fixed[fixed.fold == fold],
                    clean[clean.fold == fold],
                )
                daily.append(meta | {"fold": int(fold)} | metric)
            for reference, table in (("selected", ref), ("fixed", fixed)):
                _, paired = point_metrics(frame, table, fixed, clean)
                for length in cfg["bootstrap_blocks_s"]:
                    intervals.append(
                        meta | {"reference": reference} | paired_intervals(paired, length, cfg)
                    )
                paired["difference"] = paired.loss - paired.ref_loss
                correlations.extend(
                    meta | {"reference": reference} | row
                    for row in dependence(paired, "difference", 900)
                )
            errors = (
                frame.assign(error=frame.prediction - frame.y)
                .groupby(["fold", "origin_s"], as_index=False)
                .error.mean()
            )
            correlations.extend(
                meta | {"reference": "forecast_error"} | row
                for row in dependence(errors, "error", 900)
            )
    summary, daily_table = pd.DataFrame(rows), pd.DataFrame(daily)
    summary.to_csv(out / "summary.csv", index=False)
    daily_table.to_csv(out / "daily.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "uncertainty.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "dependence.csv", index=False)
    write_json(out / "summary.json", rows)
    failures = json.loads((source / "failures.json").read_text())
    decisions = []
    for strategy in sorted(predictions.strategy.unique()):
        table = summary[(summary.strategy == strategy) & (summary.method == "adaptive")]
        clean = table[table.scenario == "clean"]
        node_checks = []
        for recording in sorted(table.recording.unique()):
            result = clean[clean.recording == recording].iloc[0]
            folds = daily_table[
                (daily_table.strategy == strategy)
                & (daily_table.recording == recording)
                & (daily_table.method == "adaptive")
                & (daily_table.scenario == "clean")
            ]
            positive = float(((folds.mss_vs_fixed > 0) & (folds.mss_vs_selected > 0)).mean())
            degraded = table[(table.recording == recording) & (table.scenario != "clean")]
            ratios = {
                row.scenario: {"fixed": 1 - row.mss_vs_fixed, "selected": 1 - row.mss_vs_selected}
                for row in degraded.itertuples()
            }
            valid_skill = all(
                value is not None and np.isfinite(value) and value >= cfg["minimum_clean_skill"]
                for value in (result.mss_vs_fixed, result.mss_vs_selected)
            )
            robust = all(
                np.isfinite(value) and value <= cfg["maximum_degraded_ratio"]
                for ratio in ratios.values()
                for value in ratio.values()
            )
            folds_pass = (
                recording == "LW1_later" or positive >= cfg["minimum_positive_fold_fraction"]
            )
            node_checks.append(
                {
                    "recording": recording,
                    "clean_skill_fixed": float(result.mss_vs_fixed),
                    "clean_skill_selected": float(result.mss_vs_selected),
                    "clean_pass": bool(valid_skill),
                    "positive_fold_fraction": positive,
                    "fold_count": len(folds),
                    "fold_pass": bool(folds_pass),
                    "scenario_ratios": ratios,
                    "robustness_pass": bool(robust),
                }
            )
        decisions.append(
            {
                "strategy": strategy,
                "recordings": node_checks,
                "qualifies": all(
                    c["clean_pass"] and c["fold_pass"] and c["robustness_pass"] for c in node_checks
                )
                and not failures,
                "failures_recorded": len(failures),
                "minimum_clean_skill": min(
                    min(c["clean_skill_fixed"], c["clean_skill_selected"]) for c in node_checks
                ),
                "mean_clean_skill": float(
                    np.mean(
                        [
                            min(c["clean_skill_fixed"], c["clean_skill_selected"])
                            for c in node_checks
                        ]
                    )
                ),
            }
        )
    qualifying = [d for d in decisions if d["qualifies"]]
    winner = (
        min(
            qualifying,
            key=lambda d: (
                -d["minimum_clean_skill"],
                -d["mean_clean_skill"],
                ["bias", "aggregation", "ridge_history", "ridge_fine"].index(d["strategy"]),
            ),
        )
        if qualifying
        else None
    )
    write_json(
        out / "advancement.json",
        {
            "status": "qualified" if winner else "not_qualified",
            "flagship": winner["strategy"] if winner else "history",
            "all_results_exploratory": True,
            "strategies": decisions,
        },
    )
    feedback = pd.read_parquet(source / "feedback.parquet")
    feedback_summary = (
        feedback.assign(
            usable=feedback.update_reason == "updated",
            latency_s=feedback.release_s - feedback.target_end_s,
        )
        .groupby(["recording", "scenario", "seed"], as_index=False)
        .agg(
            windows=("origin_s", "size"),
            usable_windows=("usable", "sum"),
            mean_latency_s=("latency_s", "mean"),
        )
    )
    feedback_summary.to_csv(out / "feedback_summary.csv", index=False)
    periods = (
        predictions.assign(loss=(predictions.prediction - predictions.y) ** 2)
        .groupby(["recording", "strategy", "scenario", "method", "period"], as_index=False)
        .agg(n_forecasts=("origin_s", "size"), mse=("loss", "mean"))
    )
    periods.to_csv(out / "period_summary.csv", index=False)
    event_report(source, out, predictions, cfg)
    figures(predictions, summary, pd.DataFrame(intervals), out)


def event_report(source, out, predictions, cfg):
    validation = pd.read_parquet(source / "validation_predictions.parquet")
    selections = json.loads((source / "selections.json").read_text())
    rows = []
    for selection in selections:
        source_recording = selection["selection_source"]
        if source_recording == "LW1_first_final_fold":
            source_recording = "LW1_first"
            fold = int(validation[validation.recording == source_recording].fold.max())
        else:
            fold = selection["fold"]
        for strategy, candidate in selection["chosen"].items():
            val = validation[
                (validation.recording == source_recording)
                & (validation.fold == fold)
                & (validation.candidate == candidate)
            ]
            for budget in cfg["false_busy_budgets"]:
                calibration = operating_threshold(
                    val.y.to_numpy(), val.prediction.to_numpy(), budget, cfg
                )
                frame = predictions[
                    (predictions.recording == selection["recording"])
                    & (predictions.fold == selection["fold"])
                    & (predictions.strategy == strategy)
                    & (predictions.method == "adaptive")
                ]
                for scenario, part in frame.groupby("scenario"):
                    threshold = calibration["threshold"]
                    event = part.y >= cfg["event_cutoff"]
                    npos, nneg = int(event.sum()), int((~event).sum())
                    rows.append(
                        {
                            "recording": selection["recording"],
                            "fold": selection["fold"],
                            "strategy": strategy,
                            "scenario": scenario,
                            "budget": budget,
                            **calibration,
                            "evaluation_events": npos,
                            "evaluation_nonevents": nneg,
                            "false_busy": float(
                                (part.loc[~event, "prediction"] >= threshold).mean()
                            )
                            if threshold is not None and nneg
                            else None,
                            "false_idle": float((part.loc[event, "prediction"] < threshold).mean())
                            if threshold is not None and npos
                            else None,
                        }
                    )
    write_json(out / "event_thresholds.json", rows)


def figures(predictions, summary, uncertainty, out):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    synthetic = bool(predictions.synthetic.any())
    title_prefix = "SYNTHETIC REHEARSAL — " if synthetic else "Exploratory development — "
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, recording in zip(axes, sorted(predictions.recording.unique()), strict=True):
        frame = predictions[
            (predictions.recording == recording)
            & (predictions.scenario == "clean")
            & (predictions.strategy == "bias")
        ]
        for method, label, color in (
            ("fixed_history", "Two-hour history", "#555555"),
            ("adaptive", "Adaptive bias strategy", "#0072B2"),
        ):
            part = frame[frame.method == method].sort_values("origin_s")
            x = (part.origin_s - part.origin_s.min()) / 3600
            ax.plot(x, part.prediction, label=label, color=color, linewidth=1)
        part = frame[frame.method == "adaptive"].sort_values("origin_s")
        ax.plot(
            (part.origin_s - part.origin_s.min()) / 3600,
            part.y,
            label="Observed busy fraction",
            color="#D55E00",
            alpha=0.7,
            linewidth=1,
        )
        ax.set(title=recording, ylabel="Busy fraction", ylim=(-0.02, 1.02))
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Hours since first scored forecast (+3 dB, 15-minute target)")
    fig.suptitle(title_prefix + "immutable forecasts and subsequent observations")
    fig.savefig(out / "forecasts.png", dpi=150)
    plt.close(fig)
    clean = summary[(summary.scenario == "clean") & summary.method.isin(["adaptive", "controller"])]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    table = (
        clean[clean.method == "adaptive"].pivot(
            index="strategy", columns="recording", values="mss_vs_fixed"
        )
        * 100
    )
    table.plot.bar(ax=ax, color=["#0072B2", "#E69F00", "#009E73"])
    ax.axhline(5, color="black", linestyle="--", label="5% promotion criterion")
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set(
        ylabel="MSE reduction vs two-hour history (%)",
        xlabel="Declared strategy",
        title=title_prefix + "clean performance; negative values mean higher error",
    )
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(out / "clean_skill.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    scenario_labels = {
        "clean": "Clean",
        "dropout_0.5": "Drop 50%",
        "dropout_0.9": "Drop 90%",
        "dropout_0.99": "Drop 99%",
        "outage_900s": "Outage 15m",
        "outage_3600s": "Outage 1h",
        "delay_60s": "Delay 1m",
        "delay_600s": "Delay 10m",
        "delay_1800s": "Delay 30m",
        "delay_3600s": "Delay 1h",
        "delay_7200s": "Delay 2h",
    }
    for ax, recording in zip(axes, sorted(summary.recording.unique()), strict=True):
        frame = summary[(summary.recording == recording) & (summary.method == "adaptive")]
        for strategy, part in frame.groupby("strategy"):
            order = {name: i for i, name in enumerate(scenario_labels)}
            part = part.sort_values("scenario", key=lambda values: values.map(order))
            ax.plot(
                part.scenario.map(scenario_labels),
                1 - part.mss_vs_fixed,
                marker=".",
                label=strategy,
            )
        ax.axhline(1.1, linestyle="--", color="black")
        ax.set(title=recording, ylabel="MSE / same-scenario two-hour MSE")
        ax.tick_params(axis="x", rotation=25, labelsize=10)
    axes[0].legend(fontsize=10, ncol=2)
    fig.suptitle(title_prefix + "degraded evidence with sensor-derived feedback")
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    colors = {
        "aggregation": "#0072B2",
        "bias": "#E69F00",
        "ridge_fine": "#009E73",
        "ridge_history": "#CC79A7",
    }
    for ax, recording in zip(axes, sorted(summary.recording.unique()), strict=True):
        frame = uncertainty[
            (uncertainty.recording == recording)
            & (uncertainty.method == "adaptive")
            & (uncertainty.scenario == "clean")
            & (uncertainty.reference == "fixed")
        ]
        for strategy, part in frame.groupby("strategy"):
            ax.plot(
                part.block_s / 3600,
                part.difference_lo,
                marker=".",
                label=strategy,
                color=colors[strategy],
            )
            ax.plot(
                part.block_s / 3600,
                part.difference_hi,
                marker=".",
                linestyle="--",
                color=colors[strategy],
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks([2, 4, 8, 12])
        ax.set_xlim(1.5, 12.5)
        ax.set(
            title=recording,
            xlabel="Temporal block length (hours)",
            ylabel="Paired MSE-difference interval",
        )
    axes[0].legend(fontsize=10, ncol=2)
    fig.suptitle(title_prefix + "intervals omitted below four blocks; <10 blocks is weak")
    fig.savefig(out / "block_sensitivity.png", dpi=150)
    plt.close(fig)
