"""Prediction-only challenger comparisons, advancement decision, and research figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .followup import scenarios
from .followup_reporting import paired_losses, summarize


def attach_reference(predictions, reference):
    """Copy the canonical v2 forecast at exactly the challenger scored origins."""
    ref = reference[(reference.task == "primary") & (reference.method == "selected_learned")]
    keys = ["node", "fold", "scenario", "seed", "origin_s"]
    frames = [predictions]
    for _, frame in predictions[
        (predictions.task == "primary") & (predictions.method == "selected_learned")
    ].groupby("strategy_id"):
        joined = frame.merge(
            ref[keys + ["y", "prediction", "candidate_id"]],
            on=keys,
            suffixes=("", "_v2"),
            validate="one_to_one",
            how="left",
        )
        if joined.prediction_v2.isna().any() or not np.array_equal(joined.y, joined.y_v2):
            raise ValueError("canonical v2 reference does not match every scored origin and label")
        joined["prediction"] = joined.pop("prediction_v2")
        joined["candidate_id"] = joined.pop("candidate_id_v2")
        joined = joined.drop(columns="y_v2")
        joined["method"] = "v2_learned"
        joined["correction_strength"] = 0.0
        joined["selected_penalty"] = np.nan
        joined["correction_value"] = 0.0
        joined["fallback_reason"] = "canonical_reference"
        frames.append(joined)
    return pd.concat(frames, ignore_index=True)


def advancement(predictions, selections, cfg, *, nodes=("CC1", "LW1")):
    """Apply every predeclared criterion on identical common folds and paired mask losses."""
    results = []
    primary = predictions[predictions.task == "primary"]
    for hours in cfg["validation_hours"]:
        for regime in cfg["training_regimes"]:
            sid = f"{hours}h_{regime}"
            criteria, skills = ([], [])
            for node in nodes:
                rec = [
                    r
                    for r in selections
                    if r["node"] == node and r["task"] == "primary" and (r["strategy_id"] == sid)
                ]
                eligible = [
                    r["fold"]
                    for r in rec
                    if r["common_fold"]
                    and r["selection_status"] == "selected"
                    and (r["status"] == "completed")
                ]
                failed = [
                    r["fold"]
                    for r in rec
                    if r["common_fold"]
                    and (r["status"] != "completed" or r["selection_status"] != "selected")
                ]

                def criterion(name, value, passed, criteria=criteria, node=node):
                    criteria.append(
                        {"node": node, "criterion": name, "value": value, "passed": bool(passed)}
                    )

                criterion(
                    "common_eligible_folds",
                    len(eligible),
                    len(eligible) >= cfg["advancement_minimum_folds"],
                )
                criterion("failed_common_folds", failed, not failed)
                group = primary[
                    (primary.node == node)
                    & (primary.strategy_id == sid)
                    & primary.fold.isin(eligible)
                ]
                clean_keys = (
                    group[(group.method == "selected_learned") & (group.scenario == "clean")][
                        ["fold", "origin_s", "y"]
                    ]
                    .sort_values(["fold", "origin_s"])
                    .reset_index(drop=True)
                )
                anchor_sid = f"{max(cfg['validation_hours'])}h_{cfg['training_regimes'][0]}"
                anchor = (
                    primary[
                        (primary.node == node)
                        & (primary.strategy_id == anchor_sid)
                        & primary.common_fold
                        & (primary.method == "selected_learned")
                        & (primary.scenario == "clean")
                    ][["fold", "origin_s", "y"]]
                    .sort_values(["fold", "origin_s"])
                    .reset_index(drop=True)
                )
                criterion("identical_common_origins", len(clean_keys), clean_keys.equals(anchor))
                clean_loss = None
                for scenario in scenarios(cfg):
                    part = group[group.scenario == scenario]
                    forecast = part[part.method == "selected_learned"]
                    history = part[part.method == "selected_history"]
                    if forecast.empty:
                        criterion(f"{scenario}_mse_ratio", None, False)
                        continue
                    loss = paired_losses(forecast, history)
                    expected_seeds = (
                        cfg["mask_seeds"]
                        if scenarios(cfg)[scenario].dropout_p
                        or scenarios(cfg)[scenario].outage_frac
                        else [0]
                    )
                    complete = set(forecast.seed) == set(expected_seeds) and all(
                        (
                            len(p) == len(expected_seeds)
                            for _, p in forecast.groupby(["fold", "origin_s"])
                        )
                    )
                    scenario_keys = (
                        forecast[["fold", "origin_s", "y"]]
                        .drop_duplicates()
                        .sort_values(["fold", "origin_s"])
                        .reset_index(drop=True)
                    )
                    complete = complete and scenario_keys.equals(clean_keys)
                    ratio = (
                        float(loss.loss.mean() / loss.ref_loss.mean())
                        if loss.ref_loss.mean() > 0
                        else None
                    )
                    criterion(f"{scenario}_paired_complete", len(forecast), complete)
                    if scenario == "clean":
                        clean_loss = loss
                        skill = 1 - ratio if ratio is not None else None
                        skills.append(skill)
                        criterion(
                            "clean_skill",
                            skill,
                            skill is not None and skill >= cfg["advancement_minimum_skill"],
                        )
                    else:
                        criterion(
                            f"{scenario}_mse_ratio",
                            ratio,
                            ratio is not None
                            and ratio <= cfg["advancement_maximum_degraded_mse_ratio"],
                        )
                positive = 0
                if clean_loss is not None:
                    positive = sum(
                        (p.loss.mean() < p.ref_loss.mean() for _, p in clean_loss.groupby("fold"))
                    )
                fraction = positive / len(eligible) if eligible else None
                criterion(
                    "positive_fold_fraction",
                    fraction,
                    fraction is not None and fraction >= cfg["advancement_positive_fold_fraction"],
                )
            finite = [s for s in skills if s is not None]
            results.append(
                {
                    "strategy_id": sid,
                    "validation_hours": hours,
                    "training_regime": regime,
                    "qualifies": all(c["passed"] for c in criteria),
                    "criteria": criteria,
                    "minimum_node_clean_skill": min(finite) if len(finite) == len(nodes) else None,
                    "mean_node_clean_skill": float(np.mean(finite))
                    if len(finite) == len(nodes)
                    else None,
                }
            )
    qualified = [r for r in results if r["qualifies"]]
    winner = (
        min(
            qualified,
            key=lambda r: (
                -r["minimum_node_clean_skill"],
                -r["mean_node_clean_skill"],
                r["validation_hours"],
                r["training_regime"] != "clean",
            ),
        )
        if qualified
        else None
    )
    return {
        "interpretation": "exploratory reuse; not an independent superiority test",
        "advances": winner is not None,
        "chosen_strategy": winner,
        "strategies": results,
        "secondary_status": "required_for_qualifier" if winner else "not_run_no_strategy_qualified",
        "confirmation_design": "challenger_pending_freeze" if winner else "retain_canonical_v2",
    }


def report(predictions, selections, cfg, out):
    if "CC2" in set(predictions.node):
        raise ValueError("CC2 is prohibited in the exploratory mixture report")
    out = Path(out)
    interpretation = (
        "synthetic rehearsal; not measurement evidence"
        if set(predictions.node) == {"SYN"}
        else "exploratory reused CC1/LW1; confirmation pending"
    )
    tables = {}
    for sid, part in predictions.groupby("strategy_id", sort=True):
        directory = out / "_parts" / "reports" / sid
        directory.mkdir(parents=True, exist_ok=True)
        selected = [r for r in selections if r["strategy_id"] == sid]
        for name, table in summarize(part, selected, cfg, directory).items():
            tables.setdefault(name, []).append(table.assign(strategy_id=sid))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    decision = advancement(predictions, selections, cfg)
    if set(predictions.node) == {"SYN"}:
        decision["interpretation"] = interpretation
    for sid, part in predictions[
        (predictions.task == "primary")
        & predictions.common_fold
        & (predictions.scenario == "clean")
    ].groupby("strategy_id"):
        directory = out / "_parts" / "reports" / f"common_{sid}"
        directory.mkdir(parents=True, exist_ok=True)
        selected = [
            r
            for r in selections
            if r["strategy_id"] == sid and r["common_fold"] and (r["task"] == "primary")
        ]
        for name, table in summarize(part, selected, cfg, directory).items():
            if name in ("summary", "daily", "bootstrap"):
                tables.setdefault(f"common_{name}", []).append(table.assign(strategy_id=sid))
    for name in ("common_summary", "common_daily", "common_bootstrap"):
        if name in tables:
            tables[name] = pd.concat(tables[name], ignore_index=True)
            tables[name].to_csv(out / f"{name}.csv", index=False)
    (out / "advancement.json").write_text(json.dumps(decision, indent=2, allow_nan=False))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "interpretation": interpretation,
                "metrics": json.loads(
                    tables["summary"].to_json(orient="records", double_precision=15)
                ),
                "advances": decision["advances"],
                "main_bootstrap_block_s": cfg["main_bootstrap_block_s"],
                "mask_uncertainty": "paired losses averaged over seeds; separate node/fold blocks",
            },
            indent=2,
            allow_nan=False,
        )
    )
    predictions[predictions.method == "selected_learned"].groupby(
        ["node", "task", "strategy_id", "fold", "scenario", "fallback_reason"], dropna=False
    ).size().rename("n_prediction_rows").reset_index().to_csv(out / "fallbacks.csv", index=False)
    scores = [
        {
            "node": r["node"],
            "fold": r["fold"],
            "task": r["task"],
            "strategy_id": r["strategy_id"],
            **c,
        }
        for r in selections
        for c in r["candidates"]
    ]
    pd.DataFrame(scores).to_csv(out / "candidate_scores.csv", index=False)
    if "strategies" in decision:
        pd.DataFrame(
            [
                {"strategy_id": s["strategy_id"], **c}
                for s in decision["strategies"]
                for c in s["criteria"]
            ]
        ).to_csv(out / "advancement_criteria.csv", index=False)
    return tables


def make_figures(predictions, tables, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = tables["summary"]
    comparison = tables.get("common_summary", rows)
    clean = comparison[
        (comparison.task == "primary")
        & (comparison.scenario == "clean")
        & (comparison.method == "selected_learned")
    ]
    fig, axes = plt.subplots(1, len(clean.node.unique()), figsize=(12, 4), squeeze=False)
    for ax, (node, part) in zip(axes.flat, clean.groupby("node"), strict=True):
        ax.bar(part.strategy_id, part.mss_vs_selected, color="#3274a1")
        ax.axhline(0.05, color="#b33b32", linestyle="--", label="5% advancement criterion")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(
            title=f"{node}: common eligible folds (exploratory)",
            ylabel="MSE skill vs selected history",
        )
        ax.tick_params(axis="x", labelrotation=35)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "clean_skill.png", dpi=160)
    plt.close(fig)
    interval = tables.get("common_bootstrap", tables["bootstrap"])
    interval = interval[
        (interval.task == "primary")
        & (interval.scenario == "clean")
        & (interval.method == "selected_learned")
    ]
    fig, axes = plt.subplots(1, len(clean.node.unique()), figsize=(12, 5), squeeze=False)
    for ax, (node, part) in zip(axes.flat, interval.groupby("node"), strict=True):
        for index, (sid, group) in enumerate(part.groupby("strategy_id")):
            color = plt.get_cmap("tab10")(index)
            ax.plot(group.block_s / 3600, group.difference_hi, marker=".", label=sid, color=color)
            ax.plot(
                group.block_s / 3600, group.difference_lo, linestyle=":", linewidth=0.8, color=color
            )
        counts = part.groupby("block_s").n_blocks.first()
        ax.set_xticks(counts.index / 3600, [f"{b / 3600:g}\nN={n}" for b, n in counts.items()])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(
            title=f"{node}: common folds, exploratory 95% intervals",
            xlabel="Block hours",
            ylabel="Paired MSE difference (bounds)",
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "bootstrap_sensitivity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(clean.node.unique()), 1, figsize=(12, 7), squeeze=False)
    degraded = rows[(rows.task == "primary") & (rows.method == "selected_learned")]
    for ax, (node, part) in zip(axes.flat, degraded.groupby("node"), strict=True):
        table = part.pivot(index="strategy_id", columns="scenario", values="mss_vs_selected")
        im = ax.imshow(table, aspect="auto", cmap="RdBu", vmin=-0.2, vmax=0.2)
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_title(f"{node}: paired same-scenario skill; all folds")
        fig.colorbar(im, ax=ax, label="MSE skill")
    fig.tight_layout()
    fig.savefig(out / "degradation.png", dpi=160)
    plt.close(fig)
    durations = sorted(predictions.validation_hours.unique())
    fig, axes = plt.subplots(
        len(durations), len(clean.node.unique()), figsize=(13, 3 * len(durations)), squeeze=False
    )
    for j, node in enumerate(sorted(clean.node.unique())):
        for i, hours in enumerate(durations):
            ax = axes[i, j]
            part = predictions[
                (predictions.node == node)
                & (predictions.task == "primary")
                & (predictions.scenario == "clean")
                & (predictions.validation_hours == hours)
            ]
            origin = part.origin_s.min()
            truth = (
                part[part.method == "selected_history"]
                .drop_duplicates("origin_s")
                .sort_values("origin_s")
            )
            ax.plot((truth.origin_s - origin) / 3600, truth.y, color=".7", label="observed")
            for regime in sorted(part.training_regime.unique()):
                f = part[
                    (part.method == "selected_learned") & (part.training_regime == regime)
                ].sort_values("origin_s")
                ax.plot((f.origin_s - origin) / 3600, f.prediction, linewidth=0.8, label=regime)
            ax.set(
                title=f"{node}, {hours} h validation",
                xlabel="Hours from first evaluated origin",
                ylabel="Busy fraction",
            )
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=160)
    plt.close(fig)
