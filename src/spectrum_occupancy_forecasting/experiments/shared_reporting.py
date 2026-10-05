"""Saved-prediction comparisons and the declared shared-learning decision."""

from pathlib import Path

import matplotlib
import pandas as pd

from .fine_spectrum import write_json
from .followup_reporting import paired_intervals, paired_losses, point_metrics, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def compare_profiles(predictions, records, cfg, out):
    """Failures cannot disappear from common-fold eligibility or advancement."""
    rows, daily = [], []
    for name in ("CC1", "LW1"):
        node_records = [r for r in records if r["node"] == name]
        common = sorted(
            {
                r["fold"]
                for r in node_records
                if all(
                    t["selection_status"] == "selected" and t["status"] == "completed"
                    for t in node_records
                    if t["fold"] == r["fold"]
                )
            }
        )
        aligned = None
        for _, part in predictions[
            (predictions.node == name)
            & predictions.fold.isin(common)
            & (predictions.method == "selected_learned")
        ].groupby("profile_id"):
            labels = part.set_index(["fold", "origin_s"]).y.sort_index()
            if aligned is not None:
                pd.testing.assert_series_equal(aligned, labels)
            aligned = labels
        for profile, part in predictions[predictions.node == name].groupby("profile_id"):
            frame = part[part.fold.isin(common)]
            if not len(frame):
                rows.append({"node": name, "profile_id": profile, "status": "no_common_folds"})
                continue
            fixed = frame[frame.method == "fixed_history"]
            selected = frame[frame.method == "selected_history"]
            for method, subset in frame.groupby("method"):
                metric, _ = point_metrics(subset, selected, fixed, selected)
                rows.append(
                    {
                        "node": name,
                        "profile_id": profile,
                        "method": method,
                        "common_folds": len(common),
                        "status": "completed",
                        **metric,
                    }
                )
                for fold, f in subset.groupby("fold"):
                    metric, _ = point_metrics(
                        f,
                        selected[selected.fold == fold],
                        fixed[fixed.fold == fold],
                        selected[selected.fold == fold],
                    )
                    daily.append(
                        {
                            "node": name,
                            "profile_id": profile,
                            "method": method,
                            "fold": fold,
                            **metric,
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(Path(out) / "profile_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(daily).to_csv(Path(out) / "profile_daily.csv", index=False, lineterminator="\n")
    winners = []
    for profile, part in table[table.method == "selected_learned"].groupby("profile_id"):
        if set(part.node) != {"CC1", "LW1"}:
            continue
        rec = next(r for r in records if r["profile_id"] == profile)
        family = max(
            {"history": 0, "ridge": 1, "hgb": 2}[r["choice"]["family"]]
            for r in records
            if r["profile_id"] == profile
        )
        skill = part.mss_vs_fixed.to_numpy()
        winners.append(
            (
                (
                    -float(skill.min()),
                    -float(skill.mean()),
                    rec["profile"]["variant"] != "history",
                    family,
                    rec["profile"]["spacing_s"] != 900,
                    rec["profile"]["hours"] != 36,
                    profile,
                ),
                profile,
            )
        )
    if not winners:
        raise ValueError("no complete common-fold profile; publish failure before assessment")
    chosen = min(winners)[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ("CC1", "LW1"), strict=True):
        rows = table[(table.node == name) & (table.method == "selected_learned")]
        ax.barh(rows.profile_id, rows.mss_vs_fixed * 100, color="#0072B2")
        ax.axvline(0, color="black", lw=0.8)
        ax.set(title=name, xlabel="MSE reduction vs two-hour history (%)")
    fig.suptitle("SYNTHETIC REHEARSAL ONLY" if cfg.get("synthetic") else "Common eligible folds")
    fig.tight_layout()
    fig.savefig(Path(out) / "profile_comparison.png", dpi=150)
    plt.close(fig)
    write_json(
        Path(out) / "profile_choice.json",
        {
            "profile_id": chosen,
            "interpretation": "adaptive exploratory selection; not independent superiority",
        },
    )
    return chosen


def chosen_report(predictions, records, cfg, out):
    out = Path(out)
    summarize(predictions, records, cfg, out)
    criteria, sensitivity, common_metrics = [], [], []
    for name, node in predictions.groupby("node"):
        common_folds = sorted(
            {r["fold"] for r in records if r["node"] == name and r["common_fold"]}
        )
        part = node[node.fold.isin(common_folds)]
        for scenario, scenario_rows in part.groupby("scenario"):
            fixed = scenario_rows[scenario_rows.method == "fixed_history"]
            selected = scenario_rows[scenario_rows.method == "selected_history"]
            clean_reference = part[
                (part.scenario == "clean") & (part.method == "selected_history")
            ]
            for method, rows in scenario_rows.groupby("method"):
                metric, _ = point_metrics(rows, selected, fixed, clean_reference)
                common_metrics.append(
                    {"node": name, "scenario": scenario, "method": method, **metric}
                )
        learned = part[part.method == "selected_learned"]
        criteria.append(
            {
                "node": name,
                "criterion": "common_folds",
                "value": len(common_folds),
                "passed": len(common_folds) >= cfg["minimum_common_folds"],
            }
        )
        for reference in ("fixed_history", "selected_history"):
            for scenario, frame in learned.groupby("scenario"):
                baseline = part[(part.method == reference) & (part.scenario == scenario)]
                losses = paired_losses(frame, baseline)
                ratio = (
                    float(losses.loss.mean() / losses.ref_loss.mean())
                    if losses.ref_loss.mean()
                    else None
                )
                clean = scenario == "clean"
                passed = ratio is not None and ratio <= (
                    1 - cfg["minimum_clean_skill"] if clean else cfg["maximum_degraded_ratio"]
                )
                criteria.append(
                    {
                        "node": name,
                        "criterion": f"{scenario}_vs_{reference}",
                        "value": ratio,
                        "passed": bool(passed),
                    }
                )
                for seconds in cfg["bootstrap_blocks_s"]:
                    sensitivity.append(
                        {
                            "node": name,
                            "scenario": scenario,
                            "reference": reference,
                            **paired_intervals(losses, seconds, cfg),
                        }
                    )
                if clean:
                    folds = losses.groupby("fold")[["loss", "ref_loss"]].mean()
                    fraction = float((folds.loss < folds.ref_loss).mean())
                    criteria.append(
                        {
                            "node": name,
                            "criterion": f"positive_folds_vs_{reference}",
                            "value": fraction,
                            "passed": fraction >= 0.5,
                        }
                    )
        failures = [
            r
            for r in records
            if r["node"] == name
            and (r["status"] != "completed" or any(s["status"] == "failed" for s in r["scores"]))
        ]
        criteria.append(
            {
                "node": name,
                "criterion": "reported_failures",
                "value": len(failures),
                "passed": not failures,
            }
        )
    pd.DataFrame(sensitivity).to_csv(
        out / "paired_reference_bootstrap.csv", index=False, lineterminator="\n"
    )
    write_json(
        out / "advancement.json",
        {
            "qualifies": all(c["passed"] for c in criteria),
            "criteria": criteria,
            "interpretation": "exploratory; original decisions unchanged",
        },
    )
    # All values are read from saved prediction-derived summaries.
    table = pd.DataFrame(common_metrics)
    table.to_csv(out / "common_summary.csv", index=False, lineterminator="\n")
    size = table.node.nunique()
    fig, axes = plt.subplots(1, size, figsize=(5 * size, 4), squeeze=False)
    for ax, (name, rows) in zip(axes.ravel(), table.groupby("node"), strict=True):
        clean = rows[rows.scenario == "clean"]
        ax.bar(clean.method, clean.mse, color="#0072B2")
        ax.tick_params(axis="x", rotation=25)
        ax.set(title=name, ylabel="Mean squared error (lower is better)")
    if cfg.get("synthetic"):
        fig.suptitle("SYNTHETIC REHEARSAL ONLY")
    fig.tight_layout()
    fig.savefig(out / "clean_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, rows in table[table.method == "selected_learned"].groupby("node"):
        ax.plot(rows.scenario, rows.mss_vs_fixed * 100, "o-", label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="MSE reduction vs same-scenario two-hour history (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    if cfg.get("synthetic"):
        fig.suptitle("SYNTHETIC REHEARSAL ONLY")
    fig.tight_layout()
    fig.savefig(out / "degradation_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    boot = pd.DataFrame(sensitivity)
    for name, rows in boot[
        (boot.scenario == "clean") & (boot.reference == "fixed_history")
    ].groupby("node"):
        valid = rows.dropna(subset=["difference_lo", "difference_hi"])
        ax.plot(valid.block_s / 3600, valid.difference_lo, "o--", label=f"{name} lower")
        ax.plot(valid.block_s / 3600, valid.difference_hi, "o-", label=f"{name} upper")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Temporal block length (hours)", ylabel="Paired 95% MSE-difference interval")
    if ax.lines and len(ax.lines) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bootstrap_sensitivity.png", dpi=150)
    plt.close(fig)
    if size == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        clean = predictions[predictions.scenario == "clean"]
        first = clean.origin_s.min()
        observed = clean[clean.method == "fixed_history"]
        ax.plot((observed.origin_s - first) / 3600, observed.y, color="black", label="Observed")
        for method, rows in clean.groupby("method"):
            ax.plot((rows.origin_s - first) / 3600, rows.prediction, label=method, alpha=0.7)
        ax.set(xlabel="Hours after first assessment origin", ylabel="15-minute busy fraction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "later_forecasts.png", dpi=150)
        plt.close(fig)
