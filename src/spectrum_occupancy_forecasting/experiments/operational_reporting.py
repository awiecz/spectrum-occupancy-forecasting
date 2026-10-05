"""Prediction-only reporting and numerical rehearsal for activity-count learning."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .activity_counts import CountFilter, candidates
from .adaptive_reporting import regenerate as adaptive_report
from .fine_spectrum import sha, write_json
from .followup import scenarios
from .operational_selection import STRATEGIES


def numerical_suite(cfg, out):
    rng = np.random.default_rng(1701)
    cases = {
        "persistent": np.full(72, 0.2),
        "changing": np.r_[np.full(36, 0.05), np.full(36, 0.65)],
        "gaps": np.r_[np.full(24, 0.1), np.full(24, 0.7), np.full(24, 0.1)],
        "independent_future": rng.uniform(0.02, 0.8, 72),
        "all_idle": np.zeros(72),
        "all_busy": np.ones(72),
    }
    rows = []
    for low, high in ((801, 1601), (1601, 3201)):
        current = []
        for name, probabilities in cases.items():
            counts = rng.binomial(30, probabilities)
            mu = float(np.log((counts[:24].sum() + 0.5) / (24 * 30 - counts[:24].sum() + 0.5)))
            for h in cfg["count_half_lives_h"]:
                for sd in cfg["count_standard_deviations"]:
                    for rho in cfg["count_dispersions"]:
                        filters = [CountFilter(mu, h, sd, rho, nodes=n) for n in (low, high)]
                        differences = []
                        for j, count in enumerate(counts):
                            t = (j + 1) * 300
                            if not (name == "gaps" and 24 <= j < 48):
                                for state in filters:
                                    state.assimilate(t, int(count), 30)
                            forecasts = [
                                state.predict([t + 1200, t + 1500, t + 1800]) for state in filters
                            ]
                            differences.append(abs(forecasts[0] - forecasts[1]))
                        row = {
                            "case": name,
                            "half_life": h,
                            "sd": sd,
                            "rho": rho,
                            "low_nodes": low,
                            "high_nodes": high,
                            "maximum_forecast_difference": max(differences),
                        }
                        current.append(row)
        rows.extend(current)
        if max(r["maximum_forecast_difference"] for r in current) <= cfg["numerical_tolerance"]:
            selected = low
            break
    else:
        raise ValueError("grid convergence failed; do not execute real-data study")
    pd.DataFrame(rows).to_csv(out / "numerical_convergence.csv", index=False)
    write_json(
        out / "numerical_gate.json",
        {
            "passes": True,
            "selected_grid_nodes": selected,
            "tolerance": cfg["numerical_tolerance"],
            "synthetic": True,
        },
    )
    return selected


def regenerate(source, out, cfg):
    manifest_path = source / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # During initial reporting only hashes available so far exist; published regeneration
        # verifies every input byte against the completed fitting manifest.
        if manifest.get("status") == "completed":
            for name, expected in manifest["artifact_hashes"].items():
                path = source / name
                if not path.exists() or sha(path) != expected:
                    raise ValueError(f"corrupted saved artifact: {name}")
    frame = pd.read_parquet(source / "predictions.parquet")
    expected = {"CC1_first": 439, "LW1_first": 239, "LW1_later": 81}
    expected_cohorts = {
        (recording, scenario, seed, method)
        for recording in expected
        for scenario, degradation in scenarios(cfg).items()
        for seed in (
            cfg["mask_seeds"] if degradation.dropout_p or degradation.outage_len_s else [0]
        )
        for method in ("adaptive", "controller", "fixed_history", "selected_history")
    }
    observed_cohorts = set(frame.groupby(["recording", "scenario", "seed", "method"]).groups)
    if not frame.synthetic.any() and observed_cohorts != expected_cohorts:
        raise ValueError("a required recording/scenario/seed/method is missing")
    for _recording, part in frame.groupby("recording"):
        labels = part.groupby("origin_s").agg(
            y_values=("y", "nunique"), counts=("target_count", "nunique")
        )
        if not ((labels.y_values == 1) & (labels.counts == 1)).all():
            raise ValueError("degradation changed scoring labels or counts")
    for (recording, _scenario, _seed, _method), part in frame.groupby(
        ["recording", "scenario", "seed", "method"]
    ):
        if part.origin_s.duplicated().any() or (
            not part.synthetic.any() and len(part) != expected[recording]
        ):
            raise ValueError("missing or duplicated scored origins")
        if not np.isfinite(part.prediction).all():
            raise ValueError("failed forecasts must have explicit finite history fallback")
    for recording in frame.recording.unique():
        issued = pd.read_parquet(source / f"{recording}_candidate_forecasts.parquet")
        counts = issued.groupby("origin_s").candidate.nunique()
        if not (counts == len(candidates(cfg)) + 8).all():
            raise ValueError("incomplete candidate accounting")
        fallback = issued.fallback_reason.fillna("") != ""
        if not np.array_equal(
            issued.loc[fallback, "prediction"], issued.loc[fallback, "fixed_history"]
        ):
            raise ValueError("count safety fallback differs from fixed history")
    adaptive_report(
        source,
        out,
        cfg,
        candidate_factory=candidates,
        strategies=STRATEGIES,
        figure_function=figures,
    )
    path = out / "advancement.json"
    decision = json.loads(path.read_text())
    for strategy in decision["strategies"]:
        for recording in strategy["recordings"]:
            recording["minimum_fold_count_pass"] = (
                recording["recording"] == "LW1_later" or recording["fold_count"] >= 2
            )
        strategy["qualifies"] &= all(r["minimum_fold_count_pass"] for r in strategy["recordings"])
    qualified = [s for s in decision["strategies"] if s["qualifies"]]
    decision["status"] = "qualified" if qualified else "not_qualified"
    decision["flagship"] = "count" if qualified else "history"
    write_json(path, decision)


def figures(predictions, summary, uncertainty, out):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    prefix = "SYNTHETIC — " if predictions.synthetic.any() else "Exploratory development — "
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, recording in zip(axes, sorted(predictions.recording.unique()), strict=True):
        frame = predictions[
            (predictions.recording == recording) & (predictions.scenario == "clean")
        ]
        for method, label, color in (
            ("adaptive", "Selected count strategy", "#0072B2"),
            ("fixed_history", "Two-hour history", "#555555"),
        ):
            part = frame[frame.method == method].sort_values("origin_s")
            x = (part.origin_s - part.origin_s.min()) / 3600
            ax.plot(x, part.prediction, label=label, color=color, linewidth=1)
        part = frame[frame.method == "adaptive"].sort_values("origin_s")
        ax.plot(
            (part.origin_s - part.origin_s.min()) / 3600,
            part.y,
            color="#D55E00",
            label="Observed",
            alpha=0.7,
        )
        ax.set(title=recording, ylabel="Busy fraction", ylim=(-0.02, 1.02))
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Hours since first scored origin; +3 dB, target starts 15 minutes ahead")
    fig.suptitle(prefix + "forecasts saved before their outcomes")
    fig.savefig(out / "forecasts.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    clean = summary[(summary.method == "adaptive") & (summary.scenario == "clean")]
    x = np.arange(len(clean))
    axes[0].bar(x - 0.18, clean.mss_vs_fixed * 100, 0.36, color="#0072B2", label="Fixed history")
    axes[0].bar(
        x + 0.18, clean.mss_vs_selected * 100, 0.36, color="#E69F00", label="Selected history"
    )
    axes[0].set_xticks(x, clean.recording, rotation=15)
    axes[0].axhline(5, linestyle="--", color="black")
    axes[0].axhline(0, color="gray")
    axes[0].set(ylabel="MSE reduction (%)", title="Clean performance; dashed line = 5%")
    axes[0].legend(fontsize=9)
    for recording, frame in summary[summary.method == "adaptive"].groupby("recording"):
        axes[1].plot(frame.scenario, 1 - frame.mss_vs_fixed, marker=".", label=recording)
    axes[1].axhline(1.1, color="black", linestyle="--")
    axes[1].set(ylabel="MSE / same-scenario fixed history MSE", title="Sensor degradation")
    axes[1].tick_params(axis="x", rotation=80, labelsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(prefix + "mandatory comparisons")
    fig.savefig(out / "performance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    for ax, recording in zip(axes, sorted(predictions.recording.unique()), strict=True):
        frame = uncertainty[
            (uncertainty.recording == recording)
            & (uncertainty.method == "adaptive")
            & (uncertainty.scenario == "clean")
            & (uncertainty.reference == "fixed")
        ]
        ax.plot(
            frame.block_s / 3600, frame.difference_lo, "o-", color="#0072B2", label="95% limits"
        )
        ax.plot(frame.block_s / 3600, frame.difference_hi, "o--", color="#0072B2")
        ax.axhline(0, color="gray")
        ax.set(title=recording, xticks=[2, 4, 8, 12], ylabel="Paired MSE difference")
    axes[-1].set_xlabel("Block hours; omitted below four blocks; fewer than ten = weak evidence")
    fig.suptitle(prefix + "block-length sensitivity")
    fig.savefig(out / "block_sensitivity.png", dpi=150)
    plt.close(fig)


def audit_figures(protocol, controller, out):
    """Plot saved audit tables, without changing selection or fitting anything."""
    names = ["bias", "aggregation", "ridge_history", "ridge_fine", "level", "trend"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": True, "axes.spines.right": True}):
        fig = plt.figure(figsize=(12, 9), layout="constrained")
        grid = fig.add_gridspec(2, 3)
        source = []
        for j, recording in enumerate(["CC1_first", "LW1_first", "LW1_later"]):
            ax = fig.add_subplot(grid[0, j])
            for i, mode in enumerate(["historical-cold-start", "continuous-operational"]):
                part = protocol[(protocol.recording == recording) & (protocol.protocol == mode)]
                pooled = []
                for strategy in names:
                    q = part[part.strategy == strategy]
                    mse = np.average(q.mse, weights=q.n)
                    fixed = np.average(q.mse / (1 - q.skill_fixed), weights=q.n)
                    skill = 100 * (1 - mse / fixed)
                    pooled.append(skill)
                    source.append(
                        {
                            "recording": recording,
                            "protocol": mode,
                            "strategy": strategy,
                            "mse": mse,
                            "fixed_mse": fixed,
                            "skill_fixed_percent": skill,
                        }
                    )
                ax.barh(
                    np.arange(6) + (i - 0.5) * 0.32,
                    pooled,
                    0.32,
                    color=["#0072B2", "#E69F00"][i],
                    label=mode,
                )
            ax.axvline(0, color="gray", lw=0.8)
            ax.set(
                yticks=np.arange(6),
                yticklabels=[n.replace("_", " ") for n in names],
                title=recording,
                xlabel="MSE reduction vs fixed history (%)",
            )
            ax.invert_yaxis()
            if j == 0:
                ax.legend(fontsize=8, loc="upper right")
        ax = fig.add_subplot(grid[1, :])
        for j, mode in enumerate(["selected-hard", "fixed-hard", "fixed-soft"]):
            for learned, style in [(False, "--"), (True, "-")]:
                q = controller[
                    (controller.recording == "LW1_first")
                    & (controller.controller == mode)
                    & (controller.candidate.str.startswith("aggregation") == learned)
                ].sort_values("scenario")
                ax.plot(
                    q.scenario,
                    100 * q.skill_fixed,
                    style,
                    marker=".",
                    label=mode + (" + aggregation" if learned else " only"),
                    color=["#0072B2", "#E69F00", "#009E73"][j],
                )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(
            title="First rural recording: controller-only versus pinned aggregation",
            ylabel="MSE reduction vs same-scenario fixed history (%)",
        )
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.legend(fontsize=9, ncol=3)
        fig.suptitle(
            "Exploratory audits: separate validation, fallback and learned contributions",
            fontsize=14,
        )
        fig.savefig(out / "audit_comparison.png", dpi=150)
        plt.close(fig)
        pd.DataFrame(source).to_csv(out / "audit_plot_source.csv", index=False)


def adaptation_figure(predictions, states, out):
    """First validation-selected learned fold; descriptive selection does not use losses."""
    learned = predictions[
        (predictions.method == "adaptive")
        & (predictions.scenario == "clean")
        & (predictions.candidate != "count_history")
    ].sort_values(["recording", "origin_s"])
    if learned.empty:
        return
    first = learned.iloc[0]
    q = learned[(learned.recording == first.recording) & (learned.fold == first.fold)]
    key = "(" + ", ".join(first.candidate.split("_")[1:4]) + ")"
    latent = (
        states[
            (states.recording == first.recording)
            & (states.scenario == "clean")
            & (states.state == key)
        ]
        .merge(q[["origin_s"]], on="origin_s")
        .sort_values("origin_s")
    )
    with plt.rc_context({"font.size": 10, "axes.spines.top": True, "axes.spines.right": True}):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True)
        x = (q.origin_s - q.origin_s.min()) / 3600
        axes[0].plot(x, q.y, color="#D55E00", label="Observed future fraction")
        axes[0].plot(x, q.prediction, color="#0072B2", label="Count forecast")
        axes[0].plot(x, q.fixed_history, color="#555555", label="Two-hour history")
        axes[0].set(ylabel="Busy fraction", ylim=(-0.02, 1.02))
        axes[0].legend(fontsize=9, ncol=3)
        x = (latent.origin_s - q.origin_s.min()) / 3600
        axes[1].plot(x, latent.mean_logit, color="#0072B2", label="Posterior mean logit")
        sd = np.sqrt(latent.variance_logit.clip(lower=0))
        axes[1].fill_between(
            x,
            latent.mean_logit - sd,
            latent.mean_logit + sd,
            color="#0072B2",
            alpha=0.2,
            label="One posterior SD",
        )
        axes[1].set(ylabel="Latent activity log-odds", xlabel="Hours since fold start")
        axes[1].legend(fontsize=9, ncol=2)
        fig.suptitle(
            f"{first.recording}, fold {first.fold}: validation-selected count filter; +3 dB labels"
        )
        fig.savefig(out / "count_adaptation.png", dpi=150)
        plt.close(fig)
