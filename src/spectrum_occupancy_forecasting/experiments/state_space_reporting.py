"""Presentation for the fixed two-family study; numerical reports reuse audited utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .adaptive_reporting import regenerate as adaptive_regenerate
from .residual_state_space import STRATEGIES, candidates


def figures(predictions, summary, uncertainty, out):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    prefix = "SYNTHETIC — " if predictions.synthetic.any() else "Exploratory — "
    recordings = sorted(predictions.recording.unique())
    fig, axes = plt.subplots(
        len(recordings), 1, figsize=(10, 8), constrained_layout=True, squeeze=False
    )
    for ax, recording in zip(axes[:, 0], recordings, strict=True):
        group = predictions[
            (predictions.recording == recording) & (predictions.scenario == "clean")
        ]
        for family, color in (("level", "#0072B2"), ("trend", "#009E73")):
            part = group[(group.strategy == family) & (group.method == "adaptive")].sort_values(
                "origin_s"
            )
            x = (part.origin_s - part.origin_s.min()) / 3600
            ax.plot(x, part.prediction, label=family, color=color)
        ax.plot(x, part.fixed_history, label="Two-hour history", color="#555555", alpha=0.7)
        ax.plot(x, part.y, label="Observed", color="#D55E00", alpha=0.7)
        ax.set(title=recording, ylabel="Busy fraction", ylim=(-0.02, 1.02))
    axes[0, 0].legend(ncol=4, fontsize=9)
    axes[-1, 0].set_xlabel("Hours since first scored origin; +3 dB, 15-minute targets")
    fig.suptitle(prefix + "forecasts issued before outcomes")
    fig.savefig(out / "forecasts.png", dpi=150)
    plt.close(fig)
    clean = summary[(summary.scenario == "clean") & (summary.method == "adaptive")]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    (clean.pivot(index="strategy", columns="recording", values="mss_vs_fixed") * 100).plot.bar(
        ax=ax
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(5, linestyle="--", color="black")
    ax.set(
        ylabel="MSE reduction vs two-hour history (%)",
        xlabel="Model family",
        title=prefix + "negative values mean higher error",
    )
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(out / "clean_skill.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for ax, recording in zip(axes, recordings, strict=True):
        part = summary[(summary.recording == recording) & (summary.method == "adaptive")]
        for strategy, data in part.groupby("strategy"):
            ax.plot(data.scenario, 1 - data.mss_vs_fixed, marker=".", label=strategy)
        ax.axhline(1.1, color="black", linestyle="--")
        ax.set(title=recording, ylabel="MSE / same-scenario history MSE")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    axes[0].legend()
    fig.suptitle(prefix + "sensor-specific delivered feedback; paired seed losses")
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    for ax, recording in zip(axes, recordings, strict=True):
        part = uncertainty[
            (uncertainty.recording == recording)
            & (uncertainty.method == "adaptive")
            & (uncertainty.scenario == "clean")
            & (uncertainty.reference == "fixed")
        ]
        for strategy, data in part.groupby("strategy"):
            color = "#0072B2" if strategy == "level" else "#009E73"
            ax.plot(
                data.block_s / 3600, data.difference_lo, color=color, marker=".", label=strategy
            )
            ax.plot(
                data.block_s / 3600, data.difference_hi, color=color, marker=".", linestyle="--"
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(title=recording, xlabel="Block length (hours)", ylabel="Paired MSE difference")
        ax.set_xticks([2, 4, 8, 12])
    axes[0].legend()
    fig.suptitle(prefix + "intervals suppressed below four blocks; <10 blocks is weak")
    fig.savefig(out / "block_sensitivity.png", dpi=150)
    plt.close(fig)


def regenerate(source, out, cfg):
    adaptive_regenerate(
        source,
        out,
        cfg,
        candidate_factory=candidates,
        strategies=STRATEGIES,
        figure_function=figures,
    )
    state_path = Path(source) / "filter_states.parquet"
    if not state_path.exists():
        return
    states = pd.read_parquet(state_path)
    predictions = pd.read_parquet(Path(source) / "predictions.parquet")
    recordings = sorted(states.recording.unique())
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    plot_rows = []
    for row, recording in enumerate(recordings):
        trace = states[
            (states.recording == recording)
            & (states.scenario == "clean")
            & (states.coefficient == 0.01)
        ].copy()
        start = trace.origin_s.min()
        trace["hours"] = (trace.origin_s - start) / 3600
        for family, group in trace.groupby("family"):
            axes[row, 0].plot(group.hours, group.predicted_residual, label=family)
        trace = trace.assign(panel="candidate residual").rename(
            columns={"predicted_residual": "value"}
        )
        plot_rows.append(trace[["recording", "family", "origin_s", "hours", "panel", "value"]])
        issued = predictions[
            (predictions.recording == recording)
            & (predictions.scenario == "clean")
            & (predictions.method == "adaptive")
        ].copy()
        issued["hours"] = (issued.origin_s - start) / 3600
        for family, group in issued.groupby("strategy"):
            axes[row, 1].plot(group.hours, group.correction, label=family)
        issued = issued.assign(panel="issued correction").rename(
            columns={"strategy": "family", "correction": "value"}
        )
        plot_rows.append(issued[["recording", "family", "origin_s", "hours", "panel", "value"]])
        for ax in axes[row]:
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set(
                title=recording, ylabel="Busy-fraction residual", xlabel="Hours after calibration"
            )
    axes[0, 0].legend()
    axes[0, 1].legend()
    prefix = "SYNTHETIC — " if predictions.synthetic.any() else "Exploratory — "
    fig.suptitle(prefix + "candidate states (coefficient 0.01) versus actually issued corrections")
    fig.savefig(Path(out) / "residual_adaptation.png", dpi=150)
    plt.close(fig)
    pd.concat(plot_rows, ignore_index=True).to_csv(
        Path(out) / "residual_adaptation.csv", index=False
    )
