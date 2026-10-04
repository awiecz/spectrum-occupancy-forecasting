"""Figures rebuilt from v2 prediction artifacts; no refitting or raw recording access."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(predictions, tables, out, *, synthetic=False, confirmation=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = tables["summary"]
    nodes = sorted(summary["node"].unique())
    title = "SYNTHETIC REHEARSAL ONLY" if synthetic else "Exploratory v2: CC1/LW1 data reused"
    if confirmation:
        title = "CC2 frozen-design confirmation: same campaign"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name, data):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"{name}.png", dpi=160)
        fig.savefig(out / f"{name}.pdf")
        data.to_csv(out / f"{name}.csv", index=False)
        plt.close(fig)

    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 3 * len(nodes)), squeeze=False)
    primary = predictions[(predictions["task"] == "primary") & (predictions["scenario"] == "clean")]
    for node, ax in zip(nodes, axes[:, 0], strict=True):
        part = primary[primary["node"] == node]
        for method, label in (
            ("selected_history", "Selected history"),
            ("selected_learned", "Selected learned"),
        ):
            series = part[part["method"] == method].sort_values("target_start_s")
            t = pd.to_datetime(series["target_start_s"], unit="s", utc=True)
            ax.plot(t, series["prediction"], label=label, lw=1)
        series = part[part["method"] == "selected_history"].sort_values("target_start_s")
        ax.plot(
            pd.to_datetime(series["target_start_s"], unit="s", utc=True),
            series["y"],
            label="Observed future sweep fraction",
            color="black",
            alpha=0.65,
            lw=0.8,
        )
        ax.set(title=node, ylabel="Busy sweep fraction", xlabel="Target start (UTC)", ylim=(0, 1))
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    save(fig, "predictions", primary)

    fig, axes = plt.subplots(1, len(nodes), figsize=(5 * len(nodes), 3.5), squeeze=False)
    lead = summary[(summary["task"].str.startswith("lead_")) | (summary["task"] == "primary")]
    lead = lead[lead["scenario"] == "clean"]
    for node, ax in zip(nodes, axes[0], strict=True):
        for method in ("selected_history", "selected_learned", "fixed_history", "climatology"):
            part = lead[(lead["node"] == node) & (lead["method"] == method)].sort_values("lead_s")
            ax.plot(part["lead_s"] / 60, part["rmse"], "o-", label=method.replace("_", " "))
        ax.set(title=node, xlabel="Lead before 15-minute target (min)", ylabel="RMSE")
        ax.legend(fontsize=8)
    save(fig, "lead_times", lead)

    fig, axes = plt.subplots(1, len(nodes), figsize=(5 * len(nodes), 3.5), squeeze=False)
    skill_data = []
    for node, ax in zip(nodes, axes[0], strict=True):
        climate = lead[(lead["node"] == node) & (lead["method"] == "climatology")].set_index(
            "lead_s"
        )["mse"]
        for method in ("selected_history", "selected_learned", "fixed_history"):
            part = (
                lead[(lead["node"] == node) & (lead["method"] == method)]
                .sort_values("lead_s")
                .copy()
            )
            part["skill_vs_climatology"] = 1 - part["mse"] / part["lead_s"].map(climate)
            skill_data.append(part)
            ax.plot(
                part["lead_s"] / 60,
                part["skill_vs_climatology"],
                "o-",
                label=method.replace("_", " "),
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set(
            title=node,
            xlabel="Lead before 15-minute target (min)",
            ylabel="MSE skill vs training climatology",
        )
        ax.legend(fontsize=8)
    save(fig, "lead_skill", pd.concat(skill_data, ignore_index=True))

    label = summary[(summary["task"].str.startswith("label_")) | (summary["task"] == "primary")]
    label = label[
        (label["scenario"] == "clean")
        & (label["method"].isin(["selected_history", "selected_learned"]))
    ]
    fig, axes = plt.subplots(1, len(nodes), figsize=(5 * len(nodes), 3.5), squeeze=False)
    for node, ax in zip(nodes, axes[0], strict=True):
        for method in ("selected_history", "selected_learned"):
            part = label[(label["node"] == node) & (label["method"] == method)].sort_values(
                "offset_db"
            )
            ax.plot(part["offset_db"], part["rmse"], "o-", label=method.replace("_", " "))
        ax.set(title=node, xlabel="Offset above first-day noise floor (dB)", ylabel="RMSE")
        ax.legend(fontsize=8)
    save(fig, "label_sensitivity", label)

    ablation = summary[
        (summary["task"].str.startswith("ablation_")) | (summary["task"] == "primary")
    ]
    ablation = ablation[
        (ablation["scenario"] == "clean") & (ablation["method"] == "selected_learned")
    ]
    fig, axes = plt.subplots(1, len(nodes), figsize=(6 * len(nodes), 3.5), squeeze=False)
    for node, ax in zip(nodes, axes[0], strict=True):
        part = ablation[ablation["node"] == node].sort_values("variant")
        ax.barh(part["variant"].str.replace("history_", "+ "), part["mss_vs_selected"])
        ax.axvline(0, color="black", lw=0.7)
        ax.set(title=node, xlabel="MSE skill vs validation-selected history")
    save(fig, "ablations", ablation)

    degraded = summary[summary["task"] == "primary"]
    fig, axes = plt.subplots(len(nodes), 3, figsize=(12, 3.2 * len(nodes)), squeeze=False)
    methods = ["selected_history", "selected_learned", "ridge", "hgb", "ridge_aug", "hgb_aug"]
    for i, node in enumerate(nodes):
        for j, (prefix, label_name) in enumerate(
            (
                ("dropout_", "Sweep dropout (%)"),
                ("delay_", "Evidence delay (min)"),
                ("outage_", "Outage length (min; nominal 25% span)"),
            )
        ):
            ax = axes[i, j]
            for method in methods:
                part = degraded[
                    (degraded["node"] == node)
                    & (degraded["method"] == method)
                    & (
                        degraded["scenario"].str.startswith(prefix)
                        | (degraded["scenario"] == "clean")
                    )
                ].copy()
                suffix = part["scenario"].str.removeprefix(prefix).str.removesuffix("s")
                part["x"] = pd.to_numeric(suffix, errors="coerce").fillna(0)
                part["x"] *= 100 if prefix == "dropout_" else 1 / 60
                part = part.sort_values("x")
                ax.plot(part["x"], part["rmse"], "o-", label=method, lw=1, ms=3)
            ax.set(title=node, xlabel=label_name, ylabel="RMSE")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    save(fig, "degradations", degraded)

    boot = tables["bootstrap"]
    boot = boot[
        (boot["task"] == "primary")
        & (boot["scenario"] == "clean")
        & (boot["method"] == "selected_learned")
    ]
    fig, axes = plt.subplots(1, len(nodes), figsize=(5 * len(nodes), 3.5), squeeze=False)
    for node, ax in zip(nodes, axes[0], strict=True):
        part = boot[boot["node"] == node].sort_values("block_s")
        point = degraded[
            (degraded["node"] == node)
            & (degraded["method"] == "selected_learned")
            & (degraded["scenario"] == "clean")
        ]["paired_mse_difference"].iloc[0]
        for row in part.itertuples():
            x = row.block_s / 3600
            if pd.notna(row.difference_lo) and pd.notna(row.difference_hi):
                ax.plot([x, x], [row.difference_lo, row.difference_hi], color="tab:blue")
            ax.plot(x, point, "o", color="tab:orange" if row.weak_evidence else "tab:blue")
            ax.annotate(
                f"{row.n_blocks} blocks",
                (x, point),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set(
            title=node,
            xlabel="Bootstrap block length (hours)",
            ylabel="MSE(learned) - MSE(selected history)",
        )
    save(fig, "bootstrap_sensitivity", boot)
