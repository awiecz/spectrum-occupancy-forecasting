"""Figures for the sparse-regime study, labelled in plain language."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "fixed_history": "Two-hour history",
    "selected_history": "Best tuned history",
    "shrunk_history": "History shrunk toward the average",
    "climatology": "Always the average",
    "hstar": "Best simple baseline",
    "selected_learned": "Selected learned model",
    "ridge": "Ridge",
    "hgb": "Boosting",
    "resid_hgb": "Boosting on history residual",
}
SHORT = {
    "fixed_history": "Two-hour\nhistory",
    "climatology": "Always the\naverage",
    "shrunk_history": "History\nshrunk",
    "selected_learned": "Selected\nlearned",
    "hgb": "Boosting",
}
COLORS = {"CC1": "#0072B2", "LW1": "#D55E00"}
NODE_NAMES = {"CC1": "Urban (CC1)", "LW1": "Rural (LW1)"}
CELLS = (("CC1", "all_folds"), ("LW1", "later_folds"))


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def main_comparison(methods, scenario, path, synthetic=False):
    """Squared error of each strategy in the primary scenario for the two primary cells."""
    order = [
        "fixed_history",
        "climatology",
        "shrunk_history",
        "selected_learned",
        "hgb",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False, constrained_layout=True)
    for ax, (node, region) in zip(axes, CELLS, strict=True):
        frame = methods[
            (methods["node"] == node)
            & (methods["region"] == region)
            & (methods["scenario"] == scenario)
        ].set_index("method")
        values = [float(frame.loc[m, "mse"]) if m in frame.index else float("nan") for m in order]
        ax.bar(
            range(len(order)),
            values,
            color=[COLORS[node] if m in {"selected_learned", "hgb"} else "0.65" for m in order],
        )
        ax.set_xticks(range(len(order)), [SHORT[m] for m in order], fontsize=8)
        ax.set(
            title=f"{NODE_NAMES[node]}, {region.replace('_', ' ')}",
            ylabel="Squared error (lower is better)",
        )
        _style(ax)
    note = "SYNTHETIC REHEARSAL, not measurement evidence. " if synthetic else ""
    fig.suptitle(f"{note}Forecast error with only about 1 sweep in 100 visible", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gain_by_scenario(cells, delta, path, synthetic=False):
    """Learned gain over the best simple baseline (bars, with one-sided 95% lower bound) and
    over plain two-hour history (diamonds)."""
    base = cells[
        (cells["learned"] == "selected_learned")
        & (cells["loss"] == "squared")
        & cells.apply(lambda r: (r["node"], r["region"]) in CELLS, axis=1)
    ]
    best = base[base["comparator"] == "hstar"]
    plain = base[base["comparator"] == "fixed_history"]
    scenarios = list(dict.fromkeys(best["scenario"]))
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    width = 0.36
    for k, (node, _) in enumerate(CELLS):
        part = best[best["node"] == node].set_index("scenario").reindex(scenarios)
        ref = plain[plain["node"] == node].set_index("scenario").reindex(scenarios)
        x = [i + (k - 0.5) * width for i in range(len(scenarios))]
        ax.bar(
            x,
            part["gain"] * 100,
            width,
            color=COLORS[node],
            alpha=0.35,
            label=f"{NODE_NAMES[node]}: vs best simple baseline",
        )
        ax.scatter(x, part["lower95"] * 100, color=COLORS[node], marker="_", s=300, zorder=3)
        ax.scatter(
            x,
            ref["gain"] * 100,
            color="black",
            marker="D",
            s=22,
            zorder=4,
            label="Same, vs plain two-hour history" if k == 0 else None,
        )
    ax.axhline(0, color="0.2", lw=1)
    ax.axhline(delta * 100, color="0.4", lw=1, ls="--")
    ax.text(
        len(scenarios) - 0.5,
        delta * 100 + 3,
        "claim threshold",
        ha="right",
        fontsize=8,
        color="0.3",
    )
    ax.set_ylim(-125, 80)
    ax.set_xticks(range(len(scenarios)), [s.replace("_", "\n") for s in scenarios], fontsize=8)
    ax.set(ylabel="Error reduction (%)")
    prefix = "SYNTHETIC REHEARSAL. " if synthetic else ""
    ax.set_title(
        prefix + "Bars: gain over the best simple baseline. Dashes: one-sided 95% lower bound",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel(
        "Calm rural scenarios: the validation-chosen baseline was worse than plain "
        "history, so those bars overstate the learned model.",
        fontsize=7.5,
        color="0.3",
    )
    _style(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def interval_calibration(table, path, synthetic=False):
    """Share of outcomes inside the central 80% interval, by method."""
    methods = ["empirical", "conformal_hstar", "conformal_learned", "qhgb"]
    names = {
        "empirical": "Typical range",
        "conformal_hstar": "Baseline + past errors",
        "conformal_learned": "Learned + past errors",
        "qhgb": "Quantile boosting",
    }
    cells = [(n, r, s) for n, r in CELLS for s in sorted(set(table["scenario"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (value, title) in zip(
        axes,
        (
            ("coverage_80", "Coverage of the 80% interval"),
            ("mean_width_80", "Average interval width"),
        ),
        strict=True,
    ):
        for k, m in enumerate(methods):
            heights = []
            for node, region, scenario in cells:
                row = table[
                    (table["node"] == node)
                    & (table["region"] == region)
                    & (table["scenario"] == scenario)
                    & (table["method"] == m)
                ]
                heights.append(float(row[value].iloc[0]) if len(row) else float("nan"))
            ax.bar([i + (k - 1.5) * 0.2 for i in range(len(cells))], heights, 0.2, label=names[m])
        if value == "coverage_80":
            ax.axhline(0.8, color="0.3", lw=1, ls="--")
        ax.set_xticks(
            range(len(cells)), [f"{n}\n{s.replace('_', ' ')}" for n, _, s in cells], fontsize=7
        )
        ax.set_title(("SYNTHETIC. " if synthetic else "") + title, fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(methods, cells, intervals, scenario, delta, out, synthetic=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    main_comparison(methods, scenario, out / "main_comparison.png", synthetic)
    gain_by_scenario(cells, delta, out / "gain_by_scenario.png", synthetic)
    if intervals is not None and len(intervals):
        interval_calibration(intervals, out / "interval_calibration.png", synthetic)
