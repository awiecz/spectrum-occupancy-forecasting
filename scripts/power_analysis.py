"""Statistical power of the repository's forecast comparisons, from saved predictions only.

No model is fitted and no data is read beyond the saved operational-study forecasts, so this
analysis opens no new evidence and adds no candidate to the experiment registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from spectrum_occupancy_forecasting.evaluation.power import (
    ALPHA,
    POWER,
    Z,
    relative_se,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/operational_learning/run"
OUT = ROOT / "results/power_analysis"
RECORDINGS = ("CC1_first", "LW1_first", "LW1_later")
WINDOWS_PER_DAY = 96
TARGET = 0.05
BLOCKS = (1, 16, 32, 48)  # windows of 15 minutes: independent, 4 h, 8 h (primary), 12 h
PRIMARY = 32
MIN_BLOCKS = 4  # the repository omits block intervals below four blocks
LEVELS = {"optimistic": "p10", "typical": "median", "aggressive": "p90"}


def primary_block(n):
    """Eight-hour blocks, or four-hour blocks when eight hours leave fewer than four blocks."""
    return PRIMARY if n / PRIMARY >= MIN_BLOCKS else 16


def scored_origins():
    clean = pd.read_parquet(RUN / "predictions.parquet")
    clean = clean[(clean.scenario == "clean") & (clean.method == "fixed_history")]
    return {
        name: group.drop_duplicates("origin_s").sort_values("origin_s").set_index("origin_s")
        for name, group in clean.groupby("recording")
    }


def candidate_forecasts(name):
    frame = pd.read_parquet(RUN / f"{name}_candidate_forecasts.parquet")
    return frame.pivot(index="origin_s", columns="candidate", values="prediction")


def tail_statistics(squared):
    ordered = np.sort(squared)[::-1]
    top = max(1, int(round(0.1 * len(ordered))))
    centered = squared - squared.mean()
    lag = [float(np.corrcoef(centered[:-k], centered[k:])[0, 1]) for k in (1, 2, 4)]
    return dict(
        top_decile_error_share=float(ordered[:top].sum() / ordered.sum()),
        heavy_tail_effective_n=float(squared.sum() ** 2 / (squared**2).sum()),
        loss_autocorr_lag1=lag[0],
        loss_autocorr_lag2=lag[1],
        loss_autocorr_lag4=lag[2],
    )


def analyse(replicates, seed, family="all"):
    rng = np.random.default_rng(seed)
    scored = scored_origins()
    rows, tails, candidate_rows = [], [], []
    for name in RECORDINGS:
        frame = scored[name]
        y = frame.y.to_numpy()
        predictions = candidate_forecasts(name).reindex(frame.index)
        learned = predictions.drop(columns="count_history")
        if family == "count":
            learned = learned[[c for c in learned.columns if c.startswith("count_")]]
        for baseline in ("fixed_history", "selected_history"):
            base = frame[baseline].to_numpy()
            base_loss = (y - base) ** 2
            if baseline == "fixed_history":
                tails.append(dict(recording=name, n=len(y), **tail_statistics(base_loss)))
            difference = base_loss[:, None] - (y[:, None] - learned.to_numpy()) ** 2
            active = (np.abs(difference).max(axis=0) > 0) & learned.notna().all().to_numpy()
            difference = difference[:, active]
            gain = difference.mean(axis=0) / base_loss.mean()
            for length in BLOCKS:
                realistic = relative_se(difference, base_loss, length, replicates, rng)
                rows.append(
                    dict(
                        recording=name,
                        baseline=baseline,
                        block_windows=length,
                        n_origins=len(y),
                        days=len(y) / WINDOWS_PER_DAY,
                        baseline_mse=float(base_loss.mean()),
                        n_candidates=int(active.sum()),
                        n_blocks=len(y) / length,
                        valid=len(y) / length >= MIN_BLOCKS or length == 1,
                        primary=length == primary_block(len(y)),
                        se_p10=float(np.quantile(realistic, 0.1)),
                        se_median=float(np.median(realistic)),
                        se_p90=float(np.quantile(realistic, 0.9)),
                    )
                )
                if length == primary_block(len(y)):
                    candidate_rows.append(
                        pd.DataFrame(
                            dict(
                                recording=name,
                                baseline=baseline,
                                candidate=learned.columns[active],
                                relative_gain=gain,
                                se=realistic,
                                z=gain / realistic,
                            )
                        )
                    )
    table = pd.DataFrame(rows)
    for level, column in LEVELS.items():
        table[f"mde_{level}"] = Z * table[f"se_{column}"]
        table[f"days_for_target_{level}"] = table.days * (table[f"mde_{level}"] / TARGET) ** 2
    return table, pd.DataFrame(tails), pd.concat(candidate_rows, ignore_index=True)


def power_at(gain, se, threshold=0.0):
    """Probability that the observed gain's one-sided lower confidence bound exceeds threshold."""
    return norm.cdf((gain - threshold) / se - norm.ppf(1 - ALPHA))


def power_curves(table):
    primary = table[(table.baseline == "fixed_history") & table.primary]
    grid = np.linspace(0, 0.3, 61)
    rows = []
    for _, row in primary.iterrows():
        for kind in LEVELS:
            se = row[f"se_{LEVELS[kind]}"]
            rows += [
                dict(recording=row.recording, kind=kind, true_gain=g, power=float(power_at(g, se)))
                for g in grid
            ]
    curves = pd.DataFrame(rows)
    joint = []
    for kind in LEVELS:
        for gain in (0.05, 0.10, 0.20, 0.30):
            ses = primary[f"se_{LEVELS[kind]}"].to_numpy()
            powers = power_at(gain, ses)
            joint.append(
                dict(
                    kind=kind,
                    true_gain=gain,
                    **{r: float(p) for r, p in zip(primary.recording, powers, strict=True)},
                    all_three=float(np.prod(powers)),
                )
            )
    return curves, pd.DataFrame(joint)


def figure(table, curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CC1_first": "#0072B2", "LW1_first": "#D55E00", "LW1_later": "#009E73"}
    labels = {
        "CC1_first": "Urban first session (4.6 days)",
        "LW1_first": "Rural first session (2.5 days)",
        "LW1_later": "Rural later session (0.8 days)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    for name in RECORDINGS:
        for kind, style in (("typical", "-"), ("optimistic", "--")):
            sub = curves[(curves.recording == name) & (curves.kind == kind)]
            ax.plot(sub.true_gain * 100, sub.power, style, color=colors[name], lw=2)
        ax.plot([], [], color=colors[name], lw=2, label=labels[name])
    ax.axvline(5, color="0.4", lw=1)
    ax.axhline(0.8, color="0.4", lw=1, ls=":")
    ax.set(
        xlabel="True MSE reduction against fixed two-hour history (%)",
        ylabel="Power to show any improvement",
        ylim=(0, 1.02),
        xlim=(0, 30),
    )
    ax.set_title("Power by true gain (solid: typical, dashed: optimistic candidate)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    primary = table[(table.baseline == "fixed_history") & table.primary]
    ax = axes[1]
    x = np.arange(len(RECORDINGS))
    ax.bar(x - 0.2, primary.mde_optimistic * 100, 0.38, color="#9ec5e3", label="Optimistic")
    ax.bar(x + 0.2, primary.mde_typical * 100, 0.38, color="#0072B2", label="Typical")
    ax.axhline(5, color="0.4", lw=1)
    ax.text(
        1.0,
        4.4,
        "5% promotion rule",
        ha="center",
        va="top",
        fontsize=8,
        color="0.2",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    ax.set_xticks(x, ["Urban\nfirst", "Rural\nfirst", "Rural\nlater"])
    ax.set(ylabel="Smallest reliably detectable gain (%)")
    ax.set_title(
        "Minimum detectable gain (80% power; 8 h blocks, 4 h for later rural)", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sampling_floor():
    """Binomial noise if snapshots were independent; a heuristic, not an irreducible bound."""
    rows = []
    for name, frame in scored_origins().items():
        p = frame.fixed_history.clip(0, 1)
        floor = (p * (1 - p) / frame.target_count).mean()
        mse = ((frame.y - frame.fixed_history) ** 2).mean()
        rows.append(
            dict(
                recording=name,
                fixed_history_mse=mse,
                independent_binomial_floor=floor,
                ratio=mse / floor,
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--candidates",
        choices=["all", "count"],
        default="all",
        help="all: count models plus history variants; count: the 24 count-model settings only",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table, tails, candidates = analyse(args.replicates, args.seed, args.candidates)
    curves, joint = power_curves(table)
    floor = sampling_floor()
    table.to_csv(args.out_dir / "minimum_detectable_gain.csv", index=False)
    tails.to_csv(args.out_dir / "loss_structure.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_evidence.csv", index=False)
    curves.to_csv(args.out_dir / "power_curves.csv", index=False)
    joint.to_csv(args.out_dir / "joint_power.csv", index=False)
    floor.to_csv(args.out_dir / "sampling_floor.csv", index=False)
    figure(table, curves, args.out_dir / "power.png")
    (args.out_dir / "settings.json").write_text(
        json.dumps(
            dict(
                alpha=ALPHA,
                power=POWER,
                target_gain=TARGET,
                primary_block_windows=PRIMARY,
                replicates=args.replicates,
                seed=args.seed,
                source=str(RUN.relative_to(ROOT)),
            ),
            indent=2,
        )
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
