"""The three research figures. Static (matplotlib), light surface, validated categorical palette.

Fig 1 uses the DEVELOPMENT region only (never the held-out test region). Fig 2 and Fig 3 read the
result JSON written by ``scripts/run_window_dev.py`` (or the final-test equivalent). Each figure
also writes a CSV of the plotted values (the accessible "table view").

Palette: categorical slots 1-5 of the validated reference palette (adjacent CVD dE >= 9.1,
normal-vision dE >= 19.6); aqua/yellow/magenta fall below 3:1 contrast on the light surface, so
every series also has its own marker shape and there is a legend; sequential = blue ramp.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from spectrum_occupancy_forecasting.data.aerpaw import (  # noqa: E402
    band_table_from_blocks,
    label_band_table,
)
from spectrum_occupancy_forecasting.evaluation.splits import development_region_end  # noqa: E402
from spectrum_occupancy_forecasting.features.windows import (  # noqa: E402
    origin_grid,
    window_fraction,
)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQ)
TZ = "America/New_York"
CHANNELS = (2412.0, 2437.0, 2462.0)

LABELS = {
    "climatology": "Train mean (climatology)",
    "hour_of_day": "Hour-of-day prior",
    "recent_300s": "Last 5 min",
    "recent_1800s": "Last 30 min",
    "recent_7200s": "Last 2 h",
    "decayed": "Decayed history (15 min)",
    "ridge": "Ridge",
    "hgb": "Gradient boosting",
    "resid_ridge": "Last 2 h + ridge correction",
    "resid_hgb": "Last 2 h + GB correction",
    "ridge_aug": "Ridge (degradation-trained)",
    "hgb_aug": "Gradient boosting (degradation-trained)",
}
FAMILY = {  # colour = family of method, not rank
    "climatology": 0, "hour_of_day": 0,
    "recent_300s": 1, "recent_1800s": 1, "recent_7200s": 1, "decayed": 1,
    "ridge": 2, "hgb": 2, "resid_ridge": 2, "resid_hgb": 2,
}
FAMILY_NAMES = ["Priors", "History averages", "Learned models"]
FAMILY_COLORS = [BLUE, ORANGE, AQUA]


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 6,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def figure1_characterisation(npz_path: str | Path, out_dir: str | Path, x_db: float = 3.0):
    """Measurement characterisation, development region only."""
    style()
    out_dir = Path(out_dir)
    z = np.load(npz_path)
    t, f, mean_dbm = z["times_s"], z["centres_mhz"], z["mean_dbm"].astype(float)
    main = np.cumsum(np.concatenate([[0], np.diff(t) > 300])) == 0
    t, mean_dbm = t[main], mean_dbm[main]
    t0, t_last = float(t[0]), float(t[-1])
    end = development_region_end(t0, t_last)
    dev = t <= end
    day = 86400.0

    # (a) 2.4 GHz spectrogram: mean power over 5 min, minus each block's own noise floor
    band = (f >= 2400) & (f <= 2490)
    fb = f[band]
    lin = 10 ** (mean_dbm[dev][:, band] / 10)
    td = t[dev]
    nb = int((td[-1] - t0) // 300) + 1
    idx = ((td - t0) // 300).astype(int)
    sums = np.zeros((nb, band.sum()))
    cnt = np.bincount(idx, minlength=nb).astype(float)
    np.add.at(sums, idx, lin)
    with np.errstate(invalid="ignore", divide="ignore"):
        binned = 10 * np.log10(sums / cnt[:, None])
    floor = np.nanpercentile(binned, 5, axis=0)
    excess = binned - floor

    # busy labels for the channel-6 band, floor from the training period only (first 60 %)
    raw = band_table_from_blocks(npz_path, CHANNELS, 20.0, session=0)
    train_hi = t0 + 0.6 * (t_last - t0)
    thr = {c: float(np.percentile(raw.loc[(raw["freq_mhz"] == c) & (raw["t_start_s"] < train_hi),
                                          "mean_dbm"], 5)) + x_db for c in CHANNELS}
    lab = label_band_table(raw, thr, stat="mean_dbm")
    ch = lab[(lab["freq_mhz"] == 2437.0) & (lab["t_start_s"] <= end)]
    org = origin_grid(t0, end, 900.0, 900.0)
    frac, _ = window_fraction(ch["t_start_s"], ch["occ_any"], org, 900.0)
    ma = pd.Series(frac).rolling(8, min_periods=4).mean().to_numpy()  # 8 x 15 min = 2 h

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1], hspace=0.55, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(excess.T, aspect="auto", origin="lower", cmap=CMAP, vmin=0, vmax=12,
                    extent=[0, nb * 300 / day, fb[0] - 2.5, fb[-1] + 2.5], interpolation="nearest")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title("a   2.4 GHz band over the development period: power above each block's "
                  "noise floor")
    for c in CHANNELS:
        ax1.axhline(c, color=INK2, lw=0.6, ls=(0, (2, 3)))
        ax1.text(nb * 300 / day + 0.03, c, f"Wi-Fi {int((c - 2407) / 5)}", va="center", fontsize=8,
                 color=INK2)
    ax1.grid(False)
    cb = fig.colorbar(im, ax=ax1, pad=0.06, fraction=0.025)
    cb.set_label("dB above floor (5 min mean)")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    xo = (org - t0) / day
    ax2.plot(xo, frac, color=SEQ[2], lw=1.0, label="busy fraction, 15 min windows")
    ax2.plot(xo, ma, color=BLUE, lw=2.0, label="2 h moving average")
    for k, (a, b) in enumerate([(2, 3), (3, 4), (4, (end - t0) / day)]):
        ax2.axvspan(a, b, color=INK2, alpha=0.06 if k % 2 == 0 else 0.12, lw=0)
        ax2.text((a + b) / 2, 0.93, f"fold {k + 1} (scored)", ha="center", fontsize=8, color=INK2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(f"busy fraction (> floor + {x_db:g} dB)")
    ax2.set_xlabel("days since start (Tue 8 Feb 2022, 12:50 local)")
    ax2.set_title("b   Channel-6 band (2437 MHz): activity drifts down; folds 1-3 are scored")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8)
    ax2.set_xlim(0, nb * 300 / day)

    ax3 = fig.add_subplot(gs[1:, 1])
    ts = pd.to_datetime(ch["t_start_s"].to_numpy(), unit="s", utc=True).tz_convert(TZ)
    dfh = pd.DataFrame({"date": ts.date, "hour": ts.hour, "busy": ch["occ_any"].to_numpy(float)})
    piv = dfh.pivot_table(index="date", columns="hour", values="busy", aggfunc="mean")
    im3 = ax3.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP, vmin=0, vmax=0.5,
                     interpolation="nearest")
    ax3.set_yticks(range(len(piv)))
    ax3.set_yticklabels([pd.Timestamp(d).strftime("%a %d") for d in piv.index], fontsize=8)
    ax3.set_xticks([0, 6, 12, 18, 23])
    ax3.set_xlabel("hour of day (local)")
    ax3.set_title("c   busy share by day and hour")
    ax3.grid(False)
    cb3 = fig.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.18, fraction=0.05)
    cb3.set_label("busy share of sweeps")

    ax4 = fig.add_subplot(gs[2, 0])
    y = (ch["occ_any"].to_numpy(float) - ch["occ_any"].mean())
    lags = np.array([1, 2, 5, 10, 30, 100, 300, 1000])
    ac = [float(np.mean(y[:-k] * y[k:]) / y.var()) for k in lags]
    ax4.plot(lags * 19 / 60, ac, color=BLUE, marker="o", ms=6, mfc=SURFACE, mew=1.6)
    ax4.set_xscale("log")
    ax4.set_xlabel("lag (minutes)")
    ax4.set_ylabel("autocorrelation of\nbusy / idle sweeps")
    ax4.set_ylim(0, 0.3)
    ax4.set_title(f"d   single-sweep labels are noisy: lag-1 autocorrelation is only {ac[0]:.2f}")
    fig.suptitle("AERPAW node CC1, February 2022: what the measurements look like "
                 "(development region only; test region not shown)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig1_characterisation")
    pd.DataFrame({"lag_min": lags * 19 / 60, "autocorr": ac}).to_csv(
        out_dir / "fig1_autocorrelation.csv", index=False)
    piv.to_csv(out_dir / "fig1_busy_share_by_day_hour.csv")


# --------------------------------------------------------------------------------------------
def figure2_methods(results_json: str | Path, out_dir: str | Path, region: str = "dev"):
    """Baseline / model comparison across forecast windows (pooled rolling-origin folds)."""
    style()
    out_dir = Path(out_dir)
    R = json.loads(Path(results_json).read_text())["windows"]
    wins = [("300s", "5-minute window"), ("900s", "15-minute window"),
            ("3600s", "60-minute window")]
    wins = [w for w in wins if w[0] in R]  # draw whichever window lengths the run produced
    if not wins:
        raise ValueError("results contain none of the 300s / 900s / 3600s windows")
    order = [m for m in FAMILY]  # priors, then history averages, then learned (insertion order)
    fig, axes = plt.subplots(2, len(wins), figsize=(3.85 * len(wins), 6.8), squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    rows = []
    for j, (key, title) in enumerate(wins):
        clean = R[key]["clean"]
        for i, (metric, xl) in enumerate([
            ("mss", "MSE skill vs train mean (95 % CI)"),
            ("r2_within_fold", {"dev": "R$^2$ within each dev day",
                                "test": "R$^2$ within the test region"}.get(
                                    region, "R$^2$ within each fold")),
        ]):
            ax = axes[i, j]
            for k, m in enumerate(order):
                v = clean[m][metric]
                col = FAMILY_COLORS[FAMILY[m]]
                if metric == "mss" and m != "climatology":
                    lo, hi = clean[m]["mss_ci95"]
                    ax.plot([lo, hi], [k, k], color=col, lw=1.4, solid_capstyle="round")
                ax.plot(v, k, "o", color=col, ms=7, mec=SURFACE, mew=1.2, zorder=3)
                rows.append({"window": key, "metric": metric, "method": m, "value": v,
                             "ci_lo": clean[m]["mss_ci95"][0] if metric == "mss" else None,
                             "ci_hi": clean[m]["mss_ci95"][1] if metric == "mss" else None})
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_xlabel(xl)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([LABELS[m] for m in order] if j == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(title)
                ax.set_xlim(-1.2, 1.0)
            else:
                ax.set_xlim(-1.8, 1.0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=n)
               for n, c in zip(FAMILY_NAMES, FAMILY_COLORS, strict=True)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.075),
               fontsize=9)
    if region == "dev":
        title = ("Development region (3 rolling folds): a 2-hour moving average matches the "
                 "learned models")
        note = ("Top row is inflated by level drift (dev days quieter than training); the bottom "
                "row removes the level and shows how little within-day variation any method "
                "explains.")
    elif region == "synthetic":
        title = "SYNTHETIC DEMO ONLY: 15-minute forecast"
        note = "Workflow demonstration on synthetic data;\nnot evidence for the real-data study."
    elif region == "lw1":
        title = ("Second node LW1 (rural, 3 rolling folds, design frozen on CC1): method "
                 "comparison, channel-6 band")
        note = ("Design fixed on CC1; selection overlap disclosed in docs/lw1_results.md; "
                "30-min and decayed memories score better than the "
                "2 h average on point estimates, and learned models match but do not beat them; "
                "see docs/lw1_results.md.")
    else:
        title = ("Held-out test region (Sun evening to Tue midday, one block): method "
                 "comparison, AERPAW CC1, channel-6 band")
        note = ("Evaluated once with the frozen design. The level rises on Monday: shorter "
                "memories "
                "score better than the 2 h average on point estimates (intervals overlap) and "
                "gradient boosting is unstable; see docs/test_results.md.")
    fig.suptitle(title, x=0.01, ha="left", y=0.99, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.035, note, fontsize=8, color=INK2)
    _save(fig, out_dir, "fig2_method_comparison")
    pd.DataFrame(rows).to_csv(out_dir / "fig2_method_comparison.csv", index=False)


# --------------------------------------------------------------------------------------------
def figure3_robustness(
    results_json: str | Path, out_dir: str | Path, window: str = "900s", region: str = "dev"
):
    """RMSE of the 15-min forecast as predictor inputs are degraded (clean-trained unless _aug)."""
    style()
    out_dir = Path(out_dir)
    R = json.loads(Path(results_json).read_text())["windows"][window]
    if not R.get("robustness"):
        print(f"no robustness results for window {window}; skipping figure 3")
        return
    rob, clim_rmse = R["robustness"], R["clean"]["climatology"]["rmse"]
    series = [("recent_7200s", BLUE, "o"), ("decayed", ORANGE, "s"), ("ridge", AQUA, "^"),
              ("hgb", YELLOW, "D"), ("hgb_aug", MAGENTA, "v")]
    panels = [
        ("a   Sweeps dropped at random", "share of sweeps dropped",
         [("clean", "0"), ("dropout_0.5", "50 %"), ("dropout_0.9", "90 %"),
          ("dropout_0.99", "99 %")]),
        ("b   Evidence delayed", "age of the newest usable sweep (min)",
         [("clean", "0"), ("delay_60s", "1"), ("delay_600s", "10"), ("delay_1800s", "30"),
          ("delay_3600s", "60"), ("delay_7200s", "120")]),
        ("c   Contiguous outages (25 % of time)", "outage length (min)",
         [("clean", "none"), ("outage_900s_x25pct", "15"), ("outage_3600s_x25pct", "60")]),
    ]
    ymax = 0.22
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    rows = []
    for ax, (title, xl, scen) in zip(axes, panels, strict=True):
        offscale: dict[int, int] = {}
        xs = np.arange(len(scen))
        ax.axhline(clim_rmse, color=INK2, lw=1.0, ls=(0, (4, 3)))
        for m, col, mk in series:
            v = np.array([rob[s][m]["rmse"] for s, _ in scen])
            sd = np.array([rob[s][m].get("rmse_sd", 0.0) for s, _ in scen])
            ax.plot(xs, np.minimum(v, ymax), color=col, marker=mk, ms=6.5, mec=SURFACE, mew=1.0,
                    label=LABELS[m])
            if (sd > 0).any():  # spread over random degradation masks (stochastic scenarios)
                ax.errorbar(xs, np.minimum(v, ymax), yerr=sd, fmt="none", ecolor=col,
                            elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=1)
            for x, val, (s, _) in zip(xs, v, scen, strict=True):
                if val > ymax:  # stagger labels when several series are off scale at one x
                    k = offscale.get(x, 0)
                    offscale[x] = k + 1
                    ax.annotate(f"{val:.2f}", (x, ymax), xytext=(0, 4 + 9 * k),
                                textcoords="offset points", ha="center", fontsize=7.5, color=col)
                rows.append({"panel": title[0], "scenario": s, "method": m, "rmse": val,
                             "rmse_sd_over_masks": float(sd[list(xs).index(x)])})
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in scen])
        ax.set_xlabel(xl)
        ax.set_title(title)
        ax.set_ylim(0.04, ymax + 0.02)
        ax.set_xlim(-0.3, len(scen) - 0.7)
    axes[0].set_ylabel("RMSE of forecast busy fraction (lower is better)")
    axes[0].text(0.02, clim_rmse + 0.003, "train mean (climatology)", color=INK2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.09),
               fontsize=8.5)
    prefix = {"test": "Held-out test region: r", "lw1": "Node LW1: r"}.get(region, "R")
    fig.suptitle(f"{prefix}obustness of the 15-minute forecast to degraded inputs "
                 "(mean of 5 random masks, bars = SD; off-scale values printed)", x=0.01, ha="left",
                 y=1.02, fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig3_robustness")
    pd.DataFrame(rows).to_csv(out_dir / "fig3_robustness.csv", index=False)


# --------------------------------------------------------------------------------------------
def figure4_transfer(results_json: str | Path, out_dir: str | Path, window: str = "900s"):
    """Transfer penalty (MSE vs in-node training, log scale) of models trained on another node."""
    style()
    out_dir = Path(out_dir)
    R = json.loads(Path(results_json).read_text())["directions"]
    models = [("climatology", "Train mean (climatology)"), ("hour_of_day", "Hour-of-day prior"),
              ("decayed", "Decayed history (15 min)"), ("ridge", "Ridge"),
              ("hgb", "Gradient boosting"), ("resid_ridge", "Last 2 h + ridge correction"),
              ("resid_hgb", "Last 2 h + GB correction")]
    panels = [("CC1_to_LW1", "a   Train on CC1, forecast LW1 (primary)"),
              ("LW1_to_CC1", "b   Train on LW1, forecast CC1 (exploratory)")]
    arms = [("transfer", "Trained on the other node only", ORANGE, "o"),
            ("joint", "Trained on both nodes", BLUE, "s")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    rows = []
    for ax, (key, title) in zip(axes, panels, strict=True):
        ratios = R[key]["windows"][window]["ratios"]
        for k, (m, _) in enumerate(models):
            for j, (arm, _, col, mk) in enumerate(arms):
                v, lo, hi = ratios[arm][m]
                y = k + (j - 0.5) * 0.28
                ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round")
                ax.plot(v, y, mk, color=col, ms=6.5, mec=SURFACE, mew=1.0, zorder=3)
                rows.append({"direction": key, "window": window, "model": m, "arm": arm,
                             "mse_ratio_vs_in_node": v, "ci_lo": lo, "ci_hi": hi})
        ax.axvline(1.0, color=INK2, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 12)
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
        ax.set_xlabel("MSE vs in-node training (1 = no penalty)")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([lab for _, lab in models])
        ax.tick_params(axis="y", length=0)
        ax.set_title(title)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker=mk, ls="", color=col, ms=7, label=lab)
               for _, lab, col, mk in arms]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    fig.suptitle("Cross-node transfer, 15-minute forecast: history averages are unaffected; "
                 "learned models carry the source node", x=0.01, ha="left", y=1.02, fontsize=11,
                 fontweight="bold")
    _save(fig, out_dir, "fig4_transfer")
    pd.DataFrame(rows).to_csv(out_dir / "fig4_transfer.csv", index=False)
