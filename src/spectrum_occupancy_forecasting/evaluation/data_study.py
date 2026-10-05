"""Descriptive data study with auditable plot sources; never fits a forecasting model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks, sessions
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData
from spectrum_occupancy_forecasting.features.windows import window_fraction

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def temporal_pairs(values, eligible, lag):
    """Return only pairs whose whole intervening window sequence is eligible."""
    values = np.asarray(values, float)
    eligible = np.asarray(eligible, bool) & np.isfinite(values)
    if lag < 1:
        raise ValueError("lag must be positive")
    idx = [i for i in range(lag, len(values)) if eligible[i - lag : i + 1].all()]
    return values[np.asarray(idx, int) - lag], values[np.asarray(idx, int)]


def pair_correlation(values, eligible, lag):
    a, b = temporal_pairs(values, eligible, lag)
    reason = (
        "insufficient_pairs"
        if len(a) < 3
        else ("constant_data" if np.std(a) == 0 or np.std(b) == 0 else None)
    )
    return {
        "lag_minutes": lag * 15,
        "pairs": len(a),
        "status": reason or "available",
        "correlation": None if reason else float(np.corrcoef(a, b)[0, 1]),
    }


def window_table(t, busy, dt, cfg):
    width = cfg["window_s"]
    starts = np.arange(t[0], t[-1] - width + 1e-9, width)
    fractions, counts = window_fraction(t, busy, starts, width, cfg["minimum_coverage"], dt)
    return pd.DataFrame(
        {
            "start_s": starts,
            "end_s": starts + width,
            "count": counts,
            "fraction": fractions,
            "eligible": np.isfinite(fractions),
            "period": np.where(
                starts + width <= t[0] + cfg["calibration_s"], "calibration", "post_calibration"
            ),
        }
    )


def analyze_node(path, node, cfg, out):
    out = Path(out)
    with np.load(path) as z:
        t, f, power = z["times_s"], z["centres_mhz"], z["mean_dbm"]
        if (
            power.shape != (len(t), len(f))
            or len(t) < 2
            or not np.isfinite(t).all()
            or not np.isfinite(power).all()
            or not np.isfinite(f).all()
            or not np.all(np.diff(t) > 0)
            or not np.all(np.diff(f) > 0)
        ):
            raise ValueError(
                "invalid block arrays: finite ordered unique times/frequencies required"
            )
        ids = sessions(t)
        inventory = []
        for session in np.unique(ids):
            ts = t[ids == session]
            quantiles = np.quantile(np.diff(ts), [0.1, 0.5, 0.9]) if len(ts) > 1 else [np.nan] * 3
            inventory.append(
                {
                    "session": int(session),
                    "start_s": ts[0],
                    "end_s": ts[-1],
                    "start_utc": pd.Timestamp(ts[0], unit="s", tz="UTC").isoformat(),
                    "end_utc": pd.Timestamp(ts[-1], unit="s", tz="UTC").isoformat(),
                    "sweeps": len(ts),
                    "elapsed_hours": (ts[-1] - ts[0]) / 3600,
                    **dict(
                        zip(
                            ("interval_p10_s", "interval_p50_s", "interval_p90_s"),
                            quantiles,
                            strict=True,
                        )
                    ),
                }
            )
        pd.DataFrame(inventory).to_csv(out / f"{node}_sessions.csv", index=False)
        medians = np.quantile(power[ids == 0], [0.1, 0.5, 0.9], axis=0)
        pd.DataFrame(
            {"mhz": f, "p10_dbm": medians[0], "median_dbm": medians[1], "p90_dbm": medians[2]}
        ).to_csv(out / f"{node}_spectrum.csv", index=False)
    raw = band_table_from_blocks(path, cfg["channels_mhz"], site=node, session=0)
    n = NodeData(raw, cfg, cfg["primary_offset_db"])
    ch = n.ch
    times = ch.t_start_s.to_numpy()
    p = ch.mean_dbm.to_numpy()
    busy = ch.occ_any.to_numpy(float)
    windows = window_table(times, busy, n.dt, cfg)
    windows.to_csv(out / f"{node}_windows.csv", index=False)
    hist = []
    edges = np.linspace(p.min(), p.max() + 1e-9, 61)
    cal = times + ch.dwell_s.to_numpy() <= n.t0 + cfg["calibration_s"]
    for period, mask in (("calibration", cal), ("post_calibration", ~cal)):
        count, _ = np.histogram(p[mask], edges)
        hist.extend(
            {
                "period": period,
                "left_dbm": lo,
                "right_dbm": hi,
                "count": int(c),
                "total": int(mask.sum()),
            }
            for lo, hi, c in zip(edges[:-1], edges[1:], count, strict=True)
        )
    pd.DataFrame(hist).to_csv(out / f"{node}_power_histogram.csv", index=False)
    example = (times >= n.t0 + 86400) & (times < n.t0 + 90000)
    pd.DataFrame({"time_s": times[example], "power_dbm": p[example], "busy": busy[example]}).to_csv(
        out / f"{node}_example.csv", index=False
    )
    daily = ch.assign(
        local_day=pd.to_datetime(times, unit="s", utc=True)
        .tz_convert(cfg["timezone"])
        .strftime("%Y-%m-%d"),
        period=np.where(cal, "calibration", "post_calibration"),
    )
    stats = (
        daily.groupby(["local_day", "period"])
        .agg(
            sweeps=("occ_any", "size"),
            busy_sweeps=("occ_any", "sum"),
            prevalence=("occ_any", "mean"),
        )
        .reset_index()
    )
    # Window quantiles are descriptive, not independent-sample uncertainty estimates.
    windows["local_day"] = (
        pd.to_datetime(windows.start_s, unit="s", utc=True)
        .dt.tz_convert(cfg["timezone"])
        .dt.strftime("%Y-%m-%d")
    )
    for i, row in stats.iterrows():
        w = windows[(windows.local_day == row.local_day) & (windows.period == row.period)]
        stats.loc[i, "eligible_windows"] = w.eligible.sum()
        stats.loc[i, "excluded_windows"] = (~w.eligible).sum()
        for q, label in ((0.1, "p10"), (0.5, "median"), (0.9, "p90")):
            stats.loc[i, f"fraction_{label}"] = w.fraction.quantile(q)
    stats.to_csv(out / f"{node}_daily.csv", index=False)
    post = windows.period.eq("post_calibration").to_numpy()
    valid = windows.eligible.to_numpy() & post
    acf = [pair_correlation(windows.fraction, valid, k) for k in (1, 2, 4, 8)]
    pd.DataFrame(acf).to_csv(out / f"{node}_autocorrelation.csv", index=False)
    a, b = temporal_pairs(windows.fraction, valid, 1)
    changes = b - a
    bins = np.linspace(-1, 1, 41)
    counts, _ = np.histogram(changes, bins)
    pd.DataFrame({"left": bins[:-1], "right": bins[1:], "count": counts}).to_csv(
        out / f"{node}_changes.csv", index=False
    )
    sensitivity = []
    for offset in cfg["offsets_db"]:
        labels = p > n.floors[cfg["target_mhz"]] + offset
        w = window_table(times, labels, n.dt, cfg)
        for period, mask in (("calibration", cal), ("post_calibration", ~cal)):
            sensitivity.append(
                {
                    "offset_db": offset,
                    "period": period,
                    "sweeps": int(mask.sum()),
                    "busy_sweeps": int(labels[mask].sum()),
                    "prevalence": float(labels[mask].mean()),
                    "eligible_windows": int(w[w.period == period].eligible.sum()),
                }
            )
    pd.DataFrame(sensitivity).to_csv(out / f"{node}_sensitivity.csv", index=False)
    tail = times[-1] - (windows.end_s.iloc[-1] if len(windows) else times[0])
    return {
        "node": node,
        "total_sweeps": len(t),
        "sessions": len(inventory),
        "elapsed_days": float((t[-1] - t[0]) / 86400),
        "analyzed_session": 0,
        "analyzed_sweeps": len(times),
        "analyzed_days": float((times[-1] - times[0]) / 86400),
        "noise_floor_dbm": n.floors[cfg["target_mhz"]],
        "threshold_dbm": n.thresholds[cfg["target_mhz"]],
        "nominal_interval_s": n.dt,
        "calibration_end_s": n.t0 + cfg["calibration_s"],
        "complete_windows": len(windows),
        "eligible_windows": int(windows.eligible.sum()),
        "excluded_low_coverage": int((~windows.eligible).sum()),
        "excluded_partial_tail": int(tail > 1e-9),
        "partial_tail_s": float(tail),
        "periods": {
            period: {"sweeps": int(mask.sum()), "busy_prevalence": float(busy[mask].mean())}
            for period, mask in (("calibration", cal), ("post_calibration", ~cal))
        },
        "autocorrelation": acf,
        "adjacent_pairs": len(changes),
        "absolute_change_p90": float(np.quantile(np.abs(changes), 0.9)) if len(changes) else None,
    }


def make_figures(out, cfg, *, synthetic=False):
    """All figure values come from saved summary/CSV files; no recording is reread."""
    out = Path(out)
    summaries = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    names = {"CC1": "Urban site (CC1)", "LW1": "Rural site (LW1)"}
    colors = {"CC1": "#0072B2", "LW1": "#D55E00"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def read(node, kind):
        return pd.read_csv(out / f"{node}_{kind}.csv")

    def save(fig, name, title):
        fig.suptitle(("SYNTHETIC DEMONSTRATION — " if synthetic else "") + title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"{name}.png", dpi=140, metadata={"Software": "data_study"})
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5))
    for ax, s in zip(axes, summaries, strict=True):
        records = read(s["node"], "sessions")
        t0 = records.start_s.min()
        for row in records.itertuples():
            if row.start_s == row.end_s:
                ax.scatter((row.start_s - t0) / 86400, 0, color="0.7", s=45)
            ax.plot(
                [(row.start_s - t0) / 86400, (row.end_s - t0) / 86400],
                [0, 0],
                lw=10,
                color=colors[s["node"]] if row.session == 0 else "0.7",
            )
        ax.axvspan(0, 1, color="#E69F00", alpha=0.25, label="First-day calibration")
        ax.set(title=names[s["node"]], yticks=[], xlabel="Days from first recorded sweep")
        ax.legend(loc="upper right")
    save(fig, "recording_quality", "Recording timeline: colored session is used for analysis")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, s in zip(axes, summaries, strict=True):
        d = read(s["node"], "spectrum")
        ax.fill_between(
            d.mhz,
            d.p10_dbm,
            d.p90_dbm,
            color=colors[s["node"]],
            alpha=0.2,
            label="10th–90th percentile",
        )
        ax.plot(d.mhz, d.median_dbm, color=colors[s["node"]], label="Median")
        ax.axvline(2437, color="black", ls="--", label="Forecast band center")
        ax.set(title=names[s["node"]], ylabel="Received power (dBm)")
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Frequency (MHz); existing coarse blocks, analyzed session")
    save(fig, "spectrum_context", "The forecasting band is one small part of the recorded spectrum")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for axs, s in zip(axes, summaries, strict=True):
        node = s["node"]
        h = read(node, "power_histogram")
        for period, color in (("calibration", "#E69F00"), ("post_calibration", colors[node])):
            d = h[h.period == period]
            axs[0].step(
                (d.left_dbm + d.right_dbm) / 2,
                d["count"] / d.total * 100,
                color=color,
                label=period.replace("_", " "),
            )
        axs[0].axvline(s["threshold_dbm"], color="black", ls="--", label="+3 dB threshold")
        axs[0].set(title=names[node], xlabel="Band power (dBm)", ylabel="Snapshots per bin (%)")
        axs[0].legend(fontsize=7)
        e = read(node, "example")
        minutes = (e.time_s - s["calibration_end_s"]) / 60
        axs[1].plot(minutes, e.power_dbm, lw=0.7, color=colors[node])
        axs[1].axhline(s["threshold_dbm"], color="black", ls="--")
        axs[1].set(
            title="First post-calibration hour", xlabel="Minutes after calibration", ylabel="dBm"
        )
        axs[2].scatter(minutes, e.busy, s=4, color="0.6", label="Snapshot busy label")
        w = read(node, "windows")
        w = w[(w.start_s >= s["calibration_end_s"]) & (w.end_s <= s["calibration_end_s"] + 3600)]
        if len(w):
            axs[2].stairs(
                w.fraction,
                np.r_[w.start_s, w.end_s.iloc[-1]] / 60 - s["calibration_end_s"] / 60,
                color=colors[node],
                lw=2,
                label="15-minute fraction",
            )
        else:
            axs[2].text(
                0.5,
                0.5,
                "No complete post-calibration window",
                transform=axs[2].transAxes,
                ha="center",
                fontsize=8,
            )
        axs[2].set(
            xlabel="Minutes after calibration", ylabel="Busy label / fraction", ylim=(-0.05, 1.1)
        )
        axs[2].legend(fontsize=7)
    save(
        fig,
        "measurement_to_label",
        "A power threshold creates labels; window averages create the target",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    for axs, s in zip(axes, summaries, strict=True):
        w = read(s["node"], "windows")
        dates = pd.to_datetime(w.start_s, unit="s", utc=True).dt.tz_convert(cfg["timezone"])
        axs[0].plot(
            (w.start_s - w.start_s.min()) / 86400, w.fraction * 100, color=colors[s["node"]], lw=1
        )
        axs[0].axvspan(0, 1, color="#E69F00", alpha=0.2)
        axs[0].set(
            title=names[s["node"]],
            xlabel="Days from analyzed session start",
            ylabel="Busy fraction (%)",
            ylim=(0, 100),
        )
        # Missing cells stay masked; an hour requires all four valid quarter-hours.
        h = w.assign(day=dates.dt.strftime("%m-%d"), hour=dates.dt.hour)
        groups = h.groupby(["day", "hour"]).fraction.agg(["mean", "count"])
        groups.loc[groups["count"] < 4, "mean"] = np.nan
        grid = groups["mean"].unstack().reindex(columns=range(24))
        im = axs[1].imshow(grid * 100, vmin=0, vmax=100, aspect="auto", cmap="cividis")
        axs[1].set(
            yticks=range(len(grid)),
            yticklabels=grid.index,
            xticks=[0, 6, 12, 18, 23],
            xlabel="Local hour (America/New_York)",
            title="Blank = incomplete evidence",
        )
        fig.colorbar(im, ax=axs[1], label="Busy fraction (%)", fraction=0.03)
    save(
        fig,
        "activity_patterns",
        "Activity changes over time; a few days cannot establish a daily cycle",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s in summaries:
        node = s["node"]
        d = read(node, "autocorrelation")
        axes[0].plot(d.lag_minutes, d.correlation, "o-", label=names[node], color=colors[node])
        h = read(node, "changes")
        if h["count"].sum():
            axes[1].step(
                (h.left + h.right) / 2 * 100,
                h["count"] / h["count"].sum() * 100,
                label=names[node],
                color=colors[node],
            )
    axes[0].set(xlabel="Lag (minutes)", ylabel="Correlation", ylim=(-1, 1))
    axes[1].set(
        xlabel="Change between adjacent windows (percentage points)", ylabel="Pairs per bin (%)"
    )
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No eligible adjacent pairs",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
    save(
        fig,
        "persistence_changes",
        "Post-calibration persistence and changes, without bridging missing windows",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for s in summaries:
        d = read(s["node"], "sensitivity")
        d = d[d.period == "post_calibration"]
        ax.plot(
            d.offset_db, d.prevalence * 100, "o-", color=colors[s["node"]], label=names[s["node"]]
        )
    ax.set(
        xlabel="Detection offset above first-day floor (dB)",
        ylabel="Busy snapshots (%)",
        xticks=cfg["offsets_db"],
    )
    ax.legend()
    save(
        fig,
        "label_sensitivity",
        "The same received powers produce different labels as the threshold changes",
    )
