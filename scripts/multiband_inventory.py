"""Band inventory and pre-model power for the multi-band protocol (no learned model is fitted).

Eligibility uses each recording's first 24 hours only. Power inputs are descriptive properties
of the fixed two-hour-history baseline and the earlier single-band noise estimates; no learned
forecast is created or scored. The history baseline here approximates the repository's fixed
two-hour average and is for planning only; the registered run uses the repository implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multiband.yaml"
OUT = ROOT / "results/multiband"
CACHE = ROOT / "data/processed/aerpaw_{site}_blocks5MHz.npz"
SESSION_ZERO_END = {"CC1": 1644947500.0, "LW1": 1644753000.0}
LEVELS = {"optimistic": "se_p10", "typical": "se_median", "aggressive": "se_p90"}


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def band_power(site, cfg):
    """Session-zero power in 20 MHz bands: linear mean of aligned 5 MHz blocks, in dBm."""
    z = np.load(str(CACHE).format(site=site))
    t, centres, power = z["times_s"], z["centres_mhz"], z["mean_dbm"].astype(np.float64)
    keep = (t >= t[0]) & (t <= SESSION_ZERO_END[site])
    t, power = t[keep], power[keep]
    n = cfg["blocks_per_band"]
    usable = (len(centres) - cfg["band_alignment_index"]) // n * n
    start = cfg["band_alignment_index"]
    linear = 10 ** (power[:, start : start + usable] / 10)
    band = 10 * np.log10(linear.reshape(len(t), usable // n, n).mean(axis=2))
    fc = centres[start : start + usable].reshape(-1, n).mean(axis=1)
    return t, fc, band


def window_fractions(t, busy, start, width, need):
    """Busy fraction per non-overlapping window from ``start``; NaN below ``need`` sweeps."""
    edges = start + width * np.arange(int((t[-1] - start) // width) + 1)
    index = np.searchsorted(t, edges)
    cumulative = np.vstack([np.zeros((1, busy.shape[1])), np.cumsum(busy, axis=0)])
    counts = np.diff(index)
    sums = cumulative[index[1:]] - cumulative[index[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = sums / counts[:, None]
    fractions[counts < need] = np.nan
    return fractions


def inventory(site, cfg):
    """First-24-hour eligibility descriptors for every band."""
    t, fc, band = band_power(site, cfg)
    first_day = t < t[0] + cfg["calibration_s"]
    floor = np.percentile(band[first_day], 5, axis=0)
    busy = (band > floor + cfg["primary_offset_db"]).astype(np.float64)
    interval = float(np.median(np.diff(t[first_day])))
    need = cfg["minimum_coverage"] * cfg["window_s"] / interval
    day = window_fractions(t[first_day], busy[first_day], t[0], cfg["window_s"], need)
    rule = cfg["eligibility"]
    share = busy[first_day].mean(axis=0)
    spread = np.nanstd(day, axis=0)
    covered = (~np.isnan(day)).all(axis=0)
    table = pd.DataFrame(
        dict(
            site=site,
            centre_mhz=fc,
            first_day_busy_share=share,
            first_day_window_std=spread,
            first_day_windows=np.sum(~np.isnan(day), axis=0),
            eligible=covered
            & (share >= rule["min_busy_share"])
            & (share <= rule["max_busy_share"])
            & (spread >= rule["min_window_std"]),
        )
    )
    table["group"] = ((table.centre_mhz - table.centre_mhz.min()) // cfg["band_groups_mhz"]).astype(
        int
    )
    table["band_fold"] = table.group % cfg["band_folds"]
    return table, (t, busy, interval)


def reference_losses(t, busy, interval, cfg, bands):
    """Squared error of fixed two-hour history for the evaluated target windows."""
    width, lead, history = cfg["window_s"], cfg["lead_s"], cfg["history_s"]
    first_eval = t[0] + cfg["first_eval_s"]
    starts = first_eval + width * np.arange(int((t[-1] - first_eval) // width))
    targets = window_fractions(
        t, busy[:, bands], first_eval, width, cfg["minimum_coverage"] * width / interval
    )
    hist_end = starts - lead
    lo, hi = np.searchsorted(t, hist_end - history), np.searchsorted(t, hist_end)
    cumulative = np.vstack([np.zeros((1, len(bands))), np.cumsum(busy[:, bands], axis=0)])
    counts = (hi - lo).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        forecast = (cumulative[hi] - cumulative[lo]) / counts[:, None]
    forecast[counts < cfg["minimum_coverage"] * history / interval] = np.nan
    n = min(len(targets), len(forecast))
    return (targets[:n] - forecast[:n]) ** 2


def block_means(losses, windows_per_block):
    """Per-band mean loss in consecutive blocks; NaN-aware, partial blocks dropped."""
    n = losses.shape[0] // windows_per_block * windows_per_block
    shaped = losses[:n].reshape(-1, windows_per_block, losses.shape[1])
    return np.nanmean(shaped, axis=1)


def relative_cv(losses, windows_per_block, replicates, rng):
    """Bootstrap standard error of each band's mean loss, relative to that mean."""
    valid = ~np.isnan(losses).all(axis=1)
    blocks = block_means(losses[valid], windows_per_block)
    count = len(blocks)
    draws = rng.integers(0, count, size=(replicates, count))
    means = blocks[draws].mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return means.std(axis=0, ddof=1) / blocks.mean(axis=0), count


def mean_correlation(blocks):
    """Average pairwise Pearson correlation of per-block loss across bands."""
    corr = np.corrcoef(blocks.T)
    off = corr[np.triu_indices_from(corr, 1)]
    return float(np.nanmean(off)), float(np.nanmedian(off))


def pooled_se(se, rho):
    """Standard error of the equal-weight mean of correlated per-band estimates."""
    se = np.asarray(se, dtype=float)
    k = len(se)
    cross = se.sum() ** 2 - (se**2).sum()
    return float(np.sqrt(((se**2).sum() + rho * cross) / k**2))


def decision_power(gain, se, alpha, margin):
    """P(lower one-sided bound > 0 and point estimate >= margin) under a normal estimate."""
    return float(norm.cdf((gain - max(margin, norm.ppf(1 - alpha) * se)) / se))


def site_noise(cfg, site, windows_per_block, target_cv):
    """Per-level single-band standard errors observed for the target band on this recording."""
    table = pd.read_csv(ROOT / cfg["power"]["noise_source"])
    row = table[
        (table.recording == f"{site}_first")
        & (table.baseline == "fixed_history")
        & (table.block_windows == windows_per_block)
    ].iloc[0]
    return {level: float(row[column]) / target_cv for level, column in LEVELS.items()}


def analyse(cfg, replicates, seed):
    rng = np.random.default_rng(seed)
    windows_per_block = int(cfg["bootstrap"]["main_block_s"] // cfg["window_s"])
    corr_windows = int(cfg["power"]["correlation_block_s"] // cfg["window_s"])
    evaluation_rule = cfg["evaluation_eligibility"]
    tables, rows = [], []
    for site in cfg["sites"]:
        table, (t, busy, interval) = inventory(site, cfg)
        eligible = np.flatnonzero(table.eligible.to_numpy())
        target = int(np.argmin(np.abs(table.centre_mhz - cfg["target_mhz"])))
        bands = np.union1d(eligible, [target])
        losses = reference_losses(t, busy, interval, cfg, bands)
        target_col = int(np.flatnonzero(bands == target)[0])
        cv, nblocks = relative_cv(losses, windows_per_block, replicates, rng)
        mse = np.nanmean(losses, axis=0)
        evaluated = (mse >= evaluation_rule["min_reference_mse"]) & (
            nblocks >= evaluation_rule["min_blocks"]
        )
        usable = np.array([b in eligible and evaluated[i] for i, b in enumerate(bands)])
        table["evaluated"] = False
        table.loc[bands[usable], "evaluated"] = True
        table["reference_mse"] = np.nan
        table.loc[bands, "reference_mse"] = mse
        table["relative_cv"] = np.nan
        table.loc[bands, "relative_cv"] = cv
        noise = site_noise(cfg, site, windows_per_block, cv[target_col])
        corr_blocks = block_means(losses[:, usable], corr_windows)
        corr_blocks = corr_blocks[~np.isnan(corr_blocks).any(axis=1)]
        rho_mean, rho_median = mean_correlation(corr_blocks)
        tables.append(table)
        rows.append(
            dict(
                site=site,
                eligible_bands=int(table.eligible.sum()),
                evaluated_bands=int(usable.sum()),
                target_eligible=bool(table.eligible.iloc[target]),
                main_blocks=nblocks,
                correlation_blocks=int(len(corr_blocks)),
                mean_loss_correlation=rho_mean,
                median_loss_correlation=rho_median,
                target_relative_cv=float(cv[target_col]),
                **{f"single_band_se_{k}": v for k, v in noise.items()},
                _cv=cv[usable],
                _noise=noise,
            )
        )
    return pd.concat(tables, ignore_index=True), rows


def power_table(cfg, rows):
    spec = cfg["power"]
    adv = cfg["advancement"]
    out = []
    correlations = {}
    for row in rows:
        base = max(row["mean_loss_correlation"], 0.0)
        for multiplier in spec["correlation_multipliers"]:
            correlations.setdefault(f"observed x{multiplier}", {})[row["site"]] = base * multiplier
        for value in spec["fixed_correlations"]:
            correlations.setdefault(f"fixed {value}", {})[row["site"]] = value
    for label, by_site in correlations.items():
        for level in spec["noise_levels"]:
            site_se = {
                row["site"]: pooled_se(row["_cv"] * row["_noise"][level], by_site[row["site"]])
                for row in rows
            }
            overall = float(np.sqrt(sum(v**2 for v in site_se.values())) / len(site_se))
            z = norm.ppf(1 - adv["alpha"]) + norm.ppf(spec["target_power"])
            record = dict(correlation=label, noise=level, se_pooled=overall, mde=z * overall)
            record.update({f"se_{s}": v for s, v in site_se.items()})
            for gain in spec["true_gains"]:
                # Each site must also be positive; approximate by the weaker site's probability.
                joint = decision_power(gain, overall, adv["alpha"], adv["margin"])
                per_site = min(norm.cdf(gain / v) for v in site_se.values())
                record[f"power_gain_{gain:g}"] = joint * per_site
            out.append(record)
    return pd.DataFrame(out)


def figure(inventories, power, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"CC1": "#0072B2", "LW1": "#D55E00"}
    ax = axes[0]
    for site, frame in inventories.groupby("site"):
        ax.scatter(
            frame.centre_mhz[~frame.eligible],
            frame.first_day_window_std[~frame.eligible],
            s=6,
            color="0.8",
        )
        sel = frame[frame.eligible]
        ax.scatter(
            sel.centre_mhz,
            sel.first_day_window_std,
            s=14,
            color=colors[site],
            label=f"{site}: {len(sel)} eligible",
        )
    ax.set(
        xlabel="Band centre (MHz)",
        ylabel="First-day 15-minute busy-fraction SD",
        title="Eligible bands (first 24 hours only)",
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    gains = [c for c in power.columns if c.startswith("power_gain_")]
    xs = [float(c.split("_")[-1]) * 100 for c in gains]
    for level, style in (("optimistic", "--"), ("typical", "-"), ("aggressive", ":")):
        row = power[(power.correlation == "observed x1") & (power.noise == level)].iloc[0]
        ax.plot(xs, [row[c] for c in gains], style, marker="o", color="#0072B2", label=level)
    ax.axhline(0.8, color="0.4", lw=1, ls=":")
    ax.set(
        xlabel="True per-band MSE reduction (%)",
        ylabel="Probability of advancing",
        ylim=(0, 1.02),
        title="Pooled decision power (observed correlation)",
    )
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventories, rows = analyse(cfg, cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"])
    power = power_table(cfg, rows)
    inventories.to_csv(args.out_dir / "band_inventory.csv", index=False)
    power.to_csv(args.out_dir / "pre_model_power.csv", index=False)
    summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (args.out_dir / "dependence_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    figure(inventories, power, args.out_dir / "multiband_planning.png")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
