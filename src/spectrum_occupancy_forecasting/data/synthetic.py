"""SYNTHETIC data generator for tests and CI only.

Nothing produced here may appear in research results or figures. It mimics the *structure* of the
NIST campaign (shuffled frequency sweeps, irregular revisits, outages, bursty dwells) so that
leakage, windowing and degradation code can be unit-tested without the real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.occupancy.detect import (
    DWELL_BINS,
    DWELL_S,
    dwell_labels,
    occupied_bins,
)
from spectrum_occupancy_forecasting.occupancy.power import dbm_to_mw, mw_to_dbm

EPOCH0 = 1_600_000_000.0  # 2020-09-13, inside the plausible-epoch range


@dataclass(frozen=True)
class SynthConfig:
    n_days: float = 0.5
    freqs_mhz: tuple[float, ...] = (2412.0, 2437.0, 2462.0, 5200.0)
    sweep_s: float = 4.0  # nominal time between sweep starts (jittered)
    busy_mean_s: float = 90.0
    idle_mean_s: float = 240.0
    p_occ_busy: float = 0.8
    p_occ_idle: float = 0.05
    noise_dbm: float = -85.0
    burst_dbm: float = -55.0
    n_outages: int = 2
    outage_s: float = 900.0
    seed: int = 0


def _regime_intervals(rng, span_s, busy_mean, idle_mean):
    """Alternating idle/busy sojourn boundaries; returns (boundaries, first_state_busy)."""
    bounds, t, busy = [], 0.0, bool(rng.integers(2))
    first = busy
    while t < span_s:
        t += rng.exponential(busy_mean if busy else idle_mean)
        bounds.append(t)
        busy = not busy
    return np.asarray(bounds), first


def generate(cfg: SynthConfig | None = None, site: str = "synth"):
    """Return ``(meta, bins_dbm)``: dwell metadata and an (n, DWELL_BINS) power array in dBm."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    span = cfg.n_days * 86400.0
    n_f = len(cfg.freqs_mhz)

    outages = [(s, s + cfg.outage_s) for s in rng.uniform(0, span - cfg.outage_s, cfg.n_outages)]
    regimes = {f: _regime_intervals(rng, span, cfg.busy_mean_s, cfg.idle_mean_s)
               for f in cfg.freqs_mhz}

    rows = []
    t_sweep = 0.0
    while t_sweep < span:
        order = rng.permutation(n_f)
        for k, fi in enumerate(order):
            t = t_sweep + k * (cfg.sweep_s / n_f) + rng.uniform(0, 0.05)
            if t >= span or any(a <= t < b for a, b in outages):
                continue
            rows.append((cfg.freqs_mhz[fi], t))
        t_sweep += cfg.sweep_s * rng.uniform(0.8, 1.2)

    meta = pd.DataFrame(rows, columns=["freq_mhz", "t_rel"]).sort_values("t_rel", ignore_index=True)
    n = len(meta)
    busy = np.empty(n, dtype=bool)
    for f, idx in meta.groupby("freq_mhz").groups.items():
        bounds, first = regimes[f]
        k = np.searchsorted(bounds, meta.loc[idx, "t_rel"].to_numpy(), side="right")
        busy[meta.index.get_indexer(idx)] = (k % 2 == 0) == first
    p = np.where(busy, cfg.p_occ_busy, cfg.p_occ_idle)
    active = rng.random(n) < p

    # Noise: exponential linear power around the noise floor (chi-square-like), then bursts.
    lin = dbm_to_mw(cfg.noise_dbm) * rng.exponential(1.0, size=(n, DWELL_BINS))
    for i in np.nonzero(active)[0]:
        for _ in range(rng.integers(1, 4)):
            length = int(rng.integers(2, 11))
            start = int(rng.integers(2, DWELL_BINS - length - 2))
            lin[i, start : start + length] += dbm_to_mw(cfg.burst_dbm + rng.normal(0, 3))
    bins_dbm = mw_to_dbm(lin)

    meta["site"] = site
    meta["t_start_s"] = EPOCH0 + meta.pop("t_rel")
    meta["dwell_s"] = DWELL_S
    return meta[["site", "freq_mhz", "t_start_s", "dwell_s"]], bins_dbm


def to_dwell_table(meta: pd.DataFrame, bins_dbm: np.ndarray, threshold_dbm: float) -> pd.DataFrame:
    """Label synthetic bins with the production detector and emit the canonical schema."""
    labels = dwell_labels(occupied_bins(bins_dbm, threshold_dbm))
    out = meta.copy()
    out["occ_any"] = labels.occ_any
    out["duty"] = labels.duty
    out["mean_dbm"] = mw_to_dbm(dbm_to_mw(bins_dbm).mean(axis=1))
    out["max_dbm"] = bins_dbm.max(axis=1)
    return out.sort_values("t_start_s", ignore_index=True)


def write_synthetic_blocks(path, n_days: float = 5.6, dt_s: float = 30.0, seed: int = 0):
    """TEST HELPER: a small SYNTHETIC block spectrogram in the layout written by
    ``scripts/extract_aerpaw_blocks.py`` (arrays ``times_s``, ``centres_mhz``, ``mean_dbm``,
    ``max_dbm``): 5 MHz blocks across 2400-2490 MHz with independent busy/idle regimes in the three
    Wi-Fi channel bands, plus two idle 5 GHz blocks. Never used for results."""
    rng = np.random.default_rng(seed)
    n = int(n_days * 86400.0 / dt_s)
    t = EPOCH0 + np.arange(n) * dt_s
    centres = np.concatenate([np.arange(2402.5, 2490.0, 5.0), [5202.5, 5242.5]])
    mean = -134.0 + rng.normal(0.0, 0.3, size=(n, len(centres)))
    for ch in (2412.0, 2437.0, 2462.0):
        bounds, first = _regime_intervals(rng, n * dt_s, 3600.0, 7200.0)
        k = np.searchsorted(bounds, t - EPOCH0, side="right")
        regime_busy = (k % 2 == 0) == first
        busy = np.where(regime_busy, rng.random(n) < 0.8, rng.random(n) < 0.03)
        band = np.abs(centres - ch) <= 10.0
        mean[np.ix_(busy, band)] += 6.0
    mx = mean + 5.0 + rng.normal(0.0, 0.3, size=mean.shape)
    np.savez(path, times_s=t, centres_mhz=centres, mean_dbm=mean.astype(np.float32),
             max_dbm=mx.astype(np.float32))
    return path
