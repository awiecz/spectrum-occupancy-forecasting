"""Block bootstrap respecting temporal dependence (resample whole blocks, e.g. days)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_ids(t_s, block_s: float = 86400.0) -> np.ndarray:
    """Integer block id per sample (default: one block per 24 h of wall-clock time)."""
    t = np.asarray(t_s, dtype=float)
    return np.floor((t - t.min()) / block_s).astype(int)


def block_bootstrap_ci(
    y,
    p,
    blocks,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for ``metric(y, p)`` with a percentile block-bootstrap CI."""
    y = np.asarray(y)
    p = np.asarray(p)
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    groups = [np.nonzero(blocks == u)[0] for u in uniq]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        v = metric(y[idx], p[idx])
        if np.isfinite(v):
            stats.append(v)
    point = float(metric(y, p))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
