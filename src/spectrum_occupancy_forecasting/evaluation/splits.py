"""Chronological train / dev / test splits with an embargo gap."""

from __future__ import annotations

import numpy as np


def chronological_split(
    t_s,
    fractions: tuple[float, float, float] = (0.6, 0.15, 0.25),
    embargo_s: float = 3600.0,
):
    """Split target times into train / dev / test index arrays (indices into ``t_s``).

    Boundaries are the sample-count quantiles of ``t_s``. The earlier segment of each boundary
    ends ``embargo_s`` before it, so the gap between consecutive segments is >= ``embargo_s``.
    Set ``embargo_s`` >= the longest feature look-back window (plus any label horizon).
    """
    t = np.asarray(t_s, dtype=float)
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1")
    if embargo_s < 0:
        raise ValueError("embargo_s must be >= 0")
    b1 = np.quantile(t, fractions[0])
    b2 = np.quantile(t, fractions[0] + fractions[1])
    train = np.nonzero(t < b1 - embargo_s)[0]
    dev = np.nonzero((t >= b1) & (t < b2 - embargo_s))[0]
    test = np.nonzero(t >= b2)[0]
    return train, dev, test


def assert_no_leakage(t_s, train, dev, test, embargo_s: float) -> None:
    """Raise AssertionError unless the segments are ordered and separated by ``embargo_s``."""
    t = np.asarray(t_s, dtype=float)
    segs = [("train", train), ("dev", dev), ("test", test)]
    for (na, a), (nb, b) in zip(segs[:-1], segs[1:], strict=True):
        if len(a) == 0 or len(b) == 0:
            raise AssertionError(f"empty segment: {na} or {nb}")
        gap = t[b].min() - t[a].max()
        if gap < embargo_s - 1e-9:
            raise AssertionError(f"{na}->{nb} gap {gap:.1f}s < embargo {embargo_s:.1f}s")
    if set(train) & set(dev) or set(dev) & set(test) or set(train) & set(test):
        raise AssertionError("segments overlap")


def development_region_end(
    t_first: float,
    t_last: float,
    fractions: tuple[float, float, float] = (0.6, 0.15, 0.25),
    step_s: float = 900.0,
) -> float:
    """End time of the development region (train + dev); everything after it is the test region.

    Computed exactly as the experiment scripts do (sample-count quantiles of a regular origin
    grid), so figures and experiments agree on what the held-out test region is.
    """
    grid = np.arange(t_first, t_last - step_s + 1e-9, step_s)
    _, dev, _ = chronological_split(grid, fractions=fractions, embargo_s=0.0)
    return float(grid[dev].max())
