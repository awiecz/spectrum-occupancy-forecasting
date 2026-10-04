"""Window targets: the busy fraction of the next ``window_s`` seconds after a forecast origin.

The label for origin ``o`` uses observations with ``o <= t < o + window_s``. Predictor inputs only
ever use observations that ended before ``o`` (minus any delay), so inputs and targets cannot
overlap in time. Labels always come from the intact stream, never from a degraded one.
"""

from __future__ import annotations

import numpy as np


def origin_grid(t_first: float, t_last: float, stride_s: float, window_s: float) -> np.ndarray:
    """Origins ``t_first, t_first+stride, ...`` whose whole window ends by ``t_last``."""
    if stride_s <= 0 or window_s <= 0:
        raise ValueError("stride_s and window_s must be positive")
    return np.arange(t_first, t_last - window_s + 1e-9, stride_s)


def window_fraction(
    t_obs,
    busy,
    origins,
    window_s: float,
    min_coverage: float = 0.8,
    nominal_dt_s: float = 19.0,
):
    """Busy fraction and observation count in ``[o, o + window_s)`` for each origin ``o``.

    Windows with fewer than ``min_coverage * window_s / nominal_dt_s`` observations (recording
    gaps, session edges) get NaN, so they are dropped rather than scored on partial evidence.
    """
    t = np.asarray(t_obs, dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    b = np.asarray(busy, dtype=float)[order]
    cs = np.concatenate([[0.0], np.cumsum(b)])
    o = np.asarray(origins, dtype=float)
    lo = np.searchsorted(t, o, side="left")
    hi = np.searchsorted(t, o + window_s, side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (cs[hi] - cs[lo]) / n
    ok = n >= min_coverage * window_s / nominal_dt_s
    return np.where(ok, frac, np.nan), n
