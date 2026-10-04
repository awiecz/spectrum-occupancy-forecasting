"""Small diagnostics used by the sanity checks: dependence and the value of old evidence."""

from __future__ import annotations

import numpy as np


def lag_autocorr(x, lag: int = 1) -> float:
    """Sample autocorrelation of ``x`` at ``lag`` (NaN if constant or too short)."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean()
    var = float(np.mean(z**2))
    if lag < 1 or len(z) <= lag or var == 0:
        return float("nan")
    return float(np.mean(z[:-lag] * z[lag:]) / var)


def effective_n(x, lag: int = 1) -> float:
    """AR(1) effective number of independent samples: ``n (1 - rho) / (1 + rho)``.

    ``rho`` is the lag-``lag`` autocorrelation clipped to [0, 0.99]. A rough guide to how much
    independent information a strongly autocorrelated window series holds; it is an
    approximation and says nothing about non-AR(1) structure such as a daily cycle.
    """
    n = len(x)
    rho = lag_autocorr(x, lag)
    if not np.isfinite(rho):
        return float("nan")
    rho = min(max(rho, 0.0), 0.99)
    return float(n * (1.0 - rho) / (1.0 + rho))


def half_life(delays, skill) -> float:
    """First delay at which ``skill`` falls to half of its value at ``delays[0]``.

    Linear interpolation between grid points; NaN if the skill is not positive at zero delay,
    ``inf`` if it never falls to half within the grid. Delays must be increasing.
    """
    d = np.asarray(delays, dtype=float)
    s = np.asarray(skill, dtype=float)
    if len(d) != len(s) or len(d) < 2 or not np.all(np.diff(d) > 0) or s[0] <= 0:
        return float("nan")
    target = 0.5 * s[0]
    for i in range(1, len(s)):
        if s[i] <= target:
            if s[i - 1] == s[i]:
                return float(d[i])
            frac = (s[i - 1] - target) / (s[i - 1] - s[i])
            return float(d[i - 1] + frac * (d[i] - d[i - 1]))
    return float("inf")
