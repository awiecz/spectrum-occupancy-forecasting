"""Scores for quantile and interval forecasts of a busy fraction in [0, 1]."""

from __future__ import annotations

import numpy as np


def monotone(quantiles):
    """Clip to [0, 1] and sort each row so quantile levels never cross."""
    return np.sort(np.clip(np.asarray(quantiles, dtype=float), 0.0, 1.0), axis=1)


def pinball(y, quantiles, levels):
    """Mean pinball (quantile) loss per level; ``quantiles`` has shape (n, len(levels))."""
    y = np.asarray(y, dtype=float)[:, None]
    q = np.asarray(quantiles, dtype=float)
    tau = np.asarray(levels, dtype=float)[None, :]
    error = y - q
    return np.mean(np.maximum(tau * error, (tau - 1.0) * error), axis=0)


def mean_pinball(y, quantiles, levels):
    return float(np.mean(pinball(y, quantiles, levels)))


def interval_score(y, lower, upper, alpha):
    """Winkler interval score for a central (1 - alpha) interval; lower is better."""
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    width = upper - lower
    below = (2.0 / alpha) * np.maximum(lower - y, 0.0)
    above = (2.0 / alpha) * np.maximum(y - upper, 0.0)
    return float(np.mean(width + below + above))


def coverage(y, lower, upper):
    """Share of outcomes inside the closed interval; ties on a boundary count as covered."""
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))
