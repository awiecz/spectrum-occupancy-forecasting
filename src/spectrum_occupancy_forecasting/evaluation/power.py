"""Moving-block bootstrap and detectable-effect arithmetic shared by the power analyses.

Neighbouring 15-minute windows are alike, so uncertainty is estimated by re-sampling whole
blocks of consecutive windows rather than single windows.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

ALPHA = 0.05
POWER = 0.80
Z = norm.ppf(1 - ALPHA) + norm.ppf(POWER)


def block_indices(n, length, replicates, rng):
    """Moving-block bootstrap indices, shape (replicates, n)."""
    count = -(-n // length)
    starts = rng.integers(0, n - length + 1, size=(replicates, count))
    return (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[:, :n]


def relative_se(losses_difference, baseline_loss, length, replicates, rng):
    """Block-bootstrap standard error of mean(difference) / mean(baseline loss).

    ``losses_difference`` has shape (n, k): k candidate loss-difference series.
    """
    n = len(baseline_loss)
    idx = block_indices(n, min(length, n), replicates, rng)
    ratio = losses_difference[idx].mean(axis=1) / baseline_loss[idx].mean(axis=1)[:, None]
    return ratio.std(axis=0, ddof=1)
