"""Degradations of the *observation stream available to the predictor*.

These only ever produce a boolean visibility mask over observations (plus an extra delay that
the feature builder applies). Evaluation targets and labels are never modified, and a hidden
observation is *missing*, never encoded as zero power or as an idle channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Degradation:
    dropout_p: float = 0.0  # each observation independently hidden with this probability
    outage_len_s: float = 0.0  # length of each contiguous outage
    outage_frac: float = 0.0  # approx. fraction of the time span covered by outages
    delay_s: float = 0.0  # extra latency: observations become usable only delay_s after they end

    def __post_init__(self):
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if not 0.0 <= self.outage_frac < 1.0:
            raise ValueError("outage_frac must be in [0, 1)")
        if self.outage_frac > 0 and self.outage_len_s <= 0:
            raise ValueError("outage_len_s must be > 0 when outage_frac > 0")
        if self.delay_s < 0 or self.outage_len_s < 0:
            raise ValueError("delay_s and outage_len_s must be >= 0")

    @property
    def is_identity(self) -> bool:
        return self.dropout_p == 0 and self.outage_frac == 0 and self.delay_s == 0


def visibility_mask(t_s, deg: Degradation, seed: int = 0) -> np.ndarray:
    """Boolean mask over observations: True = visible to the predictor.

    ``deg.delay_s`` is not applied here (it depends on the target time); pass it to the feature
    builder. Deterministic for a given ``seed``.
    """
    t = np.asarray(t_s, dtype=float)
    rng = np.random.default_rng(seed)
    visible = np.ones(len(t), dtype=bool)
    if deg.dropout_p > 0:
        visible &= rng.random(len(t)) >= deg.dropout_p
    if deg.outage_frac > 0 and len(t) > 0:
        t0, t1 = float(t.min()), float(t.max())
        span = t1 - t0
        n_out = int(round(deg.outage_frac * span / deg.outage_len_s))
        if n_out > 0 and span > deg.outage_len_s:
            starts = rng.uniform(t0, t1 - deg.outage_len_s, n_out)
            for s in starts:
                visible &= ~((t >= s) & (t < s + deg.outage_len_s))
    return visible


def sweep_visibility_mask(t_s, deg: Degradation, seed: int = 0) -> np.ndarray:
    """Per-observation mask where all rows sharing a timestamp (one sweep, several channels) are
    seen or lost together, as when a whole sweep is dropped."""
    t = np.asarray(t_s, dtype=float)
    uniq, inv = np.unique(t, return_inverse=True)
    return visibility_mask(uniq, deg, seed)[inv]
