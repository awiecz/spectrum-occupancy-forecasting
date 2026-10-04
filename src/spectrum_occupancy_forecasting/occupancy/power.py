"""Power-unit helpers.

Averaging dBm values arithmetically is wrong for physical average power; always convert to
linear power (mW), average, then convert back. Use :func:`mean_power_dbm` for that.
"""

from __future__ import annotations

import numpy as np


def dbm_to_mw(dbm):
    """dBm -> milliwatts."""
    return np.power(10.0, np.asarray(dbm, dtype=float) / 10.0)


def mw_to_dbm(mw):
    """Milliwatts -> dBm. Zero power maps to -inf; negative power is invalid (NaN)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.asarray(mw, dtype=float))


def mean_power_dbm(dbm, axis=None):
    """Physical mean power (dBm) of dBm samples: convert to linear, average, convert back."""
    return mw_to_dbm(np.mean(dbm_to_mw(dbm), axis=axis))
