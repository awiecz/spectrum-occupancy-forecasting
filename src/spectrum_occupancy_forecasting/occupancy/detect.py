"""Derive *detected occupancy* from band-power bins.

Everything here is **detected activity at one sensor**: a bin above a power threshold. It is not
transmitter activity, interference at another receiver, or permission to transmit.

A dwell is a contiguous run of ``DWELL_BINS`` power bins of width ``BIN_S`` (NIST: 210 x 0.5 ms
= 105 ms). Following NIST TN-2240 / Kuester et al., an *occupancy event* is a maximal run of
occupied bins that does not touch the edge of the dwell window (its true extent is unknown).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_S = 0.5e-3
DWELL_BINS = 210
DWELL_S = DWELL_BINS * BIN_S  # 0.105 s


def estimate_noise_floor_dbm(power_dbm, q: float = 0.05) -> float:
    """Crude noise-floor estimate: the ``q``-quantile of training-period bin power (dBm).

    Must be fitted on training data only. Placeholder until validated against the dataset's own
    noise information (calibration / denoising notes, histograms).
    """
    return float(np.quantile(np.asarray(power_dbm, dtype=float), q))


def occupied_bins(power_dbm, threshold_dbm: float) -> np.ndarray:
    """Boolean mask: bin power strictly above ``threshold_dbm``."""
    return np.asarray(power_dbm, dtype=float) > threshold_dbm


@dataclass(frozen=True)
class DwellLabels:
    """Per-dwell labels derived from an (n_dwells, n_bins) occupied-bin mask."""

    occ_any: np.ndarray  # bool: >=1 occupancy event not touching the window edge
    duty: np.ndarray  # float: fraction of bins occupied (all occupied bins, incl. edge runs)
    n_events: np.ndarray  # int: events not touching the window edge
    event_bins: np.ndarray  # int: lengths (bins) of all valid events, row-major order
    event_row: np.ndarray  # int: dwell index of each valid event
    gap_bins: np.ndarray  # int: gaps (bins) between consecutive valid events in the same dwell


def _runs(mask: np.ndarray):
    n, b = mask.shape
    padded = np.zeros((n, b + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    sr, sc = np.nonzero(d == 1)  # run starts (index into mask)
    er, ec = np.nonzero(d == -1)  # run ends, exclusive
    return sr, sc, ec  # starts and ends are paired row-major


def dwell_labels(mask: np.ndarray) -> DwellLabels:
    """Compute occupancy labels for every dwell from a boolean (n, n_bins) mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D (n_dwells, n_bins)")
    n, b = mask.shape
    rows, starts, ends = _runs(mask)
    valid = (starts > 0) & (ends < b)
    vrows, vstarts, vends = rows[valid], starts[valid], ends[valid]
    n_events = np.bincount(vrows, minlength=n).astype(int)
    same_row = vrows[1:] == vrows[:-1]
    gaps = (vstarts[1:] - vends[:-1])[same_row]
    return DwellLabels(
        occ_any=n_events > 0,
        duty=mask.mean(axis=1),
        n_events=n_events,
        event_bins=(vends - vstarts).astype(int),
        event_row=vrows.astype(int),
        gap_bins=gaps.astype(int),
    )
