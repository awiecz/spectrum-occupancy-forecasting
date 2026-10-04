"""Canonical processed schema: one row per dwell (a contiguous burst of power bins).

Every dataset reader (NIST, later AERPAW) must emit this table. Raw recordings stay immutable;
this is a deterministic derived product.
"""

from __future__ import annotations

import pandas as pd

DWELL_COLUMNS: dict[str, str] = {
    "site": "string identifier of the monitoring site / node",
    "freq_mhz": "float, channel centre frequency (MHz)",
    "t_start_s": "float, dwell start, seconds since Unix epoch (UTC)",
    "dwell_s": "float, valid duration of the dwell (s)",
    "occ_any": "bool, detected-occupancy label (>=1 event, see occupancy.detect)",
    "duty": "float in [0,1], fraction of bins occupied",
    "mean_dbm": "float, physical mean band power over the dwell (dBm, linear average)",
    "max_dbm": "float, maximum bin power in the dwell (dBm)",
}


def validate_dwells(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` violates the dwell-table contract."""
    missing = [c for c in DWELL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[["site", "freq_mhz", "t_start_s", "dwell_s", "occ_any", "duty"]].isna().any().any():
        raise ValueError(
            "NaN in required columns (site, freq_mhz, t_start_s, dwell_s, occ_any, duty)"
        )
    if not df["duty"].between(0, 1).all():
        raise ValueError("duty must lie in [0, 1]")
    if (df["dwell_s"] <= 0).any():
        raise ValueError("dwell_s must be positive")
    if df.duplicated(["site", "freq_mhz", "t_start_s"]).any():
        raise ValueError("duplicate (site, freq_mhz, t_start_s) rows")
    # Times must be plausible epoch seconds (2000-01-01 .. 2100-01-01), i.e. not ms/us/ns.
    if not df["t_start_s"].between(9.46e8, 4.10e9).all():
        raise ValueError("t_start_s outside plausible epoch-seconds range (wrong unit?)")
    # Dwells of one channel at one site cannot overlap in time.
    for _, g in df.sort_values("t_start_s").groupby(["site", "freq_mhz"], sort=False):
        t = g["t_start_s"].to_numpy()
        end = t + g["dwell_s"].to_numpy()
        if (end[:-1] > t[1:] + 1e-9).any():
            raise ValueError("overlapping dwells within a (site, freq_mhz) stream")
