"""Causal, age-aware features for occupancy forecasting.

For a forecast at time ``t`` the predictor may use only observations that *ended* no later than
``t - delay_s`` and that are marked visible. Missing evidence is encoded as NaN plus an explicit
age, never as 0 power or an idle state. ``age_*`` = ``t - end_of_last_obs`` (so any delay is
included in the age).

Targets can be rows of the observation table (next-sweep forecasting) or arbitrary forecast
origins (``targets=``), e.g. the start of a future window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PW_REF_DBM = -130.0  # reference level for linear-power accumulation (numerical range only)


@dataclass(frozen=True)
class FeatureSpec:
    tau_s: float = 60.0  # time constant of the decayed history
    windows_s: tuple[float, ...] = (60.0, 600.0)  # count windows
    cross_channel: bool = True  # last state / age of the other channels at the same site

    @property
    def max_lookback_s(self) -> float:
        """Longest look-back used; use as a lower bound for the split embargo."""
        return float(max(self.windows_s + (5.0 * self.tau_s,)))


class _Channel:
    """Visible observations of one (site, channel) stream, with causal query structures."""

    def __init__(
        self,
        t_end: np.ndarray,
        occ: np.ndarray,
        tau: float,
        power_dbm: np.ndarray | None = None,
    ):
        order = np.argsort(t_end, kind="stable")
        self.t_end = t_end[order]
        self.occ = occ[order].astype(float)
        self.cs = np.concatenate([[0.0], np.cumsum(self.occ)])
        self.cs_pw = None
        if power_dbm is not None:  # cumulative LINEAR power (never average dBm arithmetically)
            lin = 10.0 ** ((np.asarray(power_dbm, dtype=float)[order] - PW_REF_DBM) / 10.0)
            self.cs_pw = np.concatenate([[0.0], np.cumsum(lin)])
        n = len(self.t_end)
        e = np.empty(n)
        w = np.empty(n)
        for i in range(n):  # decayed sums with irregular spacing
            if i == 0:
                e[i], w[i] = self.occ[0], 1.0
            else:
                d = np.exp(-(self.t_end[i] - self.t_end[i - 1]) / tau)
                e[i] = e[i - 1] * d + self.occ[i]
                w[i] = w[i - 1] * d + 1.0
        self.e, self.w = e, w

    def last_index(self, t_query: np.ndarray) -> np.ndarray:
        """Index of the latest observation with t_end <= t_query (-1 if none)."""
        return np.searchsorted(self.t_end, t_query, side="right") - 1


def build_features(
    dwells: pd.DataFrame,
    target_mask: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    *,
    targets: pd.DataFrame | None = None,
    delay_s: float = 0.0,
    spec: FeatureSpec | None = None,
    power_col: str | None = None,
) -> pd.DataFrame:
    """Features for each target (single site), aligned to the target rows.

    ``dwells`` is the full observation table; ``visible`` hides observations from the
    predictor (see :mod:`spectrum_occupancy_forecasting.robustness.degrade`). Targets are either the
    rows ``dwells[target_mask]`` (a target never sees itself: its end time exceeds its start) or
    an explicit frame ``targets`` with columns ``site, freq_mhz, t_start_s`` (forecast origins).
    With ``power_col`` (e.g. ``"mean_dbm"``) the physical mean power of the observations in each
    window is added as ``pw_<w>s`` (dBm, linear average; NaN if the window is empty).
    """
    spec = spec or FeatureSpec()
    if dwells["site"].nunique() != 1:
        raise ValueError("build_features expects a single site")
    if (target_mask is None) == (targets is None):
        raise ValueError("pass exactly one of target_mask or targets")
    dwells = dwells.reset_index(drop=True)
    visible = np.ones(len(dwells), dtype=bool) if visible is None else np.asarray(visible, bool)

    t_end_all = (dwells["t_start_s"] + dwells["dwell_s"]).to_numpy()
    occ_all = dwells["occ_any"].to_numpy(dtype=float)
    pw_all = dwells[power_col].to_numpy(dtype=float) if power_col else None
    freqs = sorted(dwells["freq_mhz"].unique())
    chans = {}
    for f in freqs:
        sel = (dwells["freq_mhz"].to_numpy() == f) & visible
        chans[f] = _Channel(
            t_end_all[sel], occ_all[sel], spec.tau_s, None if pw_all is None else pw_all[sel]
        )

    tgt = dwells[np.asarray(target_mask, dtype=bool)] if targets is None else targets
    t = tgt["t_start_s"].to_numpy(dtype=float)
    f_t = tgt["freq_mhz"].to_numpy()
    t_dec = t - delay_s
    out = pd.DataFrame(
        {"site": tgt["site"].to_numpy(), "freq_mhz": f_t, "t_start_s": t}, index=tgt.index
    )

    names = ["last_occ", "age_s", "ewma_occ", "ewma_w"]
    names += [f"cnt_{int(w)}s" for w in spec.windows_s] + [f"n_{int(w)}s" for w in spec.windows_s]
    if pw_all is not None:
        names += [f"pw_{int(w)}s" for w in spec.windows_s]
    cols: dict[str, np.ndarray] = {k: np.full(len(t), np.nan) for k in names}
    for f in freqs:
        rows = np.nonzero(f_t == f)[0]
        if len(rows) == 0:
            continue
        ch = chans[f]
        i = ch.last_index(t_dec[rows])
        has = i >= 0
        ii = np.where(has, i, 0)
        if len(ch.t_end):
            cols["last_occ"][rows] = np.where(has, ch.occ[ii], np.nan)
            age = t[rows] - ch.t_end[ii]
            cols["age_s"][rows] = np.where(has, age, np.nan)
            cols["ewma_occ"][rows] = np.where(has, ch.e[ii] / ch.w[ii], np.nan)
            cols["ewma_w"][rows] = np.where(has, ch.w[ii] * np.exp(-age / spec.tau_s), 0.0)
            for w in spec.windows_s:
                lo = np.searchsorted(ch.t_end, t_dec[rows] - w, side="right")
                n_in = np.where(has, i + 1 - lo, 0)
                lo_c = np.minimum(lo, ii + 1)
                s_in = np.where(has, ch.cs[ii + 1] - ch.cs[lo_c], 0.0)
                cols[f"n_{int(w)}s"][rows] = n_in
                cols[f"cnt_{int(w)}s"][rows] = s_in
                if ch.cs_pw is not None:
                    s_pw = np.where(has, ch.cs_pw[ii + 1] - ch.cs_pw[lo_c], 0.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mean_lin = s_pw / np.maximum(n_in, 1)
                        db = 10.0 * np.log10(mean_lin) + PW_REF_DBM
                    cols[f"pw_{int(w)}s"][rows] = np.where(n_in > 0, db, np.nan)
        else:
            cols["ewma_w"][rows] = 0.0
            for w in spec.windows_s:
                cols[f"n_{int(w)}s"][rows] = 0
                cols[f"cnt_{int(w)}s"][rows] = 0.0

    if spec.cross_channel:
        for f in freqs:
            ch = chans[f]
            lo_col = np.full(len(t), np.nan)
            ag_col = np.full(len(t), np.nan)
            if len(ch.t_end):
                i = ch.last_index(t_dec)
                has = (i >= 0) & (f_t != f)
                ii = np.where(i >= 0, i, 0)
                lo_col = np.where(has, ch.occ[ii], np.nan)
                ag_col = np.where(has, t - ch.t_end[ii], np.nan)
            cols[f"x_last_occ_{f:g}"] = lo_col
            cols[f"x_age_s_{f:g}"] = ag_col

    return pd.concat([out, pd.DataFrame(cols, index=tgt.index)], axis=1)
