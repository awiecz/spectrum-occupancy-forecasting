"""V2 causal data preparation. Legacy experiment behavior is deliberately untouched."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.data.aerpaw import label_band_table
from spectrum_occupancy_forecasting.features.age_aware import FeatureSpec, build_features
from spectrum_occupancy_forecasting.features.windows import window_fraction
from spectrum_occupancy_forecasting.models.regressors import with_derived
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, sweep_visibility_mask


def folds(t0, last, cfg):
    """Daily origin folds; retain a final partial fold of at least the configured duration."""
    start = t0 + cfg["first_eval_s"]
    out = []
    while start < last:
        end = min(start + cfg["fold_s"], last)
        if end - start >= cfg["minimum_partial_fold_s"]:
            out.append((start, end))
        start += cfg["fold_s"]
    return out


def maximum_lookback(cfg):
    return max(max(cfg["history_windows_s"]), cfg["decay_support_taus"] * max(cfg["decay_taus_s"]))


def bounded_decay(d, origins, visible, tau, support, delay, target_mhz):
    """Exponentially weighted busy history with a finite, explicit support at available time.

    All observations must END by origin-delay. The decay weight is aged to the actual origin.
    Truncation makes the maximum look-back used in v2 embargoes an actual bound, not a heuristic.
    """
    use = (d["freq_mhz"].to_numpy() == target_mhz) & visible
    ends = (d["t_start_s"] + d["dwell_s"]).to_numpy()[use]
    busy = d["occ_any"].to_numpy(dtype=float)[use]
    order = np.argsort(ends)
    ends, busy = ends[order], busy[order]
    total, weights = np.zeros(len(ends)), np.zeros(len(ends))
    for i in range(len(ends)):
        decay = np.exp(-(ends[i] - ends[i - 1]) / tau) if i else 0.0
        total[i] = (total[i - 1] * decay if i else 0.0) + busy[i]
        weights[i] = (weights[i - 1] * decay if i else 0.0) + 1.0
    q = np.asarray(origins) - delay
    hi = np.searchsorted(ends, q, side="right") - 1
    lo = np.searchsorted(ends, q - support, side="right")
    valid = (hi >= lo) & (hi >= 0)
    numerator, denominator = np.zeros(len(q)), np.zeros(len(q))
    idx = np.flatnonzero(valid)
    if len(idx):
        h, lower = hi[idx], lo[idx] - 1
        subtract = lower >= 0
        n, w = total[h].copy(), weights[h].copy()
        factor = np.exp(-(ends[h[subtract]] - ends[lower[subtract]]) / tau)
        n[subtract] -= total[lower[subtract]] * factor
        w[subtract] -= weights[lower[subtract]] * factor
        age = np.exp(-(np.asarray(origins)[idx] - ends[h]) / tau)
        numerator[idx], denominator[idx] = np.maximum(n, 0) * age, np.maximum(w, 0) * age
    fraction = np.divide(numerator, denominator, out=np.full(len(q), np.nan), where=denominator > 0)
    return fraction, denominator


@dataclass
class NodeData:
    raw: pd.DataFrame
    cfg: dict
    offset: float

    def __post_init__(self):
        self.t0 = float(self.raw["t_start_s"].min())
        self.last = float(self.raw["t_start_s"].max())
        self.node = str(self.raw["site"].iloc[0])
        end = self.t0 + self.cfg["calibration_s"]
        if self.last < end:
            raise ValueError("recording is shorter than the calibration period")
        calibration = self.raw[(self.raw["t_start_s"] + self.raw["dwell_s"]) <= end]
        self.floors = {}
        for channel in self.cfg["channels_mhz"]:
            values = calibration.loc[calibration["freq_mhz"] == channel, "mean_dbm"]
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError("insufficient finite calibration measurements")
            self.floors[channel] = float(np.percentile(values, 5))
        self.thresholds = {c: floor + self.offset for c, floor in self.floors.items()}
        self.d = label_band_table(self.raw, self.thresholds, stat="mean_dbm")
        self.ch = self.d[self.d["freq_mhz"] == self.cfg["target_mhz"]]
        times = np.sort(
            calibration.loc[calibration["freq_mhz"] == self.cfg["target_mhz"], "t_start_s"].unique()
        )
        self.dt = float(np.median(np.diff(times)))
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid calibrated sweep interval")
        step = self.cfg["training_stride_s"]
        self.grid = np.arange(self.t0, self.last + 1e-6, step)
        self._features = {}

    def labels(self, origins, lead):
        return window_fraction(
            self.ch["t_start_s"],
            self.ch["occ_any"],
            np.asarray(origins) + lead,
            self.cfg["window_s"],
            min_coverage=self.cfg["minimum_coverage"],
            nominal_dt_s=self.dt,
        )

    def origins(self, lo, hi, lead, stride, *, common=False):
        """Origin grid with targets fully contained in [lo, hi]."""
        horizon = max(self.cfg["leads_s"]) if common else lead
        return np.arange(lo, hi - horizon - self.cfg["window_s"] + 1e-6, stride)

    def dataset(self, origins, lead):
        y, count = self.labels(origins, lead)
        keep = np.isfinite(y)
        return np.asarray(origins)[keep], y[keep], count[keep]

    def features(self, origins, deg=None, seed=0, *, calibration_clean=False):
        deg = deg or Degradation()
        key = (deg, seed, True) if calibration_clean else (deg, seed)
        if key not in self._features:
            cfg = self.cfg
            vis = sweep_visibility_mask(self.d["t_start_s"], deg, seed)
            if calibration_clean:
                vis |= self.d["t_start_s"].to_numpy() + 0.001 <= self.t0 + 86400
            tg = pd.DataFrame(
                {"site": self.node, "freq_mhz": cfg["target_mhz"], "t_start_s": self.grid}
            )
            spec = FeatureSpec(
                tau_s=cfg["feature_tau_s"],
                windows_s=tuple(cfg["history_windows_s"]),
                cross_channel=True,
            )
            X = build_features(
                self.d,
                targets=tg,
                visible=vis,
                delay_s=deg.delay_s,
                spec=spec,
                power_col="mean_dbm",
            )
            X = with_derived(X, cfg["timezone"])
            # Match the finite-history lower boundary (origin-delay-lookback, origin-delay].
            # Expired ages must also disappear: retaining them still refers to an older sweep.
            lookback = maximum_lookback(cfg)
            expired = X["age_s"] - deg.delay_s >= lookback
            X.loc[expired, ["last_occ", "age_s"]] = np.nan
            for column in X:
                if column.startswith("x_age_s_"):
                    expired = X[column] - deg.delay_s >= lookback
                    state = column.replace("x_age_s_", "x_last_occ_")
                    X.loc[expired, [state, column]] = np.nan
            for tau in cfg["decay_taus_s"]:
                fraction, weight = bounded_decay(
                    self.d,
                    self.grid,
                    vis,
                    tau,
                    cfg["decay_support_taus"] * tau,
                    deg.delay_s,
                    cfg["target_mhz"],
                )
                X[f"decay_fraction_{tau}s"], X[f"decay_weight_{tau}s"] = fraction, weight
            tau = cfg["feature_tau_s"]
            X["ewma_occ"], X["ewma_w"] = X[f"decay_fraction_{tau}s"], X[f"decay_weight_{tau}s"]
            X.index = self.grid
            self._features[key] = X
        try:
            return self._features[key].loc[np.asarray(origins)].reset_index(drop=True)
        except KeyError as exc:
            raise ValueError("forecast origins must lie on the configured training grid") from exc


def feature_columns(X, variant):
    """Explicit allowlists: no timestamp, counts, power or calendar leak into history-only."""
    families = {
        "history": ["last_occ", "ewma_occ"] + [c for c in X if c.startswith("frac_")],
        "power": [c for c in X if c.startswith("pw_")],
        "age": ["age_s", "ewma_w"] + [c for c in X if c.startswith("n_")],
        "calendar": ["tod_sin", "tod_cos"],
        "cross": [c for c in X if c.startswith("x_last_occ_")],
    }
    names = list(families) if variant == "full" else variant.split("_")
    if names[0] != "history" or any(n not in families for n in names):
        raise ValueError(f"unknown feature variant {variant}")
    return [c for n in names for c in families[n] if X[c].notna().any()]
