"""Models that forecast the busy FRACTION of a future window (value in [0, 1]).

Same contract as the classifiers: ``fit(X, y)`` / ``predict(X)`` on a frame from
``build_features``; every statistic is fitted on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectrum_occupancy_forecasting.models.baselines import feature_columns


def with_derived(X: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Add busy fractions of the look-back windows and a smooth time-of-day encoding."""
    out = X.copy()
    for c in [c for c in X.columns if c.startswith("cnt_")]:
        w = c[len("cnt_") :]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"frac_{w}"] = np.where(X[f"n_{w}"] > 0, X[c] / X[f"n_{w}"], np.nan)
    ts = pd.to_datetime(X["t_start_s"].to_numpy(), unit="s", utc=True).tz_convert(tz)
    hour = np.asarray(ts.hour) + np.asarray(ts.minute) / 60.0
    out["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


class ClimatologyFraction:
    """Predict the training mean window fraction."""

    name = "climatology"

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class HourOfDayFraction:
    """Training mean fraction per local hour-of-day (global mean where a slot is thin)."""

    name = "hour_of_day"

    def __init__(self, tz: str = "UTC", min_count: int = 5):
        self.tz, self.min_count = tz, min_count

    def _slot(self, X):
        ts = pd.to_datetime(X["t_start_s"].to_numpy(), unit="s", utc=True).tz_convert(self.tz)
        return np.asarray(ts.hour)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean_ = float(y.mean())
        slot = self._slot(X)
        self.table_ = np.array(
            [
                y[slot == h].mean() if (slot == h).sum() >= self.min_count else self.mean_
                for h in range(24)
            ]
        )
        return self

    def predict(self, X):
        return self.table_[self._slot(X)]


class RecentFraction:
    """'Persistence of the recent past': busy fraction over the last ``window_s`` seconds.

    Needs ``window_s`` among the feature windows; falls back to the training mean without
    evidence.
    """

    def __init__(self, window_s: float):
        self.w = int(window_s)
        self.name = f"recent_{self.w}s"

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = X[f"n_{self.w}s"].to_numpy(dtype=float)
        c = X[f"cnt_{self.w}s"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, c / n, self.mean_)


class DecayedFraction:
    """Time-decayed occupancy estimate shrunk to the training mean when evidence is thin."""

    name = "decayed"

    def __init__(self, prior_weight: float = 1.0):
        self.a = prior_weight

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        e = np.nan_to_num(X["ewma_occ"].to_numpy(dtype=float), nan=0.0)
        w = X["ewma_w"].to_numpy(dtype=float)
        return (e * w + self.a * self.mean_) / (w + self.a)


class _SkRegressor:
    def __init__(self, est, name, tz="UTC", clip=True):
        self.est, self.name, self.tz, self.clip = est, name, tz, clip

    def _mat(self, X):
        D = with_derived(X, self.tz)
        return D[self.cols_].to_numpy(dtype=float)

    def fit(self, X, y):
        D = with_derived(X, self.tz)
        base = feature_columns(D) + [c for c in D.columns if c.startswith(("frac_", "tod_", "pw_"))]
        self.cols_ = [c for c in dict.fromkeys(base) if D[c].notna().any()]
        self.est.fit(D[self.cols_].to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        out = self.est.predict(self._mat(X))
        return np.clip(out, 0.0, 1.0) if self.clip else out


def ridge_regressor(tz: str = "UTC", alpha: float = 10.0):
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha)
    )
    return _SkRegressor(pipe, "ridge", tz)


def gbt_regressor(tz: str = "UTC", seed: int = 0, **kw):
    kw = {"max_iter": 150, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 40, **kw}
    return _SkRegressor(HistGradientBoostingRegressor(random_state=seed, **kw), "hgb", tz)


class ResidualRegressor:
    """Level + learned correction: ``clip(base + f(features))`` where ``base`` is the busy
    fraction over the previous ``base_window_s`` seconds and ``f`` is fitted on the training
    residual ``y - base``. Tests whether any feature adds information beyond the recent level.
    """

    def __init__(self, inner: _SkRegressor, base_window_s: float = 7200.0, name: str = "resid"):
        self.inner, self.base, self.name = inner, RecentFraction(base_window_s), name
        self.inner.clip = False

    def fit(self, X, y):
        self.base.fit(X, y)
        self.inner.fit(X, np.asarray(y, dtype=float) - self.base.predict(X))
        return self

    def predict(self, X):
        return np.clip(self.base.predict(X) + self.inner.predict(X), 0.0, 1.0)


def residual_ridge(tz: str = "UTC", alpha: float = 300.0, base_window_s: float = 7200.0):
    inner = ridge_regressor(tz, alpha=alpha)
    return ResidualRegressor(inner, base_window_s, "resid_ridge")


def residual_gbt(tz: str = "UTC", seed: int = 0, base_window_s: float = 7200.0):
    inner = gbt_regressor(
        tz, seed=seed, max_iter=100, learning_rate=0.03, max_depth=2, min_samples_leaf=100
    )
    return ResidualRegressor(inner, base_window_s, "resid_hgb")
