"""Baselines and lightweight learned models. All consume the frame from ``build_features``.

Every model exposes ``fit(X, y) -> self`` and ``predict_proba(X) -> P(busy)`` (1-D array), and
uses only features available at prediction time. Training-set statistics (rates, tables) are
fitted on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_AGE_EDGES_S = (0.0, 10.0, 30.0, 60.0, 300.0, 1800.0, np.inf)


def age_bucket(age_s, edges=DEFAULT_AGE_EDGES_S) -> np.ndarray:
    """Bucket index per age (NaN age -> -1, i.e. 'no evidence')."""
    age = np.asarray(age_s, dtype=float)
    idx = np.searchsorted(np.asarray(edges), age, side="right") - 1
    return np.where(np.isnan(age), -1, np.minimum(idx, len(edges) - 2))


class GlobalRate:
    """Climatology: predict the training busy rate for everything."""

    name = "global_rate"

    def fit(self, X, y):
        self.rate_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate_)


class HourOfWeekRate:
    """Training busy rate per local hour-of-week (``period="week"``) or hour-of-day (``"day"``).

    With only a few days of data use ``period="day"``: an hour-of-week table would memorise a
    single week. Falls back to the global rate where a slot has too few training samples.
    """

    def __init__(self, tz: str = "UTC", min_count: int = 20, period: str = "week"):
        if period not in ("week", "day"):
            raise ValueError("period must be 'week' or 'day'")
        self.tz, self.min_count, self.period = tz, min_count, period
        self.name = "hour_of_week_rate" if period == "week" else "hour_of_day_rate"

    def _slot(self, X):
        ts = pd.to_datetime(X["t_start_s"].to_numpy(), unit="s", utc=True).tz_convert(self.tz)
        if self.period == "day":
            return np.asarray(ts.hour)
        return np.asarray(ts.dayofweek * 24 + ts.hour)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        slot = self._slot(X)
        n = 24 if self.period == "day" else 168
        self.rate_ = float(y.mean())
        self.table_ = np.full(n, self.rate_)
        for h in range(n):
            m = slot == h
            if m.sum() >= self.min_count:
                self.table_[h] = y[m].mean()
        return self

    def predict_proba(self, X):
        return self.table_[self._slot(X)]


class Persistence:
    """Carry forward the most recent valid state; training rate if there is none."""

    name = "persistence"

    def fit(self, X, y):
        self.rate_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X):
        last = X["last_occ"].to_numpy(dtype=float)
        return np.where(np.isnan(last), self.rate_, last)


class DecayedHistory:
    """Time-decayed rolling occupancy, shrunk to the training rate when evidence is thin."""

    name = "decayed_history"

    def __init__(self, prior_weight: float = 1.0):
        self.a = prior_weight

    def fit(self, X, y):
        self.rate_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X):
        e = X["ewma_occ"].to_numpy(dtype=float)
        w = X["ewma_w"].to_numpy(dtype=float)
        e = np.where(np.isnan(e), 0.0, e)
        return (e * w + self.a * self.rate_) / (w + self.a)


class AgeConditionedPersistence:
    """Empirical P(busy | last state, age bucket) from training data, Laplace-smoothed.

    A cheap Markov-style baseline: how fast does the last observation go stale?
    """

    name = "age_conditioned"

    def __init__(self, edges=DEFAULT_AGE_EDGES_S, alpha: float = 1.0):
        self.edges, self.alpha = edges, alpha

    def _key(self, X):
        last = X["last_occ"].to_numpy(dtype=float)
        b = age_bucket(X["age_s"], self.edges)
        return np.where(np.isnan(last), -1, last * 1000 + b).astype(int)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.rate_ = float(y.mean())
        key = self._key(X)
        self.table_ = {}
        for k in np.unique(key):
            m = key == k
            self.table_[int(k)] = (y[m].sum() + self.alpha * self.rate_) / (m.sum() + self.alpha)
        return self

    def predict_proba(self, X):
        return np.array([self.table_.get(int(k), self.rate_) for k in self._key(X)])


FEATURE_PREFIXES = ("last_occ", "age_s", "ewma_", "cnt_", "n_", "x_last_occ_", "x_age_s_")


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith(FEATURE_PREFIXES)]


class _Sklearn:
    def __init__(self, est, name):
        self.est, self.name = est, name

    def fit(self, X, y):
        # Columns with no observed value in training (e.g. the own-channel cross-channel slot)
        # carry no information and break some estimators; drop them, keep the rest.
        self.cols_ = [c for c in feature_columns(X) if X[c].notna().any()]
        self.est.fit(X[self.cols_].to_numpy(dtype=float), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X[self.cols_].to_numpy(dtype=float))[:, 1]


def logistic_regression(seed: int = 0, C: float = 1.0):
    """Logistic regression with median imputation + missing-indicators + scaling."""
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000, random_state=seed),
    )
    return _Sklearn(pipe, "logreg")


def gradient_boosting(seed: int = 0, **kw):
    """Histogram gradient boosting; handles NaN natively."""
    est = HistGradientBoostingClassifier(random_state=seed, **kw)
    return _Sklearn(est, "hgb")
