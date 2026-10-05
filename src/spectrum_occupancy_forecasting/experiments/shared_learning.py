"""Declared local/shared residual study; targets and legacy studies remain unchanged."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from spectrum_occupancy_forecasting.experiments.fine_spectrum import FINE_COLUMNS, raw_table
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData, maximum_lookback
from spectrum_occupancy_forecasting.experiments.followup_models import history_prediction
from spectrum_occupancy_forecasting.experiments.mixture_support import history_candidates
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask

FIXED = {"id": "recent_7200s", "window": 7200}


def configuration(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec["study"] != "shared-learning":
        raise ValueError("configuration/study mismatch")
    expected = {
        "feature_sets": ["history", "history_fine"],
        "training_spacings_s": [180, 900],
        "validation_hours": [12, 36],
        "ridge_penalties": [1, 10, 100, 1000],
        "shared_weights": [0, 0.25, 0.5, 0.75, 1],
        "correction_strengths": [0.25, 0.5, 1],
    }
    if any(spec[k] != value for k, value in expected.items()):
        raise ValueError("declared grid changed")
    cfg = load_config(Path(path).parent / spec["base_config"])
    cfg.update(spec)
    # Only the unchanged target channel is needed; no cross-channel features are used.
    cfg["channels_mhz"] = [2437.0]
    return cfg


def profiles(cfg):
    return [
        {"id": f"{variant}_{step}s_{hours}h", "variant": variant, "spacing_s": step, "hours": hours}
        for variant, step, hours in itertools.product(
            cfg["feature_sets"], cfg["training_spacings_s"], cfg["validation_hours"]
        )
    ]


def candidates(cfg):
    out = [{"id": "history_only", "family": "history", "strength": 0.0, "shared": 0.0}]
    for a, share, strength in itertools.product(
        cfg["ridge_penalties"], cfg["shared_weights"], cfg["correction_strengths"]
    ):
        out.append(
            {
                "id": f"ridge_a{a}_w{share:g}_s{strength:g}",
                "family": "ridge",
                "alpha": a,
                "shared": share,
                "strength": strength,
            }
        )
    for depth, leaf, share, strength in itertools.product(
        cfg["boosting_depths"], cfg["boosting_leaf_sizes"], [0.0, 1.0], cfg["correction_strengths"]
    ):
        out.append(
            {
                "id": f"hgb_d{depth}_l{leaf}_w{share:g}_s{strength:g}",
                "family": "hgb",
                "depth": depth,
                "leaf": leaf,
                "shared": share,
                "strength": strength,
            }
        )
    return out


def tie_key(choice):
    return (
        choice["strength"] != 0,
        choice["strength"],
        {"history": 0, "ridge": 1, "hgb": 2}[choice["family"]],
        -choice.get("alpha", 0),
        choice.get("shared", 0),
        choice.get("depth", 0),
        -choice.get("leaf", 0),
    )


def balanced_weights(nodes):
    nodes = np.asarray(nodes)
    return np.asarray(
        [len(nodes) / (len(np.unique(nodes)) * np.sum(nodes == node)) for node in nodes]
    )


class LearningNode:
    def __init__(self, frame, name, cfg):
        if np.any(np.diff(frame.time_s) <= 0):
            raise ValueError("fine observations must be unique and ordered")
        self.frame, self.cfg = frame.copy(), cfg
        self.data = NodeData(raw_table(frame, name), cfg, 3.0)
        cal = frame.time_s.to_numpy() + 0.001 <= self.data.t0 + 86400
        self.fine_floors = np.quantile(frame.loc[cal, FINE_COLUMNS], 0.05, axis=0)
        self.normalized = frame[FINE_COLUMNS].to_numpy() - self.fine_floors
        self._fine_cache = {}

    def features(self, origins, mean, variant, deg=None, seed=0):
        deg = deg or Degradation()
        X = self.data.features(origins, deg, seed)
        out = pd.DataFrame(
            {h["id"]: history_prediction(h, X, mean) for h in history_candidates(self.cfg)}
        )
        out["diff_5_30"] = X.frac_300s - X.frac_1800s
        out["diff_15_120"] = X.frac_900s - X.frac_7200s
        out["last_occ"], out["age_s"] = X.last_occ, X.age_s
        for w in self.cfg["history_windows_s"]:
            out[f"coverage_{w}s"] = X[f"n_{w}s"] / (w / self.data.dt)
        if variant == "history_fine":
            key = (deg, seed)
            if key not in self._fine_cache:
                times = self.frame.time_s.to_numpy()
                visible = visibility_mask(times, deg, seed)
                ends = times[visible] + 0.001
                values = self.normalized[visible]
                total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
                self._fine_cache[key] = (ends, total)
            ends, total = self._fine_cache[key]
            query = np.asarray(origins) - deg.delay_s
            for w in (300, 900, 1800):
                hi = np.searchsorted(ends, query, side="left")
                lo = np.searchsorted(ends, query - w, side="right")
                count = hi - lo
                values = np.divide(
                    total[hi] - total[lo],
                    count[:, None],
                    out=np.full((len(query), len(FINE_COLUMNS)), np.nan),
                    where=count[:, None] > 0,
                )
                for j, name in enumerate(FINE_COLUMNS):
                    out[f"fine_{name}_{w}s"] = values[:, j]
        base = history_prediction(FIXED, X, mean)
        valid = (X.age_s.to_numpy() <= 900) & (X.n_900s.to_numpy() / (900 / self.data.dt) >= 0.5)
        return out, base, valid

    def training(self, cutoff, spacing):
        end = min(cutoff, self.data.last)
        origins = self.data.origins(self.data.t0, end, 900, spacing)
        o, y, count = self.data.dataset(origins, 900)
        if not len(y):
            raise ValueError("no eligible training labels")
        assert np.max(o + 1800) <= cutoff + 1e-6
        return o, y, count


class StableScaler:
    """Training-only site-weighted scaling with nonnegative centered variance."""

    def fit(self, values, sample_weight):
        self.mean_ = np.average(values, axis=0, weights=sample_weight)
        self.var_ = np.average((values - self.mean_) ** 2, axis=0, weights=sample_weight)
        eps = np.finfo(float).eps
        bound = len(values) * eps * self.var_ + (len(values) * self.mean_ * eps) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.var_ <= bound] = 1.0
        return self

    def transform(self, values):
        return (values - self.mean_) / self.scale_


class Regressor:
    def fit(self, X, y, weights, choice, cfg):
        self.columns = X.columns.tolist()
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        z = self.imputer.fit_transform(X)
        self.scaler = StableScaler().fit(z, sample_weight=weights)
        z = self.scaler.transform(z)
        if choice["family"] == "ridge":
            self.estimator = Ridge(alpha=choice["alpha"])
        else:
            self.estimator = HistGradientBoostingRegressor(
                max_depth=choice["depth"],
                min_samples_leaf=choice["leaf"],
                max_iter=cfg["residual_boosting_iterations"],
                learning_rate=cfg["residual_boosting_learning_rate"],
                early_stopping=False,
                random_state=cfg["model_seed"],
            )
        # A uniform local objective needs no weights; avoid costly weighted binning.
        sample_weight = None if np.all(weights == weights[0]) else weights
        self.estimator.fit(z, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.estimator.predict(
            self.scaler.transform(self.imputer.transform(X[self.columns]))
        )


class Bundle:
    def __init__(self, nodes, node, cutoff, profile, cfg):
        self.profile, self.node, self.cfg = profile, node, cfg
        self.rows, self.means, self.features, self.targets, self.bases = {}, {}, {}, {}, {}
        self.regressors, self.failures = {}, {}
        self.input_cache, self.prediction_cache = {}, {}
        for name, data in nodes.items():
            o, y, _ = data.training(cutoff, profile["spacing_s"])
            self.rows[name], self.means[name] = o, float(y.mean())
            X, base, _ = data.features(o, self.means[name], profile["variant"])
            self.features[name], self.targets[name], self.bases[name] = X, y, base
        self.cutoff = cutoff

    def estimator(self, choice, scope):
        key = (
            choice["family"],
            choice.get("alpha"),
            choice.get("depth"),
            choice.get("leaf"),
            scope,
        )
        if key in self.failures:
            raise ValueError(self.failures[key])
        if key not in self.regressors:
            names = [self.node] if scope == "local" else list(self.features)
            X = pd.concat([self.features[n] for n in names], ignore_index=True)
            y = np.concatenate([self.targets[n] - self.bases[n] for n in names])
            labels = np.concatenate([np.full(len(self.targets[n]), n) for n in names])
            try:
                self.regressors[key] = Regressor().fit(
                    X, y, balanced_weights(labels), choice, self.cfg
                )
            except Exception as exc:
                self.failures[key] = f"{type(exc).__name__}: {exc}"
                raise
        return self.regressors[key]

    def details(self, data, origins, choice, deg=None, seed=0):
        key = (id(data), tuple(np.asarray(origins)), deg or Degradation(), seed)
        if key not in self.input_cache:
            self.input_cache[key] = data.features(
                origins, self.means[self.node], self.profile["variant"], deg, seed
            )
        X, base, valid = self.input_cache[key]

        def residual(scope):
            prediction_key = (
                key,
                choice["family"],
                choice.get("alpha"),
                choice.get("depth"),
                choice.get("leaf"),
                scope,
            )
            if prediction_key not in self.prediction_cache:
                self.prediction_cache[prediction_key] = self.estimator(choice, scope).predict(X)
            return self.prediction_cache[prediction_key]

        correction = np.zeros(len(base))
        if choice["strength"]:
            share = choice["shared"]
            local = residual("local") if share < 1 else 0.0
            pooled = residual("shared") if share > 0 else 0.0
            correction = choice["strength"] * np.clip(
                (1 - share) * local + share * pooled, -0.1, 0.1
            )
            correction[~valid] = 0
        prediction = np.clip(base + correction, 0, 1)
        prediction[~valid] = base[~valid]
        reasons = np.where(valid, "none", "stale_or_low_coverage")
        if not choice["strength"]:
            prediction = base.copy()
            reasons[:] = "history_only"
        return prediction, base, prediction - base, reasons


class Study:
    def __init__(self, nodes, cfg):
        self.nodes, self.cfg, self.bundles = nodes, cfg, OrderedDict()

    def bundle(self, name, cutoff, profile):
        key = (name, cutoff, profile["variant"], profile["spacing_s"])
        if key not in self.bundles:
            self.bundles[key] = Bundle(self.nodes, name, cutoff, profile, self.cfg)
            if len(self.bundles) > 64:
                self.bundles.popitem(last=False)
        self.bundles.move_to_end(key)
        return self.bundles[key]

    def select(self, name, evaluation_origin, profile):
        node = self.nodes[name].data
        embargo = maximum_lookback(self.cfg)
        start = evaluation_origin - embargo - profile["hours"] * 3600
        choices = candidates(self.cfg)
        if start < node.t0 + 86400:
            return choices[0], FIXED, [], pd.DataFrame(), "insufficient_earlier_data"
        predictions, failures = {}, {}
        histories = history_candidates(self.cfg)
        for begin in np.arange(start, evaluation_origin - embargo, 21600):
            cutoff = begin - embargo
            bundle = self.bundle(name, cutoff, profile)
            o, y, _ = node.dataset(node.origins(begin, begin + 21600, 900, 900, common=True), 900)
            if not len(y):
                return choices[0], FIXED, [], pd.DataFrame(), "insufficient_validation_targets"
            rawX = node.features(o)
            for h in histories:
                predictions.setdefault(h["id"], []).append(
                    pd.DataFrame(
                        {
                            "origin_s": o,
                            "y": y,
                            "prediction": history_prediction(h, rawX, bundle.means[name]),
                            "candidate_id": h["id"],
                            "fit_cutoff_s": cutoff,
                            "validation_start_s": begin,
                        }
                    )
                )
            for c in choices:
                try:
                    p, base, correction, reasons = bundle.details(self.nodes[name], o, c)
                    predictions.setdefault(c["id"], []).append(
                        pd.DataFrame(
                            {
                                "origin_s": o,
                                "y": y,
                                "prediction": p,
                                "history_prediction": base,
                                "correction_value": correction,
                                "fallback_reason": reasons,
                                "candidate_id": c["id"],
                                "fit_cutoff_s": cutoff,
                                "validation_start_s": begin,
                            }
                        )
                    )
                except Exception as exc:
                    failures[c["id"]] = f"{type(exc).__name__}: {exc}"
        scores, frames = [], []
        for c in [*histories, *choices]:
            parts = predictions.get(c["id"], [])
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            reason = failures.get(c["id"])
            mse = (
                float(np.mean((frame.prediction - frame.y) ** 2))
                if len(frame) and not reason
                else None
            )
            scores.append(
                c
                | {
                    "validation_mse": mse,
                    "validation_n": len(frame),
                    "status": "failed" if reason else "completed",
                    "failure": reason,
                }
            )
            if len(frame):
                frames.append(frame.assign(candidate_status="failed" if reason else "completed"))
        lookup = {s["id"]: s["validation_mse"] for s in scores}
        choice = min(
            choices,
            key=lambda c: (
                lookup[c["id"]] if lookup[c["id"]] is not None else float("inf"),
                tie_key(c),
            ),
        )
        history = min(histories, key=lambda c: lookup[c["id"]])
        return choice, history, scores, pd.concat(frames, ignore_index=True), "selected"
