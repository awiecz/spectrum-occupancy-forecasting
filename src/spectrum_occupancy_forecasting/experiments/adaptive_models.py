"""Declared online candidates. Updates accept delivered feedback, never scoring labels."""

import pickle
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .learning_inputs import StableScaler


@dataclass(frozen=True)
class Candidate:
    family: str = "history"
    half_life: float = 0
    strength: float = 0
    penalty: float = 0
    rate: float = 0
    features: str = "history"

    @property
    def key(self):
        return (self.family, self.half_life, self.penalty, self.rate, self.features)

    @property
    def id(self):
        return "_".join(map(str, (*self.key, self.strength)))

    @property
    def tie(self):
        return (
            self.family != "history",
            self.strength,
            {"history": 0, "bias": 1, "aggregation": 2, "ridge": 3}[self.family],
            self.features != "history",
            -self.penalty,
            -self.half_life,
            self.rate,
        )


def candidates(cfg):
    result = [Candidate()]
    for half in cfg["bias_half_lives_h"]:
        result.extend(Candidate("bias", half, s) for s in cfg["correction_strengths"])
    for half in cfg["aggregation_half_lives_h"]:
        for rate in cfg["aggregation_rates"]:
            result.extend(
                Candidate("aggregation", half, s, rate=rate) for s in cfg["correction_strengths"]
            )
    for half in cfg["ridge_half_lives_h"]:
        for penalty in cfg["ridge_penalties"]:
            for features in cfg["feature_sets"]:
                result.extend(
                    Candidate("ridge", half, s, penalty, features=features)
                    for s in cfg["correction_strengths"]
                )
    if len(result) != 145 or len({c.id for c in result}) != 145:
        raise ValueError("the declared search must contain exactly 145 candidates")
    return result


class FrozenPreprocessor:
    """Median imputation, all-column missing indicators, scaling; initialization only."""

    def fit(self, X):
        self.columns = list(X.columns)
        values = X.to_numpy(float)
        self.median = np.array(
            [
                np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0
                for column in values.T
            ]
        )
        self.scaler = StableScaler().fit(self.impute(values), np.ones(len(values)))
        return self

    def impute(self, values):
        missing = ~np.isfinite(values)
        return np.concatenate([np.where(missing, self.median, values), missing], axis=1)

    def transform(self, X):
        if list(X.columns) != self.columns:
            raise ValueError("feature allowlist changed after initialization")
        values = self.scaler.transform(self.impute(X.to_numpy(float)))
        return np.column_stack([np.ones(len(values)), values])


class OnlineState:
    def __init__(self, candidate, dimension=0):
        self.candidate = candidate
        self.time = None
        self.mass = 0.0
        self.total = 0.0
        self.loss = np.zeros(8)
        self.gram = np.zeros((dimension, dimension))
        self.rhs = np.zeros(dimension)

    def decay(self, time):
        if self.time is not None:
            if time < self.time:
                raise ValueError("feedback release times must be ordered")
            factor = np.exp2(-(time - self.time) / (3600 * self.candidate.half_life))
            self.mass *= factor
            self.total *= factor
            self.loss *= factor
            self.gram *= factor
            self.rhs *= factor
        self.time = time

    def update(self, time, y, base, experts, x):
        self.decay(time)
        self.mass += 1
        self.total += y - base
        self.loss += (experts - y) ** 2
        if self.candidate.family == "ridge":
            self.gram += np.outer(x, x)
            self.rhs += x * (y - base)

    def weights(self):
        prior = np.full(8, 0.5 / 7)
        prior[3] = 0.5
        mean_loss = self.loss / self.mass if self.mass else self.loss
        log_weights = np.log(prior) - self.candidate.rate * mean_loss / (mean_loss[3] + 1e-6)
        weights = np.exp(log_weights - log_weights.max())
        return weights / weights.sum()

    def correction(self, time, base, experts, x):
        self.decay(time)
        if self.candidate.family == "bias":
            return self.total / self.mass if self.mass else 0.0
        if self.candidate.family == "aggregation":
            return float(self.weights() @ experts - base)
        prior = np.eye(len(self.rhs)) * self.candidate.penalty
        # The intercept also has the declared ridge prior; forgetting never removes it.
        coefficients = cho_solve(cho_factor(self.gram + prior, lower=True), self.rhs)
        return float(x @ coefficients)


class CandidateBank:
    def __init__(self, cfg, initial_features):
        self.cfg = cfg
        self.candidates = candidates(cfg)
        self.preprocessing = {
            name: FrozenPreprocessor().fit(frame) for name, frame in initial_features.items()
        }
        self.states = {}
        for candidate in self.candidates[1:]:
            dimension = len(self.preprocessing[candidate.features].columns) * 2 + 1
            self.states.setdefault(candidate.key, OnlineState(candidate, dimension))
        self.feedback = []
        self.failures = []

    def vectors(self, features):
        return {
            name: processor.transform(features[name])[0]
            for name, processor in self.preprocessing.items()
        }

    def update(self, time, target_end, y, base, experts, vectors):
        if target_end > time:
            raise ValueError("feedback cannot arrive before its target finishes")
        self.feedback.append((time, target_end))
        for key, state in self.states.items():
            state.update(time, y, base, experts, vectors[key[-1]])

    def forecasts(self, time, base, selected, experts, vectors, age, coverage, active=None):
        recent = [
            end
            for release, end in self.feedback
            if release <= time and time - self.cfg["feedback_window_s"] < end <= time
        ]
        reason = (
            "stale_input"
            if not np.isfinite(age) or age > self.cfg["maximum_observation_age_s"]
            else "low_input_coverage"
            if not np.isfinite(coverage) or coverage < self.cfg["minimum_predictor_coverage"]
            else "insufficient_feedback"
            if len(recent) < self.cfg["minimum_feedback_windows"]
            else "stale_feedback"
            if time - max(recent) > self.cfg["maximum_feedback_age_s"]
            else ""
        )
        raw = {}
        keys = {
            c.key
            for c in self.candidates
            if c.family != "history" and (active is None or c.id in active)
        }
        for key, state in self.states.items():
            if key not in keys or reason:
                raw[key] = 0.0
                continue
            try:
                raw[key] = state.correction(time, base, experts, vectors[key[-1]])
                if not np.isfinite(raw[key]):
                    raise ValueError("nonfinite correction")
            except (ValueError, np.linalg.LinAlgError) as error:
                self.failures.append({"time": time, "state": list(key), "reason": str(error)})
                raw[key] = np.nan
        result = []
        for candidate in self.candidates:
            if active is not None and candidate.id not in active:
                continue
            correction = 0 if candidate.family == "history" else raw[candidate.key]
            fallback = reason or ("model_failure" if not np.isfinite(correction) else "")
            bounded = candidate.strength * np.clip(
                correction, -self.cfg["correction_cap"], self.cfg["correction_cap"]
            )
            prediction = selected if fallback else float(np.clip(base + bounded, 0, 1))
            result.append(
                {
                    "candidate": candidate.id,
                    "prediction": prediction,
                    "correction": 0.0 if fallback else float(bounded),
                    "raw_correction": float(correction) if np.isfinite(correction) else None,
                    "fallback_reason": fallback,
                    "feedback_windows": len(recent),
                }
            )
        return result

    def checkpoint(self):
        return pickle.dumps(self, protocol=5)

    @staticmethod
    def resume(payload):
        return pickle.loads(payload)  # Only checkpoints written by this local replay are accepted.
