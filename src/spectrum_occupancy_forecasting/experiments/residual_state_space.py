"""Bounded residual Kalman filters, indexed by target completion rather than release."""

import pickle
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class StateCandidate:
    family: str = "history"
    coefficient: float = 0
    strength: float = 0
    features: str = "history"

    @property
    def id(self):
        return f"{self.family}_{self.coefficient}_{self.strength}"

    @property
    def tie(self):
        return (
            self.family != "history",
            self.strength,
            {"history": 0, "level": 1, "trend": 2}[self.family],
            self.coefficient,
        )


def candidates(cfg):
    grid = [StateCandidate()]
    grid.extend(
        StateCandidate(family, q, strength)
        for family in cfg["model_families"]
        for q in cfg["noise_coefficients"]
        for strength in cfg["state_strengths"]
    )
    if len(grid) != 19 or len({c.id for c in grid}) != 19:
        raise ValueError("the declared state-space grid must contain 19 candidates")
    return grid


STRATEGIES = {
    name: (lambda c, name=name: c.family in ("history", name)) for name in ("level", "trend")
}


class ResidualFilter:
    def __init__(self, family, coefficient, variance, half_life=2):
        self.family, self.r = family, variance
        self.q, self.k = coefficient * variance, np.log(2) / half_life
        self.mean = np.zeros(1 if family == "level" else 2)
        self.cov = np.eye(len(self.mean)) * variance
        self.time = None
        self._transitions = {}

    def transition(self, hours):
        if hours < 0:
            raise ValueError("target times must be chronological")
        if hours not in self._transitions:
            if self.family == "level":
                f, q = np.ones((1, 1)), np.array([[self.q * hours]])
            else:
                # Van Loan integral is stable also for tiny irregular time increments.
                a = np.array([[0.0, 1.0], [0.0, -self.k]])
                g = np.array([[0.0, 0.0], [0.0, self.q]])
                block = np.block([[a, g], [np.zeros((2, 2)), -a.T]])
                exponential = expm(block * hours)
                f = exponential[:2, :2]
                q = exponential[:2, 2:] @ f.T
                q = (q + q.T) / 2
            self._transitions[hours] = f, q
        return self._transitions[hours]

    def projected(self, time):
        hours = 0 if self.time is None else (time - self.time) / 3600
        f, q = self.transition(hours)
        return f @ self.mean, f @ self.cov @ f.T + q

    def update(self, time, residual):
        if not np.isfinite(residual):
            raise ValueError("nonfinite delivered residual")
        mean, covariance = self.projected(time)
        gain = covariance[:, 0] / (covariance[0, 0] + self.r)
        mean += gain * (residual - mean[0])
        h = np.zeros_like(covariance)
        h[:, 0] = gain
        identity = np.eye(len(mean)) - h
        covariance = identity @ covariance @ identity.T + self.r * np.outer(gain, gain)
        covariance = (covariance + covariance.T) / 2
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
            or np.linalg.eigvalsh(covariance).min() < -1e-10
        ):
            raise ValueError("invalid filter covariance or mean")
        self.mean, self.cov, self.time = mean, covariance, time


class EmptyProcessor:
    def transform(self, frame):
        return np.zeros((len(frame), 0))


class StateBank:
    snapshot_features = "history"

    def __init__(self, cfg, features, initial_y, initial_base):
        self.cfg, self.candidates = cfg, candidates(cfg)
        self.preprocessing = {name: EmptyProcessor() for name in features}
        residuals = np.asarray(initial_y) - np.asarray(initial_base)
        residuals = residuals[np.isfinite(residuals)]
        self.initialization_reason = (
            "insufficient_initialization" if len(residuals) < cfg["minimum_initialization"] else ""
        )
        variance = max(
            float(np.var(residuals, ddof=1)) if len(residuals) > 1 else 0, cfg["variance_floor"]
        )
        self.states = {
            (c.family, c.coefficient): ResidualFilter(
                c.family, c.coefficient, variance, cfg["trend_half_life_h"]
            )
            for c in self.candidates[1:]
        }
        self.feedback, self.failures, self.state_records, self.update_records = [], [], [], []
        if self.initialization_reason:
            self.failures.append(
                {"reason": self.initialization_reason, "usable_examples": len(residuals)}
            )
        self.failed = set()
        self.last_release = None

    def update(self, release, target_end, y, base, experts, vectors):
        if target_end > release or (self.last_release is not None and release < self.last_release):
            raise ValueError("feedback delivery precedes target completion or previous release")
        self.last_release = release
        self.feedback.append((release, target_end))
        self.update_records.append(
            {
                "release_s": release,
                "target_end_s": target_end,
                "feedback_y": float(y),
                "issued_fixed_history": float(base),
                "delivered_residual": float(y - base),
            }
        )
        if self.initialization_reason:
            return
        for key, state in self.states.items():
            if key in self.failed:
                continue
            try:
                state.update(target_end, y - base)
            except (ValueError, np.linalg.LinAlgError) as error:
                self.failed.add(key)
                self.failures.append(
                    {
                        "release_s": release,
                        "target_end_s": target_end,
                        "state": list(key),
                        "reason": str(error),
                    }
                )

    def forecasts(self, time, base, selected, experts, vectors, age, coverage, active=None):
        recent = [
            end
            for release, end in self.feedback
            if release <= time and time - self.cfg["feedback_window_s"] < end <= time
        ]
        reason = (
            "stale_input"
            if not np.isfinite(age) or age > 900
            else "low_input_coverage"
            if not np.isfinite(coverage) or coverage < 0.5
            else "insufficient_feedback"
            if len(recent) < 4
            else "stale_feedback"
            if time - max(recent) > 3600
            else ""
        )
        raw = {}
        for key, state in self.states.items():
            try:
                mean, covariance = state.projected(time + 1800)
                if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
                    raise ValueError("nonfinite projected state")
            except (ValueError, np.linalg.LinAlgError) as error:
                if key not in self.failed:
                    self.failures.append(
                        {"origin_s": time, "state": list(key), "reason": str(error)}
                    )
                self.failed.add(key)
                mean = np.full_like(state.mean, np.nan)
                covariance = np.full_like(state.cov, np.nan)
            raw[key] = float(mean[0])
            self.state_records.append(
                {
                    "origin_s": time,
                    "family": key[0],
                    "coefficient": key[1],
                    "last_target_end_s": state.time,
                    "last_release_s": self.last_release,
                    "level": float(state.mean[0]),
                    "slope": float(state.mean[1]) if len(state.mean) == 2 else None,
                    "cov_level": float(state.cov[0, 0]),
                    "cov_cross": float(state.cov[0, 1]) if len(state.mean) == 2 else None,
                    "cov_slope": float(state.cov[1, 1]) if len(state.mean) == 2 else None,
                    "predicted_residual": float(mean[0]),
                    "predicted_variance": float(covariance[0, 0]),
                    "observation_variance": state.r,
                }
            )
        records = []
        for candidate in self.candidates:
            if active is not None and candidate.id not in active:
                continue
            correction = (
                0.0
                if candidate.family == "history"
                else raw[candidate.family, candidate.coefficient]
            )
            fallback = reason
            if candidate.family != "history":
                fallback = (
                    fallback
                    or self.initialization_reason
                    or (
                        "model_failure"
                        if (candidate.family, candidate.coefficient) in self.failed
                        or not np.isfinite(correction)
                        else ""
                    )
                )
            bounded = candidate.strength * np.clip(correction, -0.1, 0.1)
            records.append(
                {
                    "candidate": candidate.id,
                    "prediction": selected if fallback else float(np.clip(base + bounded, 0, 1)),
                    "correction": 0.0 if fallback else float(bounded),
                    "raw_correction": float(correction) if np.isfinite(correction) else None,
                    "fallback_reason": fallback,
                    "feedback_windows": len(recent),
                }
            )
        return records

    def checkpoint(self):
        return pickle.dumps(self, protocol=5)

    @staticmethod
    def resume(payload):
        return pickle.loads(payload)
