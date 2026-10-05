"""Constrained mixtures of the original eight histories; exploratory reuse only."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .followup_models import history_prediction
from .mixture_support import BoundedCorrection, history_candidates


def sample_weights(origins, half_life_s):
    """Normalize to mean one so penalty magnitudes remain comparable between strategies."""
    origins = np.asarray(origins, dtype=float)
    if half_life_s is None:
        return np.ones(len(origins))
    weights = np.exp2((origins - origins.max()) / half_life_s)
    return weights / weights.mean()


class MixtureEstimator:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, F):
        return F.iloc[:, :8].to_numpy() @ self.weights - F.selected_history.to_numpy()


class HistoryMixture(BoundedCorrection):
    half_life_s = None

    def audit_state(self):
        return {
            "history_order": [h["id"] for h in history_candidates(self.cfg)],
            "weights": self.est.weights.tolist() if hasattr(self, "est") else None,
            "training_prior": self.mean,
            "half_life_s": self.half_life_s,
            "optimization": getattr(self, "optimization", None),
        }

    @staticmethod
    def candidates(cfg):
        return [dict(id="history_only", alpha=None, strength=0)] + [
            dict(id=f"mixture_p{a}_s{s}", alpha=a, strength=s)
            for s in sorted(cfg["correction_strengths"])
            for a in sorted(cfg["mixture_penalties"], reverse=True)
        ]

    def feature_frame(self, X, base):
        F = pd.DataFrame(
            {h["id"]: history_prediction(h, X, self.mean) for h in history_candidates(self.cfg)},
            index=X.index,
        )
        F["selected_history"] = base
        return F

    def fit(self, X, y):
        self.mean = float(np.mean(y))
        if self.choice["strength"] == 0:
            return self
        F = self.feature_frame(X, history_prediction(self.history, X, self.mean))
        self.columns = list(F)
        H = F.iloc[:, :8].to_numpy()
        prior = np.zeros(8)
        prior[list(F).index(self.history["id"])] = 1
        weights = sample_weights(X.t_start_s, self.half_life_s)
        weights /= weights.sum()
        alpha = self.choice["alpha"]

        def objective(w):
            errors = H @ w - np.asarray(y)
            return np.sum(weights * errors**2) + alpha * np.sum((w - prior) ** 2)

        def gradient(w):
            return 2 * H.T @ (weights * (H @ w - np.asarray(y))) + 2 * alpha * (w - prior)

        result = minimize(
            objective,
            prior,
            jac=gradient,
            method="SLSQP",
            bounds=[(0, 1)] * 8,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(8)},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise ValueError(f"history mixture optimization failed: {result.message}")
        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        self.est = MixtureEstimator(w)
        self.optimization = {"objective": float(objective(w)), "iterations": int(result.nit)}
        return self


class RecentSixHourMixture(HistoryMixture):
    half_life_s = 21600


class RecentDayMixture(HistoryMixture):
    half_life_s = 86400


MODELS = {
    "mixture_expanding": HistoryMixture,
    "mixture_6h": RecentSixHourMixture,
    "mixture_24h": RecentDayMixture,
}


def validate_config(cfg):
    expected = {
        "study": "history-mixture",
        "validation_hours": [36],
        "training_regimes": list(MODELS),
        "mixture_penalties": [0, 0.001, 0.01, 0.1],
        "correction_strengths": [0.25, 0.5, 1],
        "correction_cap": 0.1,
        "primary_lead_s": 900,
        "window_s": 900,
        "primary_offset_db": 3,
        "maximum_correction_age_s": 900,
        "minimum_predictor_coverage": 0.5,
        "confirmation_strategy": None,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"history-mixture declared setting mismatch: {key}")
