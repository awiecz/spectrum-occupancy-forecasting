"""Quantile and interval forecasts for the sparse-regime study.

Learned intervals come from quantile boosting; comparator intervals add the empirical quantiles
of earlier validation errors to a point forecast (split-conformal style) or use the empirical
quantiles of earlier outcomes. Only earlier data is used to build any interval.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from spectrum_occupancy_forecasting.evaluation.quantile_metrics import monotone
from spectrum_occupancy_forecasting.experiments.followup_data import feature_columns


def fit_quantile_models(X, y, variant, levels, depth, leaf, cfg):
    """One boosted quantile regressor per level, fitted on the stacked matched training set."""
    columns = feature_columns(X, variant)
    matrix = X[columns].to_numpy()
    models = [
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=tau,
            max_iter=cfg["boosting_iterations"],
            learning_rate=cfg["boosting_learning_rate"],
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=cfg["model_seed"],
            early_stopping=False,
        ).fit(matrix, np.asarray(y))
        for tau in levels
    ]
    return columns, models


def predict_quantiles(columns, models, X):
    matrix = X[columns].to_numpy()
    return monotone(np.column_stack([model.predict(matrix) for model in models]))


def conformal_quantiles(point, validation_error, levels):
    """Point forecast plus signed quantiles of earlier validation errors (y - prediction)."""
    offsets = np.quantile(np.asarray(validation_error, dtype=float), levels)
    return monotone(np.asarray(point, dtype=float)[:, None] + offsets[None, :])


def empirical_quantiles(training_y, n, levels):
    """The same quantiles of earlier outcomes, repeated for ``n`` forecasts."""
    row = np.quantile(np.asarray(training_y, dtype=float), levels)
    return monotone(np.tile(row, (n, 1)))
