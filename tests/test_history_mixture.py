"""Convexity, chronological isolation and quality fallback for history mixtures."""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData
from spectrum_occupancy_forecasting.experiments.followup_models import training
from spectrum_occupancy_forecasting.experiments.history_mixture import MODELS, HistoryMixture
from spectrum_occupancy_forecasting.experiments.mixture_support import select


def test_recovers_convex_signal_and_exports_weights():
    cfg = load_config("configs/history_mixture.yaml")
    rng = np.random.default_rng(2)
    X = pd.DataFrame({f"frac_{w}s": rng.random(400) for w in cfg["history_windows_s"]})
    for tau in cfg["decay_taus_s"]:
        X[f"decay_fraction_{tau}s"] = rng.random(400)
        X[f"decay_weight_{tau}s"] = 1
    X["t_start_s"] = np.arange(400) * 180
    X["n_900s"] = 15
    X["age_s"] = 0
    y = 0.75 * X.frac_900s + 0.25 * X.frac_7200s
    m = HistoryMixture(dict(alpha=0, strength=1), dict(id="recent_300s", window=300), cfg, 60).fit(
        X, y
    )
    w = m.est.weights
    assert np.all(w >= 0) and w.sum() == pytest.approx(1)
    F = m.feature_frame(X, m.details(X)["history_prediction"])
    assert np.mean((F.iloc[:, :8].to_numpy() @ w - y) ** 2) < 1e-8
    assert m.audit_state()["weights"] == w.tolist()
    assert len(HistoryMixture.candidates(cfg)) == 13


@pytest.mark.parametrize("name", list(MODELS))
def test_future_corruption_and_exact_fallback(tmp_path, name):
    cfg = load_config("configs/history_mixture.yaml")
    path = tmp_path / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=4.1, dt_s=60, seed=3)
    raw = band_table_from_blocks(path, [2412, 2437, 2462])
    at = raw.t_start_s.min() + 3 * 86400
    changed = raw.copy()
    changed.loc[changed.t_start_s >= at, "mean_dbm"] += 100
    a, b = NodeData(raw, cfg, 3), NodeData(changed, cfg, 3)
    cls = MODELS[name]
    with threadpool_limits(1):
        first = select(a, at, 900, 36, name, model_class=cls)
        second = select(b, at, 900, 36, name, model_class=cls)
    assert first[:3] == second[:3]
    X, y = training(a, at - 36000, 900)
    XX, yy = training(b, at - 36000, 900)
    m = cls(first[1], first[0], cfg, a.dt).fit(X, y)
    mm = cls(second[1], second[0], cfg, b.dt).fit(XX, yy)
    np.testing.assert_array_equal(m.predict(a.features([at])), mm.predict(b.features([at])))
    if hasattr(m, "est"):
        np.testing.assert_array_equal(m.est.weights, mm.est.weights)
    F = X.iloc[:3].copy()
    F["age_s"] = [901, 0, np.nan]
    F["n_900s"] = [15, 0, 15]
    np.testing.assert_array_equal(m.predict(F), m.details(F)["history_prediction"])
    d = m.details(X)
    assert np.max(np.abs(d["correction_value"])) <= 0.1 + 1e-15


def test_allowlist_boundary_zero_and_frozen_training_state(tmp_path):
    cfg = load_config("configs/history_mixture.yaml")
    path = tmp_path / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=3.1, dt_s=60, seed=3)
    node = NodeData(band_table_from_blocks(path, cfg["channels_mhz"]), cfg, 3)
    X, y = training(node, node.t0 + 2 * 86400, 900)
    history = dict(id="recent_7200s", window=7200)
    model = HistoryMixture(dict(alpha=0.1, strength=1), history, cfg, node.dt).fit(X, y)
    F = X.iloc[:5].copy()
    F["age_s"] = [900, 901, float("nan"), 0, 0]
    F["n_900s"] = [7.5, 15, 15, 7.49, 15]
    details = model.details(F)
    assert list(details["fallback_reason"]) == [
        "none",
        "stale",
        "missing_age",
        "low_coverage",
        "none",
    ]
    np.testing.assert_array_equal(
        details["prediction"][[1, 2, 3]], details["history_prediction"][[1, 2, 3]]
    )
    altered = F.copy()
    for col in altered:
        if "pw_" in col or "tod_" in col or "cross" in col:
            altered[col] = 999
    before = model.audit_state()
    np.testing.assert_array_equal(model.predict(F), model.predict(altered))
    assert before == model.audit_state()
    assert model.columns == [h for h in model.feature_frame(F, details["history_prediction"])]
    assert len(model.columns) == 9
    zero = HistoryMixture(dict(alpha=None, strength=0), history, cfg, node.dt).fit(X, y)
    np.testing.assert_array_equal(zero.predict(F), zero.details(F)["history_prediction"])
