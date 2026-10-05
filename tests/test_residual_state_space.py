"""Causality and filter invariants, independent of favorable development outcomes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.adaptive_replay import replay, select
from spectrum_occupancy_forecasting.experiments.fine_spectrum import FINE_COLUMNS
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.learning_inputs import LearningNode
from spectrum_occupancy_forecasting.experiments.residual_diagnosis import residual_pairs
from spectrum_occupancy_forecasting.experiments.residual_state_space import (
    STRATEGIES,
    ResidualFilter,
    StateBank,
    candidates,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    cfg = load_config(ROOT / "configs/followup_v2.yaml")
    for name in ("adaptive_learning", "residual_state_space"):
        cfg.update(yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text()))
    cfg["channels_mhz"] = [2437.0]
    t = 1_600_000_000.0 + np.arange(0, 4 * 86400, 60.0)
    rng = np.random.default_rng(77)
    p = -100 + rng.normal(0, 0.2, len(t)) + 8 * (rng.random(len(t)) < 0.3)
    frame = pd.DataFrame({"time_s": t, "target_dbm": p, **{name: p for name in FINE_COLUMNS}})
    with threadpool_limits(limits=2):
        yield cfg, frame


def make_bank(cfg, n=30):
    features = {
        "history": pd.DataFrame({"recent_7200s": np.full(n, 0.2)}),
        "history_fine": pd.DataFrame({"undeclared_power": np.zeros(n)}),
    }
    return StateBank(cfg, features, np.linspace(0.1, 0.4, n), np.full(n, 0.2))


def test_grid_ties_and_initialization(setup):
    cfg, _ = setup
    grid = candidates(cfg)
    assert len(grid) == 19
    assert sum(c.family == "level" for c in grid) == 9
    assert min(grid, key=lambda c: c.tie).family == "history"
    bank = make_bank(cfg)
    r = next(iter(bank.states.values())).r
    assert r == pytest.approx(np.var(np.linspace(0.1, 0.4, 30) - 0.2, ddof=1))
    bank.update(100, 100, 1, 0, None, None)
    assert next(iter(bank.states.values())).r == r
    insufficient = make_bank(cfg, 19)
    for t in (100, 1000, 1900, 2800):
        insufficient.update(t, t, 0.5, 0.2, None, None)
    rows = insufficient.forecasts(2800, 0.2, 0.6, None, None, 0, 1)
    assert all(
        x["prediction"] == 0.6 and x["fallback_reason"] == "insufficient_initialization"
        for x in rows[1:]
    )


@pytest.mark.parametrize("hours", [0, 0.000001, 0.25, 2, 5.3])
def test_exact_transition_and_integral(hours):
    state = ResidualFilter("trend", 0.1, 0.01)
    f, q = state.transition(hours)
    k = state.k
    np.testing.assert_allclose(
        f, [[1, -np.expm1(-k * hours) / k], [0, np.exp(-k * hours)]], atol=1e-14
    )
    expected = state.q * np.array(
        [
            [
                quad(lambda t: (-np.expm1(-k * t) / k) ** 2, 0, hours)[0],
                quad(lambda t: -np.expm1(-k * t) / k * np.exp(-k * t), 0, hours)[0],
            ],
            [
                quad(lambda t: -np.expm1(-k * t) / k * np.exp(-k * t), 0, hours)[0],
                quad(lambda t: np.exp(-2 * k * t), 0, hours)[0],
            ],
        ]
    )
    np.testing.assert_allclose(q, expected, atol=1e-14)
    assert np.linalg.eigvalsh(q).min() >= -1e-14
    level = ResidualFilter("level", 0.1, 0.01)
    assert level.transition(hours)[1][0, 0] == pytest.approx(0.001 * hours)


def test_forecasts_do_not_mutate_assimilation_and_stable_updates():
    a, b = ResidualFilter("trend", 0.1, 0.01), ResidualFilter("trend", 0.1, 0.01)
    for i in range(150):
        time = i * 900
        y = 0.02 if i < 60 else 0.08
        a.update(time, y)
        b.update(time, y)
        before = a.mean.copy(), a.cov.copy(), a.time
        a.projected(time + 1800)
        a.projected(time + 18000)
        np.testing.assert_array_equal(a.mean, before[0])
        np.testing.assert_array_equal(a.cov, before[1])
        assert a.time == before[2]
        assert np.linalg.eigvalsh(a.cov).min() >= 0
    np.testing.assert_array_equal(a.mean, b.mean)
    assert a.projected(time + 1800)[0][0] == pytest.approx(0.08, abs=0.005)


@pytest.mark.parametrize(
    "age,coverage,reason",
    [
        (901, 1, "stale_input"),
        (0, 0.49, "low_input_coverage"),
        (np.nan, 1, "stale_input"),
    ],
)
def test_exact_fallback_and_bounds(setup, age, coverage, reason):
    cfg, _ = setup
    bank = make_bank(cfg)
    for t in (100, 1000, 1900, 2800):
        bank.update(t + 60, t, 1, 0.2, None, None)
    rows = bank.forecasts(2860, 0.2, 0.7, None, None, age, coverage)
    assert all(row["prediction"] == 0.7 and row["fallback_reason"] == reason for row in rows)
    safe = bank.forecasts(2860, 0.2, 0.7, None, None, 0, 1)
    assert safe[0]["prediction"] == 0.2
    assert all(0 <= row["prediction"] <= 1 and abs(row["correction"]) <= 0.1 for row in safe)
    assert all(row["last_target_end_s"] == 2800 for row in bank.state_records)


@pytest.mark.parametrize("site", ["CC1", "LW1"])
def test_future_corruption_replay_selection_and_power_exclusion(setup, site):
    cfg, frame = setup
    node = LearningNode(frame, site, cfg)
    origins = node.data.t0 + 86400 + np.arange(16) * 900
    expected, feedback, _ = replay(node, cfg, origins, bank_factory=StateBank)
    corrupt = frame.copy()
    corrupt.loc[corrupt.time_s >= origins[-1] + 900, "target_dbm"] = 100
    corrupt[FINE_COLUMNS] = 1000  # These columns cannot enter this model.
    actual, actual_feedback, _ = replay(
        LearningNode(corrupt, site, cfg), cfg, origins, bank_factory=StateBank
    )
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(feedback, actual_feedback)
    evaluation = node.data.t0 + 3 * 86400
    options = dict(candidate_factory=candidates, bank_factory=StateBank, strategies=STRATEGIES)
    first = select(node, cfg, evaluation, **options)
    corrupt = frame.copy()
    corrupt.loc[corrupt.time_s >= evaluation, "target_dbm"] = 100
    second = select(LearningNode(corrupt, site, cfg), cfg, evaluation, **options)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_delay_dropout_checkpoint_reset_and_insufficient_validation(setup):
    cfg, frame = setup
    node = LearningNode(frame, "LW1", cfg)
    origins = node.data.t0 + 86400 + np.arange(18) * 900
    kwargs = dict(bank_factory=StateBank, deg=Degradation(delay_s=600))
    expected, feedback, bank = replay(node, cfg, origins, **kwargs)
    _, _, stopped = replay(node, cfg, origins, stop_after=8, **kwargs)
    actual, updates, resumed = replay(node, cfg, origins, checkpoint=stopped.checkpoint(), **kwargs)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(feedback, updates)
    assert bank.state_records == resumed.state_records
    released = feedback[feedback.update_reason == "updated"]
    assert ((released.release_s - released.target_end_s) == 600).all()
    _, missing, _ = replay(
        node, cfg, origins, bank_factory=StateBank, deg=Degradation(dropout_p=0.99), seed=5
    )
    assert not (missing.update_reason == "updated").any()
    selection = select(
        node,
        cfg,
        node.data.t0 + 86400,
        candidate_factory=candidates,
        bank_factory=StateBank,
        strategies=STRATEGIES,
    )
    assert selection[-1] == "insufficient_validation"
    fresh, _, _ = replay(node, cfg, origins, **kwargs)
    pd.testing.assert_frame_equal(expected, fresh)


def test_residual_pairs_do_not_cross_gaps_or_use_future_feedback():
    labels = pd.DataFrame(
        {
            "origin_s": [0, 900, 1800, 3600, 4500, 5400],
            "fold": [0] * 6,
            "y": [0.2] * 6,
            "fixed_history": [0.1] * 6,
        }
    )
    anchors = labels[["origin_s", "fixed_history"]]
    feedback = pd.DataFrame(
        {
            "origin_s": labels.origin_s,
            "release_s": labels.origin_s + 1800,
            "feedback_y": [0.3] * 6,
            "update_reason": ["updated"] * 6,
        }
    )
    pairs = residual_pairs(labels, anchors, feedback)
    assert pairs.origin_s.tolist() == [1800, 5400]
    assert (pairs.release_s <= pairs.origin_s).all()


def test_cli_gate_and_configuration_rejections(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "state_runner", ROOT / "scripts/run_residual_state_space.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = ROOT / "configs/residual_state_space.yaml"
    runner.configuration(config)
    bad = tmp_path / "bad.yaml"
    bad.write_text("study: adaptive-learning")
    with pytest.raises(ValueError, match="mismatch"):
        runner.configuration(bad)
    (tmp_path / "gate.json").write_text(json.dumps({"passes": False, "specification_hash": "bad"}))
    with pytest.raises(ValueError, match="gate failed"):
        runner.verify_gate(tmp_path, tmp_path, config)


def test_scoring_labels_never_enter_state_updates(setup, monkeypatch):
    cfg, frame = setup
    node = LearningNode(frame, "CC1", cfg)
    origins = node.data.t0 + 86400 + np.arange(12) * 900
    kwargs = dict(bank_factory=StateBank, deg=Degradation(dropout_p=0.5), seed=2)
    initial = node.training(node.data.t0 + 86400, 900)
    expected, feedback, _ = replay(node, cfg, origins, **kwargs)
    monkeypatch.setattr(node, "training", lambda *a: initial)
    monkeypatch.setattr(node.data, "labels", lambda *a: pytest.fail("scoring labels accessed"))
    actual, delivered, _ = replay(node, cfg, origins, **kwargs)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(feedback, delivered)


def test_nonfinite_feedback_failure_is_explicit(setup):
    cfg, _ = setup
    bank = make_bank(cfg)
    for time in (100, 1000, 1900, 2800):
        bank.update(time, time, 0.5, 0.2, None, None)
    bank.update(2900, 2900, np.nan, 0.2, None, None)
    assert len(bank.failures) == 6
    rows = bank.forecasts(2900, 0.2, 0.7, None, None, 0, 1)
    assert all(
        row["fallback_reason"] == "model_failure" and row["prediction"] == 0.7 for row in rows[1:]
    )
