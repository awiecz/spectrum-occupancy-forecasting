"""Scientific invariants of feedback-driven development, independent of favorable scores."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.adaptive_models import (
    Candidate,
    CandidateBank,
    FrozenPreprocessor,
    OnlineState,
    candidates,
)
from spectrum_occupancy_forecasting.experiments.adaptive_replay import (
    delivered_feedback,
    feature_frames,
    replay,
    select,
)
from spectrum_occupancy_forecasting.experiments.fine_spectrum import FINE_COLUMNS
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.learning_inputs import LearningNode
from spectrum_occupancy_forecasting.robustness.degrade import Degradation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    cfg = load_config(ROOT / "configs/followup_v2.yaml")
    cfg.update(yaml.safe_load((ROOT / "configs/adaptive_learning.yaml").read_text()))
    cfg["channels_mhz"] = [2437.0]
    t = 1_600_000_000.0 + np.arange(0, 4 * 86400, 60.0)
    rng = np.random.default_rng(31)
    p = -100 + rng.normal(0, 0.2, len(t)) + 8 * (rng.random(len(t)) < 0.3)
    frame = pd.DataFrame(
        {"time_s": t, "target_dbm": p, **{name: p + i / 10 for i, name in enumerate(FINE_COLUMNS)}}
    )
    with threadpool_limits(limits=2):
        yield cfg, frame


def test_grid_and_ties(fixture):
    cfg, _ = fixture
    grid = candidates(cfg)
    assert len(grid) == 145
    assert [
        sum(c.family == family for c in grid)
        for family in ("history", "bias", "aggregation", "ridge")
    ] == [1, 12, 36, 96]
    assert min(grid, key=lambda c: c.tie).family == "history"
    assert Candidate("ridge", 24, 0.25, 1000).tie < Candidate("ridge", 2, 0.5, 1).tie


@pytest.mark.parametrize("site", ["CC1", "LW1"])
def test_future_corruption_and_immutable_snapshots(fixture, site):
    cfg, frame = fixture
    node = LearningNode(frame, site, cfg)
    origins = node.data.t0 + 86400 + np.arange(0, 6 * 3600, 900)
    reference, updates, bank = replay(node, cfg, origins)
    corrupted = frame.copy()
    cutoff = origins[-1] + 900
    corrupted.loc[corrupted.time_s >= cutoff, ["target_dbm", *FINE_COLUMNS]] = 500
    earlier, earlier_updates, earlier_bank = replay(
        LearningNode(corrupted, site, cfg), cfg, origins
    )
    pd.testing.assert_frame_equal(reference, earlier)
    pd.testing.assert_frame_equal(updates, earlier_updates)
    pd.testing.assert_frame_equal(bank.feature_snapshots, earlier_bank.feature_snapshots)


def test_delivered_feedback_boundaries_delay_dropout(fixture):
    cfg, frame = fixture
    node = LearningNode(frame, "CC1", cfg)
    origin = node.data.t0 + 86400
    with pytest.raises(ValueError, match="before delivery"):
        delivered_feedback(node, origin, origin + 1800, Degradation(delay_s=60), 0, cfg)
    y, count = delivered_feedback(node, origin, origin + 1800, Degradation(), 0, cfg)
    keep = (frame.time_s >= origin + 900) & (frame.time_s < origin + 1800)
    assert count == keep.sum() == 15
    assert y == ((frame.loc[keep, "target_dbm"] > node.data.thresholds[2437]).mean())
    y, count = delivered_feedback(node, origin, origin + 1800, Degradation(dropout_p=0.99), 4, cfg)
    assert y is None and count < 12


def test_controller_bounds_preprocessing_checkpoint(fixture):
    cfg, frame = fixture
    node = LearningNode(frame, "CC1", cfg)
    origins, y, _ = node.training(node.data.t0 + 86400, 900)
    X, _ = feature_frames(node, origins, y.mean(), Degradation(), 0)
    bank = CandidateBank(cfg, X)
    vectors = bank.vectors({name: value.iloc[[-1]] for name, value in X.items()})
    time = origins[-1] + 1800
    for i in range(4):
        bank.update(time - (3 - i) * 900, time - (3 - i) * 900, 1, 0.2, np.full(8, 0.2), vectors)
    safe = bank.forecasts(time, 0.2, 0.7, np.full(8, 0.2), vectors, 0, 1)
    assert safe[0]["prediction"] == 0.2
    assert all(0 <= row["prediction"] <= 1 and abs(row["correction"]) <= 0.1 for row in safe)
    for age, coverage in ((901, 1), (0, 0.49), (np.nan, 1)):
        estimates = bank.forecasts(time, 0.2, 0.7, np.full(8, 0.2), vectors, age, coverage)
        assert all(row["prediction"] == 0.7 and row["correction"] == 0 for row in estimates)
    restored = CandidateBank.resume(bank.checkpoint())
    assert bank.forecasts(time, 0.2, 0.7, np.full(8, 0.2), vectors, 0, 1) == restored.forecasts(
        time, 0.2, 0.7, np.full(8, 0.2), vectors, 0, 1
    )
    assert all(
        not any(s in name for s in ("calendar", "hour", "x_last", "site"))
        for name in X["history_fine"]
    )
    original = bank.preprocessing["history"].median.copy()
    bank.preprocessing["history"].transform(X["history"] * 100)
    np.testing.assert_array_equal(original, bank.preprocessing["history"].median)


def test_ridge_forgetting_and_aggregation_normalization():
    ridge = OnlineState(Candidate("ridge", 2, 1, 10), 3)
    x = np.array([1, 2, 3])
    ridge.update(0, 0.6, 0.2, np.full(8, 0.2), x)
    ridge.decay(7200)
    np.testing.assert_allclose(ridge.gram, 0.5 * np.outer(x, x))
    assert np.isfinite(ridge.correction(7200, 0.2, np.full(8, 0.2), x))
    aggregate = OnlineState(Candidate("aggregation", 2, 1, rate=8))
    aggregate.update(0, 1, 0.2, np.linspace(0, 1, 8), x)
    assert aggregate.weights().sum() == pytest.approx(1)
    assert (aggregate.weights() >= 0).all()


@pytest.mark.parametrize("site", ["CC1", "LW1"])
def test_validation_embargo_future_selection_and_insufficient(fixture, site):
    cfg, frame = fixture
    node = LearningNode(frame, site, cfg)
    origin = node.data.t0 + 172800
    chosen, history, scores, _, failures = select(node, cfg, origin)
    assert not failures
    assert len(scores) == 145
    assert all(row["validation_end"] == origin - 36000 for row in scores)
    corrupted = frame.copy()
    corrupted.loc[corrupted.time_s >= origin - 36000, ["target_dbm", *FINE_COLUMNS]] = 700
    changed = select(LearningNode(corrupted, site, cfg), cfg, origin)
    assert (chosen, history, scores) == changed[:3]
    insufficient = select(node, cfg, node.data.t0 + 86400)
    assert insufficient[-1] == "insufficient_validation"
    assert all(value == candidates(cfg)[0].id for value in insufficient[0].values())


def test_constant_missing_imputation():
    X = pd.DataFrame({"constant": [1, 1], "missing": [np.nan, np.nan]})
    processor = FrozenPreprocessor().fit(X)
    assert np.isfinite(processor.transform(X)).all()


def test_replay_checkpoint_resume_and_session_reset(fixture):
    cfg, frame = fixture
    node = LearningNode(frame, "CC1", cfg)
    origins = node.data.t0 + 86400 + np.arange(0, 7200, 900)
    full, feedback, _ = replay(node, cfg, origins, deg=Degradation(delay_s=60))
    _, _, stopped = replay(node, cfg, origins, deg=Degradation(delay_s=60), stop_after=4)
    resumed, resumed_feedback, _ = replay(
        node, cfg, origins, deg=Degradation(delay_s=60), checkpoint=stopped.checkpoint()
    )
    pd.testing.assert_frame_equal(full, resumed)
    pd.testing.assert_frame_equal(feedback, resumed_feedback)
    shifted = frame.copy()
    shifted.time_s += 30 * 86400
    other = LearningNode(shifted, "CC1", cfg)
    reset, _, _ = replay(other, cfg, origins + 30 * 86400, deg=Degradation(delay_s=60))
    np.testing.assert_allclose(full.prediction, reset.prediction)


def test_scoring_labels_cannot_become_delivered_feedback(fixture, monkeypatch):
    cfg, frame = fixture
    node = LearningNode(frame, "CC1", cfg)
    origins = node.data.t0 + 86400 + np.arange(0, 7200, 900)
    initial = node.training(node.data.t0 + 86400, 900)
    expected, feedback, _ = replay(node, cfg, origins, deg=Degradation(dropout_p=.5), seed=2)
    monkeypatch.setattr(node, "training", lambda *args: initial)
    monkeypatch.setattr(node.data, "labels", lambda *args: pytest.fail("scoring labels accessed"))
    actual, delivered, _ = replay(node, cfg, origins, deg=Degradation(dropout_p=.5), seed=2)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(feedback, delivered)
    updated = delivered[delivered.update_reason == "updated"]
    assert (updated.feedback_count >= .8 * 900 / node.data.dt).all()
