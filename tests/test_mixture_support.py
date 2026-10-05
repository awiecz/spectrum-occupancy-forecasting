"""Chronological selection, fallback and saved-report contracts for the retained mixture."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.followup import load_config, scenarios
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData, maximum_lookback
from spectrum_occupancy_forecasting.experiments.followup_models import augment_training, training
from spectrum_occupancy_forecasting.experiments.history_mixture import HistoryMixture
from spectrum_occupancy_forecasting.experiments.mixture_reporting import (
    advancement,
    attach_reference,
    report,
)
from spectrum_occupancy_forecasting.experiments.mixture_support import (
    InsufficientValidation,
    validation_blocks,
)
from spectrum_occupancy_forecasting.experiments.mixture_support import (
    select as chronological_select,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation


def select(*args, **kwargs):
    return chronological_select(*args, model_class=HistoryMixture, **kwargs)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/history_mixture.yaml")


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("correction") / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=4.1, dt_s=60, seed=3)
    return band_table_from_blocks(path, [2412, 2437, 2462])


def test_actual_delay_and_long_gap(config, raw):
    node = NodeData(raw, config, 3)
    at = node.t0 + 3 * 86400
    X, y = training(node, at - maximum_lookback(config), 900)
    model = HistoryMixture(
        {"strength": 1, "alpha": 0.1}, {"id": "recent_7200s", "window": 7200}, config, node.dt
    ).fit(X, y)
    delayed = node.features([at], Degradation(delay_s=1800))
    assert model.details(delayed)["fallback_reason"][0] == "stale"
    dropped = node.features([at], Degradation(dropout_p=0.999999), 7)
    np.testing.assert_array_equal(
        model.predict(dropped), model.details(dropped)["history_prediction"]
    )


def test_validation_embargo_fallback_and_future_corruption(config, raw):
    node = NodeData(raw, config, 3)
    assert validation_blocks(node, node.t0 + 2 * 86400, 900, 36) is None
    assert select(node, node.t0 + 2 * 86400, 900, 36, "clean")[-1] == "insufficient_earlier_data"
    at = node.t0 + 3 * 86400
    for X, y, XV, yy, o in validation_blocks(node, at, 900, 36):
        assert len(X) and len(y) and len(XV) and len(yy)
        assert o.max() + 900 + config["window_s"] <= at - maximum_lookback(config)
        assert X.t_start_s.max() + 1800 <= o.min() - maximum_lookback(config)
    changed = raw.copy()
    changed.loc[changed.t_start_s >= at, "mean_dbm"] += 100
    other = NodeData(changed, config, 3)
    with threadpool_limits(1):
        a = select(node, at, 900, 12, "clean")
        b = select(other, at, 900, 12, "clean")
    assert a[:3] == b[:3]
    X, y = training(node, at - maximum_lookback(config), 900)
    XX, yy = training(other, at - maximum_lookback(config), 900)
    m = HistoryMixture(a[1], a[0], config, node.dt).fit(X, y)
    mm = HistoryMixture(b[1], b[0], config, other.dt).fit(XX, yy)
    np.testing.assert_array_equal(m.predict(node.features([at])), mm.predict(other.features([at])))


def test_augmentation_keeps_labels(config, raw):
    node = NodeData(raw, config, 3)
    X, y = training(node, node.t0 + 2 * 86400, 900)
    _, yy = augment_training(node, X, y)
    np.testing.assert_array_equal(yy, np.tile(y, 6))


def test_insufficient_covered_validation_is_recorded(config, raw, monkeypatch):
    module = importlib.import_module("spectrum_occupancy_forecasting.experiments.mixture_support")

    def unavailable(*args):
        raise InsufficientValidation("insufficient_validation_targets")

    monkeypatch.setattr(module, "validation_blocks", unavailable)
    node = NodeData(raw, config, 3)
    result = select(node, node.t0 + 3 * 86400, 900, 36, "clean")
    assert result[0]["window"] == 7200 and result[1]["strength"] == 0
    assert result[-1] == "insufficient_validation_targets"


def test_exact_validation_ties_prefer_history_only(config, raw):
    constant = raw.copy()
    constant["mean_dbm"] = -100
    node = NodeData(constant, config, 3)
    with threadpool_limits(1):
        selected = select(node, node.t0 + 3 * 86400, 900, 36, "clean")
    assert selected[0]["id"] == "recent_300s"
    assert selected[1]["id"] == "history_only"
    assert all(c["validation_mse"] == 0 for c in selected[2])


def test_saved_decision_metrics_and_common_alignment(config, tmp_path):
    cfg = config | {
        "bootstrap_resamples": 20,
        "dropout_probabilities": [0.5],
        "outage_lengths_s": [],
        "delays_s": [600],
    }
    rows, records = ([], [])
    for h in cfg["validation_hours"]:
        for regime in cfg["training_regimes"]:
            sid = f"{h}h_{regime}"
            for node in ("CC1", "LW1"):
                for fold in (1, 2):
                    records.append(
                        {
                            "node": node,
                            "task": "primary",
                            "strategy_id": sid,
                            "fold": fold,
                            "common_fold": True,
                            "selection_status": "selected",
                            "status": "completed",
                            "candidates": [],
                            "operating_thresholds": [],
                        }
                    )
                    for scenario, deg in scenarios(cfg).items():
                        for seed in cfg["mask_seeds"] if deg.dropout_p else [0]:
                            for k in range(2):
                                for method, p in (
                                    ("selected_learned", 0.3),
                                    ("selected_history", 0.4),
                                    ("fixed_history", 0.4),
                                    ("climatology", 0.4),
                                ):
                                    rows.append(
                                        {
                                            "node": node,
                                            "fold": fold,
                                            "task": "primary",
                                            "strategy_id": sid,
                                            "scenario": scenario,
                                            "seed": seed,
                                            "origin_s": fold * 86400 + k * 900,
                                            "y": 0.2,
                                            "prediction": p,
                                            "method": method,
                                            "variant": "full",
                                            "lead_s": 900,
                                            "offset_db": 3,
                                            "target_count": 15,
                                            "common_fold": True,
                                            "fallback_reason": "none",
                                        }
                                    )
    frame = pd.DataFrame(rows)
    decision = advancement(frame, records, cfg)
    assert (
        decision["advances"]
        and decision["chosen_strategy"]["strategy_id"] == "36h_mixture_expanding"
    )
    frame.to_parquet(tmp_path / "predictions.parquet", index=False)
    saved = pd.read_parquet(tmp_path / "predictions.parquet")
    assert decision == advancement(saved, records, cfg)
    tables = report(saved, records, cfg, tmp_path)
    assert decision == json.loads((tmp_path / "advancement.json").read_text())
    clean = tables["common_summary"].query("method == 'selected_learned' and scenario == 'clean'")
    np.testing.assert_allclose(clean.mss_vs_selected, 0.75)
    broken = frame.drop(
        frame[(frame.strategy_id == "36h_mixture_expanding") & (frame.scenario == "clean")].index[
            :4
        ]
    )
    assert not advancement(broken, records, cfg)["strategies"][0]["qualifies"]


def test_advancement_missing_scenarios_and_reference_alignment(config):
    records, frames = ([], [])
    for node in ("CC1", "LW1"):
        for fold in (1, 2):
            records.append(
                {
                    "node": node,
                    "task": "primary",
                    "strategy_id": "36h_mixture_expanding",
                    "fold": fold,
                    "common_fold": True,
                    "selection_status": "selected",
                    "status": "completed",
                }
            )
            for method, prediction in (("selected_learned", 0.21), ("selected_history", 0.3)):
                frames.append(
                    {
                        "node": node,
                        "fold": fold,
                        "origin_s": fold * 86400,
                        "seed": 0,
                        "scenario": "clean",
                        "task": "primary",
                        "strategy_id": "36h_mixture_expanding",
                        "method": method,
                        "y": 0.2,
                        "prediction": prediction,
                        "candidate_id": "test",
                        "common_fold": True,
                    }
                )
    frame = pd.DataFrame(frames)
    decision = advancement(frame, records, config)
    assert not decision["advances"]
    criteria = decision["strategies"][0]["criteria"]
    assert all(c["passed"] for c in criteria if c["criterion"] == "clean_skill")
    assert any(not c["passed"] for c in criteria if "drop" in c["criterion"])
    canonical = frame[frame.method == "selected_learned"].copy()
    assert len(attach_reference(frame, canonical)) == len(frame) + len(canonical)
    canonical.loc[canonical.index[0], "y"] = 0.9
    with pytest.raises(ValueError, match="match every"):
        attach_reference(frame, canonical)
