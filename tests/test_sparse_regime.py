import numpy as np
import pytest

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments import sparse_regime as sr
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData, folds
from spectrum_occupancy_forecasting.robustness.degrade import Degradation

CONFIG = sr.ROOT / "configs/sparse_regime.yaml"


@pytest.fixture(scope="module")
def cfg():
    return sr.load_config(CONFIG)


@pytest.fixture(scope="module")
def node(tmp_path_factory, cfg):
    path = tmp_path_factory.mktemp("syn") / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=3.1, dt_s=60.0, seed=3)
    raw = band_table_from_blocks(path, cfg["channels_mhz"], site="SYN", session=0)
    return NodeData(raw, cfg, cfg["primary_offset_db"])


def test_config_has_no_third_site_and_roles_use_distinct_seeds(cfg):
    assert "CC2" not in CONFIG.read_text(encoding="utf-8")
    assert cfg["nodes"] == ["CC1", "LW1"]
    roles = [set(cfg[k]) for k in ("train_seeds", "validation_seeds", "score_seeds")]
    assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    assert not set().union(*roles) & set(cfg["mask_seeds"])


def test_loader_refuses_a_drifting_grid(tmp_path, cfg):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "ridge_alphas: [1, 10, 100, 1000]", "ridge_alphas: [1, 10]"
    )
    changed = tmp_path / "sparse_regime.yaml"
    changed.write_text(text, encoding="utf-8")
    (tmp_path / "followup_v2.yaml").write_text(
        (sr.ROOT / "configs/followup_v2.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ridge_alphas"):
        sr.load_config(changed)


def test_deterministic_degradations_use_a_single_copy(cfg):
    assert sr.effective_seeds(Degradation(delay_s=3600), (1, 2, 3)) == (0,)
    assert sr.effective_seeds(Degradation(dropout_p=0.9), (1, 2, 3)) == (1, 2, 3)


def test_shrinkage_weight_is_clipped_and_recovers_known_weight():
    mean = np.full(6, 0.2)
    history = np.array([0.0, 0.1, 0.3, 0.5, 0.9, 0.2])
    y = mean + 0.4 * (history - mean)
    assert sr.shrinkage_weight(y, history, mean) == pytest.approx(0.4)
    assert sr.shrinkage_weight(y * 10, history, mean) == 1.0
    assert sr.shrinkage_weight(y, mean, mean) == 0.0


def test_matched_training_targets_finish_before_the_cutoff_and_stack_seeds(node, cfg):
    lo, _ = folds(node.t0, node.last, cfg)[0]
    cutoff = lo - sr.maximum_lookback(cfg)
    deg = Degradation(dropout_p=0.9)
    X1, y1 = sr.matched_training(node, cutoff, cfg["primary_lead_s"], deg, (101,))
    X3, y3 = sr.matched_training(node, cutoff, cfg["primary_lead_s"], deg, (101, 102, 103))
    assert len(X3) == 3 * len(X1) and len(y3) == 3 * len(y1)
    assert np.all(np.isfinite(y3))


def test_labels_do_not_change_with_the_scenario(node, cfg):
    lo, hi = folds(node.t0, node.last, cfg)[0]
    origins = node.origins(lo, hi, cfg["primary_lead_s"], cfg["window_s"], common=True)
    o_clean, y_clean, _ = node.dataset(origins, cfg["primary_lead_s"])
    node.features(o_clean, Degradation(dropout_p=0.99), 21)
    o_after, y_after, _ = node.dataset(origins, cfg["primary_lead_s"])
    assert np.array_equal(y_clean, y_after)


def test_validation_masks_differ_from_scoring_masks(node, cfg):
    lo, hi = folds(node.t0, node.last, cfg)[0]
    o = node.origins(lo, hi, cfg["primary_lead_s"], cfg["window_s"], common=True)
    deg = Degradation(dropout_p=0.9)
    validation = node.features(o, deg, cfg["validation_seeds"][0])
    scoring = node.features(o, deg, cfg["score_seeds"][0])
    assert not validation.equals(scoring)


def test_one_synthetic_scenario_runs_end_to_end_and_is_causal(node, cfg, tmp_path):
    lo, hi = folds(node.t0, node.last, cfg)[0]
    deg = Degradation(dropout_p=0.99)
    small = dict(
        cfg,
        train_seeds=cfg["train_seeds"][:1],
        validation_seeds=cfg["validation_seeds"][:1],
        score_seeds=cfg["score_seeds"][:2],
    )
    forecasts, intervals, record = sr.run_scenario(
        node,
        lo,
        hi,
        0,
        "dropout_0.99",
        deg,
        cfg["primary_lead_s"],
        small,
        intervals=True,
        ablate=True,
    )
    assert record["status"] == "completed" and record["hstar"] in sr.SIMPLE
    methods = set(forecasts["method"])
    assert {
        "hstar",
        "selected_learned",
        "fixed_history",
        "shrunk_history",
        "climatology",
    } <= methods
    assert any(m.startswith("hgb__") for m in methods)
    assert forecasts["prediction"].between(0, 1).all()
    assert set(forecasts["seed"]) == set(small["score_seeds"])
    assert (forecasts["target_end_s"] <= hi + cfg["primary_lead_s"] + cfg["window_s"]).all()
    quantile_columns = [c for c in intervals if c.startswith("q")]
    assert len(quantile_columns) == len(cfg["quantiles"])
    values = intervals[quantile_columns].to_numpy()
    assert (np.diff(values, axis=1) >= -1e-12).all() and values.min() >= 0 and values.max() <= 1
    assert set(intervals["method"]) == {"qhgb", "conformal_learned", "conformal_hstar", "empirical"}


def test_run_node_writes_resumable_parts(tmp_path, cfg):
    path = tmp_path / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=3.1, dt_s=60.0, seed=3)
    raw = band_table_from_blocks(path, cfg["channels_mhz"], site="SYN", session=0)
    small = dict(
        cfg,
        train_seeds=cfg["train_seeds"][:1],
        validation_seeds=cfg["validation_seeds"][:1],
        score_seeds=cfg["score_seeds"][:1],
        interval_scenarios=[],
    )
    out = tmp_path / "parts"
    first = sr.run_node(raw, small, out, scenarios=["delay_3600s"], progress=lambda *a, **k: None)
    assert first["folds"][0]["status"] == "completed"
    with pytest.raises(FileExistsError):
        sr.run_node(raw, small, out, scenarios=["delay_3600s"], progress=lambda *a, **k: None)
    again = sr.run_node(
        raw, small, out, scenarios=["delay_3600s"], resume=True, progress=lambda *a, **k: None
    )
    assert again["folds"][0]["status"] == "completed"
    assert again["folds"][0]["n_evaluation"] == first["folds"][0]["n_evaluation"]


def test_conformal_bands_use_only_validation_errors():
    from spectrum_occupancy_forecasting.experiments.intervals import conformal_quantiles

    point = np.array([0.1, 0.5])
    errors = np.array([-0.05, 0.0, 0.05, 0.1])
    q = conformal_quantiles(point, errors, [0.25, 0.75])
    assert q.shape == (2, 2)
    assert np.allclose(q[0], np.clip(point[0] + np.quantile(errors, [0.25, 0.75]), 0, 1))


def test_the_power_step_fits_no_model():
    import inspect

    from spectrum_occupancy_forecasting.experiments import sparse_power

    source = inspect.getsource(sparse_power)
    assert "sklearn" not in source and ".fit(" not in source


def test_scenario_names_with_dots_get_distinct_part_files(tmp_path, cfg):
    path = tmp_path / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=3.1, dt_s=60.0, seed=3)
    raw = band_table_from_blocks(path, cfg["channels_mhz"], site="SYN", session=0)
    small = dict(
        cfg,
        train_seeds=cfg["train_seeds"][:1],
        validation_seeds=cfg["validation_seeds"][:1],
        score_seeds=cfg["score_seeds"][:1],
        interval_scenarios=[],
    )
    out = tmp_path / "parts"
    result = sr.run_node(
        raw,
        small,
        out,
        scenarios=["dropout_0.5", "dropout_0.9", "dropout_0.99"],
        progress=lambda *a, **k: None,
    )
    names = sorted(p.name for p in out.glob("*.json"))
    assert names == [
        "SYN__dropout_0.5__0.json",
        "SYN__dropout_0.99__0.json",
        "SYN__dropout_0.9__0.json",
    ]
    assert len(result["folds"]) == 3
    assert len(list(out.glob("*.parquet"))) == 3
