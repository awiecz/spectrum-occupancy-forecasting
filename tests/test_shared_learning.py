"""Causal, spectral, pooled-weight and freeze-boundary checks."""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.fine_spectrum import (
    FINE_COLUMNS,
    restricted_times,
    spectrum_row,
)
from spectrum_occupancy_forecasting.experiments.shared_learning import (
    Bundle,
    LearningNode,
    Study,
    balanced_weights,
    candidates,
    configuration,
    profiles,
    tie_key,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):
        yield


def setup():
    from pathlib import Path

    cfg = configuration(Path(__file__).resolve().parents[1] / "configs/shared_learning.yaml")
    cfg["residual_boosting_iterations"] = 3  # Rehearsal speed only; production grid unchanged.
    frames = {}
    for i, name in enumerate(("CC1", "LW1")):
        t = 1_600_000_000.0 + i * 180 + np.arange(0, 4 * 86400, 300.0)
        rng = np.random.default_rng(i)
        power = -100 + rng.normal(0, 0.3, len(t)) + 6 * (rng.random(len(t)) < 0.25)
        frames[name] = pd.DataFrame(
            {
                "time_s": t,
                "target_dbm": power,
                **{column: power + j / 10 for j, column in enumerate(FINE_COLUMNS)},
            }
        )
    return cfg, frames


def test_spectral_bins_linear_mean_and_access_boundary():
    f = np.arange(2427.0, 2447.0, 0.1)
    p = np.full(len(f), -100.0)
    p[f < 2428] = -90.0
    row = spectrum_row(p, f)
    assert row[0] == pytest.approx(-90)
    assert row[2] == pytest.approx(-100)
    assert row[-1] == pytest.approx(np.quantile(p, 0.9) - np.quantile(p, 0.1))
    times = np.r_[np.arange(0.0, 2 * 86400, 30), 3 * 86400, np.arange(4 * 86400, 6 * 86400, 20)]
    calibration = restricted_times(times, "LW1", 2, "calibration")
    assert calibration.max() + 0.001 <= calibration[0] + 86400
    with pytest.raises(PermissionError):
        restricted_times(times, "LW1", 2, "assessment")
    with pytest.raises(ValueError):
        restricted_times(times, "CC2", 0, "development")


def test_weights_grid_and_exact_history_fallback():
    weights = balanced_weights(["CC1"] * 4 + ["LW1"] * 2)
    assert weights[:4].sum() == weights[4:].sum()
    cfg, frames = setup()
    assert len(profiles(cfg)) == 8
    assert len(candidates(cfg)) == 85
    assert min(candidates(cfg), key=tie_key)["strength"] == 0
    nodes = {name: LearningNode(f, name, cfg) for name, f in frames.items()}
    profile = profiles(cfg)[0]
    origin = nodes["CC1"].data.t0 + 3 * 86400
    bundle = Bundle(nodes, "CC1", origin - 36000, profile, cfg)
    node = nodes["CC1"]
    choice = candidates(cfg)[0]
    p, base, correction, _ = bundle.details(node, [origin], choice)
    assert np.array_equal(p, base)
    assert np.array_equal(correction, [0])
    from spectrum_occupancy_forecasting.robustness.degrade import Degradation

    for deg in (Degradation(delay_s=1800), Degradation(dropout_p=0.99)):
        p, base, correction, reasons = bundle.details(node, [origin], candidates(cfg)[1], deg, 1)
        assert np.array_equal(p, base)
        assert np.array_equal(correction, [0])
        assert reasons[0] != "none"


def test_future_corruption_both_nodes_preserves_selection_and_features():
    cfg, frames = setup()
    profile = next(p for p in profiles(cfg) if p["variant"] == "history_fine" and p["hours"] == 12)
    nodes = {name: LearningNode(f, name, cfg) for name, f in frames.items()}
    origin = nodes["CC1"].data.t0 + 3 * 86400
    study = Study(nodes, cfg)
    choice, history, scores, validation, status = study.select("CC1", origin, profile)
    b = study.bundle("CC1", origin - 36000, profile)
    expected = b.details(nodes["CC1"], [origin], choice)[0]
    changed = {}
    for name, frame in frames.items():
        frame = frame.copy()
        mask = frame.time_s > origin
        frame.loc[mask, ["target_dbm", *FINE_COLUMNS]] += 60
        changed[name] = LearningNode(frame, name, cfg)
    later = Study(changed, cfg)
    c, h, s, v, state = later.select("CC1", origin, profile)
    assert (c, h, s, state) == (choice, history, scores, status)
    pd.testing.assert_frame_equal(validation, v)
    actual = later.bundle("CC1", origin - 36000, profile).details(changed["CC1"], [origin], c)[0]
    assert np.array_equal(expected, actual)
    assert validation.fit_cutoff_s.max() <= validation.validation_start_s.max() - 36000
    features, _, _ = nodes["CC1"].features([origin], b.means["CC1"], "history_fine")
    assert not any(c.startswith(("tod_", "x_")) or c in ("site", "node") for c in features)
    assert np.all(np.abs(b.details(nodes["CC1"], [origin], candidates(cfg)[1])[2]) <= 0.1)


def test_insufficient_validation_and_training_only_statistics():
    cfg, frames = setup()
    nodes = {name: LearningNode(f, name, cfg) for name, f in frames.items()}
    study = Study(nodes, cfg)
    profile = next(p for p in profiles(cfg) if p["hours"] == 36)
    origin = nodes["CC1"].data.t0 + 2 * 86400
    choice, _, _, _, status = study.select("CC1", origin, profile)
    assert status == "insufficient_earlier_data" and choice["strength"] == 0
    b = study.bundle("CC1", origin - 36000, profile)
    assert all(np.max(o + 1800) <= b.cutoff for o in b.rows.values())
    estimator = b.estimator(candidates(cfg)[1], "shared")
    expected = pd.concat(list(b.features.values()), ignore_index=True).median().to_numpy()
    assert np.allclose(estimator.imputer.statistics_, expected, equal_nan=True)


def test_stable_weighted_scaling_and_calibration_isolation():
    from spectrum_occupancy_forecasting.experiments.shared_learning import StableScaler

    X = np.array([[1e4, 0.1], [1e4, 0.2], [1e4, 0.3]])
    weights = np.array([0.2, 0.3, 0.5])
    scaler = StableScaler().fit(X, weights)
    assert np.all(np.isfinite(scaler.transform(X)))
    assert scaler.scale_[0] == 1
    assert np.allclose(scaler.mean_, np.average(X, axis=0, weights=weights))
    cfg, frames = setup()
    original = LearningNode(frames["LW1"], "LW1", cfg)
    changed = frames["LW1"].copy()
    changed.loc[changed.time_s >= original.data.t0 + 86400, FINE_COLUMNS] += 60
    modified = LearningNode(changed, "LW1", cfg)
    assert np.array_equal(original.fine_floors, modified.fine_floors)


def test_assessment_rejects_mismatch_and_existing_lock_before_access(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "shared_runner", Path(__file__).resolve().parents[1] / "scripts/run_shared_learning.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frozen = {"mode": "freeze", "source_fingerprint": "wrong"}
    monkeypatch.setattr(runner, "verify_run", lambda _: frozen)
    monkeypatch.setattr(runner, "extract", lambda *a, **k: pytest.fail("assessment accessed"))
    args = SimpleNamespace(freeze=tmp_path, cache=tmp_path, config=tmp_path / "config")
    with pytest.raises(ValueError, match="source mismatch"):
        runner.run_assessment(args, {}, {})
    monkeypatch.setattr(runner, "fingerprint", lambda *a, **k: "match")
    frozen.update(source_fingerprint="match", config_fingerprint="match", runtime_versions={})
    (tmp_path / "later_LW1_assessment.lock.json").write_text("{}")
    with pytest.raises(FileExistsError):
        runner.run_assessment(args, {}, {"runtime_versions": {}})


def test_extraction_resume_corruption_and_budget(tmp_path, monkeypatch):
    from spectrum_occupancy_forecasting.data.aerpaw import write_synthetic_sweeps
    from spectrum_occupancy_forecasting.experiments import fine_spectrum as fine

    archive = write_synthetic_sweeps(
        tmp_path / "ResultsCC1Feb2022_SigMF.zip", n_sweeps=3, as_zip=True
    )
    cache = tmp_path / "cache"
    first = fine.extract(archive, "CC1", 0, cache)
    pd.testing.assert_frame_equal(first, fine.extract(archive, "CC1", 0, cache))
    shard = next(cache.rglob("*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="corrupted"):
        fine.extract(archive, "CC1", 0, cache)
    monkeypatch.setattr(fine, "LIMIT", 1)
    small = tmp_path / "limited"
    with pytest.raises(ValueError, match="250 MiB"):
        fine.extract(archive, "CC1", 0, small)
    assert not list(small.rglob("*.parquet"))


def test_saved_reporting_pairs_masks_and_clean_reference(tmp_path):
    from spectrum_occupancy_forecasting.experiments.shared_reporting import chosen_report

    cfg, _ = setup()
    cfg["bootstrap_resamples"] = 20
    rows, records = [], []
    for node in ("CC1", "LW1"):
        records.append(
            dict(
                node=node,
                fold=0,
                task="primary",
                common_fold=True,
                status="completed",
                scores=[],
                operating_thresholds=[],
            )
        )
        for scenario, seeds in (("clean", [0]), ("dropout", [1, 2])):
            for seed in seeds:
                for method in ("fixed_history", "selected_history", "selected_learned"):
                    pred = 0.1 if method == "fixed_history" else 0.12
                    if scenario != "clean":
                        pred += 0.03
                    for i in range(3):
                        rows.append(
                            dict(
                                node=node,
                                fold=0,
                                task="primary",
                                variant="history",
                                lead_s=900,
                                offset_db=3,
                                scenario=scenario,
                                seed=seed,
                                method=method,
                                origin_s=i * 900,
                                y=0.2 + i * 0.01,
                                prediction=pred,
                                target_count=45,
                            )
                        )
    frame = pd.DataFrame(rows)
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    chosen_report(pd.read_parquet(path), records, cfg, tmp_path)
    table = pd.read_csv(tmp_path / "common_summary.csv")
    degraded = table[(table.scenario == "dropout") & (table.method == "selected_learned")]
    assert (degraded.mss_vs_clean_selected != degraded.mss_vs_selected).all()
    assert (degraded.n_seeds == 2).all()
    assert (degraded.n_windows == 3).all()
