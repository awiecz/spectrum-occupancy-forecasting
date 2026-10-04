"""V2 causality, selection, temporal inference, artifact and confirmation-gate contracts."""

from __future__ import annotations

import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.followup import load_config, run_node, tasks
from spectrum_occupancy_forecasting.experiments.followup_data import (
    NodeData,
    bounded_decay,
    feature_columns,
    maximum_lookback,
)
from spectrum_occupancy_forecasting.experiments.followup_gate import (
    create_bundle,
    verify_archive,
    verify_bundle,
)
from spectrum_occupancy_forecasting.experiments.followup_models import (
    ForecastModel,
    fit_selected,
    operating_threshold,
    select_models,
    training,
)
from spectrum_occupancy_forecasting.experiments.followup_reporting import (
    paired_intervals,
    paired_losses,
    summarize,
    temporal_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    cfg = load_config(ROOT / "configs/followup_v2.yaml")
    return cfg | {
        "ridge_alphas": [1, 10],
        "boosting_depths": [2],
        "boosting_leaf_sizes": [20],
        "boosting_iterations": 8,
        "residual_boosting_iterations": 8,
        "bootstrap_resamples": 30,
        "dropout_probabilities": [0.9],
        "outage_lengths_s": [],
        "delays_s": [600],
        "mask_seeds": [1, 2],
    }


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("v2") / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=3.1, dt_s=60, seed=3)
    return band_table_from_blocks(path, [2412, 2437, 2462])


def test_nonzero_lead_and_common_grid(config, raw):
    node = NodeData(raw, config, 3)
    start = node.t0 + 2 * 86400
    o = node.origins(start, start + 86400, 900, 900, common=True)
    assert o[-1] + max(config["leads_s"]) + 900 <= start + 86400
    assert np.array_equal(o, node.origins(start, start + 86400, 7200, 900, common=True))
    y, n = node.labels(o[:1], 900)
    use = (node.ch["t_start_s"] >= o[0] + 900) & (node.ch["t_start_s"] < o[0] + 1800)
    assert y[0] == pytest.approx(node.ch.loc[use, "occ_any"].mean())
    assert n[0] == use.sum()


def test_calibration_and_prediction_do_not_use_future_power(config, raw):
    node = NodeData(raw, config, 3)
    at = node.t0 + 2 * 86400
    changed = raw.copy()
    changed.loc[changed["t_start_s"] >= at, "mean_dbm"] += 100
    other = NodeData(changed, config, 3)
    assert node.floors == other.floors and node.dt == other.dt
    pd.testing.assert_frame_equal(node.features([at]), other.features([at]))
    with threadpool_limits(1):
        a, records, _ = select_models(node, at, 900, "full")
        b, modified, _ = select_models(other, at, 900, "full")
        models_a = fit_selected(node, at, 900, "full", a, names=("selected_learned",))
        models_b = fit_selected(other, at, 900, "full", b, names=("selected_learned",))
        np.testing.assert_array_equal(
            models_a["selected_learned"].predict(node.features([at])),
            models_b["selected_learned"].predict(other.features([at])),
        )
    assert a == b and records == modified


def test_initial_calibration_is_not_reestimated(config, raw):
    node = NodeData(raw, config, 3)
    altered = raw.copy()
    altered.loc[altered["t_start_s"] >= node.t0 + config["calibration_s"], "mean_dbm"] += 20
    other = NodeData(altered, config, 3)
    assert node.thresholds == other.thresholds


def test_bounded_decay_observation_end_delay_and_support():
    d = pd.DataFrame(
        {
            "t_start_s": [0.0, 5.0, 9.5],
            "dwell_s": [1.0, 1.0, 1.0],
            "occ_any": [1.0, 0.0, 1.0],
            "freq_mhz": 2437.0,
        }
    )
    f, w = bounded_decay(d, [10.0], np.ones(3, bool), 2, 5, 0, 2437.0)
    assert f[0] == 0 and w[0] == pytest.approx(np.exp(-4 / 2))
    f, w = bounded_decay(d, [10.0], np.ones(3, bool), 2, 5, 5, 2437.0)
    assert f[0] == 1 and w[0] == pytest.approx(np.exp(-9 / 2))


def test_training_labels_finish_before_embargo(config, raw):
    node = NodeData(raw, config, 3)
    cutoff = node.t0 + config["first_eval_s"] - maximum_lookback(config)
    X, _ = training(node, cutoff, 7200)
    assert (X["t_start_s"] + 7200 + config["window_s"] <= cutoff).all()


def test_missing_target_coverage_filters_without_changing_features(config, raw):
    node = NodeData(raw, config, 3)
    o = node.t0 + 2 * 86400
    removed = raw[~raw["t_start_s"].between(o + 900, o + 1800, inclusive="left")]
    other = NodeData(removed, config, 3)
    assert np.isnan(other.labels([o], 900)[0][0])
    pd.testing.assert_frame_equal(node.features([o]), other.features([o]))


def test_history_allowlist_excludes_calendar_power_age_and_cross(config, raw):
    node = NodeData(raw, config, 3)
    X = node.features([node.t0 + 2 * 86400])
    cols = feature_columns(X, "history")
    assert all(c in ("last_occ", "ewma_occ") or c.startswith("frac_") for c in cols)
    assert "age_s" in feature_columns(X, "history_age")
    assert "tod_sin" in feature_columns(X, "history_calendar")
    assert not any(c.startswith("x_age_") for c in feature_columns(X, "history_cross"))
    with pytest.raises(ValueError):
        feature_columns(X, "unknown")


def test_model_history_ablation_ignores_excluded_values(config, raw):
    node = NodeData(raw, config, 3)
    X, y = training(node, node.t0 + 1.5 * 86400, 900)
    model = ForecastModel({"family": "ridge", "alpha": 10}, config, "history").fit(X, y)
    changed = X.copy()
    for c in changed:
        if c not in model.columns:
            changed[c] = 999
    assert np.array_equal(model.predict(X), model.predict(changed))


def test_operating_threshold_handles_ties_and_insufficient_events(config):
    cfg = config | {"minimum_validation_events": 2, "minimum_validation_nonevents": 4}
    y, score = [0, 0, 0, 0, 1, 1], [0.1, 0.1, 0.5, 0.5, 0.8, 0.9]
    result = operating_threshold(y, score, 0.25, cfg)
    assert result["threshold"] == 0.8
    assert result["validation_false_busy"] <= 0.25
    assert operating_threshold(np.zeros(6), score, 0.25, cfg)["threshold"] is None


def test_validation_selection_ties_follow_configuration_order(config, raw):
    node = NodeData(raw.assign(mean_dbm=-130.0), config, 3)
    with threadpool_limits(1):
        choices, _, _ = select_models(node, node.t0 + 2 * 86400, 900, "history")
    assert choices["selected_history"]["id"] == "recent_300s"


def test_temporal_blocks_do_not_cross_folds_or_gaps(config):
    frame = pd.DataFrame(
        {"fold": [0, 0, 0, 1, 1], "origin_s": [0, 900, 9000, 0, 900], "loss": 1.0, "ref_loss": 2.0}
    )
    blocks = temporal_blocks(frame, 7200, 900)
    assert len(blocks) == 3
    assert sorted(blocks["n"]) == [1, 2, 2]
    assert paired_intervals(frame, 7200, config)["status"] == "insufficient_blocks"


def test_paired_losses_average_losses_not_predictions():
    model = pd.DataFrame(
        {"fold": 0, "origin_s": [0, 0], "seed": [1, 2], "y": 0.5, "prediction": [0, 1]}
    )
    ref = model.assign(prediction=0.5)
    losses = paired_losses(model, ref)
    assert len(losses) == 1 and losses["loss"].iloc[0] == 0.25
    with pytest.raises(ValueError):
        paired_losses(model, ref.iloc[:1])


@pytest.fixture(scope="module")
def rehearsal(config, raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("v2_rehearsal")
    with threadpool_limits(1):
        result = run_node(raw, config, out / "parts", task_names=["primary"])
    frames = [
        pd.read_parquet(p)
        for p in (out / "parts").glob("*.parquet")
        if not p.stem.endswith("__validation")
    ]
    return out, result, pd.concat(frames, ignore_index=True)


def test_saved_predictions_recompute_and_keep_labels_across_degradation(rehearsal, config):
    out, result, frame = rehearsal
    primary = frame[frame["method"] == "selected_history"]
    labels = primary.groupby(["fold", "origin_s"])["y"].nunique()
    assert (labels == 1).all()
    frame.to_parquet(out / "roundtrip.parquet", index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(out / "roundtrip.parquet"))
    a = summarize(frame, result["folds"], config, out)
    b = summarize(pd.read_parquet(out / "roundtrip.parquet"), result["folds"], config, out)
    pd.testing.assert_frame_equal(a["summary"], b["summary"])
    row = a["summary"].query("method == 'selected_learned' and scenario == 'clean'").iloc[0]
    p = frame.query("method == 'selected_learned' and scenario == 'clean'")
    assert row["mse"] == pytest.approx(np.mean((p["prediction"] - p["y"]) ** 2))
    assert not (a["events"]["status"] == "insufficient_calibration").empty
    assert "NaN" not in (out / "summary.json").read_text()


def test_staged_task_definitions_are_not_cartesian(config):
    defined = tasks(config)
    assert len(defined) == 13
    assert sum(t[0] == "primary" for t in defined) == 1


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    for name in (
        "src/example.py",
        "scripts/run_followup.py",
        "scripts/freeze_followup.py",
        "pyproject.toml",
        "uv.lock",
    ):
        (root / name).write_text("example = 1\n")
    config, protocol = root / "config.yaml", root / "protocol.md"
    config.write_text("version: 2\n")
    protocol.write_text("Protocol frozen before CC2 activity inspection\n")
    bundle = tmp_path / "freeze.zip"
    manifest = create_bundle(root, config, protocol, bundle)
    return bundle.read_bytes(), manifest


def test_archive_gate_verifies_actual_bundle_not_a_claimed_hash(frozen):
    payload, expected = frozen
    record = {
        "draft": False,
        "immutable": True,
        "published_at": "2026-01-01T00:00:00Z",
        "assets": [
            {
                "name": "followup_v2_freeze.zip",
                "browser_download_url": "https://github.com/a/b/archive.zip",
            }
        ],
    }

    def fetcher(url, json_response=False):
        return record if json_response else payload

    verified = verify_archive(
        "https://api.github.com/repos/a/b/releases/1", expected, fetcher=fetcher
    )
    assert verified["verified"]
    record["immutable"] = False
    with pytest.raises(ValueError):
        verify_archive("https://api.github.com/repos/a/b/releases/1", expected, fetcher=fetcher)
    with pytest.raises(ValueError):
        verify_archive("https://example.com/local-claim", expected, fetcher=fetcher)


def test_tampered_freeze_payload_is_rejected(frozen):
    payload, expected = frozen
    # Changing the current expected configuration is also a gate failure.
    with pytest.raises(ValueError):
        verify_bundle(payload, expected | {"config_sha256": "wrong"})
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(buf, "w") as altered:
        for name in original.namelist():
            altered.writestr(
                name, b"tampered" if name == "followup_v2_protocol.md" else original.read(name)
            )
    with pytest.raises(ValueError):
        verify_bundle(buf.getvalue(), expected)


def test_published_zenodo_record_uses_server_updated_timestamp(frozen):
    payload, expected = frozen
    record = {
        "id": 123,
        "doi": "10.5281/zenodo.123",
        "updated": "2026-01-02T00:00:00Z",
        "files": [
            {
                "key": "followup_v2_freeze.zip",
                "links": {
                    "self": "https://zenodo.org/api/records/123/files/followup_v2_freeze.zip/content"
                },
            }
        ],
    }

    def fetcher(url, json_response=False):
        return record if json_response else payload

    verified = verify_archive("https://zenodo.org/api/records/123", expected, fetcher=fetcher)
    assert verified["archived_at"] == record["updated"]
    record["is_published"] = False
    with pytest.raises(ValueError):
        verify_archive("https://zenodo.org/api/records/123", expected, fetcher=fetcher)


def test_confirmation_refuses_before_any_cc2_read(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = importlib.import_module("run_followup")
    monkeypatch.setattr(mod, "band_table_from_blocks", lambda *a, **k: pytest.fail("CC2 read"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_followup",
            "--mode",
            "confirmation",
            "--cc2-npz",
            str(tmp_path / "must_not_read.npz"),
        ],
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_report_refuses_tampered_saved_inputs(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = importlib.import_module("run_followup")
    (tmp_path / "config.yaml").write_text("tampered")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_hashes": {"config.yaml": "does-not-match"},
                "mode": "exploratory",
            }
        )
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        mod.rebuild(tmp_path)


def test_run_refuses_existing_output_before_generating_data(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = importlib.import_module("run_followup")
    monkeypatch.setattr(mod, "write_synthetic_blocks", lambda *a, **k: pytest.fail("overwrite"))
    monkeypatch.setattr(
        sys, "argv", ["run_followup", "--mode", "synthetic-rehearsal", "--out", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        mod.main()
