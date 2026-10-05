"""Independent boundary, calibration and audit checks for the descriptive study."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.evaluation.data_study import (
    analyze_node,
    make_figures,
    pair_correlation,
    temporal_pairs,
    window_table,
)
from spectrum_occupancy_forecasting.experiments.followup import load_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/followup_v2.yaml")


def test_half_open_windows_coverage_and_tail():
    cfg = config()
    t = np.arange(0.0, 2101.0, 60.0)
    busy = (t >= 900).astype(float)
    w = window_table(t, busy, 60, cfg)
    assert w.start_s.tolist() == [0, 900]
    assert w["count"].tolist() == [15, 15]
    assert w.fraction.tolist() == [0, 1]
    keep = (t < 900) | (t >= 1140)
    w = window_table(t[keep], busy[keep], 60, cfg)
    assert not w.eligible.iloc[1]
    assert np.isnan(w.fraction.iloc[1])  # Missing is not zero activity.


def test_pairs_never_bridge_exclusion_and_constant_data():
    y = np.arange(7.0)
    ok = np.array([True, True, True, False, True, True, True])
    a, b = temporal_pairs(y, ok, 2)
    assert a.tolist() == [0, 4]
    assert b.tolist() == [2, 6]
    assert pair_correlation(y, ok, 2)["status"] == "insufficient_pairs"
    assert pair_correlation(np.ones(6), np.ones(6, bool), 1)["status"] == "constant_data"
    assert pair_correlation(y, np.ones(7, bool), 1)["correlation"] == pytest.approx(1)


def test_band_average_is_linear_power(tmp_path):
    path = tmp_path / "blocks.npz"
    np.savez(
        path,
        times_s=[0.0, 60.0],
        centres_mhz=[2432.0, 2442.0],
        mean_dbm=[[-90.0, -80.0], [-90.0, -80.0]],
        max_dbm=[[-90.0, -80.0], [-90.0, -80.0]],
    )
    result = band_table_from_blocks(path, [2437.0], site="CC1")
    expected = 10 * np.log10((10**-9 + 10**-8) / 2)
    assert result.mean_dbm.iloc[0] == pytest.approx(expected)
    assert result.mean_dbm.iloc[0] != pytest.approx(-85)


def test_saved_tables_direct_counts_determinism_and_future_calibration(tmp_path):
    source = tmp_path / "aerpaw_CC1_blocks5MHz.npz"
    write_synthetic_blocks(source, n_days=2.1, dt_s=60.0, seed=4)
    cfg = config()
    a, b, c = (tmp_path / name for name in ("first", "repeat", "changed"))
    for folder in (a, b, c):
        folder.mkdir()
    result = analyze_node(source, "CC1", cfg, a)
    assert result == analyze_node(source, "CC1", cfg, b)
    for path in a.glob("*.csv"):
        assert path.read_bytes() == (b / path.name).read_bytes()
    raw = band_table_from_blocks(source, cfg["channels_mhz"], site="CC1")
    channel = raw[raw.freq_mhz == 2437.0]
    labels = channel.mean_dbm.to_numpy() > result["threshold_dbm"]
    windows = pd.read_csv(a / "CC1_windows.csv")
    for row in windows.itertuples():
        mask = (channel.t_start_s >= row.start_s) & (channel.t_start_s < row.end_s)
        assert row.count == mask.sum()
        assert row.fraction == pytest.approx(labels[mask].mean())
    hist = pd.read_csv(a / "CC1_power_histogram.csv")
    assert hist["count"].sum() == len(channel)
    sensitivity = pd.read_csv(a / "CC1_sensitivity.csv")
    for period in sensitivity.period.unique():
        d = sensitivity[sensitivity.period == period].sort_values("offset_db")
        assert np.all(np.diff(d.prevalence) <= 0)
        assert d.eligible_windows.nunique() == 1
    with np.load(source) as z:
        arrays = {key: z[key] for key in z.files}
    future = arrays["times_s"] > arrays["times_s"][0] + 86400
    arrays["mean_dbm"][future] += 20
    np.savez(source, **arrays)
    changed = analyze_node(source, "CC1", cfg, c)
    assert changed["threshold_dbm"] == result["threshold_dbm"]
    assert changed["nominal_interval_s"] == result["nominal_interval_s"]


def test_input_gate_before_access_and_exclusive_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("study_cli", ROOT / "scripts/study_data.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(ValueError, match="prohibited"):
        cli.validate_input(tmp_path / "aerpaw_CC2_blocks5MHz.npz", "CC1")
    with pytest.raises(FileNotFoundError):
        cli.validate_input(tmp_path / "aerpaw_CC1_blocks5MHz.npz", "CC1")
    source = tmp_path / "aerpaw_CC1_blocks5MHz.npz"
    source.touch()
    command = [
        sys.executable,
        str(ROOT / "scripts/study_data.py"),
        "--cc1",
        str(source),
        "--lw1",
        str(source),
        "--out-dir",
        str(tmp_path / "prohibited"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "prohibited" in result.stderr
    assert not (tmp_path / "prohibited").exists()
    command = [
        sys.executable,
        str(ROOT / "scripts/study_data.py"),
        "--mode",
        "synthetic-rehearsal",
        "--out-dir",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr


def test_inventory_retains_other_sessions(tmp_path):
    source = tmp_path / "aerpaw_CC1_blocks5MHz.npz"
    write_synthetic_blocks(source, n_days=2.1, dt_s=60.0, seed=4)
    with np.load(source) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["times_s"] = np.r_[arrays["times_s"], arrays["times_s"][-1] + 3600]
    for key in ("mean_dbm", "max_dbm"):
        arrays[key] = np.vstack([arrays[key], arrays[key][-1]])
    np.savez(source, **arrays)
    result = analyze_node(source, "CC1", config(), tmp_path)
    assert result["sessions"] == 2
    assert result["total_sweeps"] == result["analyzed_sweeps"] + 1
    assert pd.read_csv(tmp_path / "CC1_sessions.csv").session.tolist() == [0, 1]


def test_short_post_calibration_period_is_explicit(tmp_path):
    cfg = config()
    results = []
    for node in ("CC1", "LW1"):
        path = tmp_path / f"aerpaw_{node}_blocks5MHz.npz"
        write_synthetic_blocks(path, n_days=1.002, dt_s=60, seed=4)
        results.append(analyze_node(path, node, cfg, tmp_path))
    assert all(r["adjacent_pairs"] == 0 for r in results)
    assert all(a["status"] == "insufficient_pairs" for a in results[0]["autocorrelation"])
    (tmp_path / "summary.json").write_text(json.dumps(results), encoding="utf-8")
    make_figures(tmp_path, cfg, synthetic=True)
    assert len(list(tmp_path.glob("*.png"))) == 6


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_published_checksums_survive_git_staging(tmp_path, autocrlf):
    """Exercise Git normalization rather than only hashing the local working copy."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "core.autocrlf", autocrlf], cwd=tmp_path, check=True)
    (tmp_path / ".gitattributes").write_bytes(b"* text=auto\n")
    folder = tmp_path / "results/data_study"
    shutil.copytree(ROOT / "results/data_study", folder)
    subprocess.run(
        ["git", "add", ".gitattributes", "results/data_study"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    shutil.copyfile(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    subprocess.run(["git", "add", ".gitattributes"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "--renormalize", "results/data_study"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("study_cli", ROOT / "scripts/study_data.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for name, expected in manifest["artifact_sha256"].items():
        indexed = subprocess.run(
            ["git", "show", ":results/data_study/" + name],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        ).stdout
        assert cli.hashlib.sha256(indexed).hexdigest() == expected
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert cli.text_digest(lf) == cli.text_digest(crlf)
    assert cli.digest(lf) != cli.digest(crlf)
