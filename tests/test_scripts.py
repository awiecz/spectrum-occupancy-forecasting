"""Smoke tests for the scripts, on SYNTHETIC data only: each script must still run end to end.

Heavy experiment scripts are run with a single window length and without the robustness grid by
patching their module constants; their real configuration is exercised by the actual runs.
"""

import hashlib
import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from spectrum_occupancy_forecasting.data.aerpaw import write_synthetic_sweeps
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.protocol import TestSetAlreadyUsedError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MAIN_GUARDED = ("extract_aerpaw_blocks", "run_window_dev", "run_test_once", "run_second_node",
                "run_transfer", "run_dev_baselines", "sanity_checks")


@pytest.fixture(scope="module")
def blocks(tmp_path_factory):
    path = tmp_path_factory.mktemp("blocks") / "aerpaw_SYN_blocks5MHz.npz"
    write_synthetic_blocks(path, n_days=5.6, dt_s=30.0, seed=3)
    return path


def load(monkeypatch, name):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module(name)


def run(monkeypatch, mod, *argv):
    monkeypatch.setattr(sys, "argv", [mod.__name__, *map(str, argv)])
    mod.main()


def test_every_script_with_a_main_imports(monkeypatch):
    for name in MAIN_GUARDED:
        assert callable(load(monkeypatch, name).main), name


def test_extract_script_on_a_synthetic_archive(tmp_path, monkeypatch):
    z = write_synthetic_sweeps(tmp_path / "ResultsSYNFeb2022_SigMF.zip", n_sweeps=40,
                               as_zip=True, n_bins=4000)
    mod = load(monkeypatch, "extract_aerpaw_blocks")
    run(monkeypatch, mod, z, "--out-dir", tmp_path / "out", "--workers", 2, "--shard-sweeps", 15)
    out = np.load(tmp_path / "out" / "aerpaw_SYN_blocks5MHz.npz")
    assert out["mean_dbm"].shape[0] == 40 and out["mean_dbm"].shape == out["max_dbm"].shape
    assert np.all(np.diff(out["times_s"]) > 0)
    assert len(out["centres_mhz"]) == out["mean_dbm"].shape[1]


def test_window_dev_script_and_figures(blocks, tmp_path, monkeypatch, capsys):
    mod = load(monkeypatch, "run_window_dev")
    res_path = tmp_path / "dev.json"
    run(monkeypatch, mod, blocks, "--windows", 900, "--robust-window", 0, "--out", res_path)
    res = json.loads(res_path.read_text())["windows"]["900s"]
    assert len(res["n_dev"]) == 3 and "recent_7200s" in res["clean"] and "ridge" in res["clean"]
    assert all(np.isfinite(v["rmse"]) for v in res["clean"].values())
    assert not res["robustness"]  # robustness grid was switched off

    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(blocks), str(res_path), "--out",
                                      str(tmp_path / "fig"), "--region", "dev"])
    runpy.run_path(str(SCRIPTS / "make_figures.py"), run_name="__main__")
    assert (tmp_path / "fig" / "fig1_characterisation.png").exists()
    assert (tmp_path / "fig" / "fig2_method_comparison.png").exists()
    assert not (tmp_path / "fig" / "fig3_robustness.png").exists()  # skipped without robustness
    assert "skipping figure 3" in capsys.readouterr().out


def test_second_node_script_runs_once_only(blocks, tmp_path, monkeypatch):
    mod = load(monkeypatch, "run_second_node")
    monkeypatch.setattr(mod, "WINDOWS", (900.0,))
    monkeypatch.setattr(mod, "PRIMARY_W", 0.0)  # no robustness grid
    monkeypatch.setattr(mod, "N_BOOT", 40)
    out, lock = tmp_path / "node.json", tmp_path / "lock.json"
    run(monkeypatch, mod, blocks, "--out", out, "--lock", lock)
    res = json.loads(out.read_text())
    assert res["site"] == "SYN" and "900s" in res["windows"] and not res["degenerate"]
    assert json.loads(lock.read_text())["status"] == "completed"
    with pytest.raises(TestSetAlreadyUsedError):
        run(monkeypatch, mod, blocks, "--out", out, "--lock", lock)


def test_transfer_script_rehearsal_gives_identical_arms(blocks, tmp_path, monkeypatch):
    mod = load(monkeypatch, "run_transfer")
    monkeypatch.setattr(mod, "WINDOWS", (900.0,))
    monkeypatch.setattr(mod, "N_BOOT_ARM", 30)
    monkeypatch.setattr(mod, "N_BOOT_RATIO", 30)
    out, lock = tmp_path / "tr.json", tmp_path / "trlock.json"
    run(monkeypatch, mod, "--rehearsal", blocks, "--out", out, "--lock", lock)
    res = json.loads(out.read_text())
    assert set(res["directions"]) == {"CC1_to_LW1", "LW1_to_CC1"} and res["rehearsal"]
    ratios = res["directions"]["CC1_to_LW1"]["windows"]["900s"]["ratios"]["transfer"]
    assert all(abs(v[0] - 1.0) < 1e-9 for v in ratios.values())  # identical data transfers exactly
    assert json.loads(lock.read_text())["status"] == "completed"


def test_superseded_single_sweep_script_still_runs(blocks, tmp_path, monkeypatch):
    mod = load(monkeypatch, "run_dev_baselines")
    out = tmp_path / "base.json"
    run(monkeypatch, mod, blocks, "--x-db", 3, "--out", out)
    sc = json.loads(out.read_text())["scenarios"]["x_3db"]
    assert "persistence" in sc["dev_metrics"] and "clean" in sc["robustness_dev"]


def test_small_utility_scripts(blocks, tmp_path, monkeypatch, capsys):
    data = tmp_path / "f.bin"
    data.write_bytes(b"rf-spectrum")
    monkeypatch.setattr(sys, "argv", ["sha256.py", str(data)])
    runpy.run_path(str(SCRIPTS / "sha256.py"), run_name="__main__")
    assert hashlib.sha256(b"rf-spectrum").hexdigest() in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["explore_blocks.py", str(blocks)])
    runpy.run_path(str(SCRIPTS / "explore_blocks.py"), run_name="__main__")
    assert "main session" in capsys.readouterr().out
