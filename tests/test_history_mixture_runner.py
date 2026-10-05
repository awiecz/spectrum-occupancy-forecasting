"""The retained development CLI must fail before opening prohibited inputs."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--cc1-npz", "aerpaw_CC2_never_read.npz"],
        ["--study", "transition-correction"],
        ["--config", "configs/followup_v2.yaml"],
        ["--mode", "confirmation"],
    ],
)
def test_rejects_invalid_routes_without_data_access(monkeypatch, arguments):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = importlib.import_module("run_history_mixture")
    monkeypatch.setattr(runner, "band_table_from_blocks", lambda *a, **k: pytest.fail("RF read"))
    monkeypatch.setattr(sys, "argv", ["run_history_mixture", *arguments])
    with pytest.raises((SystemExit, ValueError)):
        runner.main()


def test_existing_run_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = importlib.import_module("run_history_mixture")
    monkeypatch.setattr(runner, "band_table_from_blocks", lambda *a, **k: pytest.fail("RF read"))
    monkeypatch.setattr(sys, "argv", ["run_history_mixture", "--out", str(tmp_path)])
    with pytest.raises(FileExistsError):
        runner.main()


def test_report_rejects_cc2_manifest_before_predictions(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = importlib.import_module("run_history_mixture")
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs": {"CC2": {}}}))
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *a, **k: pytest.fail("CC2 read"))
    with pytest.raises(ValueError, match="CC2"):
        runner.rebuild(tmp_path)
