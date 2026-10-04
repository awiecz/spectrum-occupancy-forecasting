import json

import pytest

from spectrum_occupancy_forecasting.experiments.protocol import (
    TestSetAlreadyUsedError,
    acquire_lock,
    complete_lock,
    fingerprint,
    git_state,
)


def test_fingerprint_changes_with_content_and_is_order_independent(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    f1 = fingerprint([a, b])
    assert f1 == fingerprint([b, a])
    a.write_text("x = 2\n")
    assert fingerprint([a, b]) != f1
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "m.py").write_text("z = 3\n")
    (d / "notes.txt").write_text("ignored")
    assert fingerprint([d]) == fingerprint([d / "m.py"])  # *.py only; notes.txt ignored


def test_second_run_is_refused_unless_acknowledged_and_is_counted(tmp_path):
    lock = tmp_path / "res" / "LOCK.json"
    rec = acquire_lock(lock, "abc")
    assert rec["status"] == "started" and rec["reruns"] == 0
    with pytest.raises(TestSetAlreadyUsedError):
        acquire_lock(lock, "abc")
    again = acquire_lock(lock, "abc", acknowledge_rerun=True)
    assert again["reruns"] == 1 and again["previous"]["status"] == "started"


def test_complete_lock_records_result_hash(tmp_path):
    lock, res = tmp_path / "LOCK.json", tmp_path / "r.json"
    acquire_lock(lock, "fp")
    res.write_text(json.dumps({"a": 1}))
    done = complete_lock(lock, res)
    assert done["status"] == "completed" and len(done["results_sha256"]) == 64
    with pytest.raises(TestSetAlreadyUsedError):
        acquire_lock(lock, "fp")  # completed runs are locked too


def test_fingerprint_ignores_line_endings(tmp_path):
    a = tmp_path / "a.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    lf = fingerprint([a])
    a.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert fingerprint([a]) == lf


def test_lock_records_git_state_or_none(tmp_path):
    rec = acquire_lock(tmp_path / "L.json", "fp")
    assert set(rec["git"]) == {"commit", "dirty"}
    assert git_state(tmp_path / "does-not-exist") == {"commit": None, "dirty": None}
