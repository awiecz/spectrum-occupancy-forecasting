"""Cross-node transfer machinery, on SYNTHETIC data (tests only)."""

import numpy as np
import pytest

from spectrum_occupancy_forecasting.data.synthetic import SynthConfig, generate, to_dwell_table
from spectrum_occupancy_forecasting.experiments import window_forecast as wf

CH = (2412.0, 2437.0, 2462.0)


def _node(seed, site, busy_mean=1800.0, idle_mean=3600.0):
    cfg = SynthConfig(n_days=1.0, busy_mean_s=busy_mean, idle_mean_s=idle_mean, seed=seed)
    meta, bins = generate(cfg, site=site)
    t = to_dwell_table(meta, bins, threshold_dbm=-70.0)
    return t[t["freq_mhz"].isin(CH)].reset_index(drop=True)


@pytest.fixture(scope="module")
def nodes():
    return {"A": _node(21, "A"), "B": _node(22, "B", busy_mean=900.0, idle_mean=5400.0)}


def _bounds(raw, lo_frac=0.6, hi_frac=0.9):
    t0, t1 = raw["t_start_s"].min(), raw["t_start_s"].max()
    return t0 + lo_frac * (t1 - t0), t0 + hi_frac * (t1 - t0)


def test_in_node_arm_is_numerically_identical_to_run_fold(nodes):
    lo, hi = _bounds(nodes["A"])
    ref = wf.run_fold(nodes["A"], 900.0, 3.0, lo, hi, False, CH, 2437.0)
    new = wf.run_transfer_fold(nodes, ["A"], "A", 900.0, 3.0, lo, hi, CH, 2437.0)
    assert np.array_equal(ref["t"], new["t"]) and np.array_equal(ref["y"], new["y"])
    assert ref["clim"] == pytest.approx(new["clim"]) and ref["n_train"] == new["n_train"]
    for name, p in ref["preds"]["clean"].items():
        assert np.allclose(p, new["preds"]["clean"][name], atol=1e-12), name


def test_identical_source_and_target_data_transfer_exactly(nodes):
    twin = nodes["A"].copy()
    twin["site"] = "A2"
    both = {"A": nodes["A"], "A2": twin}
    lo, hi = _bounds(nodes["A"])
    in_node = wf.run_transfer_fold(both, ["A"], "A", 900.0, 3.0, lo, hi, CH, 2437.0)
    transfer = wf.run_transfer_fold(both, ["A2"], "A", 900.0, 3.0, lo, hi, CH, 2437.0)
    for name, p in in_node["preds"]["clean"].items():
        assert np.allclose(p, transfer["preds"]["clean"][name], atol=1e-12), name
    joint = wf.run_transfer_fold(both, ["A", "A2"], "A", 900.0, 3.0, lo, hi, CH, 2437.0)
    assert joint["n_train"] == 2 * in_node["n_train"]
    r = wf.paired_mse_ratio([in_node], [transfer], "ridge", n_boot=50)
    assert r[0] == pytest.approx(1.0)


def test_different_nodes_give_different_but_valid_predictions(nodes):
    lo, hi = _bounds(nodes["B"])
    in_node = wf.run_transfer_fold(nodes, ["B"], "B", 900.0, 3.0, lo, hi, CH, 2437.0)
    transfer = wf.run_transfer_fold(nodes, ["A"], "B", 900.0, 3.0, lo, hi, CH, 2437.0)
    assert np.array_equal(in_node["y"], transfer["y"])  # same target windows and labels
    assert not np.allclose(in_node["preds"]["clean"]["ridge"], transfer["preds"]["clean"]["ridge"])
    for p in transfer["preds"]["clean"].values():
        assert p.shape == in_node["y"].shape and np.all((p >= 0) & (p <= 1))
    # the history averages hardly depend on the training node (only their no-evidence fallback)
    a, b = in_node["preds"]["clean"]["recent_1800s"], transfer["preds"]["clean"]["recent_1800s"]
    assert np.mean(np.abs(a - b)) < 0.02
    r, lo_ci, hi_ci = wf.paired_mse_ratio([transfer], [in_node], "ridge", n_boot=100)
    assert lo_ci <= r <= hi_ci and r > 0


def test_source_training_data_never_passes_the_cutoff(nodes):
    """Corrupting the source node AFTER the cutoff must not change transfer predictions."""
    lo, hi = _bounds(nodes["B"])
    base = wf.run_transfer_fold(nodes, ["A"], "B", 900.0, 3.0, lo, hi, CH, 2437.0)
    A2 = nodes["A"].copy()
    late = A2["t_start_s"] >= lo - 7200.0  # nothing at/after this time may be used (cutoff + W)
    A2.loc[late, "mean_dbm"] = A2.loc[late, "mean_dbm"] + 25.0
    changed = wf.run_transfer_fold({"A": A2, "B": nodes["B"]}, ["A"], "B", 900.0, 3.0, lo, hi,
                                   CH, 2437.0)
    for name, p in base["preds"]["clean"].items():
        assert np.allclose(p, changed["preds"]["clean"][name], atol=1e-12), name


def test_absolute_threshold_arm_changes_target_labels_only(nodes):
    lo, hi = _bounds(nodes["B"])
    cal = wf.run_transfer_fold(nodes, ["A"], "B", 900.0, 3.0, lo, hi, CH, 2437.0)
    ab = wf.run_transfer_fold(nodes, ["A"], "B", 900.0, 3.0, lo, hi, CH, 2437.0,
                              abs_thresholds_from="A")
    assert np.array_equal(cal["t"], ab["t"])
    assert ab["thresholds_dbm"] != cal["thresholds_dbm"]


def test_paired_ratio_requires_identical_windows(nodes):
    lo, hi = _bounds(nodes["B"])
    a = wf.run_transfer_fold(nodes, ["B"], "B", 900.0, 3.0, lo, hi, CH, 2437.0)
    b = wf.run_transfer_fold(nodes, ["B"], "B", 900.0, 3.0, lo, hi, CH, 2437.0,
                             abs_thresholds_from="A")
    if not np.array_equal(a["y"], b["y"]):
        with pytest.raises(ValueError):
            wf.paired_mse_ratio([a], [b], "ridge", n_boot=10)
