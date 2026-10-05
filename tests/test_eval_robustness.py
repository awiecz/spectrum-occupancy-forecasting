import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.data.schema import validate_dwells
from spectrum_occupancy_forecasting.evaluation.bootstrap import block_bootstrap_ci, block_ids
from spectrum_occupancy_forecasting.evaluation.metrics import binary_report, brier, ece
from spectrum_occupancy_forecasting.evaluation.splits import assert_no_leakage, chronological_split
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask


# ---- schema ------------------------------------------------------------------------------
def test_synthetic_table_obeys_schema(synth_dwells):
    validate_dwells(synth_dwells)


def test_schema_rejects_bad_units_and_duplicates(synth_dwells):
    bad = synth_dwells.copy()
    bad["t_start_s"] = bad["t_start_s"] * 1000  # milliseconds by mistake
    with pytest.raises(ValueError, match="epoch"):
        validate_dwells(bad)
    dup = pd.concat([synth_dwells, synth_dwells.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_dwells(dup)
    with pytest.raises(ValueError, match="missing"):
        validate_dwells(synth_dwells.drop(columns=["duty"]))


# ---- splits ------------------------------------------------------------------------------
def test_split_has_embargo_and_order(synth_dwells):
    t = synth_dwells["t_start_s"].to_numpy()
    tr, dv, te = chronological_split(t, embargo_s=600.0)
    assert_no_leakage(t, tr, dv, te, 600.0)
    assert t[tr].max() < t[dv].min() < t[dv].max() < t[te].min()
    assert len(tr) > len(dv) > 0 and len(te) > 0


def test_leakage_assertion_fires_when_embargo_too_small(synth_dwells):
    t = synth_dwells["t_start_s"].to_numpy()
    tr, dv, te = chronological_split(t, embargo_s=10.0)
    with pytest.raises(AssertionError):
        assert_no_leakage(t, tr, dv, te, 5000.0)


# ---- metrics -----------------------------------------------------------------------------
def test_false_idle_denominator_is_busy_targets():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = binary_report(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 2, 1, 5)
    assert r["false_idle_rate"] == pytest.approx(2 / 4)  # FN / (FN+TP), not / n
    assert r["false_busy_rate"] == pytest.approx(1 / 6)
    assert r["recall"] == pytest.approx(0.5)
    assert r["balanced_accuracy"] == pytest.approx((0.5 + 5 / 6) / 2)
    assert r["prevalence"] == pytest.approx(0.4)


def test_brier_and_ece():
    y = np.array([1, 0, 1, 0])
    assert brier(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    y = rng.random(50_000) < p  # perfectly calibrated
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_block_bootstrap_ci_brackets_point(synth_dwells):
    t = synth_dwells["t_start_s"].to_numpy()
    rng = np.random.default_rng(0)
    y = rng.random(len(t)) < 0.4
    p = np.full(len(t), 0.4)
    blocks = block_ids(t, block_s=1800.0)
    pt, lo, hi = block_bootstrap_ci(y, p, blocks, brier, n_boot=200, seed=1)
    assert lo <= pt <= hi


# ---- degradation -------------------------------------------------------------------------
def test_identity_degradation_hides_nothing(synth_dwells):
    m = visibility_mask(synth_dwells["t_start_s"], Degradation())
    assert m.all() and Degradation().is_identity


def test_dropout_rate_and_determinism(synth_dwells):
    t = synth_dwells["t_start_s"]
    a = visibility_mask(t, Degradation(dropout_p=0.5), seed=7)
    b = visibility_mask(t, Degradation(dropout_p=0.5), seed=7)
    c = visibility_mask(t, Degradation(dropout_p=0.5), seed=8)
    assert (a == b).all() and (a != c).any()
    assert abs((~a).mean() - 0.5) < 0.03


def test_outages_are_contiguous_in_time(synth_dwells):
    t = synth_dwells["t_start_s"].to_numpy()
    deg = Degradation(outage_len_s=600.0, outage_frac=0.2)
    m = visibility_mask(t, deg, seed=2)
    hidden_t = t[~m]
    assert len(hidden_t) > 0
    # no *visible* observation inside a hidden stretch: contiguous in time order
    order = np.argsort(t)
    hidden_sorted = (~m)[order]
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, hidden_sorted.astype(int), 0])))[::2]
    assert (runs > 10).all()
    assert 0.05 < (~m).mean() < 0.4


def test_degradation_never_modifies_targets(synth_dwells):
    before = synth_dwells.copy(deep=True)
    visibility_mask(
        synth_dwells["t_start_s"],
        Degradation(dropout_p=0.9, outage_len_s=60, outage_frac=0.5, delay_s=60),
    )
    pd.testing.assert_frame_equal(before, synth_dwells)


def test_invalid_degradation_rejected():
    with pytest.raises(ValueError):
        Degradation(dropout_p=1.0)
    with pytest.raises(ValueError):
        Degradation(outage_frac=0.3)  # needs outage_len_s
    with pytest.raises(ValueError):
        Degradation(delay_s=-1)


def test_development_region_end_matches_split_boundary():
    from spectrum_occupancy_forecasting.evaluation.splits import development_region_end

    end = development_region_end(0.0, 7 * 86400.0)
    assert 5.2 * 86400 < end < 5.26 * 86400  # ~75 % of the span; the last ~25 % is test
