"""Causality / leakage tests for the feature builder."""

import numpy as np
import pytest

from spectrum_occupancy_forecasting.features.age_aware import FeatureSpec, build_features
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask


def _brute(d, row, visible, delay, spec):
    """Reference implementation: loop over observations."""
    f = d.loc[row, "freq_mhz"]
    t = d.loc[row, "t_start_s"]
    t_end = (d["t_start_s"] + d["dwell_s"]).to_numpy()
    ok = visible & (d["freq_mhz"].to_numpy() == f) & (t_end <= t - delay)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    last = idx[np.argmax(t_end[idx])]
    out = {"last_occ": float(d["occ_any"].iloc[last]), "age_s": t - t_end[last]}
    for w in spec.windows_s:
        inwin = ok & (t_end > t - delay - w)
        out[f"n_{int(w)}s"] = int(inwin.sum())
        out[f"cnt_{int(w)}s"] = float(d["occ_any"].to_numpy()[inwin].sum())
    return out


@pytest.mark.parametrize("delay", [0.0, 30.0])
def test_matches_bruteforce_reference(synth_dwells, delay):
    d = synth_dwells
    spec = FeatureSpec()
    rng = np.random.default_rng(3)
    visible = rng.random(len(d)) > 0.3
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    X = build_features(d, target, visible, delay_s=delay, spec=spec)
    for row in rng.choice(X.index.to_numpy(), 40, replace=False):
        ref = _brute(d, row, visible, delay, spec)
        if ref is None:
            assert np.isnan(X.loc[row, "last_occ"])
            continue
        for k, v in ref.items():
            assert X.loc[row, k] == pytest.approx(v), (row, k)


def test_never_uses_own_observation(synth_dwells):
    d = synth_dwells.copy()
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    X0 = build_features(d, target)
    row = X0.index[len(X0) // 2]
    d3 = d.copy()
    d3.loc[row, "occ_any"] = not d3.loc[row, "occ_any"]  # flip only the target's own label
    X3 = build_features(d3, target)
    assert X0.loc[row].equals(X3.loc[row])


def test_future_observations_cannot_change_features(synth_dwells):
    d = synth_dwells.copy()
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    cut = d["t_start_s"].quantile(0.5)
    X0 = build_features(d, target)
    d2 = d.copy()
    d2.loc[d2["t_start_s"] > cut, "occ_any"] = ~d2.loc[d2["t_start_s"] > cut, "occ_any"]
    X1 = build_features(d2, target)
    before = X0.index[X0["t_start_s"] <= cut]
    cols = [c for c in X0.columns if c != "site"]
    assert np.allclose(
        X0.loc[before, cols].to_numpy(float), X1.loc[before, cols].to_numpy(float), equal_nan=True
    )


def test_hidden_observations_do_not_influence_features(synth_dwells):
    d = synth_dwells.copy()
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    visible = visibility_mask(d["t_start_s"], Degradation(dropout_p=0.6), seed=5)
    X0 = build_features(d, target, visible)
    d2 = d.copy()
    d2.loc[~visible, "occ_any"] = ~d2.loc[~visible, "occ_any"]  # corrupt only hidden rows
    d2.loc[~visible, "mean_dbm"] = 0.0
    X1 = build_features(d2, target, visible)
    cols = [c for c in X0.columns if c != "site"]
    assert np.allclose(X0[cols].to_numpy(float), X1[cols].to_numpy(float), equal_nan=True)


def test_delay_increases_age_and_removes_recent_evidence(synth_dwells):
    d = synth_dwells
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    X0 = build_features(d, target, delay_s=0.0)
    X1 = build_features(d, target, delay_s=300.0)
    both = X0["age_s"].notna() & X1["age_s"].notna()
    assert (X1.loc[both, "age_s"] >= 300.0 - 1e-9).all()
    assert (X1.loc[both, "age_s"] >= X0.loc[both, "age_s"] - 1e-9).all()


def test_missing_evidence_is_nan_not_idle(synth_dwells):
    d = synth_dwells
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    visible = np.zeros(len(d), dtype=bool)  # nothing observed
    X = build_features(d, target, visible)
    assert X["last_occ"].isna().all() and X["age_s"].isna().all()
    assert (X["ewma_w"] == 0).all() and (X["n_60s"] == 0).all()


def test_max_lookback_covers_longest_window():
    spec = FeatureSpec(tau_s=60.0, windows_s=(60.0, 600.0))
    assert spec.max_lookback_s >= 600.0
