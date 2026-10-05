import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.evaluation.metrics import fraction_report
from spectrum_occupancy_forecasting.evaluation.splits import assert_no_leakage, chronological_split
from spectrum_occupancy_forecasting.features.age_aware import (
    PW_REF_DBM,
    FeatureSpec,
    build_features,
)
from spectrum_occupancy_forecasting.features.windows import origin_grid, window_fraction
from spectrum_occupancy_forecasting.models.regressors import (
    ClimatologyFraction,
    DecayedFraction,
    HourOfDayFraction,
    RecentFraction,
    gbt_regressor,
    ridge_regressor,
    with_derived,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation, sweep_visibility_mask


# ---- window labels -----------------------------------------------------------------------
def test_window_fraction_counts_half_open_window():
    t = np.arange(0.0, 100.0)  # one obs per second
    busy = t % 4 == 0
    frac, n = window_fraction(t, busy, [10.0], 20.0, nominal_dt_s=1.0)
    assert n[0] == 20  # [10, 30): includes 10, excludes 30
    assert frac[0] == pytest.approx(5 / 20)  # busy at 12,16,20,24,28


def test_window_with_a_recording_gap_is_dropped_not_scored():
    t = np.concatenate([np.arange(0, 40.0), np.arange(80, 120.0)])
    frac, n = window_fraction(t, np.ones(len(t)), [5.0, 45.0], 30.0, nominal_dt_s=1.0)
    assert np.isfinite(frac[0]) and np.isnan(frac[1]) and n[1] == 0


def test_origin_grid_keeps_whole_windows_inside_the_session():
    o = origin_grid(0.0, 100.0, stride_s=10.0, window_s=30.0)
    assert o[0] == 0 and o[-1] + 30.0 <= 100.0 + 1e-9 and o[-1] == 70.0
    with pytest.raises(ValueError):
        origin_grid(0, 10, 0, 5)


# ---- explicit forecast origins -----------------------------------------------------------
def test_explicit_targets_equal_row_targets(synth_dwells):
    d = synth_dwells
    mask = (d["freq_mhz"] == 2437.0).to_numpy()
    a = build_features(d, mask)
    b = build_features(d, targets=d.loc[mask, ["site", "freq_mhz", "t_start_s"]])
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        build_features(d, mask, targets=d.loc[mask, ["site", "freq_mhz", "t_start_s"]])
    with pytest.raises(ValueError):
        build_features(d)


def _origins(d, n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(d["t_start_s"].min() + 4000, d["t_start_s"].max() - 4000, n))
    return pd.DataFrame({"site": d["site"].iloc[0], "freq_mhz": 2437.0, "t_start_s": t})


def test_origin_features_match_bruteforce_and_ignore_the_window(synth_dwells):
    d = synth_dwells
    spec = FeatureSpec(tau_s=300.0, windows_s=(300.0, 1800.0))
    org = _origins(d)
    X = build_features(d, targets=org, spec=spec, delay_s=0.0)
    ch = d[d["freq_mhz"] == 2437.0]
    t_end = (ch["t_start_s"] + ch["dwell_s"]).to_numpy()
    occ = ch["occ_any"].to_numpy(float)
    for k, o in enumerate(org["t_start_s"]):
        ok = t_end <= o
        last = np.argmax(np.where(ok, t_end, -np.inf))
        assert X["last_occ"].iloc[k] == occ[last]
        assert X["age_s"].iloc[k] == pytest.approx(o - t_end[last])
        inwin = ok & (t_end > o - 300.0)
        assert X["n_300s"].iloc[k] == inwin.sum()
        assert X["cnt_300s"].iloc[k] == occ[inwin].sum()

    # Corrupt every observation inside each origin's FUTURE window: inputs must not change.
    W = 900.0
    d2 = d.copy()
    for o in org["t_start_s"]:
        m = (d2["t_start_s"] >= o) & (d2["t_start_s"] < o + W)
        d2.loc[m, "occ_any"] = ~d2.loc[m, "occ_any"]
        d2.loc[m, "mean_dbm"] = 0.0
    # (flipping windows of different origins may overlap earlier origins' inputs, so check the
    # first origin only against its own window)
    first = org.iloc[[0]]
    d3 = d.copy()
    m = (d3["t_start_s"] >= first["t_start_s"].iloc[0]) & (
        d3["t_start_s"] < first["t_start_s"].iloc[0] + W
    )
    d3.loc[m, "occ_any"] = ~d3.loc[m, "occ_any"]
    Xa = build_features(d, targets=first, spec=spec)
    Xb = build_features(d3, targets=first, spec=spec)
    pd.testing.assert_frame_equal(Xa, Xb)


def test_power_window_is_a_linear_mean(synth_dwells):
    d = synth_dwells
    spec = FeatureSpec(windows_s=(600.0,))
    org = _origins(d, n=15, seed=2)
    X = build_features(d, targets=org, spec=spec, power_col="mean_dbm")
    ch = d[d["freq_mhz"] == 2437.0]
    t_end = (ch["t_start_s"] + ch["dwell_s"]).to_numpy()
    p = ch["mean_dbm"].to_numpy()
    for k, o in enumerate(org["t_start_s"]):
        m = (t_end <= o) & (t_end > o - 600.0)
        ref = 10 * np.log10(np.mean(10 ** (p[m] / 10)))
        assert X["pw_600s"].iloc[k] == pytest.approx(ref, abs=1e-6)
    assert PW_REF_DBM == -130.0
    # arithmetic dBm mean would differ whenever the window mixes strong and weak sweeps
    assert any(
        abs(
            10 * np.log10(np.mean(10 ** (p[(t_end <= o) & (t_end > o - 600.0)] / 10)))
            - p[(t_end <= o) & (t_end > o - 600.0)].mean()
        )
        > 0.1
        for o in org["t_start_s"]
    )


def test_hidden_and_delayed_evidence_for_origins(synth_dwells):
    d = synth_dwells
    org = _origins(d, n=20, seed=3)
    X0 = build_features(d, targets=org, delay_s=0.0)
    X1 = build_features(d, targets=org, delay_s=600.0)
    both = X0["age_s"].notna() & X1["age_s"].notna()
    assert (X1.loc[both, "age_s"] >= 600.0 - 1e-9).all()
    nothing = build_features(d, targets=org, visible=np.zeros(len(d), bool))
    assert nothing["last_occ"].isna().all() and (nothing["n_60s"] == 0).all()


def test_purged_split_keeps_training_windows_out_of_dev():
    # origins every 300 s for 2 days; window 900 s; lookback 7200 s -> embargo 8100 s
    o = np.arange(0, 2 * 86400.0, 300.0)
    emb = 7200.0 + 900.0
    tr, dv, te = chronological_split(o, embargo_s=emb)
    assert_no_leakage(o, tr, dv, te, emb)
    assert o[tr].max() + 900.0 <= o[dv].min()  # no training label window reaches into dev


def test_sweep_mask_drops_whole_sweeps():
    t = np.repeat(np.arange(200.0), 3)  # 3 channels per sweep
    m = sweep_visibility_mask(t, Degradation(dropout_p=0.5), seed=4).reshape(-1, 3)
    assert (m == m[:, :1]).all() and 0.3 < (~m[:, 0]).mean() < 0.7


# ---- metrics -----------------------------------------------------------------------------
def test_fraction_report_values_and_event_denominators():
    y = np.array([0.0, 0.1, 0.5, 0.9])
    p = np.array([0.0, 0.1, 0.1, 0.9])
    r = fraction_report(y, p, climatology=0.3, theta=0.4)
    assert r["mse"] == pytest.approx(0.16 / 4) and r["bias"] == pytest.approx(-0.1)
    mse_clim = np.mean((y - 0.3) ** 2)
    assert r["mss"] == pytest.approx(1 - r["mse"] / mse_clim)
    assert r["event_prevalence"] == 0.5  # y >= 0.4: two of four
    assert r["event_false_idle"] == pytest.approx(1 / 2)  # one high window forecast low, of two
    assert r["event_false_busy"] == 0.0


# ---- regression models -------------------------------------------------------------------
@pytest.fixture(scope="module")
def slow_synth():
    """SYNTHETIC activity with regimes (30 / 60 min) much longer than the 300 s window."""
    from spectrum_occupancy_forecasting.data.synthetic import SynthConfig, generate, to_dwell_table

    meta, bins = generate(SynthConfig(n_days=1.0, busy_mean_s=1800.0, idle_mean_s=3600.0, seed=5))
    return to_dwell_table(meta, bins, threshold_dbm=-70.0)


def _window_dataset(d, W=300.0, stride=60.0):
    spec = FeatureSpec(tau_s=300.0, windows_s=(300.0, 1800.0))
    ch = d[d["freq_mhz"] == 2437.0]
    org = origin_grid(d["t_start_s"].min() + 1800, d["t_start_s"].max(), stride, W)
    y, _ = window_fraction(ch["t_start_s"], ch["occ_any"], org, W, nominal_dt_s=4.0)
    tg = pd.DataFrame({"site": d["site"].iloc[0], "freq_mhz": 2437.0, "t_start_s": org})
    X = build_features(d, targets=tg, spec=spec, power_col="mean_dbm")
    keep = np.isfinite(y)
    return X[keep].reset_index(drop=True), y[keep], spec, W


def test_regressors_beat_climatology_on_persistent_activity(slow_synth):
    X, y, spec, W = _window_dataset(slow_synth)
    t = X["t_start_s"].to_numpy()
    emb = spec.max_lookback_s + W
    tr, dv, _ = chronological_split(t, fractions=(0.5, 0.3, 0.2), embargo_s=emb)
    clim = ClimatologyFraction().fit(X.iloc[tr], y[tr])
    mse0 = np.mean((clim.predict(X.iloc[dv]) - y[dv]) ** 2)
    models = [RecentFraction(300), DecayedFraction(), ridge_regressor(), gbt_regressor()]
    for m in models:
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[dv])
        assert p.shape == (len(dv),) and np.all((p >= 0) & (p <= 1)), m.name
    mse = {m.name: np.mean((m.predict(X.iloc[dv]) - y[dv]) ** 2) for m in models}
    assert mse["decayed"] < mse0 and mse["ridge"] < mse0, (mse, mse0)


def test_recent_fraction_is_cnt_over_n_with_fallback(synth_dwells):
    X, y, _, _ = _window_dataset(synth_dwells)
    m = RecentFraction(300).fit(X, y)
    p = m.predict(X)
    ok = X["n_300s"] > 0
    assert np.allclose(p[ok], (X["cnt_300s"] / X["n_300s"])[ok])
    X2 = X.copy()
    X2["n_300s"], X2["cnt_300s"] = 0.0, 0.0
    assert np.allclose(m.predict(X2), np.mean(y))


def test_hour_of_day_fraction_and_derived_features():
    t0 = pd.Timestamp("2022-02-08", tz="UTC").timestamp()
    t = t0 + np.arange(0, 4 * 86400, 600.0)
    hour = ((t - t0) // 3600) % 24
    X = pd.DataFrame({"t_start_s": t})
    y = np.where((hour >= 12) & (hour < 18), 0.8, 0.1)
    m = HourOfDayFraction(tz="UTC").fit(X, y)
    assert np.allclose(m.predict(X), y)
    D = with_derived(X.assign(n_60s=1.0, cnt_60s=0.0), tz="UTC")
    assert {"tod_sin", "tod_cos", "frac_60s"} <= set(D.columns)


# ---- residual models and equal-false-busy comparison -----------------------------------
def test_residual_regressor_reduces_to_the_recent_level_when_nothing_is_learnable(slow_synth):
    from spectrum_occupancy_forecasting.models.regressors import residual_gbt, residual_ridge

    X, y, spec, W = _window_dataset(slow_synth)
    base = RecentFraction(1800).fit(X, y)
    for make in (residual_ridge, residual_gbt):
        m = make(base_window_s=1800.0)
        m.fit(X, base.predict(X))  # target == base level: residual is exactly zero
        p = m.predict(X)
        assert np.allclose(p, base.predict(X), atol=1e-6), m.name
    m = residual_ridge(base_window_s=1800.0).fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))


def test_event_tradeoff_equal_false_busy_budget():
    from spectrum_occupancy_forecasting.evaluation.metrics import event_tradeoff

    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(500), np.full(500, 0.6)])
    good = np.concatenate([rng.normal(0.1, 0.05, 500), rng.normal(0.5, 0.05, 500)])
    poor = rng.normal(0.3, 0.2, 1000)
    g, p = event_tradeoff(y, good, 0.3), event_tradeoff(y, poor, 0.3)
    assert g["auc"] > 0.99 and g["false_idle_at_fb10"] < 0.02
    assert 0.4 < p["auc"] < 0.6 and p["false_idle_at_fb10"] > 0.7
    assert g["event_prevalence"] == 0.5
    assert np.isnan(event_tradeoff(np.zeros(10), np.arange(10.0), 0.3)["auc"])
