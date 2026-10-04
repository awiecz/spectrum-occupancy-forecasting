"""End-to-end sanity on SYNTHETIC data: skill should exist, then erode as evidence ages."""

import numpy as np
import pytest

from spectrum_occupancy_forecasting.data.synthetic import SynthConfig, generate, to_dwell_table
from spectrum_occupancy_forecasting.evaluation.metrics import binary_report
from spectrum_occupancy_forecasting.evaluation.splits import assert_no_leakage, chronological_split
from spectrum_occupancy_forecasting.features.age_aware import FeatureSpec, build_features
from spectrum_occupancy_forecasting.models.baselines import (
    AgeConditionedPersistence,
    DecayedHistory,
    GlobalRate,
    HourOfWeekRate,
    Persistence,
    age_bucket,
    gradient_boosting,
    logistic_regression,
)


@pytest.fixture(scope="module")
def big_synth():
    cfg = SynthConfig(n_days=1.0, seed=11)
    meta, bins = generate(cfg)
    return to_dwell_table(meta, bins, threshold_dbm=-70.0)


def _run(d, delay_s=0.0, visible=None, models=None):
    spec = FeatureSpec()
    target = (d["freq_mhz"] == 2437.0).to_numpy()
    X = build_features(d, target, visible, delay_s=delay_s, spec=spec)
    y = d.loc[X.index, "occ_any"].to_numpy()
    emb = max(spec.max_lookback_s, delay_s)
    tr, dv, te = chronological_split(X["t_start_s"].to_numpy(), embargo_s=emb)
    assert_no_leakage(X["t_start_s"].to_numpy(), tr, dv, te, emb)
    res = {}
    for m in models or [GlobalRate(), Persistence(), DecayedHistory(), AgeConditionedPersistence()]:
        m.fit(X.iloc[tr], y[tr])
        res[m.name] = binary_report(y[te], m.predict_proba(X.iloc[te]))
    return res


def test_history_beats_climatology_when_evidence_is_fresh(big_synth):
    r = _run(big_synth)
    assert r["decayed_history"]["brier"] < r["global_rate"]["brier"] - 0.02
    assert r["age_conditioned"]["brier"] < r["global_rate"]["brier"] - 0.02
    assert r["decayed_history"]["balanced_accuracy"] > 0.65


def test_skill_erodes_with_observation_delay(big_synth):
    fresh = _run(big_synth, delay_s=0.0)
    stale = _run(big_synth, delay_s=1800.0)
    gap_fresh = fresh["global_rate"]["brier"] - fresh["decayed_history"]["brier"]
    gap_stale = stale["global_rate"]["brier"] - stale["decayed_history"]["brier"]
    assert gap_stale < gap_fresh


def test_learned_models_run_and_are_not_worse_than_climatology(big_synth):
    r = _run(big_synth, models=[GlobalRate(), logistic_regression(), gradient_boosting()])
    assert r["logreg"]["brier"] < r["global_rate"]["brier"]
    assert r["hgb"]["brier"] < r["global_rate"]["brier"]


def test_hour_of_week_baseline_fits_and_predicts(big_synth):
    spec = FeatureSpec()
    target = (big_synth["freq_mhz"] == 2437.0).to_numpy()
    X = build_features(big_synth, target, spec=spec)
    y = big_synth.loc[X.index, "occ_any"].to_numpy()
    m = HourOfWeekRate(tz="UTC", min_count=5).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))


def test_hour_of_day_prior_learns_a_diurnal_pattern():
    import pandas as pd

    t0 = pd.Timestamp("2022-02-08 00:00", tz="UTC").timestamp()
    t = t0 + np.arange(0, 5 * 86400, 60.0)
    hour = ((t - t0) // 3600) % 24
    X = pd.DataFrame({"t_start_s": t})
    y = (hour >= 12) & (hour < 18)  # busy every afternoon
    m = HourOfWeekRate(tz="UTC", period="day").fit(X, y)
    p = m.predict_proba(X)
    assert m.name == "hour_of_day_rate" and p[y].mean() > 0.99 and p[~y].mean() < 0.01
    with pytest.raises(ValueError):
        HourOfWeekRate(period="month")


def test_age_bucket_edges():
    assert age_bucket([0.0, 9.9, 10.0, 59.0, 4000.0, np.nan]).tolist() == [0, 0, 1, 2, 5, -1]
