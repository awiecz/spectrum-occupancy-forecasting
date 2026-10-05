import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spectrum_occupancy_forecasting.evaluation import power

spec = importlib.util.spec_from_file_location(
    "power_analysis", Path(__file__).resolve().parents[1] / "scripts/power_analysis.py"
)
power_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power_analysis)


def test_block_indices_stay_in_range_and_have_requested_length():
    idx = power.block_indices(100, 32, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_independent_losses_recover_the_analytic_standard_error():
    rng = np.random.default_rng(1)
    n = 400
    baseline = rng.gamma(2.0, 1.0, size=n)
    difference = (0.05 * baseline + rng.normal(0, 0.3, size=n))[:, None]
    se = power_analysis.relative_se(difference, baseline, 1, 4000, rng)[0]
    delta = difference[:, 0] - difference.mean() / baseline.mean() * baseline
    analytic = delta.std(ddof=1) / np.sqrt(n) / baseline.mean()
    assert se == pytest.approx(analytic, rel=0.1)


def test_block_bootstrap_widens_intervals_for_autocorrelated_losses():
    rng = np.random.default_rng(2)
    n = 600
    noise = rng.normal(size=n)
    series = np.empty(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = 0.8 * series[i - 1] + noise[i]
    baseline = np.ones(n)
    independent = power_analysis.relative_se(series[:, None], baseline, 1, 3000, rng)[0]
    blocked = power_analysis.relative_se(series[:, None], baseline, 32, 3000, rng)[0]
    assert blocked > 1.8 * independent


def test_power_is_alpha_at_the_threshold_and_monotone_in_gain():
    assert power_analysis.power_at(0.0, 0.1) == pytest.approx(power_analysis.ALPHA)
    assert power_analysis.power_at(0.2, 0.1) > power_analysis.power_at(0.1, 0.1)
    detectable = power_analysis.Z * 0.1
    assert power_analysis.power_at(detectable, 0.1) == pytest.approx(power_analysis.POWER)


def test_primary_block_falls_back_when_eight_hours_leave_too_few_blocks():
    assert power_analysis.primary_block(439) == 32
    assert power_analysis.primary_block(81) == 16
