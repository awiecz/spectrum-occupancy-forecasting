import numpy as np
import pytest

from spectrum_occupancy_forecasting.evaluation.diagnostics import (
    effective_n,
    half_life,
    lag_autocorr,
)


def test_lag_autocorr_of_white_noise_and_of_a_persistent_series():
    rng = np.random.default_rng(0)
    assert abs(lag_autocorr(rng.normal(size=20000), 1)) < 0.03
    x = np.zeros(20000)
    for i in range(1, len(x)):  # AR(1) with rho = 0.8
        x[i] = 0.8 * x[i - 1] + rng.normal()
    assert lag_autocorr(x, 1) == pytest.approx(0.8, abs=0.03)
    assert np.isnan(lag_autocorr(np.ones(10), 1)) and np.isnan(lag_autocorr([1.0, 2.0], 5))


def test_effective_n_shrinks_with_dependence():
    rng = np.random.default_rng(1)
    white = rng.normal(size=5000)
    assert effective_n(white) == pytest.approx(5000, rel=0.1)
    x = np.zeros(5000)
    for i in range(1, len(x)):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    assert effective_n(x) == pytest.approx(5000 * 0.2 / 1.8, rel=0.2)  # (1-rho)/(1+rho)
    # negative autocorrelation is clipped to 0, never inflating n
    alt = np.tile([1.0, -1.0], 500)
    assert effective_n(alt) == pytest.approx(1000.0)


def test_half_life_interpolates_and_handles_edge_cases():
    delays = [0, 10, 30, 60, 120]
    expected = 30 + (0.5 - 0.4) / 0.2 * 30  # skill crosses 0.4 between 30 and 60 min
    assert half_life(delays, [0.8, 0.7, 0.5, 0.3, 0.0]) == pytest.approx(expected)
    assert half_life(delays, [0.8, 0.79, 0.7, 0.6, 0.5]) == float("inf")  # never halves
    assert np.isnan(half_life(delays, [-0.1, 0.0, 0.0, 0.0, 0.0]))  # no skill to begin with
    assert half_life([0, 60], [0.8, 0.4]) == pytest.approx(60.0)
    assert np.isnan(half_life([0, 1, 2], [1.0, 0.5]))  # mismatched lengths
