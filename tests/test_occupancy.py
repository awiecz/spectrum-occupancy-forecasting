import numpy as np
import pytest

from spectrum_occupancy_forecasting.occupancy.detect import (
    BIN_S,
    DWELL_BINS,
    DWELL_S,
    dwell_labels,
    estimate_noise_floor_dbm,
    occupied_bins,
)
from spectrum_occupancy_forecasting.occupancy.power import dbm_to_mw, mean_power_dbm, mw_to_dbm


def test_dwell_constants_match_nist():
    assert DWELL_BINS == 210
    assert BIN_S == pytest.approx(0.5e-3)
    assert DWELL_S == pytest.approx(0.105)


def test_dbm_roundtrip():
    x = np.array([-90.0, -60.0, 0.0, 13.0])
    assert np.allclose(mw_to_dbm(dbm_to_mw(x)), x)
    assert dbm_to_mw(0.0) == pytest.approx(1.0)  # 0 dBm = 1 mW
    assert dbm_to_mw(30.0) == pytest.approx(1000.0)


def test_mean_power_is_linear_not_arithmetic():
    # One 0 dBm sample among nine -100 dBm samples: physical mean ~= -10 dBm, NOT -90 dBm.
    x = np.array([0.0] + [-100.0] * 9)
    assert mean_power_dbm(x) == pytest.approx(10 * np.log10(0.1), abs=1e-3)
    assert x.mean() == pytest.approx(-90.0)
    assert mean_power_dbm(x) > x.mean() + 70


def test_zero_power_is_minus_inf_not_idle_sentinel():
    assert np.isneginf(mw_to_dbm(0.0))


def test_noise_floor_is_low_quantile():
    rng = np.random.default_rng(0)
    p = rng.normal(-85, 1, 10_000)
    assert estimate_noise_floor_dbm(p, 0.05) == pytest.approx(-85 - 1.645, abs=0.1)


def test_threshold_is_strict():
    p = np.array([-60.0, -59.9, -60.1])
    assert occupied_bins(p, -60.0).tolist() == [False, True, False]


def _mask(rows):
    m = np.zeros((len(rows), 20), dtype=bool)
    for r, runs in enumerate(rows):
        for a, b in runs:
            m[r, a:b] = True
    return m


def test_events_exclude_edge_runs():
    m = _mask([[(0, 5)], [(15, 20)], [(5, 8)], [], [(0, 20)]])
    lab = dwell_labels(m)
    assert lab.occ_any.tolist() == [False, False, True, False, False]
    assert lab.n_events.tolist() == [0, 0, 1, 0, 0]
    assert lab.event_bins.tolist() == [3]
    # duty counts every occupied bin, including edge runs
    assert lab.duty[0] == pytest.approx(5 / 20)
    assert lab.duty[4] == pytest.approx(1.0)


def test_event_and_gap_lengths():
    m = _mask([[(2, 4), (7, 10), (14, 15)]])
    lab = dwell_labels(m)
    assert lab.event_bins.tolist() == [2, 3, 1]
    assert lab.gap_bins.tolist() == [3, 4]
    assert lab.event_row.tolist() == [0, 0, 0]


def test_gaps_do_not_cross_dwells():
    m = _mask([[(2, 4)], [(5, 6)]])
    lab = dwell_labels(m)
    assert lab.gap_bins.size == 0
    assert lab.event_row.tolist() == [0, 1]


def test_labels_reject_wrong_shape():
    with pytest.raises(ValueError):
        dwell_labels(np.zeros(10, dtype=bool))
