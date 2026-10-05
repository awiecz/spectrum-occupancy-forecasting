import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "multiband_inventory", Path(__file__).resolve().parents[1] / "scripts/multiband_inventory.py"
)
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)


@pytest.fixture
def cfg():
    return mb.load_config()


def synthetic_cache(tmp_path, monkeypatch, second_day_scale):
    """Two 20 MHz bands over two days: A varies on day one, B is quiet until day two."""
    rng = np.random.default_rng(0)
    t = 1.6e9 + 30.0 * np.arange(2 * 2880)
    power = rng.normal(-100, 0.2, size=(len(t), 8))
    first = t < t[0] + 86400
    burst = np.floor((t - t[0]) / 3600) % 2 == 0
    power[first & burst, 0:4] += 10
    power[~first, 4:8] += 10 * second_day_scale
    np.savez(
        tmp_path / "aerpaw_TST_blocks5MHz.npz",
        times_s=t,
        centres_mhz=2400.0 + 5 * np.arange(8),
        mean_dbm=power,
        max_dbm=power,
    )
    monkeypatch.setattr(mb, "CACHE", tmp_path / "aerpaw_{site}_blocks5MHz.npz")
    monkeypatch.setitem(mb.SESSION_ZERO_END, "TST", float(t[-1]))


def test_eligibility_uses_only_the_first_day(tmp_path, monkeypatch, cfg):
    synthetic_cache(tmp_path, monkeypatch, second_day_scale=1.0)
    table, _ = mb.inventory("TST", cfg)
    assert table.eligible.tolist() == [True, False]
    synthetic_cache(tmp_path, monkeypatch, second_day_scale=0.0)
    again, _ = mb.inventory("TST", cfg)
    assert again.eligible.tolist() == table.eligible.tolist()
    assert np.allclose(again.first_day_busy_share, table.first_day_busy_share)


def test_window_fractions_mark_under_covered_windows_missing():
    t = np.arange(0.0, 3000.0, 30.0)
    busy = np.ones((len(t), 1))
    busy[: len(t) // 2] = 0
    keep = np.ones(len(t), bool)
    keep[40:50] = False
    fractions = mb.window_fractions(t[keep], busy[keep], 0.0, 900.0, need=24)
    assert fractions[0, 0] == 0.0 and fractions[-1, 0] == 1.0
    assert np.isnan(fractions).sum() >= 1


def test_pooled_se_interpolates_between_independence_and_perfect_correlation():
    se = np.full(10, 0.2)
    assert mb.pooled_se(se, 0.0) == pytest.approx(0.2 / np.sqrt(10))
    assert mb.pooled_se(se, 1.0) == pytest.approx(0.2)
    assert mb.pooled_se(se, 0.3) > mb.pooled_se(se, 0.1)


def test_decision_power_requires_both_margin_and_significance():
    assert mb.decision_power(0.0, 0.01, 0.05, 0.03) < 0.002
    assert mb.decision_power(0.10, 0.01, 0.05, 0.03) > 0.999
    # With a large standard error the significance bound, not the margin, binds.
    assert mb.decision_power(0.05, 0.1, 0.05, 0.03) == pytest.approx(
        mb.norm.cdf((0.05 - 1.645 * 0.1) / 0.1), abs=1e-3
    )


def test_mean_correlation_of_independent_bands_is_near_zero():
    rng = np.random.default_rng(3)
    mean, median = mb.mean_correlation(rng.normal(size=(200, 30)))
    assert abs(mean) < 0.02 and abs(median) < 0.03
