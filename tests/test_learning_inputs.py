"""Retained spectral/cache invariants moved from the archived transfer challenger."""

import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.experiments.fine_spectrum import (
    restricted_times,
    spectrum_row,
)
from spectrum_occupancy_forecasting.experiments.learning_inputs import StableScaler, cached_frame


def test_spectral_bins_linear_mean_and_access_boundary():
    f = np.arange(2427.0, 2447.0, 0.1)
    p = np.full(len(f), -100.0)
    p[f < 2428] = -90.0
    row = spectrum_row(p, f)
    assert row[0] == pytest.approx(-90)
    assert row[2] == pytest.approx(-100)
    assert row[-1] == pytest.approx(np.quantile(p, 0.9) - np.quantile(p, 0.1))
    times = np.r_[np.arange(0.0, 2 * 86400, 30), 3 * 86400, np.arange(4 * 86400, 6 * 86400, 20)]
    calibration = restricted_times(times, "LW1", 2, "calibration")
    assert calibration.max() + 0.001 <= calibration[0] + 86400
    with pytest.raises(PermissionError):
        restricted_times(times, "LW1", 2, "assessment")
    with pytest.raises(ValueError):
        cached_frame("unused", "CC2_session0_development")


def test_stable_weighted_scaling():
    X = np.array([[1e4, 0.1], [1e4, 0.2], [1e4, 0.3]])
    weights = np.array([0.2, 0.3, 0.5])
    scaler = StableScaler().fit(X, weights)
    assert np.isfinite(scaler.transform(X)).all()
    assert scaler.scale_[0] == 1
    np.testing.assert_allclose(scaler.mean_, np.average(X, axis=0, weights=weights))


def test_extraction_resume_corruption_and_budget(tmp_path, monkeypatch):
    from spectrum_occupancy_forecasting.data.aerpaw import write_synthetic_sweeps
    from spectrum_occupancy_forecasting.experiments import fine_spectrum as fine

    archive = write_synthetic_sweeps(
        tmp_path / "ResultsCC1Feb2022_SigMF.zip", n_sweeps=3, as_zip=True
    )
    cache = tmp_path / "cache"
    first = fine.extract(archive, "CC1", 0, cache)
    pd.testing.assert_frame_equal(first, fine.extract(archive, "CC1", 0, cache))
    shard = next(cache.rglob("*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="corrupted"):
        fine.extract(archive, "CC1", 0, cache)
    monkeypatch.setattr(fine, "LIMIT", 1)
    small = tmp_path / "limited"
    with pytest.raises(ValueError, match="250 MiB"):
        fine.extract(archive, "CC1", 0, small)
    assert not list(small.rglob("*.parquet"))
