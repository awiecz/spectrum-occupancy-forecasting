import pytest

from spectrum_occupancy_forecasting.data.synthetic import SynthConfig, generate, to_dwell_table


@pytest.fixture(scope="session")
def synth_dwells():
    """Small SYNTHETIC dwell table (tests only)."""
    cfg = SynthConfig(n_days=0.25, seed=1)
    meta, bins = generate(cfg)
    return to_dwell_table(meta, bins, threshold_dbm=-70.0)
