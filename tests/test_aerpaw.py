from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.data.aerpaw import (
    N_BINS,
    SweepSource,
    band_mask,
    band_stats,
    band_table_from_blocks,
    block_edges,
    block_spectrogram,
    check_datatype,
    decode_sweep,
    extract_band_table,
    freq_axis,
    label_band_table,
    node_from_path,
    parse_sweep_time,
    sessions,
    site_from_meta,
    write_synthetic_sweeps,
)
from spectrum_occupancy_forecasting.data.schema import validate_dwells
from spectrum_occupancy_forecasting.features.age_aware import build_features


def test_filename_time_is_new_york_local():
    # README example: 2022-02-08 12:50:34 local (EST = UTC-5) -> 17:50:34 UTC
    t = parse_sweep_time("results_20220208_125034.sigmf-meta")
    assert datetime.fromtimestamp(t, UTC) == datetime(2022, 2, 8, 17, 50, 34, tzinfo=UTC)
    assert parse_sweep_time("some/dir/results_20220208_125034.sigmf-data") == t
    with pytest.raises(ValueError):
        parse_sweep_time("notes.txt")


def test_dst_offset_changes_in_summer():
    winter = parse_sweep_time("results_20220115_120000.sigmf-data")
    summer = parse_sweep_time("results_20220715_120000.sigmf-data")
    assert datetime.fromtimestamp(winter, UTC).hour == 17  # EST
    assert datetime.fromtimestamp(summer, UTC).hour == 16  # EDT


def test_node_from_path():
    assert node_from_path("ResultsCC1Feb2022_SigMF.zip") == "CC1"
    assert node_from_path("x/ResultsLW1Feb2022_SigMF") == "LW1"
    assert node_from_path("weird") == "aerpaw"


def test_decode_checks_size_and_range():
    good = np.full(N_BINS, -90.0, dtype="<f4").tobytes()
    assert decode_sweep(good).shape == (N_BINS,)
    with pytest.raises(ValueError, match="expected"):
        decode_sweep(good[:-4])
    with pytest.raises(ValueError, match="plausible"):
        decode_sweep(np.full(N_BINS, 500.0, dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="finite"):
        decode_sweep(np.full(N_BINS, np.nan, dtype="<f4").tobytes())


def test_freq_axis_priority_and_flags():
    assumed = freq_axis(None)
    assert assumed.source == "assumed" and assumed.freqs_mhz[0] == pytest.approx(87.0)
    ann = {"annotations": [{"core:freq_lower_edge": 100e6, "core:freq_upper_edge": 5900e6}]}
    ax = freq_axis(ann)
    assert ax.source == "annotation"
    assert ax.freqs_mhz[0] == pytest.approx(100.0) and ax.freqs_mhz[-1] == pytest.approx(5900.0)
    real = {"global": {"dataset:frequency_axis_MHz": [1.0, 2.0, 4.0]}, **ann}  # real key wins
    ax = freq_axis(real)
    assert ax.source == "metadata" and ax.freqs_mhz.tolist() == [1.0, 2.0, 4.0]


def test_datatype_check_matches_dataset_converter():
    check_datatype({"global": {"core:datatype": "rf32_le"}})
    check_datatype(None)
    with pytest.raises(ValueError, match="rf32_le"):
        check_datatype({"global": {"core:datatype": "cf32_le"}})


def test_band_stats_uses_linear_mean():
    f = np.linspace(2400, 2500, 101)
    p = np.full(101, -100.0)
    p[50] = -50.0
    mean_dbm, max_dbm = band_stats(p, band_mask(f, 2450, 20))
    assert max_dbm == -50.0
    assert mean_dbm > -70  # linear average dominated by the strong bin; arithmetic would be ~-98
    with pytest.raises(ValueError):
        band_stats(p, band_mask(f, 3000, 20))


@pytest.mark.parametrize("as_zip", [False, True])
def test_source_table_and_labels(tmp_path, as_zip):
    path = tmp_path / ("ResultsCC1Feb2022_SigMF.zip" if as_zip else "ResultsCC1Feb2022_SigMF")
    write_synthetic_sweeps(path, n_sweeps=12, as_zip=as_zip, n_bins=4000)
    src = SweepSource(path)
    assert len(src) == 12 and np.all(np.diff(src.times) == 15)
    df = extract_band_table(src, [2437.0, 5200.0])
    assert set(df["site"]) == {"CC1"} and set(df["axis_source"]) == {"metadata"}
    assert len(df) == 24
    lab = label_band_table(df, {2437.0: -80.0, 5200.0: -80.0})
    ch = lab[lab["freq_mhz"] == 2437.0].sort_values("t_start_s")
    assert ch["occ_any"].tolist() == [i % 2 == 0 for i in range(12)]  # burst on even sweeps
    assert not lab[lab["freq_mhz"] == 5200.0]["occ_any"].any()
    validate_dwells(lab)


def test_time_window_selection_reads_only_selected(tmp_path):
    path = write_synthetic_sweeps(tmp_path / "ResultsLW1Feb2022_SigMF", n_sweeps=10, n_bins=4000)
    src = SweepSource(path)
    t0, t1 = src.times[3], src.times[7]
    got = [t for t, _ in src.sweeps(t0, t1)]
    assert got == list(src.times[3:7])
    assert len(list(src.sweeps(limit=2))) == 2


def test_missing_threshold_rejected(tmp_path):
    path = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=3, n_bins=4000)
    df = extract_band_table(SweepSource(path), [2437.0])
    with pytest.raises(ValueError, match="threshold"):
        label_band_table(df, {})


def test_aerpaw_table_feeds_the_feature_builder(tmp_path):
    path = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=40, n_bins=4000)
    df = label_band_table(extract_band_table(SweepSource(path), [2437.0, 5200.0]),
                          {2437.0: -80.0, 5200.0: -80.0})
    target = (df["freq_mhz"] == 2437.0).to_numpy()
    X = build_features(df, target)
    assert isinstance(X, pd.DataFrame) and len(X) == 40
    assert X["age_s"].dropna().between(14.9, 15.1).all()  # revisit period is 15 s


def test_missing_axis_key_falls_back_and_says_so(tmp_path):
    path = write_synthetic_sweeps(
        tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=3, include_axis=False
    )
    df = extract_band_table(SweepSource(path), [2437.0])
    assert set(df["axis_source"]) == {"assumed"}


def test_mixed_axes_are_refused(tmp_path):
    import json

    path = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=5, n_bins=4000)
    last = sorted(path.glob("*.sigmf-meta"))[-1]
    meta = json.loads(last.read_text())
    meta["global"]["dataset:frequency_axis_MHz"][10] += 1.0
    last.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="axis differs"):
        extract_band_table(SweepSource(path), [2437.0])


def test_macos_junk_in_zip_is_ignored(tmp_path):
    import zipfile

    z = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF.zip", n_sweeps=4, as_zip=True,
                               n_bins=4000)
    with zipfile.ZipFile(z, "a") as zf:  # AppleDouble stubs like those in the real archive
        for name in list(zf.namelist()):
            if name.endswith(".sigmf-data"):
                zf.writestr(f"__MACOSX/SigMF/._{name}", b"x" * 163)
    src = SweepSource(z)
    assert len(src) == 4
    assert all(len(v) == 4000 for _, v in src.sweeps())


def test_block_spectrogram_matches_bruteforce(tmp_path):
    path = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=6, n_bins=4000)
    src = SweepSource(path)
    times, centres, mean_dbm, max_dbm = block_spectrogram(src, block_mhz=50.0)
    axis = freq_axis(src.meta(float(src.times[0]))).freqs_mhz
    starts, c2 = block_edges(axis, 50.0)
    assert np.allclose(centres, c2) and mean_dbm.shape == (6, len(centres))
    _, p0 = next(src.sweeps(limit=1))
    lo, hi = starts[3], starts[4]
    ref_mean = 10 * np.log10(np.mean(10 ** (p0[lo:hi] / 10)))
    assert mean_dbm[0, 3] == pytest.approx(ref_mean, abs=1e-3)
    assert max_dbm[0, 3] == pytest.approx(p0[lo:hi].max(), abs=1e-3)
    assert np.all(np.diff(times) > 0)


def test_site_from_meta_ignores_annotation_label():
    meta = {"global": {"dataset:site": "CC1"}, "annotations": [{"core:label": "LW1"}]}
    assert site_from_meta(meta) == "CC1"
    assert site_from_meta({}) is None


def test_sessions_split_on_gaps():
    t = np.array([0, 19, 38, 5000, 5019, 99999], dtype=float)
    assert sessions(t, gap_s=300).tolist() == [0, 0, 0, 1, 1, 2]


def test_band_table_from_blocks_and_mean_label(tmp_path):
    path = write_synthetic_sweeps(tmp_path / "ResultsCC1Feb2022_SigMF", n_sweeps=8, n_bins=6000)
    t, c, m, x = block_spectrogram(SweepSource(path), block_mhz=5.0)
    npz = tmp_path / "aerpaw_CC1_blocks5MHz.npz"
    np.savez(npz, times_s=t, centres_mhz=c, mean_dbm=m, max_dbm=x)
    df = band_table_from_blocks(npz, [2437.0, 5200.0], bw_mhz=20.0)
    assert set(df["site"]) == {"CC1"} and len(df) == 16
    lab = label_band_table(df, {2437.0: -90.0, 5200.0: -90.0}, stat="mean_dbm")
    ch = lab[lab["freq_mhz"] == 2437.0].sort_values("t_start_s")
    assert ch["occ_any"].tolist() == [i % 2 == 0 for i in range(8)]
    assert not lab[lab["freq_mhz"] == 5200.0]["occ_any"].any()
    validate_dwells(lab)
