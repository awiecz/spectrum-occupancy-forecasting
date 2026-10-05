"""AERPAW fixed-node sub-6 GHz sweep reader (Dryad doi:10.5061/dryad.hmgqnk9zn).

What the dataset README states (verified): one SigMF pair per sweep,
``results_YYYYMMDD_HHMMSS.sigmf-{data,meta}``; timestamp in the *file name*, America/New_York;
payload = ``98,868`` little-endian float32 values in **dBm**; sweep spans ~87 MHz - 6 GHz; no
missing values; ~15 s per sweep (arXiv 2510.08752 describes the helikite variant: 30.72 MS/s,
25.68 MHz hop).

Metadata layout (verified from the dataset's own ``convert_SigMF_to_mat.py``): the frequency
axis is stored explicitly per file in ``global["dataset:frequency_axis_MHz"]`` and the datatype in
``global["core:datatype"]`` (must be ``rf32_le``); the bin count is the axis length, not a
constant. Fallbacks (annotation edges, then an *assumed* linear 87-6000 MHz axis) exist only for
files lacking that key and are reported via ``axis_source``. Calibration is undocumented, so
absolute thresholds must be treated as relative to the node's own noise floor.

Snapshot caveat: a sweep gives ONE power snapshot per channel (no sub-dwell time bins), so the
canonical ``duty`` column is set to ``float(occ_any)`` and ``dwell_s`` is a nominal value.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

N_BINS = 98_868
ASSUMED_START_MHZ = 87.0
ASSUMED_STOP_MHZ = 6000.0
SNAPSHOT_S = 1e-3  # nominal; per-hop dwell is not documented
LOCAL_TZ = ZoneInfo("America/New_York")
_NAME_RE = re.compile(r"results_(\d{8})_(\d{6})\.sigmf-(data|meta)$")
_NODE_RE = re.compile(r"Results([A-Za-z0-9]+?)(?:Feb2022)?(?:_SigMF)?(?:\.zip)?$")


def _is_junk(name: str) -> bool:
    """macOS archive debris (``__MACOSX/`` and AppleDouble ``._*`` stubs) shipped inside the zip."""
    parts = name.replace("\\", "/").split("/")
    return "__MACOSX" in parts or parts[-1].startswith("._")


def parse_sweep_time(name: str) -> float:
    """Epoch seconds (UTC) from ``results_YYYYMMDD_HHMMSS.sigmf-*`` (America/New_York local)."""
    m = _NAME_RE.search(Path(name).name)
    if not m:
        raise ValueError(f"not an AERPAW sweep file name: {name!r}")
    local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=LOCAL_TZ)
    return local.timestamp()


def node_from_path(path: str | Path, default: str = "aerpaw") -> str:
    """Node id (CC1/CC2/LW1) from names like ``ResultsCC1Feb2022_SigMF[.zip]``."""
    m = _NODE_RE.search(Path(path).name)
    return m.group(1) if m else default


@dataclass(frozen=True)
class FreqAxis:
    freqs_mhz: np.ndarray
    source: str  # "metadata" or "assumed"


def site_from_meta(meta: dict | None) -> str | None:
    """Node id from ``global["dataset:site"]``. (Do not use ``annotations[].core:label``: in the
    CC1 archive it reads 'LW1'.)"""
    return ((meta or {}).get("global") or {}).get("dataset:site")


def check_datatype(meta: dict | None) -> None:
    """Raise if the metadata declares a datatype other than little-endian real float32."""
    dt = ((meta or {}).get("global") or {}).get("core:datatype")
    if dt is not None and dt != "rf32_le":
        raise ValueError(f"unsupported core:datatype {dt!r}; expected 'rf32_le'")


def freq_axis(meta: dict | None, n: int = N_BINS) -> FreqAxis:
    """Frequency axis (MHz): the metadata's explicit axis, else annotation edges, else assumed."""
    g = (meta or {}).get("global") or {}
    if "dataset:frequency_axis_MHz" in g:
        return FreqAxis(np.asarray(g["dataset:frequency_axis_MHz"], dtype=np.float64), "metadata")
    for ann in (meta or {}).get("annotations", []) or []:
        lo, hi = ann.get("core:freq_lower_edge"), ann.get("core:freq_upper_edge")
        if lo is not None and hi is not None and hi > lo:
            return FreqAxis(np.linspace(lo / 1e6, hi / 1e6, n), "annotation")
    return FreqAxis(np.linspace(ASSUMED_START_MHZ, ASSUMED_STOP_MHZ, n), "assumed")


def decode_sweep(payload: bytes, n_bins: int = N_BINS) -> np.ndarray:
    """Little-endian float32 dBm vector; raises if the bin count or values are implausible."""
    if len(payload) != 4 * n_bins:
        raise ValueError(f"expected {4 * n_bins} bytes ({n_bins} float32 bins), got {len(payload)}")
    x = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite power values in sweep")
    if x.min() < -200 or x.max() > 60:
        raise ValueError(f"power outside plausible dBm range: [{x.min():.1f}, {x.max():.1f}]")
    return x


def band_mask(freqs_mhz: np.ndarray, center_mhz: float, bw_mhz: float) -> np.ndarray:
    half = bw_mhz / 2.0
    return (freqs_mhz >= center_mhz - half) & (freqs_mhz <= center_mhz + half)


def band_stats(power_dbm: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(physical mean power dBm over the band's bins [linear average], max bin dBm)."""
    if not mask.any():
        raise ValueError("no bins inside band; check frequency axis / channel")
    sel = power_dbm[mask]
    return float(10 * np.log10(np.mean(10 ** (sel / 10)))), float(sel.max())


# ---- sources: directory or zip, no full extraction ----------------------------------------
class SweepSource:
    """Time-ordered access to sweeps in a directory or a ``.zip`` (members read on demand)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.suffix.lower() == ".zip" else None
        self._paths: dict[str, Path] = {}
        if self._zip:
            names = self._zip.namelist()
        else:
            self._paths = {
                p.name: p for p in self.path.rglob("results_*.sigmf-*") if not _is_junk(p.name)
            }
            names = list(self._paths)
        self._data: dict[float, str] = {}
        for nm in names:
            if _is_junk(nm):
                continue
            m = _NAME_RE.search(nm)
            if m and m.group(3) == "data":
                self._data[parse_sweep_time(nm)] = nm
        self.times = np.array(sorted(self._data))

    def __len__(self) -> int:
        return len(self.times)

    def _read(self, name: str) -> bytes:
        if self._zip:
            return self._zip.read(name)
        return self._paths[Path(name).name].read_bytes()

    def n_bins(self) -> int:
        """Bins per sweep: the length of the metadata frequency axis (``N_BINS`` if absent)."""
        if not len(self):
            return N_BINS
        return len(freq_axis(self.meta(float(self.times[0]))).freqs_mhz)

    def meta(self, t: float) -> dict | None:
        name = self._data[t].replace(".sigmf-data", ".sigmf-meta")
        try:
            return json.loads(self._read(name))
        except (KeyError, FileNotFoundError, json.JSONDecodeError):
            return None

    def sweeps(
        self,
        t0: float | None = None,
        t1: float | None = None,
        limit: int | None = None,
        n_bins: int | None = None,
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(t_epoch_s, dBm_vector)`` for sweeps with ``t0 <= t < t1`` (time-selected from
        file names, so unselected payloads are never read)."""
        if n_bins is None:
            n_bins = self.n_bins()
        sel = self.times
        if t0 is not None:
            sel = sel[sel >= t0]
        if t1 is not None:
            sel = sel[sel < t1]
        if limit is not None:
            sel = sel[:limit]
        for t in sel:
            yield float(t), decode_sweep(self._read(self._data[float(t)]), n_bins)


def extract_band_table(
    source: SweepSource,
    channels_mhz: Sequence[float],
    bw_mhz: float = 20.0,
    site: str | None = None,
    t0: float | None = None,
    t1: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Per-sweep, per-channel band statistics (no labels yet).

    Columns: site, freq_mhz, t_start_s, dwell_s, mean_dbm, max_dbm, axis_source.
    Labels are applied separately with thresholds fitted on training data only.
    """
    if not len(source):
        raise ValueError(f"no sweeps found in {source.path}")
    first_meta = source.meta(float(source.times[0]))
    check_datatype(first_meta)
    site = site or site_from_meta(first_meta) or node_from_path(source.path)
    axis = freq_axis(first_meta)
    n_bins = len(axis.freqs_mhz)
    # The axis is stored in every file; confirm it is identical on a sample (middle/last).
    if axis.source == "metadata":
        for t in source.times[[len(source) // 2, -1]]:
            other = freq_axis(source.meta(float(t)))
            if other.source != "metadata" or not np.array_equal(other.freqs_mhz, axis.freqs_mhz):
                raise ValueError("frequency axis differs between sweeps; refusing to mix them")
    masks = {c: band_mask(axis.freqs_mhz, c, bw_mhz) for c in channels_mhz}
    rows = []
    for t, p in source.sweeps(t0, t1, limit, n_bins):
        for c, m in masks.items():
            mean_dbm, max_dbm = band_stats(p, m)
            rows.append((site, float(c), t, SNAPSHOT_S, mean_dbm, max_dbm, axis.source))
    return pd.DataFrame(
        rows,
        columns=["site", "freq_mhz", "t_start_s", "dwell_s", "mean_dbm", "max_dbm", "axis_source"],
    )


def block_edges(freqs_mhz: np.ndarray, block_mhz: float):
    """Contiguous block index per bin and the (start index, block centre MHz) of each block."""
    k = np.floor((freqs_mhz - freqs_mhz[0]) / block_mhz).astype(int)
    blocks, starts = np.unique(k, return_index=True)
    centres = freqs_mhz[0] + (blocks + 0.5) * block_mhz
    return starts, centres


def block_spectrogram(
    source: SweepSource,
    block_mhz: float = 5.0,
    t0: float | None = None,
    t1: float | None = None,
    limit: int | None = None,
):
    """Whole-band reduction: per sweep, the linear-mean and max power per frequency block.

    Returns ``(times_s, centres_mhz, mean_dbm[n_sweeps, n_blocks], max_dbm[...])`` as float32.
    """
    if not len(source):
        raise ValueError(f"no sweeps found in {source.path}")
    meta = source.meta(float(source.times[0]))
    check_datatype(meta)
    axis = freq_axis(meta)
    starts, centres = block_edges(axis.freqs_mhz, block_mhz)
    counts = np.diff(np.append(starts, len(axis.freqs_mhz)))
    times, means, maxs = [], [], []
    for t, p in source.sweeps(t0, t1, limit, len(axis.freqs_mhz)):
        lin = np.add.reduceat(10 ** (p / 10), starts) / counts
        means.append(10 * np.log10(lin))
        maxs.append(np.maximum.reduceat(p, starts))
        times.append(t)
    return (
        np.asarray(times),
        centres,
        np.asarray(means, dtype=np.float32),
        np.asarray(maxs, dtype=np.float32),
    )


def sessions(times_s: np.ndarray, gap_s: float = 300.0) -> np.ndarray:
    """Session id per sorted sweep time; a new one starts after a gap longer than ``gap_s``."""
    return np.concatenate([[0], np.cumsum(np.diff(times_s) > gap_s)]).astype(int)


def band_table_from_blocks(
    npz_path: str | Path,
    channels_mhz: Sequence[float],
    bw_mhz: float = 20.0,
    site: str | None = None,
    session: int | None = 0,
) -> pd.DataFrame:
    """Per-sweep, per-channel band table from a block spectrogram (``extract_aerpaw_blocks.py``).

    Band mean = linear average of the 5 MHz block means whose centres lie within +-bw/2 of the
    channel centre (blocks hold equal bin counts up to +-1, so equal weights are used); band max =
    max of the block maxima. ``session`` selects one continuous recording (None = all).
    """
    z = np.load(npz_path)
    t, f = z["times_s"], z["centres_mhz"]
    mean_dbm, max_dbm = z["mean_dbm"].astype(float), z["max_dbm"].astype(float)
    keep = np.ones(len(t), bool) if session is None else sessions(t) == session
    site = site or Path(npz_path).stem.split("_")[1]
    frames = []
    for c in channels_mhz:
        sel = np.abs(f - c) <= bw_mhz / 2
        if not sel.any():
            raise ValueError(f"no blocks within +-{bw_mhz / 2} MHz of {c} MHz")
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "freq_mhz": float(c),
                    "t_start_s": t[keep],
                    "dwell_s": SNAPSHOT_S,
                    "mean_dbm": 10 * np.log10((10 ** (mean_dbm[keep][:, sel] / 10)).mean(axis=1)),
                    "max_dbm": max_dbm[keep][:, sel].max(axis=1),
                    "axis_source": "metadata",
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values("t_start_s", ignore_index=True)


def label_band_table(
    df: pd.DataFrame, thresholds_dbm: dict[float, float], stat: str = "max_dbm"
) -> pd.DataFrame:
    """Add ``occ_any`` (``stat`` above the channel threshold) and ``duty`` to a band table.

    ``thresholds_dbm`` must be fitted on training data only (see occupancy.detect).
    """
    out = df.copy()
    thr = out["freq_mhz"].map(thresholds_dbm)
    if thr.isna().any():
        raise ValueError("missing threshold for some channels")
    out["occ_any"] = out[stat] > thr
    out["duty"] = out["occ_any"].astype(float)  # snapshot data: no sub-dwell time bins
    return out.sort_values("t_start_s", ignore_index=True)


def write_synthetic_sweeps(
    path: Path,
    n_sweeps: int = 20,
    seed: int = 0,
    as_zip: bool = False,
    n_bins: int = N_BINS,
    t_first: str = "20220208_125000",
    include_axis: bool = True,
) -> Path:
    """TEST HELPER: write synthetic AERPAW-style sweeps (never used for results)."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(ASSUMED_START_MHZ, ASSUMED_STOP_MHZ, n_bins)
    base = datetime.strptime(t_first, "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
    members: dict[str, bytes] = {}
    for i in range(n_sweeps):
        ts = datetime.fromtimestamp(base.timestamp() + 15 * i, LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        p = -95 + rng.normal(0, 1, n_bins)
        if i % 2 == 0:  # burst in the 2437 MHz Wi-Fi channel on every other sweep
            p[band_mask(freqs, 2437.0, 20.0)] += 35.0
        members[f"results_{ts}.sigmf-data"] = p.astype("<f4").tobytes()
        members[f"results_{ts}.sigmf-meta"] = json.dumps(
            {
                "global": {
                    "core:datatype": "rf32_le",
                    "core:version": "1.0.0",
                    **({"dataset:frequency_axis_MHz": freqs.tolist()} if include_axis else {}),
                },
                "captures": [{"core:sample_start": 0}],
                "annotations": [],
            }
        ).encode()
    if as_zip:
        with zipfile.ZipFile(path, "w") as z:
            for k, v in members.items():
                z.writestr(k, v)
    else:
        path.mkdir(parents=True, exist_ok=True)
        for k, v in members.items():
            (path / k).write_bytes(v)
    return path


__all__ = [
    "FreqAxis",
    "N_BINS",
    "SweepSource",
    "band_mask",
    "band_stats",
    "band_table_from_blocks",
    "block_edges",
    "block_spectrogram",
    "check_datatype",
    "decode_sweep",
    "extract_band_table",
    "freq_axis",
    "label_band_table",
    "node_from_path",
    "parse_sweep_time",
    "sessions",
    "site_from_meta",
]
