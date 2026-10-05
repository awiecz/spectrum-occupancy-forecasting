"""Restricted, resumable target-band extraction; no full recording extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.data.aerpaw import (
    SweepSource,
    block_edges,
    check_datatype,
    decode_sweep,
    freq_axis,
    sessions,
)

LIMIT = 250 * 1024 * 1024
FINE_COLUMNS = [f"sub_{i:02d}" for i in range(20)] + [
    "spectral_p10",
    "spectral_p50",
    "spectral_p90",
    "spectral_spread",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def spectrum_row(power, frequencies):
    masks = [(frequencies >= 2427 + i) & (frequencies < 2428 + i) for i in range(20)]
    if any(not mask.any() for mask in masks):
        raise ValueError("frequency axis does not resolve every declared subband")
    means = [10 * np.log10(np.mean(10 ** (power[mask] / 10))) for mask in masks]
    band = power[(frequencies >= 2427) & (frequencies < 2447)]
    quantiles = np.quantile(band, [0.1, 0.5, 0.9])
    return np.r_[means, quantiles, quantiles[-1] - quantiles[0]]


def restricted_times(times, node, session, purpose, *, frozen=False):
    if node not in ("CC1", "LW1") or (session != 0 and (node, session) != ("LW1", 2)):
        raise ValueError("only first CC1/LW1 sessions and later LW1 are permitted")
    selected = np.asarray(times)[sessions(np.asarray(times)) == session]
    if not len(selected):
        raise ValueError("missing declared session")
    if session == 0:
        if purpose != "development":
            raise ValueError("first sessions are development only")
    elif purpose == "calibration":
        selected = selected[selected + 0.001 <= selected[0] + 86400]
    elif purpose != "assessment" or not frozen:
        raise PermissionError("later activity requires a verified freeze")
    return selected


def extract(archive, node, session, cache, purpose="development", *, frozen=False):
    archive, cache = Path(archive), Path(cache)
    if archive.name != f"Results{node}Feb2022_SigMF.zip":
        raise ValueError("unexpected archive; confirmation data prohibited")
    cache.mkdir(parents=True, exist_ok=True)
    if sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) > LIMIT:
        raise ValueError("fine-feature cache exceeds 250 MiB")
    folder = cache / f"{node}_session{session}_{purpose}"
    folder.mkdir(exist_ok=True)
    source = SweepSource(archive)
    times = restricted_times(source.times, node, session, purpose, frozen=frozen)
    meta = source.meta(float(times[0]))
    check_datatype(meta)
    axis = freq_axis(meta)
    if axis.source != "metadata":
        raise ValueError("explicit metadata frequency axis is required")
    frequencies = axis.freqs_mhz
    starts, centres = block_edges(frequencies, 5)
    counts = np.diff(np.r_[starts, len(frequencies)])
    selected_blocks = np.abs(centres - 2437) <= 10
    definition = {
        "node": node,
        "session": session,
        "purpose": purpose,
        "archive_sha256": sha(archive),
        "axis_sha256": hashlib.sha256(frequencies.tobytes()).hexdigest(),
        "start_s": float(times[0]),
        "end_s": float(times[-1]),
        "sweeps": len(times),
        "columns": FINE_COLUMNS,
    }
    manifest = folder / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in definition.items()):
            raise ValueError("cache inputs or extraction definition changed")
        hashes = old.get("shards", {})
    else:
        hashes = {}
    for begin in range(0, len(times), 500):
        shard = folder / f"part_{begin:06d}.parquet"
        if shard.exists():
            if hashes.get(shard.name) != sha(shard):
                raise ValueError("unverified or corrupted extraction shard")
            continue
        rows = []
        for time in times[begin : begin + 500]:
            metadata = source.meta(float(time))
            check_datatype(metadata)
            other = freq_axis(metadata)
            if other.source != "metadata" or not np.array_equal(other.freqs_mhz, frequencies):
                raise ValueError("frequency axis differs between sweeps")
            power = decode_sweep(source._read(source._data[float(time)]), len(frequencies))
            # Exactly reproduce the existing float32 5 MHz reduction and band conversion.
            linear = np.add.reduceat(10 ** (power / 10), starts) / counts
            blocks = (10 * np.log10(linear)).astype(np.float32).astype(float)
            target = 10 * np.log10(np.mean(10 ** (blocks[selected_blocks] / 10)))
            rows.append([time, target, *spectrum_row(power, frequencies)])
        pd.DataFrame(rows, columns=["time_s", "target_dbm", *FINE_COLUMNS]).to_parquet(
            shard, index=False
        )
        if sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) > LIMIT:
            shard.unlink()
            raise ValueError("fine-feature cache exceeds 250 MiB")
        hashes[shard.name] = sha(shard)
        write_json(manifest, definition | {"status": "running", "shards": hashes})
        print(
            f"{node} session {session} {purpose}: {min(begin + 500, len(times))}/{len(times)}",
            flush=True,
        )
    write_json(manifest, definition | {"status": "completed", "shards": hashes})
    if source._zip:
        source._zip.close()
    return pd.concat([pd.read_parquet(folder / name) for name in sorted(hashes)], ignore_index=True)


def raw_table(frame, node):
    return pd.DataFrame(
        {
            "site": node,
            "freq_mhz": 2437.0,
            "t_start_s": frame.time_s,
            "dwell_s": 0.001,
            "mean_dbm": frame.target_dbm,
        }
    )
