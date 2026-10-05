"""Reduce an AERPAW node archive to a compact per-sweep block spectrogram.

For every sweep: physical-mean (linear) and max power per ``--block-mhz`` frequency block.
Resumable (shards already on disk are skipped) and parallel (one zip handle per worker).

    uv run python scripts/extract_aerpaw_blocks.py data/raw/aerpaw/ResultsCC1Feb2022_SigMF.zip

Output: data/processed/aerpaw_<NODE>_blocks<N>MHz.npz with arrays
``times_s`` (epoch s UTC), ``centres_mhz``, ``mean_dbm`` and ``max_dbm`` (sweeps x blocks, float32).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from spectrum_occupancy_forecasting.data.aerpaw import (
    SweepSource,
    block_spectrogram,
    node_from_path,
    site_from_meta,
)


def _shard(args):
    zip_path, shard_dir, i, t0, t1, block_mhz = args
    out = Path(shard_dir) / f"shard_{i:04d}.npz"
    if out.exists():
        return i, 0.0
    t_start = time.time()
    times, centres, mean_dbm, max_dbm = block_spectrogram(SweepSource(zip_path), block_mhz, t0, t1)
    np.savez(
        str(out) + ".tmp.npz",
        times_s=times,
        centres_mhz=centres,
        mean_dbm=mean_dbm,
        max_dbm=max_dbm,
    )
    Path(str(out) + ".tmp.npz").replace(out)
    return i, time.time() - t_start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("zip_path")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--block-mhz", type=float, default=5.0)
    ap.add_argument("--shard-sweeps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    src = SweepSource(a.zip_path)
    meta_site = site_from_meta(src.meta(float(src.times[0])))
    path_site = node_from_path(a.zip_path, default="")
    node = path_site or meta_site or "aerpaw"  # the file name wins: metadata labels can be wrong
    if meta_site and path_site and meta_site != path_site:
        print(
            f"WARNING: archive name says {path_site!r} but metadata dataset:site says "
            f"{meta_site!r}; using {path_site!r}",
            flush=True,
        )
    out_dir = Path(a.out_dir)
    shard_dir = out_dir / f"_shards_{node}_{a.block_mhz:g}MHz"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cuts = list(range(0, len(src), a.shard_sweeps))
    jobs = []
    for i, c in enumerate(cuts):
        nxt = c + a.shard_sweeps
        t1 = float(src.times[nxt]) if nxt < len(src) else None
        jobs.append((a.zip_path, str(shard_dir), i, float(src.times[c]), t1, a.block_mhz))
    print(f"{len(src)} sweeps, {len(jobs)} shards, {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for done, (i, dt) in enumerate(ex.map(_shard, jobs), 1):
            print(
                f"[{done}/{len(jobs)}] shard {i} {'skipped' if dt == 0 else f'{dt:.0f}s'} "
                f"(elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )

    parts = [np.load(shard_dir / f"shard_{i:04d}.npz") for i in range(len(jobs))]
    merged = {
        "times_s": np.concatenate([p["times_s"] for p in parts]),
        "centres_mhz": parts[0]["centres_mhz"],
        "mean_dbm": np.concatenate([p["mean_dbm"] for p in parts]),
        "max_dbm": np.concatenate([p["max_dbm"] for p in parts]),
    }
    assert len(merged["times_s"]) == len(src), "sweep count mismatch after merge"
    assert np.all(np.diff(merged["times_s"]) > 0), "times not strictly increasing"
    out = out_dir / f"aerpaw_{node}_blocks{a.block_mhz:g}MHz.npz"
    np.savez_compressed(out, **merged)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), shape {merged['mean_dbm'].shape}")


if __name__ == "__main__":
    main()
