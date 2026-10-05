"""Describe existing recordings without training models or accessing confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.evaluation.data_study import analyze_node, make_figures
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.protocol import git_state

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_digest(path):
    """Portable text fingerprint; artifact and recording hashes remain byte-exact."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_input(path, node):
    """Check identity before opening any recording, including a symlink target."""
    path = Path(path)
    expected = f"aerpaw_{node}_blocks5MHz.npz"
    if path.name != expected or path.resolve().name != expected:
        raise ValueError(f"only {expected} is accepted; confirmation inputs are prohibited")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("exploratory", "synthetic-rehearsal"), default="exploratory")
    ap.add_argument("--cc1", type=Path, default=ROOT / "data/processed/aerpaw_CC1_blocks5MHz.npz")
    ap.add_argument("--lw1", type=Path, default=ROOT / "data/processed/aerpaw_LW1_blocks5MHz.npz")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/followup_v2.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if (cfg["window_s"], cfg["calibration_s"], cfg["primary_offset_db"]) != (900, 86400, 3):
        raise ValueError("data study requires the declared 15-minute, first-day, +3 dB settings")
    if cfg["minimum_coverage"] != 0.8 or cfg["offsets_db"] != [2, 3, 4, 6]:
        raise ValueError("data study requires 80% coverage and the declared label-offset grid")
    synthetic = args.mode == "synthetic-rehearsal"
    inputs = (
        {}
        if synthetic
        else {
            node: validate_input(path, node)
            for node, path in (("CC1", args.cc1), ("LW1", args.lw1))
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": args.mode,
        "synthetic": synthetic,
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": cfg,
        "config_sha256": text_digest(args.config),
        "config_hash_scheme": "sha256-lf-normalized-v1",
        "source_hash_scheme": "sha256-lf-normalized-v1",
        "artifact_hash_scheme": "sha256-raw-bytes",
        "git": git_state(ROOT),
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): text_digest(p)
            for p in sorted([*ROOT.glob("src/**/*.py"), ROOT / "scripts/study_data.py"])
        },
        "inputs": {},
        "seeds": {"CC1": 31, "LW1": 32} if synthetic else {},
        "analysis": "All-session inventory; session 0 analysis; nonoverlapping 15-minute windows",
    }
    target = args.out_dir / "manifest.json"
    try:
        if synthetic:
            for node, seed in manifest["seeds"].items():
                path = args.out_dir / f"aerpaw_{node}_blocks5MHz.npz"
                write_synthetic_blocks(path, n_days=3.1, dt_s=60, seed=seed)
                inputs[node] = path
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summaries = []
        for node, path in inputs.items():
            print(f"Describing {node}; no model fitting", flush=True)
            manifest["inputs"][node] = {"name": path.name, "sha256": digest(path)}
            summaries.append(analyze_node(path, node, cfg, args.out_dir))
        for summary in summaries:
            summary["synthetic"] = synthetic
        if synthetic:
            for csv in args.out_dir.glob("*.csv"):
                frame = pd.read_csv(csv)
                frame["data_kind"] = "SYNTHETIC DEMONSTRATION"
                frame.to_csv(csv, index=False)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
        )
        make_figures(args.out_dir, cfg, synthetic=synthetic)
        if synthetic:
            for path in inputs.values():
                path.unlink()  # Generated fixtures only; real recordings are never modified.
        size = sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file())
        if size > 10 * 1024 * 1024:
            raise ValueError("publication exceeds the 10 MiB artifact budget")
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {
            p.relative_to(args.out_dir).as_posix(): digest(p)
            for p in sorted(args.out_dir.rglob("*"))
            if p.is_file() and p != target
        }
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{'SYNTHETIC ONLY: ' if synthetic else ''}Study saved in {args.out_dir}")


if __name__ == "__main__":
    main()
