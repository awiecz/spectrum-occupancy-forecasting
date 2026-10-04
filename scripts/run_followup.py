"""Run the separate v2 study, rebuild reports, or verify the CC2 confirmation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.followup import load_config, run_node
from spectrum_occupancy_forecasting.experiments.followup_gate import (
    freeze_manifest,
    normalized_bytes,
    sha256_file,
    verify_archive,
)
from spectrum_occupancy_forecasting.experiments.followup_reporting import summarize
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state
from spectrum_occupancy_forecasting.viz.followup_figures import make_figures

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def artifact_hashes(out):
    return {
        p.relative_to(out).as_posix(): sha256_file(p)
        for p in out.rglob("*")
        if p.is_file()
        and "_parts" not in p.parts
        and p.name != "manifest.json"
        and p.suffix != ".npz"
    }


def rebuild(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for name in ("config.yaml", "predictions.parquet", "selections.json"):
        expected = manifest.get("artifact_hashes", {}).get(name)
        if expected and sha256_file(out / name) != expected:
            raise ValueError(f"saved input checksum mismatch: {name}")
    cfg = load_config(out / "config.yaml")
    predictions = pd.read_parquet(out / "predictions.parquet")
    selections = json.loads((out / "selections.json").read_text())
    tables = summarize(predictions, selections, cfg, out)
    make_figures(
        predictions,
        tables,
        out / "figures",
        synthetic=manifest["mode"] == "synthetic-rehearsal",
        confirmation=manifest["mode"] == "confirmation",
    )
    if manifest["mode"] == "confirmation":
        summary = json.loads((out / "summary.json").read_text())
        summary["interpretation"] = "CC2 frozen-design confirmation; same measurement campaign"
        write_json(out / "summary.json", summary)
    return tables


def legacy_lw1_sensitivity(path, stored, out):
    """Recompute legacy predictions, verify stored pooled numbers, report first/later folds."""
    import numpy as np

    from spectrum_occupancy_forecasting.experiments import window_forecast as wf

    raw = band_table_from_blocks(path, wf.CHANNELS, session=0)
    t0, last = raw["t_start_s"].min(), raw["t_start_s"].max()
    definitions = [(t0 + k * wf.DAY, min(t0 + (k + 1) * wf.DAY, last)) for k in (2, 3, 4)]
    evaluated = [
        wf.run_fold(raw, 900.0, 3.0, lo, hi, False)
        for lo, hi in definitions
        if hi - lo >= 0.5 * wf.DAY
    ]
    reference = json.loads(Path(stored).read_text())["windows"]["900s"]["clean"]
    rows, worst = [], 0.0
    for name in evaluated[0]["preds"]["clean"]:
        yy = np.concatenate([f["y"] for f in evaluated])
        pp = np.concatenate([f["preds"]["clean"][name] for f in evaluated])
        rmse = float(np.sqrt(np.mean((yy - pp) ** 2)))
        worst = max(worst, abs(rmse - reference[name]["rmse"]))
        for region, subset in (
            ("first_fold", evaluated[:1]),
            ("later_folds", evaluated[1:]),
            ("all_folds", evaluated),
        ):
            y = np.concatenate([f["y"] for f in subset])
            p = np.concatenate([f["preds"]["clean"][name] for f in subset])
            baseline = np.concatenate([f["preds"]["clean"][wf.REF] for f in subset])
            rows.append(
                {
                    "method": name,
                    "region": region,
                    "n_windows": len(y),
                    "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                    "mae": float(np.mean(np.abs(y - p))),
                    "mean_y": float(y.mean()),
                    "mss_vs_fixed": float(1 - np.sum((y - p) ** 2) / np.sum((y - baseline) ** 2)),
                    "interpretation": "retrospective legacy sensitivity; not fresh confirmation",
                }
            )
    if worst > 1e-6:
        raise ValueError(f"legacy LW1 pooled results differ from stored metrics: {worst}")
    pd.DataFrame(rows).to_csv(out / "legacy_lw1_overlap.csv", index=False)
    return {"max_pooled_rmse_difference": worst, "status": "matched_historical_metrics"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["exploratory", "synthetic-rehearsal", "confirmation", "report"],
        default="exploratory",
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs/followup_v2.yaml")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--cc1-npz", type=Path, default=ROOT / "data/processed/aerpaw_CC1_blocks5MHz.npz"
    )
    ap.add_argument(
        "--lw1-npz", type=Path, default=ROOT / "data/processed/aerpaw_LW1_blocks5MHz.npz"
    )
    ap.add_argument("--cc2-npz", type=Path)
    ap.add_argument("--archive-url")
    ap.add_argument("--verify-gate-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    gate = None
    if a.mode == "confirmation" or a.verify_gate_only:
        if not a.archive_url:
            ap.error("CC2 remains blocked: supply an independently timestamped freeze archive")
        expected = freeze_manifest(ROOT, a.config, ROOT / "docs/followup_v2_protocol.md")
        gate = verify_archive(a.archive_url, expected)
        if a.verify_gate_only:
            print(json.dumps(gate, indent=2))
            return
        if a.cc2_npz is None:
            ap.error("supply the verified CC2 block product only after the gate succeeds")
    if a.mode == "report":
        if a.out is None:
            ap.error("report mode requires --out pointing to an existing v2 run")
        with threadpool_limits(limits=1):
            rebuild(a.out.resolve())
        manifest_path = a.out.resolve() / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("report_rebuilds", []).append(
            {
                "at": timestamp(),
                "git": git_state(ROOT),
                "source_fingerprint": fingerprint(
                    [
                        ROOT / "src",
                        ROOT / "scripts/run_followup.py",
                        ROOT / "scripts/freeze_followup.py",
                        ROOT / "pyproject.toml",
                        ROOT / "uv.lock",
                    ],
                    root=ROOT,
                ),
            }
        )
        manifest["artifact_hashes"] = artifact_hashes(a.out.resolve())
        write_json(manifest_path, manifest)
        return
    if a.mode != "confirmation" and a.cc2_npz is not None:
        ap.error("CC2 is accepted only in gated confirmation mode")
    name = f"{a.mode}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out = (a.out or ROOT / "results/followup_v2" / name).resolve()
    # Do not allow an output path to overwrite or mingle with historical study artifacts.
    if out.is_relative_to(ROOT) and not (
        out.is_relative_to(ROOT / "results/followup_v2")
        or out.is_relative_to(ROOT / "reproductions")
    ):
        ap.error("inside this repository, outputs must be in results/followup_v2 or reproductions")
    out.mkdir(parents=True, exist_ok=False)
    if a.mode == "synthetic-rehearsal":
        syn = out / "aerpaw_SYN_blocks5MHz.npz"
        write_synthetic_blocks(syn, n_days=3.1, dt_s=60.0, seed=3)
        inputs = {"SYN": syn}
    elif a.mode == "confirmation":
        inputs = {"CC2": a.cc2_npz}
    else:
        inputs = {"CC1": a.cc1_npz, "LW1": a.lw1_npz}
        if any("CC2" in str(path).upper() for path in inputs.values()):
            ap.error("CC2 cannot be opened by exploratory mode")
    if a.mode == "confirmation":
        lock = ROOT / "results/followup_v2/CC2_CONFIRMATION_LOCK.json"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("x") as stream:
            json.dump({"started": timestamp(), "archive": gate, "out": out.name}, stream, indent=2)
    source = fingerprint(
        [
            ROOT / "src",
            ROOT / "scripts/run_followup.py",
            ROOT / "scripts/freeze_followup.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ],
        root=ROOT,
    )
    manifest = {
        "version": 2,
        "mode": a.mode,
        "status": "started",
        "started": timestamp(),
        "git": git_state(ROOT),
        "source_fingerprint": source,
        "fingerprint_scheme": "root-relative-v2",
        "config_sha256": hashlib.sha256(normalized_bytes(a.config)).hexdigest(),
        "archive_verification": gate,
        "inputs": {},
        "node_results": {},
    }
    (out / "config.yaml").write_bytes(normalized_bytes(a.config))
    write_json(out / "manifest.json", manifest)
    records = []
    try:
        with threadpool_limits(limits=1):
            for node, path in inputs.items():
                if node != "SYN" and f"aerpaw_{node}_" not in path.name:
                    raise ValueError(f"input filename must identify the expected {node} node")
                manifest["inputs"][node] = {"file": path.name, "sha256": sha256_file(path)}
                raw = band_table_from_blocks(path, cfg["channels_mhz"], site=node, session=0)
                result = run_node(
                    raw,
                    cfg,
                    out / "_parts",
                    task_names=["primary"] if a.mode == "synthetic-rehearsal" else None,
                )
                manifest["node_results"][node] = result
                records.extend(result["folds"])
                write_json(out / "manifest.json", manifest)
            parts = list((out / "_parts").glob("*.parquet"))
            forecasts = [p for p in parts if not p.stem.endswith("__validation")]
            validations = [p for p in parts if p.stem.endswith("__validation")]
            if not forecasts:
                manifest["status"] = "no_eligible_evaluation_targets"
            else:
                pd.concat([pd.read_parquet(p) for p in forecasts], ignore_index=True).to_parquet(
                    out / "predictions.parquet", index=False
                )
                pd.concat([pd.read_parquet(p) for p in validations], ignore_index=True).to_parquet(
                    out / "validation_predictions.parquet", index=False
                )
                write_json(out / "selections.json", records)
                print(
                    "Rebuilding metrics, intervals and figures from saved predictions", flush=True
                )
                rebuild(out)
                if a.mode == "exploratory":
                    manifest["legacy_lw1_sensitivity"] = legacy_lw1_sensitivity(
                        a.lw1_npz, ROOT / "results/node_LW1_window_forecast.json", out
                    )
                manifest["status"] = "completed"
            manifest["completed"] = timestamp()
            manifest["artifact_hashes"] = artifact_hashes(out)
            write_json(out / "manifest.json", manifest)
            if a.mode == "confirmation":
                write_json(
                    lock,
                    {
                        "started": manifest["started"],
                        "completed": timestamp(),
                        "status": manifest["status"],
                        "archive": gate,
                        "out": out.name,
                        "manifest_sha256": sha256_file(out / "manifest.json"),
                    },
                )
    except Exception as exc:
        manifest.update(status="failed", failed_at=timestamp(), error=str(exc))
        write_json(out / "manifest.json", manifest)
        raise
    print(f"V2 {manifest['status']}: {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
