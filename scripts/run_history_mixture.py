"""Reproduce the reviewed exploratory history mixture; CC2 is prohibited."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from run_followup import artifact_hashes, node_manifest, timestamp, write_json
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.followup import load_config
from spectrum_occupancy_forecasting.experiments.followup_data import NodeData
from spectrum_occupancy_forecasting.experiments.followup_gate import normalized_bytes, sha256_file
from spectrum_occupancy_forecasting.experiments.history_mixture import MODELS, validate_config
from spectrum_occupancy_forecasting.experiments.mixture_reporting import (
    attach_reference,
    make_figures,
    report,
)
from spectrum_occupancy_forecasting.experiments.mixture_support import run_node
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {"history-mixture": (validate_config, MODELS)}


def source_fingerprint():
    return fingerprint(
        [
            ROOT / "src",
            ROOT / "scripts/run_history_mixture.py",
            ROOT / "scripts/run_followup.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ],
        root=ROOT,
    )


def validate_study(cfg, study=None):
    if study is not None and cfg.get("study") != study:
        raise ValueError("configuration/study mismatch")
    if cfg.get("study") not in STUDIES:
        raise ValueError("unrecognized development study")
    STUDIES[cfg["study"]][0](cfg)


def rebuild(out, study=None):
    manifest = json.loads((out / "manifest.json").read_text())
    if "CC2" in str(out).upper() or "CC2" in manifest.get("inputs", {}):
        raise ValueError("CC2 is prohibited in this development experiment")
    for name in ("config.yaml", "predictions.parquet", "selections.json"):
        if (
            name in manifest.get("artifact_hashes", {})
            and sha256_file(out / name) != manifest["artifact_hashes"][name]
        ):
            raise ValueError(f"saved input checksum mismatch: {name}")
    cfg = load_config(out / "config.yaml")
    validate_study(cfg, study)
    predictions = pd.read_parquet(out / "predictions.parquet")
    records = json.loads((out / "selections.json").read_text())
    tables = report(predictions, records, cfg, out)
    make_figures(predictions, tables, out / "figures")
    manifest.setdefault("report_rebuilds", []).append(
        {"at": timestamp(), "source_fingerprint": source_fingerprint(), "git": git_state(ROOT)}
    )
    manifest["artifact_hashes"] = artifact_hashes(out)
    write_json(out / "manifest.json", manifest)
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=["synthetic-rehearsal", "exploratory", "report"], default="exploratory"
    )
    ap.add_argument(
        "--study",
        choices=list(STUDIES),
        default="history-mixture",
    )
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--cc1-npz", type=Path, default=ROOT / "data/processed/aerpaw_CC1_blocks5MHz.npz"
    )
    ap.add_argument(
        "--lw1-npz", type=Path, default=ROOT / "data/processed/aerpaw_LW1_blocks5MHz.npz"
    )
    a = ap.parse_args()
    if a.config is None:
        filename = a.study.replace("-", "_") + ".yaml"
        a.config = ROOT / "configs" / filename
    cfg = load_config(a.config)
    validate_study(cfg, a.study)
    if a.mode == "report":
        if a.out is None:
            ap.error("report mode requires an existing run directory")
        with threadpool_limits(1):
            rebuild(a.out.resolve(), a.study)
        return
    inputs = {"CC1": a.cc1_npz, "LW1": a.lw1_npz}
    if any("CC2" in str(path).upper() for path in inputs.values()):
        ap.error("CC2 is prohibited in this development experiment")
    out = (
        a.out
        or ROOT / "reproductions" / f"{a.study}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    ).resolve()
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "reproductions"):
        ap.error("development outputs inside this repository must be under reproductions")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_bytes(normalized_bytes(a.config))
    manifest = {
        "study": cfg["study"],
        "mode": a.mode,
        "status": "started",
        "started": timestamp(),
        "git": git_state(ROOT),
        "source_fingerprint": source_fingerprint(),
        "config_sha256": sha256_file(out / "config.yaml"),
        "inputs": {},
        "node_results": {},
        "interpretation": "exploratory development on reused recordings; CC2 untouched",
    }
    write_json(out / "manifest.json", manifest)
    records = []
    try:
        if a.mode == "synthetic-rehearsal":
            path = out / "aerpaw_SYN_blocks5MHz.npz"
            write_synthetic_blocks(path, n_days=4.75, dt_s=60, seed=3)
            inputs = {"SYN": path}
        reference_out = ROOT / cfg["reference_results"]
        reference = None
        with threadpool_limits(1):
            for node, path in inputs.items():
                if f"aerpaw_{node}_" not in path.name:
                    raise ValueError(f"input filename must identify {node}")
                manifest["inputs"][node] = {"file": path.name, "sha256": sha256_file(path)}
                if node != "SYN":
                    old = json.loads((reference_out / "manifest.json").read_text())
                    if old["inputs"][node]["sha256"] != manifest["inputs"][node]["sha256"]:
                        raise ValueError("input differs from canonical v2 recording")
                    if reference is None:
                        expected = old["artifact_hashes"]["predictions.parquet"]
                        if sha256_file(reference_out / "predictions.parquet") != expected:
                            raise ValueError("canonical v2 prediction checksum mismatch")
                        reference = pd.read_parquet(reference_out / "predictions.parquet")
                        manifest["reference_predictions_sha256"] = expected
                raw = band_table_from_blocks(path, cfg["channels_mhz"], site=node, session=0)
                models = STUDIES[a.study][1].items()
                node_cfg = cfg
                for hours in cfg["validation_hours"]:
                    for regime, model in models:
                        node_class = NodeData
                        result = run_node(
                            raw,
                            node_cfg,
                            out / "_parts",
                            strategy=(hours, regime),
                            model_class=model,
                            node_class=node_class,
                        )
                        records.extend(result["folds"])
                        manifest["node_results"][f"{node}_{hours}h_{regime}"] = node_manifest(
                            result
                        )
                        write_json(out / "manifest.json", manifest)
            parts = list((out / "_parts").glob("*.parquet"))
            forecasts = [p for p in parts if not p.stem.endswith("__validation")]
            validations = [p for p in parts if p.stem.endswith("__validation")]
            write_json(out / "selections.json", records)
            if not forecasts:
                manifest["status"] = "no_eligible_evaluation_targets"
            else:
                predictions = pd.concat([pd.read_parquet(p) for p in forecasts], ignore_index=True)
                if reference is not None:
                    predictions = attach_reference(predictions, reference)
                predictions.to_parquet(out / "predictions.parquet", index=False)
                if validations:
                    pd.concat(
                        [pd.read_parquet(p) for p in validations], ignore_index=True
                    ).to_parquet(out / "validation_predictions.parquet", index=False)
                rebuild(out)
                manifest = json.loads((out / "manifest.json").read_text())
                manifest["status"] = "completed"
            manifest["completed"] = timestamp()
            manifest["artifact_hashes"] = artifact_hashes(out)
            write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), failed_at=timestamp())
        write_json(out / "manifest.json", manifest)
        raise
    print(f"History-mixture experiment {manifest['status']}: {out}")


if __name__ == "__main__":
    main()
