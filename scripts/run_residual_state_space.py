"""Diagnose saved forecasts, then conditionally run the declared state-space study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.adaptive_replay import replay, select
from spectrum_occupancy_forecasting.experiments.fine_spectrum import sha, write_json
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state
from spectrum_occupancy_forecasting.experiments.replay_workflow import run
from spectrum_occupancy_forecasting.experiments.residual_diagnosis import diagnose, source_hashes
from spectrum_occupancy_forecasting.experiments.residual_state_space import (
    STRATEGIES,
    StateBank,
    candidates,
)
from spectrum_occupancy_forecasting.experiments.state_space_reporting import regenerate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "adaptive_runner", ROOT / "scripts/run_adaptive_learning.py"
)
adaptive_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_runner)


def configuration(path):
    declared = {
        "study": "residual-state-space",
        "base_config": "adaptive_learning.yaml",
        "model_families": ["level", "trend"],
        "noise_coefficients": [0.01, 0.1, 1],
        "state_strengths": [0.25, 0.5, 1],
        "trend_half_life_h": 2,
        "minimum_initialization": 20,
        "variance_floor": 1e-6,
        "gate_headroom_skill": 0.02,
        "gate_correlation": 0.20,
        "gate_minimum_pairs": 80,
        "gate_policy": "either",
    }
    settings = yaml.safe_load(path.read_text())
    if settings != declared:
        raise ValueError("configuration/study mismatch or declared settings changed")
    return adaptive_runner.configuration(path.parent / settings["base_config"]) | settings


def diagnostic_specification(path):
    return fingerprint(
        [path, ROOT / "src/spectrum_occupancy_forecasting/experiments/residual_diagnosis.py"],
        root=ROOT,
    )


def verify_gate(directory, source, config):
    gate = json.loads((directory / "gate.json").read_text())
    if not gate["passes"] or gate["specification_hash"] != diagnostic_specification(config):
        raise ValueError("diagnostic gate failed or specification changed")
    if gate["source_hashes"] != source_hashes(source):
        raise ValueError("diagnostic source predictions changed")
    with tempfile.TemporaryDirectory(prefix="spectrum_gate_") as scratch:
        expected = diagnose(
            source, Path(scratch), configuration(config), diagnostic_specification(config)
        )
    if gate != expected:
        raise ValueError("gate decision cannot be reproduced from source predictions")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["diagnose", "synthetic-rehearsal", "exploratory", "report"],
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/residual_state_space.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "reproductions/shared_learning_cache")
    parser.add_argument(
        "--source-adaptive-dir", type=Path, default=ROOT / "results/adaptive_learning/run"
    )
    parser.add_argument("--diagnosis-dir", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = configuration(args.config)
    gate = None
    if args.mode == "exploratory":
        if args.diagnosis_dir is None:
            raise ValueError("exploratory mode requires --diagnosis-dir")
        gate = verify_gate(args.diagnosis_dir, args.source_adaptive_dir, args.config)
    if args.mode == "report" and args.predictions_dir is None:
        raise ValueError("report mode requires --predictions-dir")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "configuration.json", cfg)
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_adaptive_learning.py",
        args.config,
        ROOT / "configs/adaptive_learning.yaml",
        ROOT / "configs/followup_v2.yaml",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    with zipfile.ZipFile(
        args.out_dir / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    manifest = {
        "study": "residual-state-space",
        "mode": args.mode,
        "started": datetime.now(UTC).isoformat(),
        "status": "started",
        "source_hash": fingerprint(files, root=ROOT),
        "config_hash": sha(args.config),
        "git": git_state(ROOT),
        "python": sys.version,
        "CC2_accessed": False,
        "all_results_exploratory": True,
        "candidate_count": len(candidates(cfg)),
    }
    if gate is not None:
        write_json(args.out_dir / "gate.json", gate)
        manifest["gate_hash"] = sha(args.diagnosis_dir / "gate.json")
    try:
        with threadpool_limits(limits=2):
            if args.mode == "diagnose":
                diagnose(
                    args.source_adaptive_dir,
                    args.out_dir,
                    cfg,
                    diagnostic_specification(args.config),
                )
            elif args.mode == "report":
                manifest["input_prediction_hash"] = sha(
                    args.predictions_dir / "predictions.parquet"
                )
                regenerate(args.predictions_dir, args.out_dir, cfg)
            else:
                synthetic = args.mode == "synthetic-rehearsal"
                nodes = adaptive_runner.load_nodes(args.cache, cfg, synthetic)
                if not synthetic:
                    manifest["cache_manifests"] = {
                        folder: sha(args.cache / folder / "manifest.json")
                        for _, folder in adaptive_runner.RECORDINGS.values()
                    }
                run(
                    nodes,
                    cfg,
                    args.out_dir,
                    synthetic,
                    selection_fn=lambda n, c, t: select(
                        n,
                        c,
                        t,
                        candidate_factory=candidates,
                        bank_factory=StateBank,
                        strategies=STRATEGIES,
                    ),
                    replay_fn=lambda *a, **k: replay(*a, **k, bank_factory=StateBank),
                    strategies=STRATEGIES,
                    candidate_factory=candidates,
                    report_fn=regenerate,
                    scoring_grid=adaptive_runner.scoring_grid,
                )
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["finished"] = datetime.now(UTC).isoformat()
        write_json(args.out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
