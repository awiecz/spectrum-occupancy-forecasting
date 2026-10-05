"""Registered operational validation/controller audits and activity-count learning."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.activity_counts import candidates, count_bins
from spectrum_occupancy_forecasting.experiments.adaptive_reporting import verify_records
from spectrum_occupancy_forecasting.experiments.fine_spectrum import sha, write_json
from spectrum_occupancy_forecasting.experiments.operational_audits import (
    controller_audit,
    controller_report,
    validation_audit,
)
from spectrum_occupancy_forecasting.experiments.operational_rehearsal import synthetic_sources
from spectrum_occupancy_forecasting.experiments.operational_reporting import (
    adaptation_figure,
    audit_figures,
    numerical_suite,
    regenerate,
)
from spectrum_occupancy_forecasting.experiments.operational_selection import (
    STRATEGIES,
    ContinuousSelector,
    count_replay,
)
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state
from spectrum_occupancy_forecasting.experiments.registry import (
    Registry,
    digest,
    implementation_digest,
    migrate,
)
from spectrum_occupancy_forecasting.experiments.replay_workflow import run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "adaptive_runner", ROOT / "scripts/run_adaptive_learning.py"
)
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


def configuration(path):
    declared = {
        "study": "operational-learning",
        "base_config": "adaptive_learning.yaml",
        "protocols": ["historical-cold-start", "continuous-operational"],
        "controllers": ["selected-hard", "fixed-hard", "fixed-soft"],
        "count_half_lives_h": [0.5, 2, 6],
        "count_standard_deviations": [0.5, 1],
        "count_dispersions": [0, 0.1],
        "count_strengths": [0.5, 1],
        "count_bin_s": 300,
        "count_coverage": 0.8,
        "minimum_count_initialization": 48,
        "grid_nodes": 801,
        "grid_limit": 16,
        "numerical_tolerance": 0.001,
        "count_fallback": "fixed-hard",
        "count_cache_budget_bytes": 10000000,
        "stopping_rule": "stop-after-declared-grid",
    }
    settings = yaml.safe_load(Path(path).read_text())
    if settings != declared:
        raise ValueError("configuration/study mismatch or declared grid changed")
    base = adaptive.configuration(Path(path).parent / settings["base_config"])
    state = yaml.safe_load((ROOT / "configs/residual_state_space.yaml").read_text())
    return base | state | settings


def scientific_files():
    # Explicit dependency closure. Presentation and registry-management code are excluded.
    folders = ["features", "occupancy", "models", "robustness", "data"]
    package = ROOT / "src/spectrum_occupancy_forecasting"
    files = [path for folder in folders for path in (package / folder).rglob("*.py")]
    names = [
        "activity_counts",
        "operational_selection",
        "operational_audits",
        "adaptive_replay",
        "adaptive_models",
        "residual_state_space",
        "followup_data",
        "followup_models",
        "learning_inputs",
        "mixture_support",
        "replay_workflow",
        "followup",
    ]
    return (
        files
        + [package / "experiments" / f"{name}.py" for name in names]
        + [
            ROOT / "scripts/run_adaptive_learning.py",
            ROOT / "uv.lock",
            package / "experiments/adaptive_reporting.py",
            package / "experiments/followup_reporting.py",
            package / "experiments/operational_reporting.py",
        ]
    )


def scientific_implementation():
    return implementation_digest(
        scientific_files(),
        ROOT,
        {
            "run_adaptive_learning.py": [
                "configuration",
                "synthetic_frame",
                "load_nodes",
                "scoring_grid",
            ],
            "adaptive_reporting.py": ["verify_records", "regenerate", "event_report"],
            "operational_reporting.py": ["numerical_suite", "regenerate"],
        },
    )


def clean_json(value):
    # Pandas missing numeric values are explicit nulls, never invalid JSON NaN.
    return (
        json.loads(pd.io.json.dumps(value))
        if hasattr(pd.io.json, "dumps")
        else json.loads(
            json.dumps(value, default=lambda x: x.item() if hasattr(x, "item") else str(x)).replace(
                "NaN", "null"
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation-audit", "controller-audit", "count-model"])
    parser.add_argument("--mode", choices=["synthetic-rehearsal", "exploratory", "report"])
    parser.add_argument(
        "--registry-command", choices=["list", "compare", "explain", "recover", "migrate"]
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "results/experiment_registry.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/operational_learning.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "reproductions/shared_learning_cache")
    parser.add_argument("--parent-attempt")
    parser.add_argument("--hypothesis")
    parser.add_argument(
        "--attempt-kind",
        choices=[
            "model",
            "protocol-audit",
            "controller-change",
            "parameter-revision",
            "seed-replication",
            "implementation-maintenance",
        ],
    )
    parser.add_argument("--evidence")
    parser.add_argument("--reproduction")
    parser.add_argument("--attempt")
    parser.add_argument("--recovery-reason")
    parser.add_argument("--recover-lock", action="store_true")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--numerical-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    registry = Registry(args.registry)
    if args.registry_command:
        if args.registry_command == "recover":
            print(registry.recover(args.recovery_reason, args.attempt, args.recover_lock))
        elif args.registry_command == "migrate":
            migrate(registry, ROOT)
        elif args.registry_command == "explain":
            print(
                json.dumps(
                    [r for r in registry.read() if r.get("attempt") == args.attempt], indent=2
                )
            )
        else:
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=False)
                registry.export(args.out_dir)
            print(
                json.dumps(
                    [
                        {
                            k: r.get(k)
                            for k in ("attempt", "hypothesis", "status", "kind", "source_reference")
                        }
                        for r in registry.read()
                        if r["event"] in ("declaration", "historical-import")
                    ],
                    indent=2,
                )
            )
            print("stale attempts:", [r["attempt"] for r in registry.stale()])
        return
    if not args.phase or not args.mode or not args.out_dir:
        parser.error("execution requires --phase, --mode and fresh --out-dir")
    cfg = configuration(args.config)
    if args.out_dir.exists():
        raise ValueError("output already exists; accidental overwrite refused")
    if args.mode == "report" and not args.predictions_dir:
        parser.error("report requires --predictions-dir")
    if args.mode == "exploratory" and not args.hypothesis:
        parser.error("real-data execution requires declared --hypothesis")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "configuration.json", cfg)
    manifest = {
        "study": "operational-learning",
        "phase": args.phase,
        "mode": args.mode,
        "status": "started",
        "CC2_accessed": False,
        "all_results_exploratory": True,
        "git": git_state(ROOT),
        "python": sys.version,
        "implementation_hash": scientific_implementation(),
        "source_byte_hash": fingerprint(scientific_files(), root=ROOT),
        "configuration_hash": digest(cfg),
    }
    attempt = None
    write_json(args.out_dir / "manifest.json", manifest)
    try:
        with threadpool_limits(2):
            if args.mode == "report":
                if args.phase == "count-model":
                    source_cfg = json.loads(
                        (args.predictions_dir / "configuration.json").read_text()
                    )
                    if source_cfg["study"] != cfg["study"]:
                        raise ValueError("report study mismatch")
                    regenerate(args.predictions_dir, args.out_dir, source_cfg)
                    adaptation_figure(
                        pd.read_parquet(args.predictions_dir / "predictions.parquet"),
                        pd.read_parquet(args.predictions_dir / "filter_states.parquet"),
                        args.out_dir,
                    )
                    audit_root = ROOT / "results/operational_learning/audits"
                    if audit_root.exists():
                        audit_figures(
                            pd.read_csv(audit_root / "validation/protocol_comparison.csv"),
                            pd.read_csv(audit_root / "controller/controller_comparison.csv"),
                            args.out_dir,
                        )
                elif args.phase == "controller-audit":
                    controller_report(
                        pd.read_parquet(args.predictions_dir / "predictions.parquet"), args.out_dir
                    )
                else:
                    # The audit itself consumes only saved prediction records.
                    validation_audit(None, cfg, args.out_dir, ROOT, None)
            elif args.mode == "synthetic-rehearsal":
                if args.phase == "count-model":
                    manifest["selected_grid_nodes"] = numerical_suite(cfg, args.out_dir)
                else:
                    nodes = adaptive.load_nodes(None, cfg, synthetic=True)
                    cfg = cfg | {"synthetic": True}
                    write_json(args.out_dir / "configuration.json", cfg)
                    fixtures = args.out_dir / "synthetic_sources"
                    synthetic_sources(
                        nodes,
                        cfg,
                        fixtures,
                        adaptive.scoring_grid,
                        studies=["adaptive_learning"] if args.phase == "controller-audit" else None,
                    )
                    if args.phase == "validation-audit":
                        validation_audit(None, cfg, args.out_dir, fixtures, None)
                    else:
                        controller_audit(
                            nodes,
                            cfg,
                            args.out_dir,
                            adaptive.scoring_grid,
                            fixtures / "results/adaptive_learning/run/selections.json",
                        )
            else:
                nodes = adaptive.load_nodes(args.cache, cfg)
                origins = {r: adaptive.scoring_grid(n, r == "LW1_later") for r, n in nodes.items()}
                if {r: len(g) for r, g in origins.items()} != {
                    "CC1_first": 439,
                    "LW1_first": 239,
                    "LW1_later": 81,
                }:
                    raise ValueError("protected scored-origin population changed")
                input_hashes = {
                    p.relative_to(args.cache).as_posix(): sha(p)
                    for p in sorted(args.cache.rglob("*.parquet"))
                    if p.parent.name in {v[1] for v in adaptive.RECORDINGS.values()}
                }
                source_hashes = {
                    p.relative_to(ROOT).as_posix(): sha(p)
                    for study in ("adaptive_learning", "residual_state_space")
                    for p in (ROOT / "results" / study / "run").glob("*")
                    if p.is_file()
                }
                specification = {
                    "phase": args.phase,
                    "resolved_settings": cfg,
                    "candidate_settings": [asdict(c) for c in candidates(cfg)],
                    "input_hashes": input_hashes,
                    "scored_origin_hashes": {
                        r: hashlib.sha256(g[["origin_s", "fold"]].to_numpy().tobytes()).hexdigest()
                        for r, g in origins.items()
                    },
                    "references": ["fixed_two_hour", "independently_selected_history"],
                    "initialization": "clean_first_day_only",
                    "features": "delivered_counts_only",
                    "expected_mechanism": {
                        "validation-audit": "continuous states match deployment; test cold-start mismatch",
                        "controller-audit": "isolate fallback and quality shrinkage from aggregation",
                        "count-model": "learn persistence from counts before target feedback",
                    }[args.phase],
                    "model_family": args.phase,
                    "protocol": "continuous-operational"
                    if args.phase == "count-model"
                    else cfg["protocols"],
                    "inputs": sorted(input_hashes.values()),
                    "parameters": [asdict(c) for c in candidates(cfg)]
                    if args.phase == "count-model"
                    else cfg,
                    "stopping_rule": "stop-after-declared-grid",
                }
                if args.phase == "validation-audit":
                    from spectrum_occupancy_forecasting.experiments.adaptive_models import (
                        candidates as adaptive_grid,
                    )
                    from spectrum_occupancy_forecasting.experiments.residual_state_space import (
                        candidates as state_grid,
                    )

                    specification["candidate_settings"] = {
                        "adaptive": [asdict(c) for c in adaptive_grid(cfg)],
                        "state_space": [asdict(c) for c in state_grid(cfg)],
                        "history_control_shared": True,
                    }
                    specification["features"] = "existing history/history_fine and residual inputs"
                elif args.phase == "controller-audit":
                    specification["candidate_settings"] = {
                        "controllers": cfg["controllers"],
                        "methods": ["no-correction", "aggregation_h6_rate0.5_strength1"],
                    }
                    specification["features"] = "existing history and quality gate"
                if args.phase == "count-model":
                    if args.numerical_dir is None:
                        raise ValueError("count execution requires passing --numerical-dir")
                    numerical_cfg = json.loads(
                        (args.numerical_dir / "configuration.json").read_text()
                    )
                    gate = json.loads((args.numerical_dir / "numerical_gate.json").read_text())
                    if numerical_cfg != cfg or not gate["passes"]:
                        raise ValueError("numerical gate/specification mismatch")
                    numerical_manifest = json.loads(
                        (args.numerical_dir / "manifest.json").read_text()
                    )
                    if numerical_manifest["status"] != "completed":
                        raise ValueError("numerical rehearsal incomplete")
                    for name, expected_hash in numerical_manifest["artifact_hashes"].items():
                        if sha(args.numerical_dir / name) != expected_hash:
                            raise ValueError("corrupted numerical rehearsal")
                    convergence = pd.read_csv(args.numerical_dir / "numerical_convergence.csv")
                    chosen_grid = convergence[convergence.low_nodes == gate["selected_grid_nodes"]]
                    if (
                        len(chosen_grid) != 72
                        or chosen_grid.maximum_forecast_difference.max()
                        > cfg["numerical_tolerance"]
                    ):
                        raise ValueError("numerical convergence evidence missing or non-passing")
                    cfg["grid_nodes"] = gate["selected_grid_nodes"]
                    specification["resolved_settings"] = cfg
                    write_json(args.out_dir / "configuration.json", cfg)
                attempt = registry.declare(
                    specification,
                    manifest["implementation_hash"],
                    hypothesis=args.hypothesis,
                    kind=args.attempt_kind
                    or {
                        "validation-audit": "protocol-audit",
                        "controller-audit": "controller-change",
                        "count-model": "model",
                    }[args.phase],
                    parent=args.parent_attempt,
                    evidence=args.evidence,
                    reproduction=args.reproduction,
                )
                registry.append({"event": "status", "attempt": attempt, "status": "running"})
                manifest.update(
                    attempt=attempt, input_hashes=input_hashes, source_evidence_hashes=source_hashes
                )
                write_json(args.out_dir / "manifest.json", manifest)
                with zipfile.ZipFile(
                    args.out_dir / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
                ) as bundle:
                    snapshot_files = scientific_files() + [
                        Path(__file__),
                        args.config,
                        ROOT / "configs/adaptive_learning.yaml",
                        ROOT / "configs/followup_v2.yaml",
                        ROOT / "configs/residual_state_space.yaml",
                        ROOT / "src/spectrum_occupancy_forecasting/experiments/registry.py",
                    ]
                    for path in dict.fromkeys(snapshot_files):
                        bundle.write(path, path.relative_to(ROOT).as_posix())
                if args.phase == "validation-audit":
                    verify_records(ROOT / "results/adaptive_learning/run", cfg)
                    validation_audit(nodes, cfg, args.out_dir, ROOT, adaptive.scoring_grid)
                elif args.phase == "controller-audit":
                    controller_audit(
                        nodes,
                        cfg,
                        args.out_dir,
                        adaptive.scoring_grid,
                        ROOT / "results/adaptive_learning/run/selections.json",
                    )
                else:
                    for recording, node in nodes.items():
                        bins = count_bins(node, cfg)
                        bins.to_parquet(
                            args.out_dir / f"{recording}_count_bins.parquet", index=False
                        )
                    if (
                        sum(p.stat().st_size for p in args.out_dir.glob("*_count_bins.parquet"))
                        > cfg["count_cache_budget_bytes"]
                    ):
                        raise ValueError("derived count cache exceeds 10 MB")
                    run(
                        nodes,
                        cfg,
                        args.out_dir,
                        False,
                        selection_fn=ContinuousSelector(),
                        replay_fn=count_replay,
                        strategies=STRATEGIES,
                        candidate_factory=candidates,
                        report_fn=regenerate,
                        scoring_grid=adaptive.scoring_grid,
                    )
                numerical_outcomes = {}
                for name in (
                    "advancement.json",
                    "selections.json",
                    "candidate_scores.json",
                    "failures.json",
                    "audit.json",
                ):
                    path = args.out_dir / name
                    if path.exists():
                        numerical_outcomes[name] = json.loads(path.read_text())
                for name in ("protocol_comparison.csv", "controller_comparison.csv", "summary.csv"):
                    path = args.out_dir / name
                    if path.exists():
                        numerical_outcomes[name] = clean_json(pd.read_csv(path).to_dict("records"))
                registry.append(
                    {
                        "event": "findings",
                        "attempt": attempt,
                        "findings": numerical_outcomes,
                        "artifact_hashes": {
                            p.name: sha(p)
                            for p in args.out_dir.iterdir()
                            if p.is_file() and p.name != "manifest.json"
                        },
                        "reproduction_of": args.reproduction,
                    }
                )
                if args.reproduction:
                    original = next(
                        r
                        for r in registry.read()
                        if r["event"] == "findings" and r["attempt"] == args.reproduction
                    )
                    agreement = digest(original["findings"]) == digest(numerical_outcomes)
                    registry.append(
                        {
                            "event": "reproduction-check",
                            "attempt": attempt,
                            "original": args.reproduction,
                            "agrees": agreement,
                            "comparison": "all numerical outcomes, exact canonical JSON",
                        }
                    )
                    if not agreement:
                        raise ValueError("reproduction numerical outcomes disagree")
                registry.append({"event": "status", "attempt": attempt, "status": "completed"})
            manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(
            status="failed" if not isinstance(error, KeyboardInterrupt) else "interrupted",
            error=str(error),
        )
        if attempt:
            registry.append(
                {
                    "event": "status",
                    "attempt": attempt,
                    "status": manifest["status"],
                    "error": str(error),
                }
            )
        raise
    finally:
        manifest["artifact_hashes"] = {
            p.name: sha(p)
            for p in args.out_dir.iterdir()
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(args.out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
