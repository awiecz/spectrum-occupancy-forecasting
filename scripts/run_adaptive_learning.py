"""Audit or replay the bounded adaptive study; never reads raw archives or CC2."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.adaptive_audit import audit
from spectrum_occupancy_forecasting.experiments.adaptive_models import candidates
from spectrum_occupancy_forecasting.experiments.adaptive_replay import STRATEGIES, replay, select
from spectrum_occupancy_forecasting.experiments.adaptive_reporting import regenerate
from spectrum_occupancy_forecasting.experiments.fine_spectrum import FINE_COLUMNS, sha, write_json
from spectrum_occupancy_forecasting.experiments.followup import load_config, scenarios
from spectrum_occupancy_forecasting.experiments.followup_data import folds
from spectrum_occupancy_forecasting.experiments.learning_inputs import LearningNode, cached_frame
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = {
    "CC1_first": ("CC1", "CC1_session0_development"),
    "LW1_first": ("LW1", "LW1_session0_development"),
    "LW1_later": ("LW1", "LW1_session2_assessment"),
}


def configuration(path):
    settings = yaml.safe_load(Path(path).read_text())
    if settings["study"] != "adaptive-learning":
        raise ValueError("configuration/study mismatch")
    declared = {
        "initialization_s": 86400,
        "replay_stride_s": 900,
        "validation_hours": 12,
        "selection_embargo_s": 36000,
        "validation_margin": 0.02,
        "bias_half_lives_h": [0.5, 2, 6, 12],
        "aggregation_half_lives_h": [2, 6, 12, 24],
        "aggregation_rates": [0.5, 2, 8],
        "ridge_half_lives_h": [2, 6, 12, 24],
        "ridge_penalties": [1, 10, 100, 1000],
        "feature_sets": ["history", "history_fine"],
        "correction_strengths": [0.25, 0.5, 1],
        "correction_cap": 0.1,
        "maximum_observation_age_s": 900,
        "minimum_predictor_coverage": 0.5,
        "feedback_coverage": 0.8,
        "feedback_window_s": 7200,
        "minimum_feedback_windows": 4,
        "maximum_feedback_age_s": 3600,
        "minimum_clean_skill": 0.05,
        "maximum_degraded_ratio": 1.1,
        "minimum_positive_fold_fraction": 0.5,
    }
    if any(settings[key] != value for key, value in declared.items()):
        raise ValueError("declared scientific settings changed")
    base = load_config(Path(path).parent / settings["base_config"])
    base["channels_mhz"] = [2437.0]
    candidates(settings)
    return base | settings


def synthetic_frame(name, days, seed):
    rng = np.random.default_rng(seed)
    times = 1_600_000_000.0 + np.arange(0, days * 86400, 60)
    probability = 0.10 + 0.15 * (np.sin(np.arange(len(times)) / 300) > 0)
    power = -100 + rng.normal(0, 0.25, len(times)) + 8 * (rng.random(len(times)) < probability)
    return pd.DataFrame(
        {
            "time_s": times,
            "target_dbm": power,
            **{column: power + rng.normal(0, 0.1, len(times)) for column in FINE_COLUMNS},
        }
    )


def load_nodes(cache, cfg, synthetic=False):
    return {
        recording: LearningNode(
            synthetic_frame(site, days, i) if synthetic else cached_frame(cache, folder), site, cfg
        )
        for i, ((recording, (site, folder)), days) in enumerate(
            zip(RECORDINGS.items(), (5, 4.75, 44.64 / 24), strict=True)
        )
    }


def scoring_grid(node, later=False):
    groups = []
    ranges = (
        [(node.data.t0 + 86400, node.data.last)]
        if later
        else folds(node.data.t0, node.data.last, node.cfg)
    )
    for fold, (begin, end) in enumerate(ranges):
        origins = node.data.origins(begin, end, 900, 900, common=not later)
        origins, y, count = node.data.dataset(origins, 900)
        groups.append(
            pd.DataFrame(
                {
                    "origin_s": origins,
                    "y": y,
                    "target_count": count,
                    "fold": fold,
                    "fold_begin_s": begin,
                }
            )
        )
    return pd.concat(groups, ignore_index=True)


def run(nodes, cfg, out, synthetic):
    selections, scores, validation, failures, all_predictions, all_feedback, snapshots = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    first_selections = {}
    for recording, node in nodes.items():
        grid = scoring_grid(node, recording == "LW1_later")
        fold_choices = {}
        for fold, group in grid.groupby("fold"):
            begin = float(group.fold_begin_s.iloc[0])
            if recording == "LW1_later":
                # Startup settings come entirely from earlier LW1 first-session validation.
                chosen, history, rows, val, failure = first_selections["LW1_first"]
            else:
                chosen, history, rows, val, failure = select(node, cfg, begin)
                first_selections[recording] = (chosen, history, rows, val, failure)
            fold_choices[int(fold)] = (chosen, history)
            selection = {
                "recording": recording,
                "fold": int(fold),
                "origin_s": begin,
                "chosen": chosen,
                "selected_history": history,
                "selection_source": "LW1_first_final_fold"
                if recording == "LW1_later"
                else recording,
                "failure": failure,
            }
            selections.append(selection)
            scores.extend(dict(row, recording=recording, fold=int(fold)) for row in rows)
            if recording != "LW1_later":
                validation.extend(dict(row, recording=recording, fold=int(fold)) for row in val)
            if failure:
                failures.append(selection)
            print(f"{recording}: selected fold {fold}", flush=True)
        origins = node.data.origins(
            node.data.t0 + 86400, node.data.last, 900, cfg["replay_stride_s"]
        )
        # Selection switches at fold boundaries; unscored forecasts continue to receive feedback.
        selected_histories = {}
        for origin in origins:
            eligible = [
                f for f, g in grid.groupby("fold") if float(g.fold_begin_s.iloc[0]) <= origin
            ]
            selected_histories[float(origin)] = (
                fold_choices[max(eligible)][1] if eligible else "recent_7200s"
            )
        for scenario, deg in scenarios(cfg).items():
            seeds = cfg["mask_seeds"] if (deg.dropout_p or deg.outage_len_s) else [0]
            for seed in seeds:
                records, feedback, bank = replay(
                    node,
                    cfg,
                    origins,
                    deg=deg,
                    seed=seed,
                    selected_history=selected_histories,
                    active=None
                    if scenario == "clean"
                    else {candidates(cfg)[0].id}
                    | {
                        candidate
                        for chosen, _ in fold_choices.values()
                        for candidate in chosen.values()
                    },
                )
                joined = records.merge(grid, on="origin_s", validate="many_to_one")
                baseline = candidates(cfg)[0].id
                views = []
                for strategy in STRATEGIES:
                    selected_rows = []
                    for fold, rows in joined.groupby("fold"):
                        chosen = fold_choices[int(fold)][0][strategy]
                        part = rows[rows.candidate == chosen].copy()
                        part["method"] = "adaptive"
                        selected_rows.append(part)
                        controller = rows[rows.candidate == baseline].copy()
                        controller["method"] = "controller"
                        selected_rows.append(controller)
                        for name, column in (
                            ("fixed_history", "fixed_history"),
                            ("selected_history", "selected_history"),
                        ):
                            reference = controller.copy()
                            reference["prediction"] = reference[column]
                            reference["method"] = name
                            selected_rows.append(reference)
                    view = pd.concat(selected_rows, ignore_index=True)
                    view["strategy"] = strategy
                    views.append(view)
                predictions = pd.concat(views, ignore_index=True)
                for frame in (predictions, feedback, bank.feature_snapshots):
                    frame["recording"], frame["scenario"], frame["seed"] = recording, scenario, seed
                predictions["target_start_s"] = predictions.origin_s + 900
                predictions["target_end_s"] = predictions.origin_s + 1800
                predictions["threshold_dbm"] = node.data.thresholds[2437.0]
                predictions["synthetic"] = synthetic
                predictions["period"] = np.where(
                    predictions.fallback_reason.fillna("") != "", "fallback", "adaptation"
                )
                predictions.loc[predictions.origin_s < node.data.t0 + 86400 + 7200, "period"] = (
                    "startup"
                )
                all_predictions.append(predictions)
                all_feedback.append(feedback)
                snapshots.append(bank.feature_snapshots)
                failures.extend(
                    dict(row, recording=recording, scenario=scenario, seed=seed)
                    for row in bank.failures
                )
                if scenario == "clean":
                    records.to_parquet(
                        out / f"{recording}_candidate_forecasts.parquet", index=False
                    )
                print(f"{recording}: {scenario} seed {seed} replay complete", flush=True)
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        out / "predictions.parquet", index=False
    )
    pd.concat(all_feedback, ignore_index=True).to_parquet(out / "feedback.parquet", index=False)
    pd.concat(snapshots, ignore_index=True).to_parquet(
        out / "feature_snapshots.parquet", index=False
    )
    pd.DataFrame(validation).to_parquet(out / "validation_predictions.parquet", index=False)
    write_json(out / "selections.json", selections)
    write_json(out / "candidate_scores.json", scores)
    write_json(out / "failures.json", failures)
    regenerate(out, out, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["audit", "synthetic-rehearsal", "exploratory", "report"], required=True
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/adaptive_learning.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "reproductions/shared_learning_cache")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = configuration(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "configuration.json", cfg)
    source_files = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__),
        args.config,
        ROOT / "configs/followup_v2.yaml",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    with zipfile.ZipFile(
        args.out_dir / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for path in source_files:
            bundle.write(path, path.resolve().relative_to(ROOT).as_posix())
    manifest = {
        "study": "adaptive-learning",
        "mode": args.mode,
        "started": datetime.now(UTC).isoformat(),
        "status": "started",
        "source_hash": fingerprint(
            [
                ROOT / "src",
                Path(__file__),
                args.config,
                ROOT / "configs/followup_v2.yaml",
                ROOT / "pyproject.toml",
                ROOT / "uv.lock",
            ],
            root=ROOT,
        ),
        "config_hash": sha(args.config),
        "portable_config_hash": fingerprint(
            [args.config, ROOT / "configs/followup_v2.yaml"], root=ROOT
        ),
        "runtime": {
            "python": sys.version.split()[0],
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")
            },
        },
        "git": git_state(ROOT),
        "candidates": [asdict(c) | {"id": c.id} for c in candidates(cfg)],
        "CC2_accessed": False,
        "all_recordings_are_development": True,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=2):
            if args.mode == "report":
                if args.predictions_dir is None:
                    raise ValueError("report requires --predictions-dir")
                frozen_cfg = json.loads((args.predictions_dir / "configuration.json").read_text())
                regenerate(args.predictions_dir, args.out_dir, frozen_cfg)
                manifest["original_manifest_hash"] = sha(args.predictions_dir / "manifest.json")
            else:
                nodes = load_nodes(args.cache, cfg, args.mode == "synthetic-rehearsal")
                manifest["inputs"] = {
                    recording: {
                        "sweeps": len(node.frame),
                        "start": node.data.t0,
                        "end": node.data.last,
                        "floor_dbm": node.data.floors[2437.0],
                        "interval_s": node.data.dt,
                        "cache_manifest_hash": None
                        if args.mode == "synthetic-rehearsal"
                        else sha(args.cache / RECORDINGS[recording][1] / "manifest.json"),
                    }
                    for recording, node in nodes.items()
                }
                if args.mode == "audit":
                    audit(ROOT, nodes, args.out_dir)
                    write_json(args.out_dir / "input_audit.json", manifest["inputs"])
                else:
                    run(nodes, cfg, args.out_dir, args.mode == "synthetic-rehearsal")
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"], manifest["failure"] = "failed", repr(error)
        raise
    finally:
        manifest["finished"] = datetime.now(UTC).isoformat()
        manifest["artifacts"] = {
            p.name: sha(p)
            for p in sorted(args.out_dir.iterdir())
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(args.out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
