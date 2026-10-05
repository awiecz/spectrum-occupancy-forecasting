"""Extended declared learning study; later LW1 activity is gated by a local frozen package."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import SweepSource, freq_axis, sessions
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments.fine_spectrum import (
    FINE_COLUMNS,
    extract,
    sha,
    write_json,
)
from spectrum_occupancy_forecasting.experiments.followup import scenarios
from spectrum_occupancy_forecasting.experiments.followup_data import folds, maximum_lookback
from spectrum_occupancy_forecasting.experiments.followup_models import (
    history_prediction,
    operating_threshold,
    training,
)
from spectrum_occupancy_forecasting.experiments.history_mixture import HistoryMixture
from spectrum_occupancy_forecasting.experiments.mixture_support import select as mixture_select
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state
from spectrum_occupancy_forecasting.experiments.shared_learning import (
    FIXED,
    LearningNode,
    Study,
    configuration,
    profiles,
)
from spectrum_occupancy_forecasting.experiments.shared_reporting import (
    chosen_report,
    compare_profiles,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    ROOT / "src",
    ROOT / "scripts/run_shared_learning.py",
    ROOT / "scripts/extract_learning_features.py",
    ROOT / "configs/history_mixture.yaml",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached_frame(cache, node):
    folder = cache / f"{node}_session0_development"
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("feature extraction is incomplete")
    for name, digest in manifest["shards"].items():
        if sha(folder / name) != digest:
            raise ValueError("corrupted feature cache")
    return pd.concat(
        [pd.read_parquet(folder / name) for name in sorted(manifest["shards"])], ignore_index=True
    )


def synthetic_frame(out, node, days, seed):
    path = out / f"aerpaw_SYN_{node}.npz"
    write_synthetic_blocks(path, n_days=days, dt_s=60, seed=seed)
    with np.load(path) as z:
        f, p, t = z["centres_mhz"], z["mean_dbm"].astype(float), z["times_s"]
        target = 10 * np.log10((10 ** (p[:, np.abs(f - 2437) <= 10] / 10)).mean(axis=1))
        rng = np.random.default_rng(seed)
        values = target[:, None] + rng.normal(0, 0.3, (len(t), len(FINE_COLUMNS)))
    path.unlink()
    return pd.DataFrame(
        {
            "time_s": t,
            "target_dbm": target,
            **{name: values[:, i] for i, name in enumerate(FINE_COLUMNS)},
        }
    )


def load_nodes(args, cfg, synthetic=False):
    frames = (
        {
            name: synthetic_frame(args.out_dir, name, days, seed)
            for name, days, seed in (("CC1", 7.0, 61), ("LW1", 4.75, 62))
        }
        if synthetic
        else {name: cached_frame(args.cache, name) for name in ("CC1", "LW1")}
    )
    return {name: LearningNode(frame, name, cfg) for name, frame in frames.items()}


def prediction_rows(
    node,
    origins,
    y,
    counts,
    bundle,
    choice,
    history,
    profile,
    fold,
    scenario="clean",
    deg=None,
    seed=0,
):
    learned, base, correction, reasons = bundle.details(node, origins, choice, deg, seed)
    X = node.data.features(origins, deg, seed)
    selected = history_prediction(history, X, bundle.means[bundle.node])
    meta = {
        "node": bundle.node,
        "fold": fold,
        "task": "primary",
        "variant": profile["variant"],
        "profile_id": profile["id"],
        "lead_s": 900,
        "offset_db": 3,
        "scenario": scenario,
        "seed": seed,
        "origin_s": origins,
        "target_start_s": origins + 900,
        "target_end_s": origins + 1800,
        "target_count": counts,
        "y": y,
        "training_mean": bundle.means[bundle.node],
        "label_threshold_dbm": node.data.thresholds[2437.0],
        "correction_strength": choice["strength"],
        "selected_penalty": choice.get("alpha"),
        "shared_weight": choice["shared"],
        "candidate_id": choice["id"],
        "fit_cutoff_s": bundle.cutoff,
        "history_prediction": base,
    }
    frames = []
    for method, values in (
        ("fixed_history", base),
        ("selected_history", selected),
        ("selected_learned", learned),
    ):
        frames.append(
            pd.DataFrame(
                meta
                | {
                    "method": method,
                    "prediction": values,
                    "correction_value": correction if method == "selected_learned" else 0.0,
                    "fallback_reason": reasons if method == "selected_learned" else "reference",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def attach_mixture(frame, reference):
    source = reference[
        (reference.strategy_id == "36h_mixture_expanding")
        & (reference.method == "selected_learned")
        & (reference.task == "primary")
    ]
    target = frame[frame.method == "selected_learned"]
    keys = ["node", "fold", "origin_s", "scenario", "seed"]
    fields = [
        "prediction",
        "y",
        "candidate_id",
        "training_mean",
        "history_prediction",
        "correction_value",
        "fallback_reason",
        "correction_strength",
        "selected_penalty",
    ]
    joined = target.merge(
        source[keys + fields],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_old"),
    )
    if joined.prediction_old.isna().any() or not np.array_equal(joined.y, joined.y_old):
        raise ValueError("retained mixture origins/labels do not exactly match")
    for column in fields:
        if column != "y":
            joined[column] = joined[column + "_old"]
    joined["shared_weight"] = np.nan
    joined.method = "history_mixture"
    joined = joined.drop(columns=[column + "_old" for column in fields])
    return pd.concat([frame, joined], ignore_index=True)


def run_development(args, cfg, manifest):
    synthetic = args.mode == "synthetic-rehearsal"
    cfg["synthetic"] = synthetic
    nodes = load_nodes(args, cfg, synthetic)
    study = Study(nodes, cfg)
    parts = args.out_dir / "parts"
    parts.mkdir()
    records, prediction_files = [], []
    reference = (
        None
        if synthetic
        else pd.read_parquet(ROOT / "results/followup_v2/history_mixture/predictions.parquet")
    )
    if not synthetic:
        manifest["inputs"] = {
            name: {
                "cache_manifest_sha256": sha(
                    args.cache / f"{name}_session0_development/manifest.json"
                ),
                "archive_sha256": read_json(
                    args.cache / f"{name}_session0_development/manifest.json"
                )["archive_sha256"],
            }
            for name in nodes
        }
        manifest["reference_predictions_sha256"] = sha(
            ROOT / "results/followup_v2/history_mixture/predictions.parquet"
        )
    for name, node in nodes.items():
        for fold, (lo, hi) in enumerate(folds(node.data.t0, node.data.last, cfg)):
            o, y, count = node.data.dataset(node.data.origins(lo, hi, 900, 900, common=True), 900)
            for profile in profiles(cfg):
                print(f"{name} fold {fold + 1}: {profile['id']}", flush=True)
                choice, history, scores, validation, status = study.select(name, lo, profile)
                bundle = study.bundle(name, lo - maximum_lookback(cfg), profile)
                frame = prediction_rows(node, o, y, count, bundle, choice, history, profile, fold)
                if reference is not None:
                    frame = attach_mixture(frame, reference)
                prefix = parts / f"{name}_{fold}_{profile['id']}"
                frame.to_parquet(prefix.with_suffix(".parquet"), index=False)
                if len(validation):
                    validation.assign(node=name, fold=fold, profile_id=profile["id"]).to_parquet(
                        prefix.with_name(prefix.name + "_validation.parquet"), index=False
                    )
                rec = {
                    "node": name,
                    "fold": fold,
                    "task": "primary",
                    "profile_id": profile["id"],
                    "profile": profile,
                    "choice": choice,
                    "history": history,
                    "scores": scores,
                    "selection_status": status,
                    "status": "completed",
                    "operating_thresholds": [],
                    "evaluation_start_s": lo,
                    "evaluation_end_s": hi,
                }
                records.append(rec)
                write_json(parts / f"{prefix.name}.json", rec)
                prediction_files.append(prefix.with_suffix(".parquet"))
    predictions = pd.concat([pd.read_parquet(p) for p in prediction_files], ignore_index=True)
    predictions.to_parquet(args.out_dir / "development_predictions.parquet", index=False)
    chosen = compare_profiles(predictions, records, cfg, args.out_dir)
    profile = next(p for p in profiles(cfg) if p["id"] == chosen)
    chosen_records = [r for r in records if r["profile_id"] == chosen]
    final = []
    for rec in chosen_records:
        name, fold = rec["node"], rec["fold"]
        node = nodes[name]
        o, y, count = node.data.dataset(
            node.data.origins(
                rec["evaluation_start_s"], rec["evaluation_end_s"], 900, 900, common=True
            ),
            900,
        )
        bundle = study.bundle(name, rec["evaluation_start_s"] - maximum_lookback(cfg), profile)
        for scenario, deg in scenarios(cfg).items():
            seeds = cfg["mask_seeds"] if deg.dropout_p or deg.outage_frac else [0]
            for seed in seeds:
                frame = prediction_rows(
                    node,
                    o,
                    y,
                    count,
                    bundle,
                    rec["choice"],
                    rec["history"],
                    profile,
                    fold,
                    scenario,
                    deg,
                    seed,
                )
                final.append(attach_mixture(frame, reference) if reference is not None else frame)
        validation_path = parts / f"{name}_{fold}_{chosen}_validation.parquet"
        if validation_path.exists():
            v = pd.read_parquet(validation_path)
            ids = {
                "fixed_history": FIXED["id"],
                "selected_history": rec["history"]["id"],
                "selected_learned": rec["choice"]["id"],
            }
            for method, identity in ids.items():
                selected = v[v.candidate_id == identity]
                for budget in cfg["false_busy_budgets"]:
                    rec["operating_thresholds"].append(
                        {
                            "method": method,
                            **operating_threshold(selected.y, selected.prediction, budget, cfg),
                        }
                    )
        same_fold = [r for r in records if r["node"] == name and r["fold"] == fold]
        rec["common_fold"] = all(r["selection_status"] == "selected" for r in same_fold)
    final = pd.concat(final, ignore_index=True)
    final.to_parquet(args.out_dir / "predictions.parquet", index=False)
    write_json(args.out_dir / "selections.json", chosen_records)
    write_json(args.out_dir / "development_selections.json", records)
    write_json(args.out_dir / "config.json", cfg)
    chosen_report(final, chosen_records, cfg, args.out_dir)
    manifest["chosen_profile"] = profile
    manifest["calibrations"] = {
        name: {"floor": n.data.floors[2437.0], "dt": n.data.dt} for name, n in nodes.items()
    }


def verify_run(folder):
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("source run incomplete")
    for name, digest in manifest["artifact_hashes"].items():
        if sha(folder / name) != digest:
            raise ValueError("source run checksum mismatch")
    return manifest


def run_freeze(args, cfg, manifest):
    original = verify_run(args.source_run)
    if original["mode"] != "exploratory":
        raise ValueError("only a reviewed real-data development run can be frozen")
    if original["config_fingerprint"] != manifest["config_fingerprint"]:
        raise ValueError("configuration changed after development")
    for name, input_record in original["inputs"].items():
        if input_record["cache_manifest_sha256"] != sha(
            args.cache / f"{name}_session0_development/manifest.json"
        ):
            raise ValueError("development cache changed")
    if original["source_fingerprint"] != fingerprint(SOURCE_PATHS, root=ROOT):
        raise ValueError("source changed since development; disclose and verify before freezing")
    profile = original["chosen_profile"]
    nodes = load_nodes(args, cfg)
    study, models, validation_records = Study(nodes, cfg), {}, []
    final_cutoff = max(node.data.grid[-1] for node in nodes.values())
    for name, node in nodes.items():
        last = node.data.grid[-1]
        artificial_origin = last + maximum_lookback(cfg)
        choice, history, scores, validation, status = study.select(name, artificial_origin, profile)
        if status != "selected":
            raise ValueError("final selection lacks eligible earlier validation")
        # At the later assessment, both first sessions are already historical evidence.
        bundle = study.bundle(name, final_cutoff, profile)
        if choice["strength"]:
            for scope in (
                ["local"]
                if choice["shared"] == 0
                else ["shared"]
                if choice["shared"] == 1
                else ["local", "shared"]
            ):
                bundle.estimator(choice, scope)
        mixture_cfg = cfg | yaml_mixture_settings()
        old_cfg = node.data.cfg
        node.data.cfg = mixture_cfg
        h, c, _, _, selected = mixture_select(
            node.data, artificial_origin, 900, 36, "mixture_expanding", model_class=HistoryMixture
        )
        if selected != "selected":
            raise ValueError("retained mixture final selection unavailable")
        X, y = training(node.data, last, 900)
        mixture = HistoryMixture(c, h, mixture_cfg, node.data.dt).fit(X, y)
        node.data.cfg = old_cfg
        models[name] = {"bundle": bundle, "choice": choice, "history": history, "mixture": mixture}
        validation.to_parquet(args.out_dir / f"{name}_final_validation.parquet", index=False)
        validation_records.append(
            {
                "node": name,
                "choice": choice,
                "history": history,
                "scores": scores,
                "fit_cutoff_s": final_cutoff,
                "parameter_selection_end_s": last,
            }
        )
    joblib.dump(models, args.out_dir / "models.joblib")
    write_json(args.out_dir / "final_selection.json", validation_records)
    write_json(args.out_dir / "config.json", cfg)
    manifest["chosen_profile"] = profile
    manifest["development_manifest_sha256"] = sha(args.source_run / "manifest.json")
    with zipfile.ZipFile(args.out_dir / "source_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in [
            *ROOT.glob("src/**/*.py"),
            ROOT / "scripts/run_shared_learning.py",
            ROOT / "scripts/extract_learning_features.py",
            ROOT / "configs/shared_learning.yaml",
            ROOT / "configs/followup_v2.yaml",
            ROOT / "configs/history_mixture.yaml",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]:
            z.write(path, path.relative_to(ROOT))


def yaml_mixture_settings():
    import yaml

    return yaml.safe_load((ROOT / "configs/history_mixture.yaml").read_text(encoding="utf-8"))


def run_assessment(args, cfg, manifest):
    frozen = verify_run(args.freeze)
    if frozen["mode"] != "freeze" or frozen["source_fingerprint"] != fingerprint(
        SOURCE_PATHS, root=ROOT
    ):
        raise ValueError("frozen package/source mismatch")
    if frozen["config_fingerprint"] != fingerprint(
        [args.config, ROOT / "configs/followup_v2.yaml"], root=ROOT
    ):
        raise ValueError("frozen configuration mismatch")
    if frozen["runtime_versions"] != manifest["runtime_versions"]:
        raise ValueError("frozen runtime version mismatch")
    lock = args.cache / "later_LW1_assessment.lock.json"
    args.cache.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {"status": "started", "freeze_sha256": sha(args.freeze / "manifest.json")}, stream
        )
    try:
        models = joblib.load(args.freeze / "models.joblib")
        frame = extract(args.lw1, "LW1", 2, args.cache, "assessment", frozen=True)
        node = LearningNode(frame, "LW1", cfg)
        frozen_model = models["LW1"]
        profile = frozen["chosen_profile"]
        start = node.data.t0 + 86400
        o, y, counts = node.data.dataset(node.data.origins(start, node.data.last, 900, 900), 900)
        if not len(y):
            raise ValueError("no eligible later-session targets; report insufficient outcome")
        predictions = prediction_rows(
            node,
            o,
            y,
            counts,
            frozen_model["bundle"],
            frozen_model["choice"],
            frozen_model["history"],
            profile,
            0,
        )
        old = frozen_model["mixture"]
        old.dt = node.data.dt  # Availability rule uses the new session's frozen calibration.
        mixture = predictions[predictions.method == "selected_learned"].copy()
        for key, value in old.details(node.data.features(o)).items():
            mixture[key] = value
        mixture["candidate_id"] = old.choice["id"]
        mixture["correction_strength"] = old.choice["strength"]
        mixture["selected_penalty"] = old.choice.get("alpha")
        mixture["training_mean"] = old.mean
        mixture["shared_weight"] = np.nan
        mixture["method"] = "history_mixture"
        predictions = pd.concat([predictions, mixture], ignore_index=True)
        predictions.to_parquet(args.out_dir / "predictions.parquet", index=False)
        rec = {
            "node": "LW1",
            "fold": 0,
            "task": "primary",
            "common_fold": True,
            "status": "completed",
            "scores": [],
            "operating_thresholds": [],
        }
        write_json(args.out_dir / "selections.json", [rec])
        write_json(args.out_dir / "config.json", cfg)
        chosen_report(predictions, [rec], cfg, args.out_dir)
        manifest.update(
            calibration_floor=node.data.floors[2437.0],
            nominal_interval_s=node.data.dt,
            targets=len(y),
            target_degenerate=bool(np.var(y) == 0),
            interpretation="reserved later-session assessment, same campaign; not independent registration",
        )
        write_json(
            lock,
            {
                "status": "completed",
                "out_name": args.out_dir.name,
                "freeze_sha256": sha(args.freeze / "manifest.json"),
            },
        )
    except Exception as exc:
        write_json(lock, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def audit_metadata(args, manifest):
    records = []
    for name, path in (("CC1", args.cc1), ("LW1", args.lw1)):
        source = SweepSource(path)
        ids = sessions(source.times)
        for sid in np.unique(ids):
            times = source.times[ids == sid]
            metadata = source.meta(float(times[0]))
            axis = freq_axis(metadata)
            records.append(
                {
                    "node": name,
                    "session": int(sid),
                    "start_s": float(times[0]),
                    "end_s": float(times[-1]),
                    "sweeps": len(times),
                    "axis_sha256": __import__("hashlib")
                    .sha256(axis.freqs_mhz.tobytes())
                    .hexdigest(),
                    "datatype": metadata.get("global", {}).get("core:datatype"),
                    "receiver": metadata.get("global", {}).get("core:hw"),
                    "metadata_site": metadata.get("global", {}).get("dataset:site"),
                    "median_interval_s": float(np.median(np.diff(times)))
                    if len(times) > 1
                    else None,
                }
            )
        source._zip.close()
    write_json(args.out_dir / "session_inventory.json", records)
    for name, path in (("CC1", args.cc1), ("LW1", args.lw1)):
        extract(path, name, 0, args.cache)
    calibration = extract(args.lw1, "LW1", 2, args.cache, "calibration")
    write_json(
        args.out_dir / "later_calibration.json",
        {
            "sweeps": len(calibration),
            "floor_dbm": float(np.quantile(calibration.target_dbm, 0.05)),
            "nominal_interval_s": float(np.median(np.diff(calibration.time_s))),
            "activity_after_calibration_accessed": False,
        },
    )
    manifest["inventory"] = records
    nodes = {
        name: LearningNode(cached_frame(args.cache, name), name, configuration(args.config))
        for name in ("CC1", "LW1")
    }
    reference = pd.read_parquet(ROOT / "results/followup_v2/history_mixture/predictions.parquet")
    selected = reference[
        (reference.strategy_id == "36h_mixture_expanding")
        & (reference.scenario == "clean")
        & (reference.method == "selected_learned")
    ]
    diagnostics = []
    for name, rows in selected.groupby("node"):
        node = nodes[name]
        origins, labels, counts = node.data.dataset(rows.origin_s.to_numpy(), 900)
        if not np.array_equal(origins, rows.origin_s.to_numpy()) or not np.array_equal(
            labels, rows.y.to_numpy()
        ):
            raise ValueError(f"{name}: extracted targets differ from retained predictions")
        if not np.array_equal(counts, rows.target_count.to_numpy()):
            raise ValueError(f"{name}: target observation counts differ")
        X = node.data.features(rows.origin_s.to_numpy())
        delta = X.frac_300s.to_numpy() - X.frac_7200s.to_numpy()
        frame = rows.copy()
        frame["past_change"] = np.where(
            delta > 0.05, "rising", np.where(delta < -0.05, "falling", "small")
        )
        frame["future_level"] = pd.cut(
            frame.y, [-1e-9, 0.05, 0.2, 0.5, 1.0], labels=["low", "moderate", "high", "very_high"]
        )
        frame["age_group"] = np.where(X.age_s.to_numpy() <= 60, "at_most_60s", "older")
        frame["availability"] = np.where(
            X.n_900s.to_numpy() / (900 / node.data.dt) >= 0.8, "at_least_80pct", "below_80pct"
        )
        frame["loss"] = (frame.prediction - frame.y) ** 2
        for column in ("fold", "past_change", "future_level", "age_group", "availability"):
            stats = (
                frame.groupby(column, observed=True).loss.agg(["size", "mean", "sum"]).reset_index()
            )
            for row in stats.to_dict("records"):
                diagnostics.append(
                    {
                        "node": name,
                        "group_type": column,
                        "group": str(row[column]),
                        "n": row["size"],
                        "mse": row["mean"],
                        "squared_loss": row["sum"],
                        "interpretation": "retrospective"
                        if column == "future_level"
                        else "past-feature diagnosis",
                    }
                )
    pd.DataFrame(diagnostics).to_csv(
        args.out_dir / "error_diagnosis.csv", index=False, lineterminator="\n"
    )
    clean = reference[
        (reference.strategy_id == "36h_mixture_expanding")
        & reference.common_fold
        & (reference.scenario == "clean")
    ]
    benchmark = (
        clean.assign(loss=(clean.prediction - clean.y) ** 2).groupby(["node", "method"]).loss.mean()
    )
    benchmark.reset_index(name="mse").to_csv(
        args.out_dir / "retained_benchmark.csv", index=False, lineterminator="\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["audit", "exploratory", "synthetic-rehearsal", "report", "freeze", "assessment"],
        required=True,
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs/shared_learning.yaml")
    ap.add_argument("--cache", type=Path, default=ROOT / "reproductions/shared_learning_cache")
    ap.add_argument(
        "--cc1", type=Path, default=ROOT / "data/raw/aerpaw/ResultsCC1Feb2022_SigMF.zip"
    )
    ap.add_argument(
        "--lw1", type=Path, default=ROOT / "data/raw/aerpaw/ResultsLW1Feb2022_SigMF.zip"
    )
    ap.add_argument("--source-run", type=Path)
    ap.add_argument("--freeze", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = configuration(args.config)
    if (
        args.cc1.name != "ResultsCC1Feb2022_SigMF.zip"
        or args.lw1.name != "ResultsLW1Feb2022_SigMF.zip"
    ):
        raise ValueError("only CC1/LW1 archives permitted; CC2 prohibited")
    if args.mode in ("report", "freeze") and args.source_run is None:
        raise ValueError("--source-run required")
    if args.mode == "assessment" and args.freeze is None:
        raise ValueError("--freeze required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": args.mode,
        "status": "running",
        "git": git_state(ROOT),
        "source_fingerprint": fingerprint(SOURCE_PATHS, root=ROOT),
        "config_fingerprint": fingerprint(
            [args.config, ROOT / "configs/followup_v2.yaml"], root=ROOT
        ),
        "CC2_accessed": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime_versions": {
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
        },
    }
    try:
        with threadpool_limits(limits=2):
            if args.mode == "audit":
                audit_metadata(args, manifest)
            elif args.mode in ("exploratory", "synthetic-rehearsal"):
                run_development(args, cfg, manifest)
            elif args.mode == "freeze":
                run_freeze(args, cfg, manifest)
            elif args.mode == "assessment":
                run_assessment(args, cfg, manifest)
            else:
                original = verify_run(args.source_run)
                p = pd.read_parquet(args.source_run / "predictions.parquet")
                rec = read_json(args.source_run / "selections.json")
                chosen_report(p, rec, read_json(args.source_run / "config.json"), args.out_dir)
                manifest["original_manifest_sha256"] = sha(args.source_run / "manifest.json")
                manifest["original_mode"] = original["mode"]
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(UTC).isoformat()
        manifest["artifact_hashes"] = {
            p.relative_to(args.out_dir).as_posix(): sha(p)
            for p in sorted(args.out_dir.rglob("*"))
            if p.is_file()
        }
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
