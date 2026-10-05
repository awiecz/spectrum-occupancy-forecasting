"""Sparse-regime study: power first, then matched-training forecasts and intervals.

    uv run python scripts/run_sparse_regime.py --phase power --out-dir results/sparse_regime/power

The power phase reads saved predictions only. It fits nothing, reads no recording and never
touches CC2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.data.synthetic import write_synthetic_blocks
from spectrum_occupancy_forecasting.experiments import sparse_regime as sr
from spectrum_occupancy_forecasting.experiments import sparse_reporting as rep
from spectrum_occupancy_forecasting.experiments.protocol import fingerprint, git_state
from spectrum_occupancy_forecasting.experiments.registry import Registry, implementation_digest
from spectrum_occupancy_forecasting.experiments.sparse_power import (
    LEARNED,
    decide,
    power_tables,
)
from spectrum_occupancy_forecasting.viz.sparse_figures import make_figures

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/followup_v2/exploratory/predictions.parquet"
SCENARIOS = (
    "clean",
    "dropout_0.5",
    "dropout_0.9",
    "dropout_0.99",
    "outage_3600s",
    "delay_3600s",
    "delay_7200s",
)
PRIMARY_SCENARIO = "dropout_0.99"
REPLICATES, SEED = 2000, 0


def noise_ratios():
    """Clean-input candidate-to-baseline noise ratios from the multi-band planning summary."""
    summary = json.loads((ROOT / "results/multiband/dependence_summary.json").read_text())
    return {
        row["site"]: {
            level: row[f"single_band_se_{level}"]
            for level in ("optimistic", "typical", "aggressive")
        }
        for row in summary
    }


def power_figure(baselines, spread, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    squared = baselines[
        (baselines["loss"] == "squared") & (baselines["scenario"] == PRIMARY_SCENARIO)
    ]
    squared = squared.assign(cell=squared["node"] + "\n" + squared["region"].str.replace("_", " "))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    names = [
        ("fixed_history", "Two-hour history", "#9ec5e3"),
        ("climatology", "Always the average", "#D55E00"),
        ("oracle_shrunk", "History shrunk to average", "#009E73"),
    ]
    width = 0.26
    x = range(len(squared))
    for k, (column, label, color) in enumerate(names):
        ax.bar(
            [i + (k - 1) * width for i in x],
            squared[f"{column}_loss"],
            width,
            color=color,
            label=label,
        )
    ax.set_xticks(list(x), squared["cell"])
    ax.set(
        ylabel="Squared error (lower is better)",
        title="At 99% dropout, simple fixes beat plain history",
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    cells = spread[(spread["scenario"] == PRIMARY_SCENARIO) & (spread["loss"] == "squared")]
    labels = (cells["node"] + "\n" + cells["region"].str.replace("_", " ")).tolist()
    ax.bar(
        [i - 0.2 for i in range(len(cells))],
        cells["mde_typical"] * 100,
        0.38,
        color="#9ec5e3",
        label="Original proxy (noisy unmatched models)",
    )
    ax.bar(
        [i + 0.2 for i in range(len(cells))],
        cells["scaled_mde_typical"] * 100,
        0.38,
        color="#0072B2",
        label="Amended proxy (baseline noise x clean-input ratio)",
    )
    ax.axhline(20, color="0.4", lw=1)
    ax.axhline(50, color="0.4", lw=1, ls="--")
    ax.text(len(cells) - 0.5, 22, "20% minimum margin", fontsize=8, color="0.2", ha="right")
    ax.text(
        len(cells) - 0.5, 52, "above 50%: claim not possible", fontsize=8, color="0.2", ha="right"
    )
    ax.set_xticks(range(len(cells)), labels)
    ax.set(
        ylabel="Typical smallest detectable gain (%)",
        title="Detectable gain against the strongest simple baseline",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_power(args):
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; choose a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(args.predictions)
    ratios = noise_ratios()
    baselines, evidence, spread = power_tables(
        predictions, SCENARIOS, args.replicates, args.seed, ratios
    )
    original_cells, original_abort = decide(spread, PRIMARY_SCENARIO, proxy="unmatched")
    cells, abort = decide(spread, PRIMARY_SCENARIO, proxy="scaled")
    baselines.to_csv(out / "baseline_diagnostic.csv", index=False)
    evidence.to_csv(out / "cell_evidence.csv", index=False)
    spread.to_csv(out / "minimum_detectable_gain.csv", index=False)
    (out / "margin.json").write_text(
        json.dumps(
            dict(
                primary_scenario=PRIMARY_SCENARIO,
                original_rule=dict(
                    proxy="noise of the six unmatched exploratory learned families",
                    cells=original_cells,
                    abort_study=original_abort,
                ),
                amended_rule=dict(
                    proxy="baseline relative noise x clean-input candidate-to-baseline noise ratio",
                    noise_ratios=ratios,
                    cells=cells,
                    abort_study=abort,
                ),
                rule="delta = max(0.20, typical MDE rounded up to the next 0.05); "
                "cells with typical MDE above 0.50 are descriptive; if every primary cell is "
                "descriptive the study runs as exploratory only",
                learned_families=list(LEARNED),
                note="Computed from already-inspected exploratory predictions; no model was fitted.",
            ),
            indent=2,
        )
    )
    power_figure(baselines, spread, out / "power.png")
    (out / "settings.json").write_text(
        json.dumps(
            dict(
                predictions=str(args.predictions.relative_to(ROOT)),
                scenarios=list(SCENARIOS),
                replicates=args.replicates,
                seed=args.seed,
                block_windows=32,
            ),
            indent=2,
        )
    )
    print(f"wrote {out}; original_abort={original_abort}; amended_abort={abort}")


CONFIG = ROOT / "configs/sparse_regime.yaml"
PROTOCOL = ROOT / "docs/technical/sparse_regime_protocol.md"
POWER_DIR = ROOT / "results/sparse_regime/power"
REGISTRY = ROOT / "results/experiment_registry.jsonl"
INPUTS = {
    "CC1": ROOT / "data/processed/aerpaw_CC1_blocks5MHz.npz",
    "LW1": ROOT / "data/processed/aerpaw_LW1_blocks5MHz.npz",
}


def timestamp():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=float, allow_nan=False))


def source_files():
    package = ROOT / "src/spectrum_occupancy_forecasting"
    files = [
        path
        for folder in ("features", "occupancy", "models", "robustness", "data")
        for path in (package / folder).rglob("*.py")
    ]
    names = ["followup_data", "followup_models", "sparse_regime", "sparse_reporting", "intervals"]
    return (
        files
        + [package / "experiments" / f"{name}.py" for name in names]
        + [package / "evaluation/quantile_metrics.py", package / "evaluation/power.py"]
        + [Path(__file__).resolve(), ROOT / "uv.lock"]
    )


def declared_margin():
    """delta = the largest per-cell margin among the primary cells of the power step."""
    margin = json.loads((POWER_DIR / "margin.json").read_text())
    cells = [c for c in margin["amended_rule"]["cells"] if c["primary"]]
    return float(max(c["delta"] for c in cells)), margin


def build_report(out, cfg, delta, synthetic=False):
    """Rebuild every table, the outcome label and the figures from saved predictions."""
    predictions = pd.read_parquet(out / "predictions.parquet")
    interval_path = out / "intervals.parquet"
    intervals = pd.read_parquet(interval_path) if interval_path.exists() else None
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    cells = rep.cell_table(predictions, cfg, cfg["bootstrap"]["resamples"], rng)
    methods = rep.method_losses(predictions, cfg)
    primary = cfg["primary_scenario"]
    descriptors = [
        rep.robustness(predictions, cfg, node, region, primary)
        for node, region in (("CC1", "all_folds"), ("LW1", "later_folds"), ("LW1", "all_folds"))
        if ((predictions["node"] == node) & (predictions["scenario"] == primary)).any()
    ]
    outcome = rep.decide(cells, descriptors, delta, cfg)
    cells.to_csv(out / "gains.csv", index=False)
    methods.to_csv(out / "method_losses.csv", index=False)
    pd.DataFrame(descriptors).to_csv(out / "robustness.csv", index=False)
    interval_summary = None
    if intervals is not None and len(intervals):
        interval_summary = rep.interval_table(intervals, cfg)
        interval_summary.to_csv(out / "intervals_summary.csv", index=False)
    write_json(out / "summary.json", {"outcome": outcome, "delta": delta, "synthetic": synthetic})
    make_figures(methods, cells, interval_summary, primary, delta, out / "figures", synthetic)
    return outcome


def run_matched(args):
    cfg = sr.load_config(args.config)
    synthetic = args.mode == "synthetic-rehearsal"
    out = args.out_dir.resolve()
    if args.mode == "report":
        was_synthetic = (
            json.loads((out / "manifest.json").read_text())["mode"] == "synthetic-rehearsal"
        )
        delta = 0.20 if was_synthetic else declared_margin()[0]
        with threadpool_limits(limits=1):
            outcome = build_report(out, cfg, delta, synthetic=was_synthetic)
        manifest = json.loads((out / "manifest.json").read_text())
        manifest.setdefault("report_rebuilds", []).append(
            {
                "at": timestamp(),
                "git": git_state(ROOT),
                "source_fingerprint": fingerprint(
                    [ROOT / "src", Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock"],
                    root=ROOT,
                ),
            }
        )
        manifest["artifact_hashes"] = {
            p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"
        }
        write_json(out / "manifest.json", manifest)
        print(json.dumps(outcome, indent=2))
        return
    if out.exists() and not (args.resume and any(out.iterdir())):
        raise SystemExit(f"{out} already exists; choose a new --out-dir or pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    registry = None
    if synthetic:
        path = out / "aerpaw_SYN_blocks5MHz.npz"
        write_synthetic_blocks(path, n_days=3.1, dt_s=60.0, seed=3)
        inputs, scenarios, delta = {"SYN": path}, ["dropout_0.99", "delay_3600s"], 0.20
        cfg = dict(
            cfg,
            interval_scenarios=["dropout_0.99"],
            train_seeds=cfg["train_seeds"][:1],
            validation_seeds=cfg["validation_seeds"][:1],
            score_seeds=cfg["score_seeds"][:2],
        )
    else:
        if not args.hypothesis:
            raise SystemExit("a real run must be declared: pass --hypothesis")
        inputs, scenarios = dict(INPUTS), None
        delta, _ = declared_margin()
        registry = Registry(args.registry)
    source = fingerprint(
        [ROOT / "src", Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock"], root=ROOT
    )
    manifest = {
        "study": "sparse-regime",
        "mode": args.mode,
        "status": "started",
        "started": timestamp(),
        "git": git_state(ROOT),
        "source_fingerprint": source,
        "config_sha256": sha(args.config),
        "protocol_sha256": sha(PROTOCOL) if PROTOCOL.exists() else None,
        "delta": delta,
        "CC2_accessed": False,
        "inputs": {name: {"file": p.name, "sha256": sha(p)} for name, p in inputs.items()},
    }
    (out / "config.yaml").write_bytes(Path(args.config).read_bytes())
    write_json(out / "manifest.json", manifest)
    attempt = None
    try:
        if registry is not None:
            specification = {
                "phase": "sparse-regime-matched",
                "model_family": "matched ridge, boosting, residual boosting and quantile boosting",
                "inputs": {n: v["sha256"] for n, v in manifest["inputs"].items()},
                "protocol": {"protocol_sha256": manifest["protocol_sha256"], "delta": delta},
                "parameters": json.loads(json.dumps(cfg, default=float)),
            }
            attempt = registry.declare(
                specification,
                implementation_digest(source_files(), ROOT),
                hypothesis=args.hypothesis,
                kind=args.attempt_kind,
                parent=args.parent_attempt,
                evidence=args.evidence,
            )
            registry.append({"event": "status", "attempt": attempt, "status": "running"})
            manifest["attempt"] = attempt
            write_json(out / "manifest.json", manifest)
        parts = out / "_parts"
        records = []
        with threadpool_limits(limits=1):
            for node, path in inputs.items():
                raw = band_table_from_blocks(path, cfg["channels_mhz"], site=node, session=0)
                result = sr.run_node(raw, cfg, parts, scenarios=scenarios, resume=args.resume)
                manifest.setdefault("calibration", {})[node] = result["calibration"]
                records += result["folds"]
                write_json(out / "manifest.json", manifest)
            forecasts = sorted(p for p in parts.glob("*.parquet") if "__intervals" not in p.stem)
            interval_parts = sorted(parts.glob("*__intervals.parquet"))
            pd.concat([pd.read_parquet(p) for p in forecasts], ignore_index=True).to_parquet(
                out / "predictions.parquet", index=False
            )
            if interval_parts:
                pd.concat(
                    [pd.read_parquet(p) for p in interval_parts], ignore_index=True
                ).to_parquet(out / "intervals.parquet", index=False)
            write_json(out / "selections.json", records)
            outcome = build_report(out, cfg, delta, synthetic=synthetic)
        manifest["status"] = "completed"
        manifest["outcome"] = outcome
        if registry is not None:
            registry.append(
                {
                    "event": "findings",
                    "attempt": attempt,
                    "findings": json.loads(json.dumps(outcome, default=float)),
                    "artifact_hashes": {
                        p.name: sha(p)
                        for p in out.iterdir()
                        if p.is_file() and p.name != "manifest.json"
                    },
                }
            )
            registry.append({"event": "status", "attempt": attempt, "status": "completed"})
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        if attempt:
            registry.append(
                {"event": "status", "attempt": attempt, "status": "failed", "error": str(error)}
            )
        raise
    finally:
        manifest["artifact_hashes"] = {
            p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"
        }
        write_json(out / "manifest.json", manifest)
    print(f"label: {outcome['label']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["power", "matched"], required=True)
    parser.add_argument(
        "--mode",
        choices=["synthetic-rehearsal", "exploratory", "report"],
        default="exploratory",
        help="matched phase only",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--hypothesis")
    parser.add_argument("--attempt-kind", default="model")
    parser.add_argument("--parent-attempt")
    parser.add_argument("--evidence")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.phase == "power":
        run_power(args)
    else:
        run_matched(args)


if __name__ == "__main__":
    main()
