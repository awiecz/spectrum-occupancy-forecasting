"""Staged v2 experiments, keeping historic code and output artifacts unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spectrum_occupancy_forecasting.experiments.followup_data import NodeData, folds
from spectrum_occupancy_forecasting.experiments.followup_models import (
    augmented_validation,
    fit_selected,
    operating_threshold,
    select_models,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["version"] != 2 or cfg["target_mhz"] != 2437.0:
        raise ValueError("v2 must keep its version and preselected target band")
    step = cfg["training_stride_s"]
    times = [
        cfg["window_s"],
        cfg["calibration_s"],
        cfg["first_eval_s"],
        cfg["fold_s"],
        cfg["validation_block_s"],
        *cfg["history_windows_s"],
        *cfg["leads_s"],
        *[t * cfg["decay_support_taus"] for t in cfg["decay_taus_s"]],
    ]
    # Look-back windows need not be aligned, but all origin boundaries must be.
    aligned = times[:5] + cfg["leads_s"] + [max(times[-len(cfg["decay_taus_s"]) :])]
    if step <= 0 or any(t % step for t in aligned):
        raise ValueError("origin boundaries must align to the training grid")
    if (
        cfg["primary_lead_s"] not in cfg["leads_s"]
        or cfg["feature_tau_s"] not in cfg["decay_taus_s"]
    ):
        raise ValueError("primary lead and feature decay must be configured candidates")
    if cfg["validation_blocks"] != 2 or len(cfg["augmentation_seeds"]) != 5:
        raise ValueError("v2 requires two validation blocks and five augmentation seeds")
    if not 0 < cfg["minimum_coverage"] <= 1 or cfg["bootstrap_resamples"] < 1:
        raise ValueError("invalid coverage or bootstrap count")
    return cfg


def tasks(cfg):
    lead, offset = cfg["primary_lead_s"], cfg["primary_offset_db"]
    out = [("primary", lead, offset, "full")]
    out += [(f"lead_{h}s", h, offset, "full") for h in cfg["leads_s"] if h != lead]
    out += [(f"label_{x}db", lead, x, "full") for x in cfg["offsets_db"] if x != offset]
    out += [(f"ablation_{v}", lead, offset, v) for v in cfg["ablations"] if v != "full"]
    return out


def scenarios(cfg):
    out = {"clean": Degradation()}
    out.update({f"dropout_{p:g}": Degradation(dropout_p=p) for p in cfg["dropout_probabilities"]})
    out.update(
        {
            f"outage_{s}s": Degradation(outage_len_s=s, outage_frac=cfg["outage_fraction"])
            for s in cfg["outage_lengths_s"]
        }
    )
    out.update({f"delay_{s}s": Degradation(delay_s=s) for s in cfg["delays_s"]})
    return out


def run_node(raw, cfg, out, *, task_names=None, progress=print):
    """Write per-fold predictions incrementally; return calibration and completion records."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nodes = {}
    calibrations, completed = [], []
    task_defs = [t for t in tasks(cfg) if task_names is None or t[0] in task_names]
    for task, lead, offset, variant in task_defs:
        if offset not in nodes:
            nodes[offset] = NodeData(raw, cfg, offset)
            n = nodes[offset]
            calibrations.append(
                {
                    "node": n.node,
                    "offset_db": offset,
                    "noise_floors_dbm": n.floors,
                    "nominal_interval_s": n.dt,
                    "calibration_end_s": n.t0 + cfg["calibration_s"],
                }
            )
        node = nodes[offset]
        for fold, (lo, hi) in enumerate(folds(node.t0, node.last, cfg)):
            progress(
                f"{node.node}: {task}, fold {fold + 1}, selecting earlier-only models", flush=True
            )
            prefix = out / f"{node.node}__{task}__{fold}"
            if prefix.with_suffix(".parquet").exists():
                raise FileExistsError(prefix)
            selected, candidate_scores, validation = select_models(node, lo, lead, variant)
            models = fit_selected(node, lo, lead, variant, selected)
            if task == "primary":
                validation.update(augmented_validation(node, lo, lead, variant, selected))
                augmented = fit_selected(
                    node, lo, lead, variant, selected, augmented=True, names=("ridge", "hgb")
                )
                models.update({name + "_aug": model for name, model in augmented.items()})
            thresholds = []
            for method, frame in validation.items():
                for budget in cfg["false_busy_budgets"]:
                    thresholds.append(
                        {
                            "method": method,
                            **operating_threshold(frame["y"], frame["prediction"], budget, cfg),
                        }
                    )
            record = {
                "node": node.node,
                "task": task,
                "fold": fold,
                "lead_s": lead,
                "offset_db": offset,
                "variant": variant,
                "selected": selected,
                "candidates": candidate_scores,
                "operating_thresholds": thresholds,
                "evaluation_start_s": lo,
                "evaluation_end_s": hi,
                "target_degenerate": False,
            }
            validation_rows = [
                f.assign(method=name, node=node.node, task=task, fold=fold)
                for name, f in validation.items()
            ]
            pd.concat(validation_rows, ignore_index=True).to_parquet(
                prefix.with_name(prefix.name + "__validation.parquet"), index=False
            )
            origins = node.origins(lo, hi, lead, cfg["window_s"], common=True)
            o, y, count = node.dataset(origins, lead)
            if not len(y):
                record["status"] = "no_eligible_evaluation_targets"
                prefix.with_suffix(".json").write_text(json.dumps(record, indent=2))
                completed.append(record)
                continue
            record["target_degenerate"] = bool(np.var(y) == 0)
            rows = []
            defs = scenarios(cfg) if task == "primary" else {"clean": Degradation()}
            for scenario, deg in defs.items():
                stochastic = deg.dropout_p > 0 or deg.outage_frac > 0
                seeds = cfg["mask_seeds"] if stochastic else [0]
                for seed in seeds:
                    X = node.features(o, deg, seed)
                    for method, model in models.items():
                        p = np.full(len(y), model) if isinstance(model, float) else model.predict(X)
                        choice = selected.get(method, selected.get(method.removesuffix("_aug")))
                        rows.append(
                            pd.DataFrame(
                                {
                                    "node": node.node,
                                    "fold": fold,
                                    "task": task,
                                    "method": method,
                                    "candidate_id": choice["id"],
                                    "variant": variant,
                                    "scenario": scenario,
                                    "seed": seed,
                                    "origin_s": o,
                                    "target_start_s": o + lead,
                                    "target_end_s": o + lead + cfg["window_s"],
                                    "lead_s": lead,
                                    "offset_db": offset,
                                    "label_threshold_dbm": node.thresholds[cfg["target_mhz"]],
                                    "target_count": count,
                                    "y": y,
                                    "prediction": p,
                                    "training_mean": models["climatology"],
                                    "dropout_p": deg.dropout_p,
                                    "delay_s": deg.delay_s,
                                    "outage_s": deg.outage_len_s,
                                }
                            )
                        )
            frame = pd.concat(rows, ignore_index=True)
            frame.to_parquet(prefix.with_suffix(".parquet"), index=False)
            record["status"], record["n_evaluation"] = "completed", len(y)
            prefix.with_suffix(".json").write_text(json.dumps(record, indent=2))
            completed.append(record)
    return {
        "calibrations": calibrations,
        "folds": completed,
        "status": "completed" if completed else "no_eligible_folds",
    }
