"""Matched-training forecasts when monitoring is sparse, stale or missing.

Earlier studies fitted and selected models on complete input and scored them under simulated
degradation. Here training, validation and scoring all see the *same kind* of degraded input,
through independent random masks, and the strongest simple baseline is selected the same way.
Existing modules are reused unchanged; nothing here reads CC2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spectrum_occupancy_forecasting.experiments.followup_data import (
    NodeData,
    folds,
    maximum_lookback,
)
from spectrum_occupancy_forecasting.experiments.followup_models import (
    ForecastModel,
    candidates,
)
from spectrum_occupancy_forecasting.experiments.intervals import (
    conformal_quantiles,
    empirical_quantiles,
    fit_quantile_models,
    predict_quantiles,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation

ROOT = Path(__file__).resolve().parents[3]
GRID_KEYS = (
    "history_windows_s",
    "decay_taus_s",
    "decay_support_taus",
    "ridge_alphas",
    "boosting_depths",
    "boosting_leaf_sizes",
    "boosting_iterations",
    "boosting_learning_rate",
    "residual_boosting_iterations",
    "residual_boosting_learning_rate",
    "model_seed",
    "training_stride_s",
    "validation_block_s",
    "validation_blocks",
    "window_s",
    "calibration_s",
    "first_eval_s",
    "fold_s",
    "minimum_partial_fold_s",
    "minimum_coverage",
)
FAMILIES = ("ridge", "hgb", "resid_hgb")
SIMPLE = ("fixed_history", "selected_history", "shrunk_history")
ABLATIONS = ("history", "history_power", "history_age", "history_calendar", "history_cross")


def load_config(path, base_path=None):
    """Read the sparse-regime configuration and fail if its declared grid differs from v2."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base_path = Path(base_path or Path(path).with_name(cfg["base_config"]))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    declared = cfg.get("expected_grid", {})
    if set(declared) != set(GRID_KEYS):
        raise ValueError("expected_grid must list exactly the frozen grid keys")
    for key in GRID_KEYS:
        if declared[key] != base[key]:
            raise ValueError(f"declared grid differs from {base_path.name}: {key}")
    if cfg["primary_scenario"] not in cfg["scenarios"]:
        raise ValueError("primary scenario must be a declared scenario")
    merged = base | {k: v for k, v in cfg.items() if k != "expected_grid"}
    merged["target_mhz"] = base["target_mhz"]
    return merged


def scenario_definitions(cfg):
    return {name: Degradation(**spec) for name, spec in cfg["scenarios"].items()}


def is_stochastic(deg):
    return deg.dropout_p > 0 or deg.outage_frac > 0


def effective_seeds(deg, seeds):
    """Deterministic degradations ignore the seed; use a single copy rather than duplicates."""
    return tuple(seeds) if is_stochastic(deg) else (0,)


def matched_training(node, cutoff, lead, deg, seeds):
    """Training rows whose targets finish by ``cutoff``, with inputs degraded like the scenario."""
    origins = node.origins(node.t0, cutoff, lead, node.cfg["training_stride_s"])
    o, y, _ = node.dataset(origins, lead)
    if not len(y):
        raise ValueError("no eligible training targets")
    assert np.max(o + lead + node.cfg["window_s"]) <= cutoff + 1e-6
    X = pd.concat([node.features(o, deg, s) for s in seeds], ignore_index=True)
    return X, np.tile(y, len(seeds))


@dataclass
class Block:
    Xtr: pd.DataFrame
    ytr: np.ndarray
    Xv: list
    y: np.ndarray
    origins: np.ndarray


def validation_blocks(node, lo, lead, deg, cfg):
    """Two chronological validation blocks before the embargoed training cutoff."""
    lookback = maximum_lookback(cfg)
    start = lo - lookback - cfg["validation_blocks"] * cfg["validation_block_s"]
    if start < node.t0 + cfg["calibration_s"]:
        raise ValueError("validation starts before initial calibration finishes")
    train_seeds = effective_seeds(deg, cfg["train_seeds"])
    validation_seeds = effective_seeds(deg, cfg["validation_seeds"])
    blocks = []
    for i in range(cfg["validation_blocks"]):
        block_lo = start + i * cfg["validation_block_s"]
        block_hi = block_lo + cfg["validation_block_s"]
        Xtr, ytr = matched_training(node, block_lo - lookback, lead, deg, train_seeds)
        o, y, _ = node.dataset(
            node.origins(block_lo, block_hi, lead, cfg["window_s"], common=True), lead
        )
        if not len(y):
            raise ValueError("no eligible validation targets")
        blocks.append(Block(Xtr, ytr, [node.features(o, deg, s) for s in validation_seeds], y, o))
    return blocks


def validation_rows(blocks, predict):
    """Stack per-seed validation predictions; ``predict(number, X)`` returns an array."""
    frames = []
    for number, block in enumerate(blocks):
        for k, X in enumerate(block.Xv):
            frames.append(
                pd.DataFrame(
                    {
                        "block": number,
                        "seed_index": k,
                        "origin_s": block.origins,
                        "y": block.y,
                        "prediction": predict(number, X),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def shrinkage_weight(y, history, mean):
    """Least-squares weight of history over the training mean, clipped to [0, 1]."""
    centred = np.asarray(history) - np.asarray(mean)
    denominator = float(np.sum(centred**2))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.sum(centred * (np.asarray(y) - mean)) / denominator, 0.0, 1.0))


def mse(frame):
    return float(np.mean((frame["prediction"] - frame["y"]) ** 2))


def matched_selection(node, lo, lead, variant, deg, cfg):
    """Select the strongest simple baseline and each learned family on matched validation.

    History candidates are scored first; residual models are conditioned on the selected
    history, as in v2. Shrunk history has one parameter fitted on the same validation blocks,
    which, if anything, favours the baseline.
    """
    blocks = validation_blocks(node, lo, lead, deg, cfg)
    pool = candidates(cfg)
    records, frames, fitted = [], {}, {}

    def score(candidate, base=None):
        models = [ForecastModel(candidate, cfg, variant, base).fit(b.Xtr, b.ytr) for b in blocks]
        fitted[candidate["id"]] = models
        frame = validation_rows(blocks, lambda number, X, m=models: m[number].predict(X))
        frames[candidate["id"]] = frame
        records.append({**candidate, "validation_mse": mse(frame), "validation_n": len(frame)})

    for candidate in pool:
        if candidate["family"] == "history":
            score(candidate)
    history = min(records, key=lambda r: r["validation_mse"])
    for candidate in pool:
        if candidate["family"] in FAMILIES:
            score(candidate, history)
    selected = {"selected_history": history}
    selected["fixed_history"] = next(r for r in records if r["id"] == "recent_7200s")
    for family in FAMILIES:
        selected[family] = min(
            (r for r in records if r["family"] == family), key=lambda r: r["validation_mse"]
        )
    selected["selected_learned"] = min(
        (selected[f] for f in FAMILIES), key=lambda r: r["validation_mse"]
    )
    fixed_id = selected["fixed_history"]["id"]
    fixed_frame = frames[fixed_id]
    block_mean = (
        fixed_frame["block"]
        .map({number: m.mean for number, m in enumerate(fitted[fixed_id])})
        .to_numpy()
    )
    weight = shrinkage_weight(fixed_frame["y"], fixed_frame["prediction"], block_mean)
    shrunk = fixed_frame.assign(
        prediction=block_mean + weight * (fixed_frame["prediction"] - block_mean)
    )
    simple = {
        "fixed_history": fixed_frame,
        "selected_history": frames[history["id"]],
        "shrunk_history": shrunk,
    }
    scores = {name: mse(frame) for name, frame in simple.items()}
    chosen = min(scores, key=scores.get)
    validation = dict(simple)
    validation["hstar"] = simple[chosen]
    validation |= {family: frames[selected[family]["id"]] for family in FAMILIES}
    validation["selected_learned"] = frames[selected["selected_learned"]["id"]]
    return {
        "selected": selected,
        "records": records,
        "shrinkage_weight": weight,
        "simple_validation_mse": scores,
        "hstar": chosen,
        "validation": validation,
    }


def final_models(node, lo, lead, variant, deg, selection):
    """Fit the selected strategies on every earlier matched training row."""
    cfg = node.cfg
    X, y = matched_training(
        node, lo - maximum_lookback(cfg), lead, deg, effective_seeds(deg, cfg["train_seeds"])
    )
    selected = selection["selected"]
    models = {
        "fixed_history": ForecastModel(selected["fixed_history"], cfg, variant).fit(X, y),
        "selected_history": ForecastModel(selected["selected_history"], cfg, variant).fit(X, y),
    }
    for family in FAMILIES:
        models[family] = ForecastModel(
            selected[family], cfg, variant, selected["selected_history"]
        ).fit(X, y)
    return models, X, y


def predict_all(models, selection, X, y_train_mean):
    """Predictions for every reported strategy, including the validation-chosen baseline."""
    out = {name: model.predict(X) for name, model in models.items()}
    mean = models["fixed_history"].mean
    weight = selection["shrinkage_weight"]
    out["shrunk_history"] = mean + weight * (out["fixed_history"] - mean)
    out["hstar"] = out[selection["hstar"]]
    out["selected_learned"] = out[selection["selected"]["selected_learned"]["family"]]
    out["climatology"] = np.full(len(X), y_train_mean)
    return out


def method_choice(selection, method):
    selected = selection["selected"]
    if method == "hstar":
        return selection["hstar"]
    if method == "shrunk_history":
        return f"shrunk_history(w={selection['shrinkage_weight']:.4f})"
    if method == "selected_learned":
        return selected["selected_learned"]["id"]
    if method == "climatology":
        return "climatology"
    return selected[method]["id"]


def ablation_predictions(node, deg, selection, X, y, scoring_X, variants):
    """Selected boosting configuration refitted on feature subsets; no new search."""
    out = {}
    for variant in variants:
        model = ForecastModel(
            selection["selected"]["hgb"],
            node.cfg,
            variant,
            selection["selected"]["selected_history"],
        ).fit(X, y)
        out[f"hgb__{variant}"] = model.predict(scoring_X)
    return out


def run_scenario(node, lo, hi, fold, name, deg, lead, cfg, *, intervals=False, ablate=False):
    """Select, fit and score one node/fold/scenario; return forecasts, intervals and a record."""
    variant = cfg["primary_variant"]
    selection = matched_selection(node, lo, lead, variant, deg, cfg)
    models, Xtr, ytr = final_models(node, lo, lead, variant, deg, selection)
    origins = node.origins(lo, hi, lead, cfg["window_s"], common=True)
    o, y, count = node.dataset(origins, lead)
    record = {
        "node": node.node,
        "fold": fold,
        "scenario": name,
        "variant": variant,
        "selected": selection["selected"],
        "hstar": selection["hstar"],
        "shrinkage_weight": selection["shrinkage_weight"],
        "simple_validation_mse": selection["simple_validation_mse"],
        "candidates": selection["records"],
        "evaluation_start_s": lo,
        "evaluation_end_s": hi,
        "n_evaluation": len(y),
    }
    if not len(y):
        record["status"] = "no_eligible_evaluation_targets"
        return pd.DataFrame(), pd.DataFrame(), record
    mean = float(np.mean(ytr))
    levels = cfg["quantiles"]
    rows, interval_rows = [], []
    for seed in effective_seeds(deg, cfg["score_seeds"]):
        X = node.features(o, deg, seed)
        predictions = predict_all(models, selection, X, mean)
        if ablate:
            predictions |= ablation_predictions(node, deg, selection, Xtr, ytr, X, ABLATIONS)
        for method, p in predictions.items():
            rows.append(
                pd.DataFrame(
                    {
                        "node": node.node,
                        "fold": fold,
                        "scenario": name,
                        "seed": seed,
                        "method": method,
                        "candidate_id": method_choice(selection, method)
                        if not method.startswith("hgb__")
                        else method,
                        "origin_s": o,
                        "target_start_s": o + lead,
                        "target_end_s": o + lead + cfg["window_s"],
                        "target_count": count,
                        "y": y,
                        "prediction": p,
                        "training_mean": mean,
                        "dropout_p": deg.dropout_p,
                        "delay_s": deg.delay_s,
                        "outage_s": deg.outage_len_s,
                    }
                )
            )
        if intervals:
            interval_rows += interval_frames(
                node, fold, name, seed, o, y, X, predictions, selection, Xtr, ytr, levels, cfg
            )
    record["status"] = "completed"
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame(),
        record,
    )


def interval_frames(node, fold, name, seed, o, y, X, predictions, selection, Xtr, ytr, levels, cfg):
    """Quantile forecasts for the learned strategy, its conformal band, the baseline and climate."""
    chosen = selection["selected"]["hgb"]
    columns, models = fit_quantile_models(
        Xtr, ytr, cfg["primary_variant"], levels, chosen["depth"], chosen["leaf"], cfg
    )
    forecasts = {"qhgb": predict_quantiles(columns, models, X)}
    for label, method in (("conformal_learned", "selected_learned"), ("conformal_hstar", "hstar")):
        validation = selection["validation"][method]
        error = (validation["y"] - validation["prediction"]).to_numpy()
        forecasts[label] = conformal_quantiles(predictions[method], error, levels)
    forecasts["empirical"] = empirical_quantiles(ytr, len(o), levels)
    frames = []
    for method, q in forecasts.items():
        frame = pd.DataFrame(q, columns=[f"q{tau:g}" for tau in levels])
        frame.insert(0, "y", y)
        frame.insert(0, "origin_s", o)
        frame.insert(0, "method", method)
        frame.insert(0, "seed", seed)
        frame.insert(0, "scenario", name)
        frame.insert(0, "fold", fold)
        frame.insert(0, "node", node.node)
        frames.append(frame)
    return frames


def part_path(prefix, suffix):
    """Append a suffix to a part name. ``with_suffix`` is wrong here: scenario names such as
    ``dropout_0.5`` contain a dot, so it would rewrite part of the name."""
    return prefix.with_name(prefix.name + suffix)


def run_node(raw, cfg, out, *, scenarios=None, progress=print, resume=False):
    """Write one part per node/scenario/fold; return fold records. Resumable by part."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    node = NodeData(raw, cfg, cfg["primary_offset_db"])
    definitions = scenario_definitions(cfg)
    records = []
    for name, deg in definitions.items():
        if scenarios is not None and name not in scenarios:
            continue
        primary = name == cfg["primary_scenario"]
        for fold, (lo, hi) in enumerate(folds(node.t0, node.last, cfg)):
            prefix = out / f"{node.node}__{name}__{fold}"
            if part_path(prefix, ".json").exists():
                if not resume:
                    raise FileExistsError(prefix)
                records.append(json.loads(part_path(prefix, ".json").read_text()))
                continue
            progress(
                f"{node.node}: {name}, fold {fold + 1}, matched selection and scoring", flush=True
            )
            forecasts, intervals, record = run_scenario(
                node,
                lo,
                hi,
                fold,
                name,
                deg,
                cfg["primary_lead_s"],
                cfg,
                intervals=name in cfg["interval_scenarios"],
                ablate=primary,
            )
            if len(forecasts):
                forecasts.to_parquet(part_path(prefix, ".parquet"), index=False)
            if len(intervals):
                intervals.to_parquet(
                    prefix.with_name(prefix.name + "__intervals.parquet"), index=False
                )
            part_path(prefix, ".json").write_text(json.dumps(record, indent=2, default=float))
            records.append(record)
    return {
        "calibration": {"noise_floors_dbm": node.floors, "nominal_interval_s": node.dt},
        "folds": records,
    }
