"""Chronological selection and bounded quality fallback for history mixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.experiments.followup import scenarios
from spectrum_occupancy_forecasting.experiments.followup_data import (
    NodeData,
    folds,
    maximum_lookback,
)
from spectrum_occupancy_forecasting.experiments.followup_models import (
    augment_training,
    history_prediction,
    operating_threshold,
    training,
)


def history_candidates(cfg):
    return [
        {"id": f"recent_{w}s", "family": "history", "window": w} for w in cfg["history_windows_s"]
    ] + [{"id": f"decay_{tau}s", "family": "history", "tau": tau} for tau in cfg["decay_taus_s"]]


class InsufficientValidation(ValueError):
    """A declared validation block cannot be scored; keep the fold as fixed history."""


class BoundedCorrection:
    def __init__(self, choice, history, cfg, dt, variant="full"):
        self.choice, self.history, self.cfg, self.dt, self.variant = (
            choice,
            history,
            cfg,
            dt,
            variant,
        )

    def feature_frame(self, X, base):
        raise NotImplementedError("Implement in the concrete history-mixture model")

    def fit(self, X, y):
        raise NotImplementedError("Implement in the concrete history-mixture model")

    def details(self, X):
        base = history_prediction(self.history, X, self.mean)
        reasons = np.full(len(X), "none", dtype=object)
        coverage = X["n_900s"].to_numpy() / (900 / self.dt)
        age = X["age_s"].to_numpy()
        reasons[(coverage < self.cfg["minimum_predictor_coverage"]) | ~np.isfinite(coverage)] = (
            "low_coverage"
        )
        reasons[age > self.cfg["maximum_correction_age_s"]] = "stale"
        reasons[~np.isfinite(age)] = "missing_age"
        if self.choice["strength"] == 0:
            correction = np.zeros(len(X))
            reasons[:] = "lambda_zero"
        else:
            raw = self.est.predict(self.feature_frame(X, base)[self.columns])
            correction = self.choice["strength"] * np.clip(
                raw, -self.cfg["correction_cap"], self.cfg["correction_cap"]
            )
            correction[reasons != "none"] = 0
        prediction = np.clip(base + correction, 0, 1)
        prediction[reasons != "none"] = base[reasons != "none"]
        return {
            "prediction": prediction,
            "history_prediction": base,
            "correction_value": prediction - base,
            "fallback_reason": reasons,
        }

    def predict(self, X):
        return self.details(X)["prediction"]


def validation_blocks(node, lo, lead, hours):
    start = lo - maximum_lookback(node.cfg) - hours * 3600
    if start < node.t0 + node.cfg["calibration_s"]:
        return None
    blocks = []
    for begin in np.arange(start, start + hours * 3600, node.cfg["validation_block_s"]):
        try:
            X, y = training(node, begin - maximum_lookback(node.cfg), lead)
        except ValueError as exc:
            if str(exc) != "no eligible training targets":
                raise
            raise InsufficientValidation("insufficient_validation_training_targets") from exc
        o, yy, _ = node.dataset(
            node.origins(
                begin,
                begin + node.cfg["validation_block_s"],
                lead,
                node.cfg["window_s"],
                common=True,
            ),
            lead,
        )
        if not len(yy):
            raise InsufficientValidation("insufficient_validation_targets")
        blocks.append((X, y, node.features(o), yy, o))
    return blocks


def select(node, lo, lead, hours, regime, variant="full", model_class=None):
    candidates = model_class.candidates(node.cfg)
    fixed = {"id": "recent_7200s", "family": "history", "window": 7200}
    try:
        blocks = validation_blocks(node, lo, lead, hours)
    except InsufficientValidation as exc:
        return (fixed, candidates[0], [], {}, str(exc))
    if blocks is None:
        return (fixed, candidates[0], [], {}, "insufficient_earlier_data")
    records, saved = ([], {})

    def save(choice, parts, family):
        frame = pd.concat(parts, ignore_index=True).assign(
            candidate_id=choice["id"],
            correction_strength=choice.get("strength", 0),
            selected_penalty=choice.get("alpha"),
        )
        saved[choice["id"]] = frame
        records.append(
            {
                **choice,
                "family": family,
                "validation_n": len(frame),
                "validation_mse": float(np.mean((frame.prediction - frame.y) ** 2)),
            }
        )

    for h in history_candidates(node.cfg):
        save(
            h,
            [
                pd.DataFrame(
                    {
                        "origin_s": o,
                        "y": yy,
                        "prediction": history_prediction(h, XV, float(y.mean())),
                    }
                )
                for X, y, XV, yy, o in blocks
            ],
            "history",
        )
    history = min(records, key=lambda r: r["validation_mse"])
    train_blocks = [
        augment_training(node, X, y) if regime == "augmented" else (X, y)
        for X, y, _, _, _ in blocks
    ]
    fitted = {}
    for choice in candidates:
        parts = []
        for i, (_, _, XV, yy, o) in enumerate(blocks):
            key = (i, choice["alpha"], choice["strength"] == 0)
            if key not in fitted:
                fitted[key] = model_class(choice, history, node.cfg, node.dt, variant).fit(
                    *train_blocks[i]
                )
            model = fitted[key]
            model.choice = choice
            parts.append(pd.DataFrame({"origin_s": o, "y": yy, **model.details(XV)}))
        save(choice, parts, "correction")
    winner = min(
        (r for r in records if r["family"] == "correction"), key=lambda r: r["validation_mse"]
    )
    return (
        history,
        winner,
        records,
        {
            "selected_history": saved[history["id"]],
            "selected_learned": saved[winner["id"]],
            "fixed_history": saved[fixed["id"]],
            **{f"candidate:{key}": value for key, value in saved.items()},
        },
        "selected",
    )


def run_node(
    raw,
    cfg,
    out,
    *,
    task_names=None,
    progress=print,
    strategy=None,
    secondary=False,
    model_class,
    node_class=NodeData,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary = [("primary", cfg["primary_lead_s"], cfg["primary_offset_db"], "full")]
    tasks = primary
    if secondary:
        tasks += [(f"lead_{lead}s", lead, 3, "full") for lead in cfg["leads_s"] if lead != 900]
        tasks += [
            (f"label_{offset}db", 900, offset, "full")
            for offset in cfg["offsets_db"]
            if offset != 3
        ]
        tasks += [
            (f"ablation_{variant}", 900, 3, variant) for variant in ("history", "power", "calendar")
        ]
    if task_names is not None:
        tasks = [task for task in tasks if task[0] in task_names]
    strategies = (
        [strategy]
        if strategy
        else [(h, r) for h in cfg["validation_hours"] for r in cfg["training_regimes"]]
    )
    nodes, records, calibrations = ({}, [], [])
    for task, lead, offset, variant in tasks:
        if offset not in nodes:
            nodes[offset] = node_class(raw, cfg, offset)
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
            common = (
                lo - maximum_lookback(cfg) - max(cfg["validation_hours"]) * 3600
                >= node.t0 + cfg["calibration_s"]
            )
            o, y, count = node.dataset(
                node.origins(lo, hi, lead, cfg["window_s"], common=True), lead
            )
            for hours, regime in strategies:
                sid = f"{hours}h_{regime}"
                progress(f"{node.node}: {task}, fold {fold + 1}, {sid}", flush=True)
                prefix = out / f"{node.node}__{task}__{sid}__{fold}"
                if prefix.with_suffix(".parquet").exists():
                    raise FileExistsError(prefix)
                history, choice, scores, validation, status = select(
                    node, lo, lead, hours, regime, variant, model_class=model_class
                )
                X, yy = training(node, lo - maximum_lookback(cfg), lead)
                mean = float(yy.mean())
                if regime == "augmented":
                    X, yy = augment_training(node, X, yy)
                model = model_class(choice, history, cfg, node.dt, variant).fit(X, yy)
                record = {
                    "node": node.node,
                    "task": task,
                    "fold": fold,
                    "strategy_id": sid,
                    "validation_hours": hours,
                    "training_regime": regime,
                    "variant": variant,
                    "lead_s": lead,
                    "offset_db": offset,
                    "selection_status": status,
                    "common_fold": common,
                    "selected": {"history": history, "correction": choice},
                    "candidates": scores,
                    "evaluation_start_s": lo,
                    "evaluation_end_s": hi,
                    "status": "completed" if len(y) else "no_eligible_evaluation_targets",
                    "n_evaluation": len(y),
                    "target_degenerate": bool(len(y) and np.var(y) == 0),
                    "operating_thresholds": [],
                }
                if hasattr(model, "audit_state"):
                    record["model_state"] = model.audit_state()
                for method in ("selected_history", "selected_learned", "fixed_history"):
                    frame = validation.get(method, pd.DataFrame(columns=["y", "prediction"]))
                    record["operating_thresholds"].extend(
                        
                            {
                                "method": method,
                                **operating_threshold(frame.y, frame.prediction, budget, cfg),
                            }
                            for budget in cfg["false_busy_budgets"]
                        
                    )
                if validation:
                    pd.concat(
                        [
                            v.assign(
                                method=k, node=node.node, task=task, fold=fold, strategy_id=sid
                            )
                            for k, v in validation.items()
                        ],
                        ignore_index=True,
                    ).to_parquet(
                        prefix.with_name(prefix.name + "__validation.parquet"), index=False
                    )
                if len(y):
                    rows = []
                    for scenario, deg in (
                        scenarios(cfg) if task == "primary" else {"clean": scenarios(cfg)["clean"]}
                    ).items():
                        seeds = cfg["mask_seeds"] if deg.dropout_p or deg.outage_frac else [0]
                        for seed in seeds:
                            F = node.features(o, deg, seed)
                            details = model.details(F)
                            fixed = history_prediction({"window": 7200}, F, mean)
                            for method, pred in (
                                ("selected_learned", details["prediction"]),
                                ("selected_history", details["history_prediction"]),
                                ("fixed_history", fixed),
                                ("climatology", np.full(len(y), mean)),
                            ):
                                rows.append(
                                    pd.DataFrame(
                                        {
                                            "node": node.node,
                                            "task": task,
                                            "fold": fold,
                                            "strategy_id": sid,
                                            "method": method,
                                            "variant": variant,
                                            "scenario": scenario,
                                            "seed": seed,
                                            "origin_s": o,
                                            "target_start_s": o + lead,
                                            "target_end_s": o + lead + cfg["window_s"],
                                            "lead_s": lead,
                                            "offset_db": offset,
                                            "y": y,
                                            "prediction": pred,
                                            "label_threshold_dbm": node.thresholds[
                                                cfg["target_mhz"]
                                            ],
                                            "target_count": count,
                                            "training_mean": mean,
                                            "candidate_id": choice["id"]
                                            if method == "selected_learned"
                                            else history["id"]
                                            if method == "selected_history"
                                            else method,
                                            "validation_hours": hours,
                                            "training_regime": regime,
                                            "common_fold": common,
                                            "correction_strength": choice["strength"],
                                            "selected_penalty": choice["alpha"],
                                            "history_prediction": details["history_prediction"],
                                            "correction_value": details["correction_value"]
                                            if method == "selected_learned"
                                            else 0,
                                            "fallback_reason": details["fallback_reason"]
                                            if status == "selected"
                                            else status,
                                            "selection_status": status,
                                            "dropout_p": deg.dropout_p,
                                            "delay_s": deg.delay_s,
                                            "outage_s": deg.outage_len_s,
                                            **{
                                                key: value
                                                for key, value in details.items()
                                                if method == "selected_learned"
                                                and key
                                                not in {
                                                    "prediction",
                                                    "history_prediction",
                                                    "correction_value",
                                                    "fallback_reason",
                                                }
                                            },
                                        }
                                    )
                                )
                    pd.concat(rows, ignore_index=True).to_parquet(
                        prefix.with_suffix(".parquet"), index=False
                    )
                prefix.with_suffix(".json").write_text(
                    json.dumps(record, indent=2, allow_nan=False)
                )
                records.append(record)
    return {
        "calibrations": calibrations,
        "folds": records,
        "status": "completed" if records else "no_eligible_folds",
    }
