"""Prediction-only paired protocol audit and fixed, untuned controller replay."""

import json

import numpy as np
import pandas as pd

from .adaptive_models import Candidate, CandidateBank
from .adaptive_models import candidates as adaptive_candidates
from .adaptive_replay import STRATEGIES as ADAPTIVE_STRATEGIES
from .adaptive_replay import replay
from .fine_spectrum import write_json
from .followup import scenarios
from .operational_selection import choose
from .residual_state_space import STRATEGIES as STATE_STRATEGIES
from .residual_state_space import candidates as state_candidates


def validation_audit(nodes, cfg, out, root, scoring_grid):
    summaries, choices, scores, validations = [], [], [], []
    for study, factory, strategies in (
        ("adaptive_learning", adaptive_candidates, ADAPTIVE_STRATEGIES),
        ("residual_state_space", state_candidates, STATE_STRATEGIES),
    ):
        source = root / "results" / study / "run"
        cold = pd.read_parquet(source / "validation_predictions.parquet")
        original = pd.read_parquet(source / "predictions.parquet")
        historic_selections = json.loads((source / "selections.json").read_text())
        first_choices = {}
        for recording in ("CC1_first", "LW1_first", "LW1_later"):
            # These are the saved uninterrupted event-driven replay states, not refitted forecasts.
            issued = pd.read_parquet(source / f"{recording}_candidate_forecasts.parquet")
            grid = (
                original[(original.recording == recording) & (original.scenario == "clean")][
                    ["origin_s", "y", "target_count", "fold", "fold_begin_s"]
                ]
                .drop_duplicates()
                .sort_values("origin_s")
            )
            for fold, population in grid.groupby("fold"):
                evaluation = float(population.fold_begin_s.iloc[0])
                for protocol in ("historical-cold-start", "continuous-operational"):
                    if recording == "LW1_later":
                        chosen, history = first_choices[(study, protocol)]
                        rows, val = [], []
                    else:
                        if protocol == "historical-cold-start":
                            frame = cold[(cold.recording == recording) & (cold.fold == fold)].copy()
                        else:
                            blocks = []
                            for b in (0, 1):
                                labels = cold[
                                    (cold.recording == recording)
                                    & (cold.fold == fold)
                                    & (cold.block == b)
                                ][["origin_s", "y", "target_count"]].drop_duplicates()
                                lo = evaluation - 36000 - 43200 + b * 21600
                                if not (
                                    (labels.origin_s >= lo) & (labels.origin_s + 1800 <= lo + 21600)
                                ).all():
                                    raise ValueError(
                                        "validation target crosses block or selection embargo"
                                    )
                                blocks.append(
                                    issued.merge(
                                        labels, on="origin_s", validate="many_to_one"
                                    ).assign(block=b)
                                )
                            frame = pd.concat(blocks, ignore_index=True)
                        if frame.empty:
                            chosen, history, rows, val = (
                                {s: factory(cfg)[0].id for s in strategies},
                                "recent_7200s",
                                [],
                                [],
                            )
                        else:
                            chosen, history, rows, val, _ = choose(
                                frame, cfg, factory(cfg), strategies
                            )
                        first_choices[(study, protocol)] = chosen, history
                        if protocol == "historical-cold-start":
                            saved = next(
                                r
                                for r in historic_selections
                                if r["recording"] == recording and r["fold"] == fold
                            )
                            if saved["chosen"] != chosen or saved["selected_history"] != history:
                                raise ValueError(
                                    "historical cold-start selection differs from saved evidence"
                                )
                    metadata = {
                        "study": study,
                        "recording": recording,
                        "fold": int(fold),
                        "protocol": protocol,
                    }
                    if cfg.get("synthetic"):
                        metadata["synthetic"] = True
                    subsequent = issued.merge(population, on="origin_s", validate="many_to_one")
                    selected_ref = (
                        subsequent[subsequent.candidate == history].set_index("origin_s").prediction
                    )
                    fallback = subsequent.fallback_reason.fillna("") != ""
                    subsequent.loc[fallback, "prediction"] = subsequent.loc[
                        fallback, "origin_s"
                    ].map(selected_ref)
                    outer_mse = (
                        subsequent.assign(loss=(subsequent.prediction - subsequent.y) ** 2)
                        .groupby("candidate")
                        .loss.mean()
                    )
                    ranks = {
                        r["candidate"]: index + 1
                        for index, r in enumerate(
                            sorted(rows, key=lambda r: (r["pooled_mse"], r["candidate"]))
                        )
                    }
                    scores.extend(
                        metadata
                        | r
                        | {
                            "validation_rank": ranks[r["candidate"]],
                            "subsequent_mse": float(outer_mse[r["candidate"]]),
                            "retrospective_only": True,
                        }
                        for r in rows
                    )
                    validations.extend(metadata | r for r in val)
                    choices.append(
                        metadata
                        | {
                            "chosen": chosen,
                            "history": history,
                            "selection_source": "LW1_first_final_fold"
                            if recording == "LW1_later"
                            else recording,
                            "status": "inherited"
                            if recording == "LW1_later"
                            else "selected"
                            if rows
                            else "insufficient_validation",
                        }
                    )
                    evaluation_rows = issued.merge(
                        population, on="origin_s", validate="many_to_one"
                    )
                    histories = (
                        evaluation_rows[evaluation_rows.candidate == history]
                        .set_index("origin_s")
                        .prediction
                    )
                    fixed = (
                        evaluation_rows[evaluation_rows.candidate == "recent_7200s"]
                        .set_index("origin_s")
                        .prediction
                    )
                    for strategy, identifier in chosen.items():
                        selected = evaluation_rows[evaluation_rows.candidate == identifier].copy()
                        fallback = selected.fallback_reason.fillna("") != ""
                        selected.loc[fallback, "prediction"] = selected.loc[
                            fallback, "origin_s"
                        ].map(histories)
                        loss = float(((selected.prediction - selected.y) ** 2).mean())
                        reference_y = selected.set_index("origin_s").y
                        fixed_loss = float(((fixed - reference_y) ** 2).mean())
                        hist_loss = float(((histories - reference_y) ** 2).mean())
                        summaries.append(
                            metadata
                            | {
                                "strategy": strategy,
                                "candidate": identifier,
                                "n": len(selected),
                                "mse": loss,
                                "skill_fixed": 1 - loss / fixed_loss,
                                "skill_selected": 1 - loss / hist_loss,
                                "fallback_share": float(fallback.mean()),
                            }
                        )
    pd.DataFrame(scores).to_csv(out / "candidate_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "protocol_comparison.csv", index=False)
    pd.DataFrame(validations).to_parquet(out / "validation_predictions.parquet", index=False)
    write_json(out / "selections.json", choices)
    write_json(
        out / "audit.json",
        {
            "historical_selections_reproduced": True,
            "continuous_states_source": "immutable uninterrupted saved operational replay",
            "adoption_reason": "matches declared deployment; not selected by favorable performance",
            "shared_history_controls": True,
        },
    )
    if cfg.get("synthetic"):
        write_json(out / "SYNTHETIC.json", {"synthetic": True, "not_real_data_evidence": True})


class ControllerBank(CandidateBank):
    def __init__(self, cfg, initial_features):
        super().__init__(cfg, initial_features)
        self.candidates = [Candidate(), Candidate("aggregation", 6, 1, rate=0.5)]
        key = self.candidates[1].key
        self.states = {key: self.states[key]}


def controller_audit(nodes, cfg, out, scoring_grid, source_selections):
    saved = json.loads(source_selections.read_text())
    output, feedback = [], []
    for recording, node in nodes.items():
        grid = scoring_grid(node, recording == "LW1_later")
        choices = {r["fold"]: r["selected_history"] for r in saved if r["recording"] == recording}
        origins = node.data.origins(node.data.t0 + 86400, node.data.last, 900, 900)
        histories = {}
        for origin in origins:
            folds = [f for f, p in grid.groupby("fold") if p.fold_begin_s.iloc[0] <= origin]
            histories[float(origin)] = choices[max(folds)] if folds else "recent_7200s"
        for scenario, deg in scenarios(cfg).items():
            for seed in cfg["mask_seeds"] if deg.dropout_p or deg.outage_len_s else [0]:
                rows, events, _ = replay(
                    node,
                    cfg,
                    origins,
                    deg=deg,
                    seed=seed,
                    selected_history=histories,
                    bank_factory=lambda settings, frames, *_: ControllerBank(settings, frames),
                )
                scored = rows.merge(grid, on="origin_s", validate="many_to_one")
                scored = scored[
                    scored.candidate.isin(
                        [Candidate().id, Candidate("aggregation", 6, 1, rate=0.5).id]
                    )
                ].copy()
                for controller in cfg["controllers"]:
                    part = scored.copy()
                    fallback = part.fallback_reason.fillna("") != ""
                    if controller != "selected-hard":
                        part.loc[fallback, "prediction"] = part.loc[fallback, "fixed_history"]
                    if controller == "fixed-soft":
                        shrink = part.coverage_900s.clip(0, 1) * np.exp(-part.age_s / 900)
                        active = ~fallback
                        part.loc[active, "prediction"] = np.clip(
                            part.loc[active, "fixed_history"]
                            + part.loc[active, "correction"] * shrink[active],
                            0,
                            1,
                        )
                    part["controller"] = controller
                    part["recording"], part["scenario"], part["seed"] = recording, scenario, seed
                    output.append(part)
                feedback.append(events.assign(recording=recording, scenario=scenario, seed=seed))
                print(f"controller {recording} {scenario} {seed}", flush=True)
    frame = pd.concat(output, ignore_index=True)
    if cfg.get("synthetic"):
        frame["synthetic"] = True
        for part in feedback:
            part["synthetic"] = True
        write_json(out / "SYNTHETIC.json", {"synthetic": True, "not_real_data_evidence": True})
    frame.to_parquet(out / "predictions.parquet", index=False)
    pd.concat(feedback, ignore_index=True).to_parquet(out / "feedback.parquet", index=False)
    controller_report(frame, out)


def controller_report(frame, out):
    rows = []
    for keys, part in frame.groupby(["recording", "scenario", "controller", "candidate"]):
        losses = (part.prediction - part.y) ** 2
        fixed_loss = (part.fixed_history - part.y) ** 2
        selected_loss = (part.selected_history - part.y) ** 2
        rows.append(
            dict(zip(["recording", "scenario", "controller", "candidate"], keys, strict=True))
            | {
                "n_origins": part.origin_s.nunique(),
                "mse": float(losses.mean()),
                "skill_fixed": float(1 - losses.mean() / fixed_loss.mean()),
                "skill_selected": float(1 - losses.mean() / selected_loss.mean()),
                "fallback_share": float((part.fallback_reason.fillna("") != "").mean()),
            }
        )
    table = pd.DataFrame(rows)
    if "synthetic" in frame and frame.synthetic.any():
        table["synthetic"] = True
    table.to_csv(out / "controller_comparison.csv", index=False)
