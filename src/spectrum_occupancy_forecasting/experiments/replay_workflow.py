"""Shared operational replay orchestration; algorithms are injected explicitly."""

import numpy as np
import pandas as pd

from .fine_spectrum import write_json
from .followup import scenarios


def run(
    nodes,
    cfg,
    out,
    synthetic,
    *,
    selection_fn,
    replay_fn,
    strategies,
    candidate_factory,
    report_fn,
    scoring_grid,
):
    state_records, update_records = [], []
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
                chosen, history, rows, val, failure = selection_fn(node, cfg, begin)
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
                records, feedback, bank = replay_fn(
                    node,
                    cfg,
                    origins,
                    deg=deg,
                    seed=seed,
                    selected_history=selected_histories,
                    active=None
                    if scenario == "clean"
                    else {candidate_factory(cfg)[0].id}
                    | {
                        candidate
                        for chosen, _ in fold_choices.values()
                        for candidate in chosen.values()
                    },
                )
                if hasattr(bank, "state_records"):
                    states = pd.DataFrame(bank.state_records)
                    states["recording"], states["scenario"], states["seed"] = (
                        recording,
                        scenario,
                        seed,
                    )
                    state_records.append(states)
                    updates = pd.DataFrame(bank.update_records)
                    updates["recording"], updates["scenario"], updates["seed"] = (
                        recording,
                        scenario,
                        seed,
                    )
                    update_records.append(updates)
                joined = records.merge(grid, on="origin_s", validate="many_to_one")
                baseline = candidate_factory(cfg)[0].id
                views = []
                for strategy in strategies:
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
    if state_records:
        pd.concat(state_records, ignore_index=True).to_parquet(
            out / "filter_states.parquet", index=False
        )
        pd.concat(update_records, ignore_index=True).to_parquet(
            out / "filter_updates.parquet", index=False
        )
    report_fn(out, out, cfg)
