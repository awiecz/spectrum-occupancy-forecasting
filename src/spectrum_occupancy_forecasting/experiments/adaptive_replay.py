"""Event-driven replay and embargoed selection for the declared adaptive study."""

import heapq

import pandas as pd

from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask

from .adaptive_models import CandidateBank, candidates
from .mixture_support import history_candidates

STRATEGIES = {
    "bias": lambda c: c.family in ("history", "bias"),
    "aggregation": lambda c: c.family in ("history", "aggregation"),
    "ridge_history": lambda c: (
        c.family == "history" or (c.family == "ridge" and c.features == "history")
    ),
    "ridge_fine": lambda c: (
        c.family == "history" or (c.family == "ridge" and c.features == "history_fine")
    ),
}


def feature_frames(node, origins, mean, deg, seed):
    fine, base, _ = node.features(origins, mean, "history_fine", deg, seed, calibration_clean=True)
    history = fine.loc[:, ~fine.columns.str.startswith("fine_")]
    return {"history": history, "history_fine": fine}, base


def delivered_feedback(node, origin, release, deg, seed, cfg):
    """Access only the completed delivered window, with original labels used nowhere."""
    end = origin + 1800
    if release < end + deg.delay_s:
        raise ValueError("target feedback requested before delivery")
    times = node.frame.time_s.to_numpy()
    visible = visibility_mask(times, deg, seed)
    keep = visible & (times >= origin + 900) & (times < end)
    values = node.frame.target_dbm.to_numpy()[keep] > node.data.floors[2437.0] + 3
    count = int(keep.sum())
    adequate = count >= cfg["feedback_coverage"] * 900 / node.data.dt
    return (float(values.mean()) if adequate else None), count


def replay(
    node,
    cfg,
    origins,
    *,
    training_cutoff=None,
    start=None,
    deg=None,
    seed=0,
    selected_history=None,
    active=None,
    checkpoint=None,
    stop_after=None,
    bank_factory=None,
):
    """Replay every unscored/scored origin; queue immutable feature and expert snapshots."""
    deg = deg or Degradation()
    cutoff = training_cutoff if training_cutoff is not None else node.data.t0 + 86400
    start = start if start is not None else node.data.t0 + 86400
    initial_origins, initial_y, _ = node.training(cutoff, cfg["replay_stride_s"])
    mean = float(initial_y.mean())
    initial_features, initial_base = feature_frames(node, initial_origins, mean, Degradation(), 0)
    bank = (
        CandidateBank(cfg, initial_features)
        if bank_factory is None
        else bank_factory(cfg, initial_features, initial_y, initial_base)
    )
    experts_initial = initial_features["history"].iloc[:, :8].to_numpy()
    vectors_initial = {
        name: processor.transform(initial_features[name])
        for name, processor in bank.preprocessing.items()
    }
    # Initial labels are already available at replay startup, but retain their actual end times.
    # Fit in target-completion order; the feedback gate uses completion, not artificial freshness.
    for j, origin in enumerate(initial_origins):
        bank.update(
            origin + 1800,
            origin + 1800,
            initial_y[j],
            initial_base[j],
            experts_initial[j],
            {name: values[j] for name, values in vectors_initial.items()},
        )
    signature = (
        tuple(origins),
        deg,
        seed,
        cutoff,
        start,
        None if active is None else frozenset(active),
    )
    if checkpoint is not None:
        bank = bank.resume(checkpoint)
        if bank.replay_signature != signature:
            raise ValueError("checkpoint replay configuration mismatch")
        pending, records, feedback, begin_index = bank.replay_progress
    else:
        pending, records, feedback, begin_index = [], [], [], 0
    features, base = feature_frames(node, origins, mean, deg, seed)
    snapshot_features = getattr(bank, "snapshot_features", "history_fine")
    bank.feature_snapshots = features[snapshot_features].assign(origin_s=origins)
    vectors = {
        name: processor.transform(features[name]) for name, processor in bank.preprocessing.items()
    }
    history = features["history"].iloc[:, :8].to_numpy()
    selected_history = selected_history or {}
    default = history_candidates(node.cfg)[3]["id"]
    history_ids = [h["id"] for h in history_candidates(node.cfg)]
    for j, origin in enumerate(origins):
        if j < begin_index:
            continue
        if stop_after is not None and j >= stop_after:
            bank.replay_signature = signature
            bank.replay_progress = (pending, records, feedback, j)
            bank.feature_snapshots = bank.feature_snapshots.iloc[:j].copy()
            return pd.DataFrame(records), pd.DataFrame(feedback), bank
        if origin < start:
            raise ValueError("replay origins precede startup")
        while pending and pending[0][0] <= origin:
            release, index, target_origin = heapq.heappop(pending)
            value, count = delivered_feedback(node, target_origin, release, deg, seed, cfg)
            event = {
                "origin_s": target_origin,
                "target_end_s": target_origin + 1800,
                "release_s": release,
                "feedback_y": value,
                "feedback_count": count,
                "update_reason": "updated" if value is not None else "insufficient_coverage",
            }
            feedback.append(event)
            if bank_factory is not None:
                event["issued_fixed_history"] = float(base[index])
                event["delivered_residual"] = None if value is None else float(value - base[index])
            if value is not None:
                bank.update(
                    release,
                    target_origin + 1800,
                    value,
                    base[index],
                    history[index],
                    {name: values[index] for name, values in vectors.items()},
                )
        selected = selected_history.get(float(origin), default)
        selected_value = history[j, history_ids.index(selected)]
        estimates = bank.forecasts(
            origin,
            base[j],
            selected_value,
            history[j],
            {name: values[j] for name, values in vectors.items()},
            features["history"].age_s.iloc[j],
            features["history"].coverage_900s.iloc[j],
            active=active,
        )
        for estimate in estimates:
            records.append(
                estimate
                | {
                    "origin_s": origin,
                    "fixed_history": base[j],
                    "selected_history": selected_value,
                    "age_s": features["history"].age_s.iloc[j],
                    "coverage_900s": features["history"].coverage_900s.iloc[j],
                }
            )
        for h, value in zip(history_ids, history[j], strict=True):
            records.append(
                {
                    "origin_s": origin,
                    "candidate": h,
                    "prediction": value,
                    "fixed_history": base[j],
                    "selected_history": selected_value,
                }
            )
        heapq.heappush(pending, (origin + 1800 + deg.delay_s, j, float(origin)))
    for release, _, target_origin in pending:
        feedback.append(
            {
                "origin_s": target_origin,
                "target_end_s": target_origin + 1800,
                "release_s": release,
                "feedback_y": None,
                "feedback_count": None,
                "update_reason": "not_released_before_replay_end",
            }
        )
    return pd.DataFrame(records), pd.DataFrame(feedback), bank


def select(
    node,
    cfg,
    evaluation_origin,
    *,
    candidate_factory=candidates,
    bank_factory=None,
    strategies=None,
):
    """Two six-hour validation replays with independent embargoed initialization."""
    end = evaluation_origin - cfg["selection_embargo_s"]
    begin = end - cfg["validation_hours"] * 3600
    validation, failures = [], []
    strategies = STRATEGIES if strategies is None else strategies
    grid = candidate_factory(cfg)
    for block in range(2):
        lo = begin + block * 21600
        hi = lo + 21600
        cutoff = lo - cfg["selection_embargo_s"]
        if lo < node.data.t0 + 86400 or cutoff <= node.data.t0 + 1800:
            return (
                {name: grid[0].id for name in strategies},
                "recent_7200s",
                [],
                [],
                "insufficient_validation",
            )
        origins = node.data.origins(lo, hi, 900, cfg["replay_stride_s"])
        predictions, _, bank = replay(
            node, cfg, origins, training_cutoff=cutoff, start=lo, bank_factory=bank_factory
        )
        scored_origins, y, count = node.data.dataset(origins, 900)
        if not len(y):
            return (
                {name: grid[0].id for name in strategies},
                "recent_7200s",
                [],
                [],
                "insufficient_validation",
            )
        labels = pd.DataFrame({"origin_s": scored_origins, "y": y, "target_count": count})
        predictions = predictions.merge(labels, on="origin_s", validate="many_to_one")
        predictions["block"] = block
        validation.append(predictions)
        failures.extend(bank.failures)
        if predictions.empty:
            return (
                {name: grid[0].id for name in strategies},
                "recent_7200s",
                [],
                [],
                "insufficient_validation",
            )
    frame = pd.concat(validation, ignore_index=True)
    frame["loss"] = (frame.prediction - frame.y) ** 2
    scores = frame.groupby(["candidate", "block"]).loss.mean().unstack()
    pooled = frame.groupby("candidate").loss.mean()
    histories = history_candidates(node.cfg)
    history_id = min(histories, key=lambda h: (pooled[h["id"]], histories.index(h)))["id"]
    # Recompute controller and safety fallbacks using independently selected history.
    ref = frame[frame.candidate == history_id].set_index(["block", "origin_s"]).prediction
    fallback = frame.fallback_reason.fillna("") != ""
    frame.loc[fallback, "prediction"] = [
        ref.loc[(row.block, row.origin_s)] for row in frame[fallback].itertuples()
    ]
    frame["loss"] = (frame.prediction - frame.y) ** 2
    scores = frame.groupby(["candidate", "block"]).loss.mean().unstack()
    pooled = frame.groupby("candidate").loss.mean()
    chosen, rows = {}, []
    for candidate in grid:
        eligible = candidate.family != "history" and not failures
        for reference in ("recent_7200s", history_id):
            eligible &= pooled[candidate.id] <= (1 - cfg["validation_margin"]) * pooled[reference]
            eligible &= bool((scores.loc[candidate.id] < scores.loc[reference]).all())
        rows.append(
            {
                "candidate": candidate.id,
                "pooled_mse": float(pooled[candidate.id]),
                "block_0_mse": float(scores.loc[candidate.id, 0]),
                "block_1_mse": float(scores.loc[candidate.id, 1]),
                "selection_eligible": bool(eligible),
                "history": history_id,
                "training_cutoffs": [begin - 36000, begin + 21600 - 36000],
                "validation_begin": begin,
                "validation_end": end,
            }
        )
        if bank_factory is not None:
            rows[-1]["selection_blocked_by_failure"] = bool(failures)
    eligibility = {row["candidate"]: row["selection_eligible"] for row in rows}
    for name, allow in strategies.items():
        options = [c for c in grid if allow(c) and eligibility[c.id]]
        chosen[name] = (
            min(options, key=lambda c: (pooled[c.id], c.tie)).id if options else grid[0].id
        )
    return chosen, history_id, rows, frame.drop(columns="loss").to_dict("records"), failures
