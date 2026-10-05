"""Selection from immutable issued forecasts; protocol choice is explicit."""

import numpy as np
import pandas as pd

from .activity_counts import CountBank, candidates, count_bins
from .adaptive_replay import replay
from .mixture_support import history_candidates

STRATEGIES = {"count": lambda c: True}


def choose(frame, cfg, grid, strategies, *, fixed_fallback=False, failures=()):
    frame = frame.copy()
    frame["loss"] = (frame.prediction - frame.y) ** 2
    pooled = frame.groupby("candidate").loss.mean()
    histories = history_candidates(cfg)
    history = min(histories, key=lambda h: (pooled[h["id"]], histories.index(h)))["id"]
    reference = (
        frame[frame.candidate == ("recent_7200s" if fixed_fallback else history)]
        .set_index(["block", "origin_s"])
        .prediction
    )
    fallback = frame.fallback_reason.fillna("") != ""
    frame.loc[fallback, "prediction"] = [
        reference.loc[(r.block, r.origin_s)] for r in frame[fallback].itertuples()
    ]
    frame["loss"] = (frame.prediction - frame.y) ** 2
    pooled = frame.groupby("candidate").loss.mean()
    blocks = frame.groupby(["candidate", "block"]).loss.mean().unstack()
    rows = []
    for c in grid:
        eligible = c.family != "history" and not failures
        for ref in ("recent_7200s", history):
            eligible &= bool(
                np.isfinite(pooled[c.id])
                and pooled[c.id] <= (1 - cfg["validation_margin"]) * pooled[ref]
                and (blocks.loc[c.id] < blocks.loc[ref]).all()
            )
        rows.append(
            {
                "candidate": c.id,
                "pooled_mse": float(pooled[c.id]),
                "block_0_mse": float(blocks.loc[c.id, 0]),
                "block_1_mse": float(blocks.loc[c.id, 1]),
                "selection_eligible": bool(eligible),
                "history": history,
                "selection_blocked_by_failure": bool(failures),
            }
        )
    allowed = {r["candidate"]: r["selection_eligible"] for r in rows}
    selected = {}
    for name, predicate in strategies.items():
        options = [c for c in grid if predicate(c) and allowed[c.id]]
        selected[name] = (
            min(options, key=lambda c: (pooled[c.id], c.tie)).id if options else grid[0].id
        )
    return selected, history, rows, frame.drop(columns="loss").to_dict("records"), list(failures)


def count_replay(node, cfg, origins, **kwargs):
    bins = count_bins(node, cfg, kwargs.get("deg"), kwargs.get("seed", 0))
    startup = node.data.t0 + 86400
    return replay(
        node,
        cfg,
        origins,
        bank_factory=lambda *_: CountBank(cfg, bins, startup, active=kwargs.get("active")),
        **kwargs,
    )


class ContinuousSelector:
    def __init__(self):
        self.saved = {}

    def __call__(self, node, cfg, evaluation):
        end = evaluation - cfg["selection_embargo_s"]
        begin = end - 43200
        if begin < node.data.t0 + 86400:
            return (
                {"count": candidates(cfg)[0].id},
                "recent_7200s",
                [],
                [],
                "insufficient_validation",
            )
        key = id(node)
        # Replay only up to the latest validation end, never future evaluation outcomes.
        origins = node.data.origins(node.data.t0 + 86400, end, 900, cfg["replay_stride_s"])
        records, _, bank = count_replay(node, cfg, origins)
        self.saved[key] = bank.checkpoint()
        blocks = []
        for b in (0, 1):
            lo, hi = begin + b * 21600, begin + (b + 1) * 21600
            available = node.data.origins(lo, hi, 900, cfg["replay_stride_s"])
            origins_b, y, n = node.data.dataset(available, 900)
            labels = pd.DataFrame({"origin_s": origins_b, "y": y, "target_count": n})
            part = records.merge(labels, on="origin_s", validate="many_to_one").assign(block=b)
            if part.empty:
                return (
                    {"count": candidates(cfg)[0].id},
                    "recent_7200s",
                    [],
                    [],
                    "insufficient_validation",
                )
            blocks.append(part)
        result = choose(
            pd.concat(blocks, ignore_index=True),
            cfg,
            candidates(cfg),
            STRATEGIES,
            fixed_fallback=True,
            failures=bank.failures,
        )
        for row in result[2]:
            row.update(
                validation_begin=begin,
                validation_end=end,
                initialization_cutoff=node.data.t0 + 86400,
                protocol="continuous-operational",
            )
        return result
