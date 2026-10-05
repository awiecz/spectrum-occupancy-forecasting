"""Generated-data cold/continuous audit fixtures; no recordings or downloads."""

import pandas as pd

from .adaptive_models import candidates as adaptive_grid
from .adaptive_replay import STRATEGIES as ADAPTIVE_STRATEGIES
from .adaptive_replay import replay, select
from .fine_spectrum import write_json
from .residual_state_space import STRATEGIES as STATE_STRATEGIES
from .residual_state_space import StateBank
from .residual_state_space import candidates as state_grid


def synthetic_sources(nodes, cfg, out, scoring_grid, studies=None):
    sources = [
        ("adaptive_learning", adaptive_grid, ADAPTIVE_STRATEGIES, None),
        ("residual_state_space", state_grid, STATE_STRATEGIES, StateBank),
    ]
    for study, factory, strategies, bank_factory in sources:
        if studies is not None and study not in studies:
            continue
        destination = out / "results" / study / "run"
        destination.mkdir(parents=True, exist_ok=False)
        validations, selections, populations = [], [], []
        inherited = {}
        for recording, node in nodes.items():
            grid = scoring_grid(node, recording == "LW1_later")
            for fold, part in grid.groupby("fold"):
                if recording == "LW1_later":
                    result = inherited["LW1_first"]
                else:
                    result = select(
                        node,
                        cfg,
                        float(part.fold_begin_s.iloc[0]),
                        candidate_factory=factory,
                        bank_factory=bank_factory,
                        strategies=strategies,
                    )
                    inherited[recording] = result
                    validations.extend(
                        dict(row, recording=recording, fold=int(fold), synthetic=True)
                        for row in result[3]
                    )
                selections.append(
                    {
                        "recording": recording,
                        "fold": int(fold),
                        "chosen": result[0],
                        "selected_history": result[1],
                        "synthetic": True,
                    }
                )
            origins = node.data.origins(node.data.t0 + 86400, node.data.last, 900, 900)
            issued, _, _ = replay(node, cfg, origins, bank_factory=bank_factory)
            issued.assign(synthetic=True).to_parquet(
                destination / f"{recording}_candidate_forecasts.parquet", index=False
            )
            reference = issued[issued.candidate == "recent_7200s"].merge(grid, on="origin_s")
            populations.append(
                reference.assign(recording=recording, scenario="clean", synthetic=True)
            )
        pd.concat(populations, ignore_index=True).to_parquet(
            destination / "predictions.parquet", index=False
        )
        pd.DataFrame(validations).to_parquet(
            destination / "validation_predictions.parquet", index=False
        )
        write_json(destination / "selections.json", selections)
