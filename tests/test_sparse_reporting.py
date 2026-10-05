import numpy as np
import pandas as pd
import pytest

from spectrum_occupancy_forecasting.experiments import sparse_regime as sr
from spectrum_occupancy_forecasting.experiments import sparse_reporting as rep

CFG = sr.load_config(sr.ROOT / "configs/sparse_regime.yaml")


def synthetic(learned_noise, baseline_noise=0.08, seed=0, scenario="dropout_0.99"):
    """CC1 5 folds x 80 windows; LW1 3 folds (95, 75, 75). Errors scale with the noise args."""
    rng = np.random.default_rng(seed)
    rows, origin = [], 0.0
    plan = [("CC1", fold, 80) for fold in range(5)] + [
        ("LW1", 0, 95),
        ("LW1", 1, 75),
        ("LW1", 2, 75),
    ]
    for node, fold, n in plan:
        origins = origin + 900.0 * np.arange(n)
        origin = origins[-1] + 900.0 * 12
        y = np.clip(0.2 + 0.1 * rng.standard_normal(n), 0, 1)
        base = y + baseline_noise * rng.standard_normal(n)
        cand = y + learned_noise * rng.standard_normal(n)
        for method, p in (
            ("hstar", base),
            ("selected_learned", cand),
            ("ridge", cand),
            ("hgb", cand),
            ("resid_hgb", cand),
            ("fixed_history", base * 1.2),
            ("selected_history", base),
            ("shrunk_history", base),
            ("climatology", np.full(n, 0.2)),
        ):
            rows.append(
                pd.DataFrame(
                    dict(
                        node=node,
                        fold=fold,
                        scenario=scenario,
                        seed=1,
                        method=method,
                        origin_s=origins,
                        y=y,
                        prediction=p,
                    )
                )
            )
    return pd.concat(rows, ignore_index=True)


def evaluate(predictions, delta=0.2):
    cells = rep.cell_table(predictions, CFG, 400, np.random.default_rng(0))
    descriptors = [
        rep.robustness(predictions, CFG, node, region, CFG["primary_scenario"])
        for node, region in (("CC1", "all_folds"), ("LW1", "later_folds"), ("LW1", "all_folds"))
    ]
    return cells, rep.decide(cells, descriptors, delta, CFG)


def test_a_large_consistent_gain_is_supported():
    cells, outcome = evaluate(synthetic(learned_noise=0.02))
    assert outcome["label"] == "large_effect_supported"
    assert all(outcome["tests"].values())


def test_matching_the_baseline_is_an_honest_negative():
    _, outcome = evaluate(synthetic(learned_noise=0.08))
    assert outcome["label"] == "honest_negative"


def test_a_modest_gain_is_directional_only():
    _, outcome = evaluate(synthetic(learned_noise=0.065, baseline_noise=0.08))
    assert outcome["label"] in {"directional_only", "honest_negative"}
    assert not all(outcome["tests"].values())


def test_cell_table_shapes_and_lower_bound_ordering():
    cells, _ = evaluate(synthetic(learned_noise=0.04))
    row = cells[
        (cells["node"] == "CC1")
        & (cells["learned"] == "selected_learned")
        & (cells["comparator"] == "hstar")
        & (cells["loss"] == "squared")
    ].iloc[0]
    assert (
        row["lower95"] <= row["gain"]
        and row["n_windows"] == 400
        and row["n_blocks"] == pytest.approx(12.5)
    )
    later = cells[(cells["node"] == "LW1") & (cells["region"] == "later_folds")]
    assert (later["n_windows"] == 150).all()
    assert set(cells[cells["node"] == "LW1"]["region"]) == {
        "all_folds",
        "later_folds",
        "first_fold",
    }


def test_the_overlap_fold_is_excluded_from_later_folds():
    keep = rep.regions("LW1", 0)["later_folds"]
    assert [keep(f) for f in (0, 1, 2)] == [False, True, True]
    assert set(rep.regions("CC1", 0)) == {"all_folds"}


def test_loss_is_averaged_over_seeds_before_comparison():
    base = synthetic(0.05)
    doubled = pd.concat([base, base.assign(seed=2, prediction=base["y"])], ignore_index=True)
    one = rep.window_losses(base, "squared")["selected_learned"].to_numpy()
    two = rep.window_losses(doubled, "squared")["selected_learned"].to_numpy()
    assert np.allclose(two, one / 2)


def test_interval_scores_reward_calibrated_intervals():
    rng = np.random.default_rng(1)
    n = 400
    y = np.clip(0.3 + 0.1 * rng.standard_normal(n), 0, 1)
    levels = CFG["quantiles"]
    from scipy.stats import norm

    def frame(method, scale):
        q = np.clip(0.3 + scale * norm.ppf(levels)[None, :], 0, 1).repeat(n, axis=0)
        data = pd.DataFrame(q, columns=[f"q{t:g}" for t in levels])
        return data.assign(
            node="CC1", fold=0, scenario="clean", seed=0, method=method, y=y, origin_s=np.arange(n)
        )

    table = rep.interval_table(pd.concat([frame("good", 0.1), frame("narrow", 0.02)]), CFG)
    good = table[table["method"] == "good"].iloc[0]
    narrow = table[table["method"] == "narrow"].iloc[0]
    assert good["coverage_80"] > narrow["coverage_80"]
    assert good["mean_pinball"] < narrow["mean_pinball"]
    assert 0.7 < good["coverage_80"] < 0.9
