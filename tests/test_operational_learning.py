"""Scientific and registry invariants for the fixed count-model experiment."""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betabinom, binom
from threadpoolctl import threadpool_limits

from spectrum_occupancy_forecasting.experiments.activity_counts import (
    CountBank,
    CountFilter,
    candidates,
    count_bins,
    log_likelihood,
    transition,
)
from spectrum_occupancy_forecasting.experiments.learning_inputs import LearningNode
from spectrum_occupancy_forecasting.experiments.operational_reporting import regenerate
from spectrum_occupancy_forecasting.experiments.operational_selection import (
    STRATEGIES,
    ContinuousSelector,
    choose,
    count_replay,
)
from spectrum_occupancy_forecasting.experiments.registry import (
    Registry,
    digest,
    implementation_digest,
    migrate,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "operational_runner", ROOT / "scripts/run_operational_learning.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def setup():
    cfg = runner.configuration(ROOT / "configs/operational_learning.yaml")
    cfg["grid_nodes"] = (
        101  # fast invariant tests; convergence tested separately at declared resolution
    )
    with threadpool_limits(2):
        yield cfg, runner.adaptive.synthetic_frame("CC1", 4, 102)


@pytest.mark.parametrize("rho", [0, 0.1, 0.000001])
@pytest.mark.parametrize("counts", [(0, 20), (7, 20), (20, 20), (0, 0)])
def test_likelihood_direct_calculation(rho, counts):
    logits = np.array([-8.0, -1.0, 0.0, 2.0, 8.0])
    p = 1 / (1 + np.exp(-logits))
    k, n = counts
    expected = (
        binom.logpmf(k, n, p)
        if rho == 0
        else betabinom.logpmf(k, n, p * (1 / rho - 1), (1 - p) * (1 / rho - 1))
    )
    np.testing.assert_allclose(log_likelihood(logits, k, n, rho), expected, atol=1e-8)
    if rho == 0.000001:
        # Near p=0/1, convergence requires rho much smaller than p/n.
        np.testing.assert_allclose(expected[1:4], binom.logpmf(k, n, p)[1:4], atol=0.003)


def test_transition_normalization_and_forecast_nonmutation():
    a, b = CountFilter(-1, 2, 1, 0.1), CountFilter(-1, 2, 1, 0.1)
    for time, k, n in [(300, 0, 1), (900, 20, 20), (2000, 7, 13), (7000, 0, 20)]:
        a.assimilate(time, k, n)
        b.assimilate(time, k, n)
        saved = a.posterior.copy(), a.time
        assert 0 <= a.predict([time + 1200, time + 1500, time + 1800]) <= 1
        a.predict([time + 40000])
        np.testing.assert_array_equal(a.posterior, saved[0])
        assert a.time == saved[1]
        np.testing.assert_array_equal(a.posterior, b.posterior)
        assert np.isclose(a.posterior.sum(), 1)
    matrix = transition(801, 16, -1, 2, 1, 137.0)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=0)), 1, atol=1e-14)
    resumed = pickle.loads(pickle.dumps(a))
    assert resumed.predict([8000]) == a.predict([8000])
    with pytest.raises(ValueError):
        a.predict([6999])


def test_bins_boundaries_delay_missingness_and_calibration(setup):
    cfg, frame = setup
    node = LearningNode(frame, "CC1", cfg)
    bins = count_bins(node, cfg)
    assert bins.iloc[0].total_count == 5
    assert bins.iloc[0].event_s == node.data.t0 + 300
    delayed = count_bins(node, cfg, Degradation(delay_s=600))
    post = delayed.event_s > node.data.t0 + 86400
    np.testing.assert_array_equal(
        delayed.loc[post, "release_s"], delayed.loc[post, "event_s"] + 600
    )
    dropped = count_bins(node, cfg, Degradation(dropout_p=0.99), 5)
    np.testing.assert_array_equal(dropped.loc[~post, "total_count"], bins.loc[~post, "total_count"])
    assert not dropped.loc[post, "usable"].any()
    assert (dropped.loc[post, "total_count"] < bins.loc[post, "total_count"]).all()
    assert (
        node.data.thresholds
        == LearningNode(
            frame.assign(
                target_dbm=np.where(frame.time_s > node.data.t0 + 86400, 999, frame.target_dbm)
            ),
            "CC1",
            cfg,
        ).data.thresholds
    )


def test_grid_controller_bounds_and_insufficient_initialization(setup):
    cfg, frame = setup
    grid = candidates(cfg)
    assert len(grid) == 25 and len({c.key for c in grid[1:]}) == 12
    assert min(grid, key=lambda c: c.tie).family == "history"
    node = LearningNode(frame, "CC1", cfg)
    bank = CountBank(cfg, count_bins(node, cfg), node.data.t0 + 86400)
    for t in np.arange(node.data.t0 + 86400 - 3600, node.data.t0 + 86400, 900):
        bank.update(t, t, 0.2, 0.2, None, None)
    time = node.data.t0 + 86400
    for age, coverage in [(901, 1), (0, 0.49), (np.nan, 1)]:
        rows = bank.forecasts(time, 0.234, 0.999, None, {}, age, coverage)
        assert all(r["prediction"] == 0.234 for r in rows)
    rows = bank.forecasts(time, 0.234, 0.999, None, {}, 0, 1)
    assert all(0 <= r["prediction"] <= 1 for r in rows)
    assert rows[0]["prediction"] == 0.234
    insufficient = CountBank(cfg, count_bins(node, cfg).iloc[:47], time)
    assert insufficient.insufficient
    assert all(
        r["prediction"] == 0.234 for r in insufficient.forecasts(time, 0.234, 0.999, None, {}, 0, 1)
    )


@pytest.mark.parametrize("site", ["CC1", "LW1"])
def test_future_corruption_checkpoint_and_feedback_isolation(setup, site):
    cfg, frame = setup
    node = LearningNode(frame, site, cfg)
    origins = node.data.t0 + 86400 + np.arange(12) * 900
    first, feedback, _ = count_replay(node, cfg, origins)
    corrupt = frame.copy()
    corrupt.loc[corrupt.time_s >= origins[-1] + 0.001, "target_dbm"] = 1000
    other = LearningNode(corrupt, site, cfg)
    second, _, _ = count_replay(other, cfg, origins)
    pd.testing.assert_frame_equal(first, second)
    partial, _, bank = count_replay(node, cfg, origins, stop_after=5)
    resumed, feedback2, _ = count_replay(node, cfg, origins, checkpoint=bank.checkpoint())
    pd.testing.assert_frame_equal(first, resumed)
    pd.testing.assert_frame_equal(feedback, feedback2)
    assert len(partial) < len(first)
    delayed, events, _ = count_replay(node, cfg, origins, deg=Degradation(delay_s=1800))
    assert (events.release_s >= events.target_end_s + 1800).all()
    assert (delayed[delayed.candidate.str.startswith("count_")].prediction.between(0, 1)).all()


def test_continuous_validation_and_future_selection(setup):
    cfg, frame = setup
    node = LearningNode(frame, "CC1", cfg)
    evaluation = node.data.t0 + 172800
    result = ContinuousSelector()(node, cfg, evaluation)
    frame.loc[frame.time_s >= evaluation - 36000, "target_dbm"] = 900
    other = ContinuousSelector()(LearningNode(frame, "CC1", cfg), cfg, evaluation)
    assert result[:3] == other[:3] and result[4] == other[4]
    pd.testing.assert_frame_equal(pd.DataFrame(result[3]), pd.DataFrame(other[3]))
    assert all(r["validation_end"] == evaluation - 36000 for r in result[2])
    # The saved validation predictions match one uninterrupted calibration-start replay.
    records, _, _ = count_replay(
        node, cfg, node.data.origins(node.data.t0 + 86400, evaluation - 36000, 900, 900)
    )
    validation = pd.DataFrame(result[3])
    active = validation.fallback_reason.fillna("") == ""
    joined = validation[active].merge(
        records[["candidate", "origin_s", "prediction"]],
        on=["candidate", "origin_s"],
        suffixes=("", "_replay"),
    )
    np.testing.assert_array_equal(joined.prediction, joined.prediction_replay)


def test_registry_identity_reproduction_lineage_recovery(tmp_path):
    registry = Registry(tmp_path / "registry.jsonl")
    spec = {"model": "count", "parameters": {"half_life": 2, "sd": 0.5}}
    attempt = registry.declare(spec, "implementation", hypothesis="test")
    with pytest.raises(ValueError, match="duplicate"):
        registry.declare(
            {"parameters": spec["parameters"], "model": "count"},
            "implementation",
            hypothesis="renamed",
        )
    reproduction = registry.declare(
        spec, "implementation", hypothesis="repeat", reproduction=attempt
    )
    assert reproduction != attempt
    with pytest.raises(ValueError, match="explanation"):
        registry.declare(spec, "changed implementation", hypothesis="repeat", reproduction=attempt)
    revised = registry.declare(
        spec,
        "changed implementation",
        hypothesis="repeat",
        reproduction=attempt,
        evidence="verified maintenance of reporting",
    )
    assert revised != attempt
    with pytest.raises(ValueError, match="parent"):
        registry.declare(
            {"model": "count", "parameters": {}},
            "implementation",
            hypothesis="revision",
            kind="parameter-revision",
        )
    registry.append({"event": "status", "attempt": attempt, "status": "running"})
    assert registry.stale(hours=0)[0]["attempt"] == attempt
    original = registry.path.read_bytes()
    with registry.path.open("ab") as stream:
        stream.write(b'{"incomplete":')
    with pytest.raises(ValueError, match="incomplete"):
        registry.read()
    registry.recover("explicit test recovery", attempt)
    assert registry.path.read_bytes().startswith(original)
    assert registry.read()[-1]["status"] == "interrupted"
    assert digest(spec) == digest(dict(reversed(list(spec.items()))))


def test_registry_serializes_concurrent_appends_and_seed_lineage(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    registry = Registry(tmp_path / "registry.jsonl")
    with ThreadPoolExecutor(4) as workers:
        list(workers.map(lambda n: registry.append({"event": "test", "sequence": n}), range(20)))
    assert {r["sequence"] for r in registry.read()} == set(range(20))
    specification = {
        "model_family": "counts",
        "inputs": ["input"],
        "protocol": "continuous",
        "seed": 1,
    }
    first = registry.declare(specification, "implementation", hypothesis="first")
    with pytest.raises(ValueError, match="seed"):
        registry.declare(specification | {"seed": 2}, "implementation", hypothesis="same idea")
    second = registry.declare(
        specification | {"seed": 2},
        "implementation",
        hypothesis="paired replication",
        kind="seed-replication",
        parent=first,
    )
    assert second != first


def test_historical_migration_preserves_source(tmp_path):
    ledger = tmp_path / "results/old/experiment_ledger.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"experiment,status\nfailed_model,failed\n")
    before = ledger.read_bytes()
    registry = Registry(tmp_path / "registry.jsonl")
    migrate(registry, tmp_path)
    assert ledger.read_bytes() == before
    assert registry.read()[0]["historical_metadata"]["status"] == "failed"
    assert registry.read()[0]["missing_metadata"]
    migrate(registry, tmp_path)
    assert len(registry.read()) == 1


def test_scientific_grid_rejects_config_routing(tmp_path):
    config = json.loads(
        json.dumps(runner.configuration(ROOT / "configs/operational_learning.yaml"))
    )
    assert len(candidates(config)) == 25
    path = tmp_path / "wrong.yaml"
    path.write_text("study: adaptive-learning\n")
    with pytest.raises(ValueError, match="mismatch"):
        runner.configuration(path)


def test_implementation_identity_ignores_documentation_and_plot_changes(tmp_path):
    source = tmp_path / "model.py"
    source.write_text(
        '"""first documentation"""\ndef update(x):\n    """doc"""\n    return x+1\n'
        'def plot():\n    return "red"\n'
    )
    selected = {"model.py": ["update"]}
    before = implementation_digest([source], tmp_path, selected)
    source.write_text(
        '"""different documentation"""\n# comment\ndef update(x):\n    """new doc"""\n'
        '    return x + 1\ndef plot():\n    return "blue"\n'
    )
    assert implementation_digest([source], tmp_path, selected) == before
    source.write_text("def update(x):\n    return x+2\n")
    assert implementation_digest([source], tmp_path, selected) != before


def test_synthetic_persistence_helps_but_random_future_has_no_information():
    rng = np.random.default_rng(113)
    state = CountFilter(np.log(0.1 / 0.9), 0.5, 1, 0.1)
    probabilities = np.r_[np.full(30, 0.1), np.full(30, 0.65), np.full(30, 0.1)]
    issued, histories, target = [], [], []
    observed = []
    for j, p in enumerate(probabilities):
        count = rng.binomial(100, p)
        observed.append(count / 100)
        state.assimilate((j + 1) * 300, count, 100)
        if j >= 24 and j + 6 < len(probabilities):
            issued.append(state.predict([(j + 5) * 300, (j + 6) * 300, (j + 7) * 300]))
            histories.append(np.mean(observed[-24:]))
            target.append(np.mean(probabilities[j + 4 : j + 7]))
    assert np.mean((np.array(issued) - target) ** 2) < np.mean((np.array(histories) - target) ** 2)
    random_state = CountFilter(0, 0.5, 0.5, 0.1)
    forecasts, outcomes = [], []
    for j in range(500):
        random_state.assimilate((j + 1) * 300, rng.binomial(60, 0.5), 60)
        forecasts.append(random_state.predict([(j + 7) * 300]))
        outcomes.append(rng.binomial(60, 0.5) / 60)  # independent future evidence
    learned_loss = np.mean((np.array(forecasts) - outcomes) ** 2)
    oracle_loss = np.mean((np.array(outcomes) - 0.5) ** 2)
    assert learned_loss >= oracle_loss - 0.001


def test_constant_labels_report_unavailable_skill_without_promotion(setup, tmp_path):
    cfg, _ = setup
    source, out = tmp_path / "source", tmp_path / "report"
    source.mkdir()
    out.mkdir()
    records = []
    grid = candidates(cfg)
    from spectrum_occupancy_forecasting.experiments.mixture_support import history_candidates

    for block in (0, 1):
        for origin in (1600000000 + block * 21600, 1600000900 + block * 21600):
            for c in grid:
                records.append(
                    {
                        "candidate": c.id,
                        "origin_s": origin,
                        "y": 0.0,
                        "target_count": 15,
                        "block": block,
                        "fallback_reason": "",
                        "prediction": 0.2 * c.strength,
                        "fixed_history": 0.0,
                        "selected_history": 0.0,
                    }
                )
            records.extend(
                {
                    "candidate": h["id"],
                    "origin_s": origin,
                    "y": 0.0,
                    "target_count": 15,
                    "block": block,
                    "fallback_reason": "",
                    "prediction": 0.0,
                    "fixed_history": 0.0,
                    "selected_history": 0.0,
                }
                for h in history_candidates(cfg)
            )
    chosen, history, scores, validation, _ = choose(
        pd.DataFrame(records), cfg, grid, STRATEGIES, fixed_fallback=True
    )
    selections, all_scores, vals, predictions, feedback = [], [], [], [], []
    for recording in ("CC1_first", "LW1_first", "LW1_later"):
        selections.append(
            {
                "recording": recording,
                "fold": 0,
                "chosen": chosen,
                "selected_history": history,
                "failure": [],
                "selection_source": "LW1_first_final_fold"
                if recording == "LW1_later"
                else recording,
            }
        )
        all_scores.extend(dict(r, recording=recording, fold=0) for r in scores)
        if recording != "LW1_later":
            vals.extend(dict(r, recording=recording, fold=0) for r in validation)
        origins = 1600000000 + np.arange(8) * 900
        issued = []
        for origin in origins:
            issued.extend(
                {
                    "candidate": c.id,
                    "origin_s": origin,
                    "prediction": 0.2 * c.strength,
                    "fallback_reason": "",
                    "fixed_history": 0.0,
                }
                for c in grid
            )
            issued.extend(
                {
                    "candidate": h["id"],
                    "origin_s": origin,
                    "prediction": 0.0,
                    "fallback_reason": "",
                    "fixed_history": 0.0,
                }
                for h in history_candidates(cfg)
            )
            predictions.extend(
                {
                    "recording": recording,
                    "fold": 0,
                    "origin_s": origin,
                    "y": 0.0,
                    "prediction": 0.0,
                    "target_count": 15,
                    "candidate": grid[0].id,
                    "strategy": "count",
                    "method": method,
                    "scenario": "clean",
                    "seed": 0,
                    "synthetic": True,
                    "period": "adaptation",
                    "correction": 0.0,
                    "fallback_reason": "",
                }
                for method in ("adaptive", "controller", "fixed_history", "selected_history")
            )
            feedback.append(
                {
                    "recording": recording,
                    "scenario": "clean",
                    "seed": 0,
                    "origin_s": origin,
                    "target_end_s": origin + 1800,
                    "release_s": origin + 1800,
                    "update_reason": "updated",
                }
            )
        pd.DataFrame(issued).to_parquet(
            source / f"{recording}_candidate_forecasts.parquet", index=False
        )
    for name, values in (
        ("candidate_scores", all_scores),
        ("selections", selections),
        ("failures", []),
    ):
        (source / f"{name}.json").write_text(json.dumps(values))
    for name, values in (
        ("predictions", predictions),
        ("validation_predictions", vals),
        ("feedback", feedback),
    ):
        pd.DataFrame(values).to_parquet(source / f"{name}.parquet", index=False)
    regenerate(source, out, cfg)
    decision = json.loads((out / "advancement.json").read_text())
    assert decision["status"] == "not_qualified"
    assert all(
        r["clean_skill_fixed"] is None and not r["clean_pass"]
        for r in decision["strategies"][0]["recordings"]
    )
