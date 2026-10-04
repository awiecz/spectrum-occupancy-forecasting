# Documentation index and repository map

New here? Read in this order (about 20 minutes):

1. [research_note/research_note.pdf](research_note/research_note.pdf): 3-page summary, results and
   open questions.
2. [../README.md](../README.md): headline findings, figures, how to reproduce.
3. [sanity_checks.md](sanity_checks.md): is the signal real, and how much independent data is there?
4. [test_results.md](test_results.md), [lw1_results.md](lw1_results.md),
   [transfer_results.md](transfer_results.md): the three one-shot evaluations, each scored against
   predictions registered beforehand (including the ones that failed).

## What is where

| Path | Contents |
|---|---|
| `src/spectrum_occupancy_forecasting/` | The library: `data/` (AERPAW reader, dwell schema, synthetic generator for tests), `occupancy/` (power units, detection), `features/` (causal age-aware features, window targets), `models/` (baselines, learned models), `evaluation/` (splits with embargo, metrics, bootstrap, diagnostics), `robustness/` (degradations of predictor inputs only), `experiments/` (the window-forecast experiment, transfer, one-shot protocol locks), `viz/` (figures) |
| `scripts/` | Entry points: `extract_aerpaw_blocks.py` (archive to 5 MHz block product), `run_window_dev.py`, `run_test_once.py`, `run_second_node.py`, `run_transfer.py` (the experiments), `make_figures.py`, `sanity_checks.py`, `explore_blocks.py`, `sha256.py`, `run_dev_baselines.py` (superseded, kept for the record) |
| `tests/` | Units, labelling, leakage and causality checks, degradations, locks, transfer, scripts smoke tests. All on synthetic data. |
| `configs/frozen_experiment.yaml` | The frozen design (asserted equal to the code constants by the test script) |
| `docs/` | This index, protocols (pre-registered), results write-ups, dataset investigation, the research note, and the historical original plan |
| `results/` | Result JSON, run logs, lock files (code fingerprints), figures with data tables |
| `data/` | Git-ignored: raw archives and derived block products (not redistributed) |

## Pre-registered protocols and locks

`test_protocol.md`, `lw1_protocol.md` and `transfer_protocol.md` were written before the corresponding
results existed and are **not edited afterwards** (their text is fingerprinted into
`results/*_RUN_LOCK.json`, together with the code). Each one-shot run refuses to execute twice. That is
why the protocol files still say "Wi-Fi channel 6" while everything else now says "channel-6 band
(consistent with Wi-Fi, not decoded)"; see [sanity_checks.md](sanity_checks.md).

## The original plan versus what happened

[original_plan.md](original_plan.md) is the approved plan, kept as a historical record.

| Planned | What happened |
|---|---|
| NIST COVID-19 occupancy data as the primary dataset (natively sparse) | NIST's file server returned errors for the whole project (metadata and DOI worked); switched to the AERPAW fixed-node archives. **Sparsity is therefore simulated**, by degrading predictor inputs |
| Target: occupancy of the next dwell/sweep | Single-sweep labels were nearly unpredictable (persistence worse than climatology); the target became the **busy fraction of the next 5/15/60 minutes**, a post-hoc change disclosed in `dev_results.md` |
| One chronological train/dev/test split | Rolling one-day folds on the development region, then a one-time held-out test region |
| One site/node | Second node (LW1) as a pre-registered confirmation run, plus a cross-node transfer analysis |
| Classical models only; GRU/HMM deferred | Kept classical; added "last 2 h + learned correction" models |
| Command-line tool `rfsi fetch/process/run/report` | Dropped; the scripts are the interface |
| Notebooks | Not produced; figures come from `scripts/make_figures.py` |
| Research note, 1-3 pages | Done, 3 pages |
| (not planned) | Self-audit (`sanity_checks.md`): the signal is real and channel-shaped, but the effective sample size is small and intervals are probably too narrow |

## Known gaps

No natively sparse data; one week per node and one band; the activity was not decoded (not verified to
be Wi-Fi); per-bin power is uncalibrated; the cause of boosting's weakness on CC1 and of ridge's failure
in reverse transfer is untested. See the "Limits" sections of the write-ups.
