# Run and reproduce the project

Run commands from the repository root. Use Python 3.11 or 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

## Start with the small demo

```bash
uv sync --locked
uv run python scripts/run_synthetic_demo.py
```

The demo uses generated measurements and saves example results/figures under
`reproductions/synthetic_demo/`. No recording download is required. Synthetic scores are
not evidence for the research findings.

If a synced folder prevents environment creation, set `UV_PROJECT_ENVIRONMENT` to a location
outside that folder before syncing; for example, in PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:TEMP "spectrum-occupancy-env"
uv sync --locked
```

## Before using real data

This requires the CC1 and LW1 archives: approximately **32 GB of downloads**, plus processed
files and working space. Recordings are excluded from Git. To understand the findings without
downloading data, use the saved outputs linked from the [results overview](results.md).

## Reproduce the data study

The [illustrated data study](data_study.md) can be read entirely from its saved figures and
tables, with no installation or recording download. To rehearse the analysis on generated data:

```bash
uv run python scripts/study_data.py --mode synthetic-rehearsal --out-dir reproductions/data_demo
```

For real-data reproduction, use the existing CC1/LW1 processed block files described below:

```bash
uv run python scripts/study_data.py --mode exploratory --out-dir reproductions/data_study_new
```

Optional `--cc1`, `--lw1` and `--config` arguments override input locations and the shared
`configs/followup_v2.yaml` specification. Inputs must retain their CC1/LW1 block-file names.
The study audits all sessions and analyzes session 0, using first-day calibration and complete,
non-overlapping 15-minute windows. Output directories must be new; CC2 inputs are rejected.
This script fits no forecasting models and never modifies recordings.

Each output includes six PNG figure groups, CSV source tables, `summary.json` and a provenance
manifest with input/configuration/source hashes and completion status. Figures use only those
saved tables. Outputs are limited to 10 MiB; synthetic fixtures are removed after analysis.

Recording and artifact checksums cover exact bytes; Git preserves data-study artifact bytes.
New runs normalize CRLF to LF when hashing configuration and Python source, so fingerprints
are portable between Windows and Linux. The published study's original manifest is preserved;
its [publication audit](../results/data_study/publication_audit.json) adds portable fingerprints
of the original run sources and records the line-ending repair without implying a new analysis.

## Reproduce the forecasting experiments

Follow [download, checksum and extraction instructions](technical/reproduction.md#1-download-and-verify)
to produce `data/processed/aerpaw_CC1_blocks5MHz.npz` and
`data/processed/aerpaw_LW1_blocks5MHz.npz`. Then choose fresh output directories:

```bash
uv run python scripts/run_followup.py --mode exploratory --out reproductions/study_new
uv run python scripts/run_history_mixture.py --mode exploratory --out reproductions/mixture_new
```

The first command runs the chronological model/history comparison and sensitivity analyses.
The second runs the complete three-regime history-mixture experiment. Neither accesses CC2.
Reruns of these previously inspected recordings remain exploratory.

For a download-free check of either pipeline, replace `--mode exploratory` with
`--mode synthetic-rehearsal`, still choosing a new output directory. The technical guide explains
[input overrides, report regeneration and saved artifacts](technical/followup_v2_reproduction.md).
Commands for original window-width and transfer experiments are in the
[original-study reference](technical/reproduction.md#3-run-independent-reproductions).

## Adaptive local learning

The [adaptive study](adaptive_learning.md) uses the three already-inspected recordings.
Completed feedback updates local model states; conservative earlier validation selects their
settings. Later LW1 is now development data. CC2 remains untouched.

```bash
uv run python scripts/run_adaptive_learning.py --mode synthetic-rehearsal --out-dir reproductions/adaptive_demo
uv run python scripts/run_adaptive_learning.py --mode report --predictions-dir results/adaptive_learning/run --out-dir reproductions/adaptive_report
```

Both commands require fresh output directories. The rehearsal downloads nothing and visibly
labels generated outputs. Report mode reads saved predictions and validation records without
refitting models. Its metrics, bootstrap intervals and advancement decision reproduce the study.

For existing checksum-verified fine-spectrum caches:

```bash
uv run python scripts/run_adaptive_learning.py --mode audit --out-dir reproductions/adaptive_audit
uv run python scripts/run_adaptive_learning.py --mode exploratory --cache reproductions/shared_learning_cache --out-dir reproductions/adaptive_new
```

The local historical audit also needs the verified `reproductions/adaptive_archive_initial`
archive produced during cleanup. The cache must contain `CC1_session0_development`,
`LW1_session0_development` and the previously
inspected `LW1_session2_assessment`, with completed manifests and valid shard checksums. This
runner never opens raw archives, downloads data or accesses CC2. A fresh clone can run the
synthetic and report workflows immediately; raw-to-cache provenance is retained in the original
shared-learning manifests and Git commit `f11e47a`. Archived standalone experiments are no longer
supported entry points. The local cleanup archive contains their exact implementations and outputs.

The original adaptive run manifest and fitting-source snapshot remain unchanged. A separately
recorded report regeneration improved figure readability and interval-line colors; all numerical
tables and advancement decisions agree. Use `results/adaptive_learning/publication_audit.json`
for the current published-file checksums and `audit/report_manifest.json` for reporting provenance.

## Residual diagnosis and state-space follow-up

The [targeted follow-up](residual_state_space.md) starts with saved forecasts, not another
model search. All output directories must be new. The first two commands need no recordings:

```bash
uv run python scripts/run_residual_state_space.py --mode diagnose --out-dir reproductions/residual_diagnosis
uv run python scripts/run_residual_state_space.py --mode synthetic-rehearsal --out-dir reproductions/residual_demo
```

Real replay uses only the three existing verified caches described above. It requires a
passing diagnosis whose specification and input hashes still match. The runner recomputes
the gate before reading caches; it rejects CC2 and never reads raw archives or downloads data.

```bash
uv run python scripts/run_residual_state_space.py --mode exploratory --diagnosis-dir reproductions/residual_diagnosis --out-dir reproductions/residual_new
uv run python scripts/run_residual_state_space.py --mode report --predictions-dir results/residual_state_space/run --out-dir reproductions/residual_report
```

`--config`, `--cache`, and `--source-adaptive-dir` accept explicit paths. Report mode
reproduces numerical tables from saved forecasts and validation predictions without fitting.
Filter updates and state snapshots additionally expose the delivered anchors and covariance.
Synthetic runs are workflow checks and are not evidence of measurement-data performance.

## Operational learning (declared in the registry)

The [operational study](operational_learning.md) audits validation and controllers before
testing one declared activity-count grid. Existing recordings are development data. These
commands download nothing; all output directories must be fresh.

```bash
uv run python scripts/run_operational_learning.py --phase count-model --mode synthetic-rehearsal --out-dir reproductions/count_numerical
uv run python scripts/run_operational_learning.py --phase validation-audit --mode synthetic-rehearsal --out-dir reproductions/validation_demo
uv run python scripts/run_operational_learning.py --phase controller-audit --mode synthetic-rehearsal --out-dir reproductions/controller_demo
uv run python scripts/run_operational_learning.py --phase count-model --mode report --predictions-dir results/operational_learning/run --out-dir reproductions/count_report
uv run python scripts/run_operational_learning.py --phase validation-audit --mode report --predictions-dir results/operational_learning/audits/validation --out-dir reproductions/validation_report
uv run python scripts/run_operational_learning.py --phase controller-audit --mode report --predictions-dir results/operational_learning/audits/controller --out-dir reproductions/controller_report
```

Rehearsal checks numerical convergence on synthetic persistent, changing, missing and random
activity, including all-idle/all-busy cases. It selected 1,601 nodes before real execution.
Report commands use saved predictions and need no recordings or fitting. Validation report
regeneration also reads the retained adaptive/residual prediction files in this repository.

The count model's 439/239/81 scoring population, frozen labels, 24 configurations, selection
margin, controller and degradation seeds are fixed by `configs/operational_learning.yaml`
and its retained base configurations. `--config`, `--cache`, `--registry`, `--parent-attempt`,
`--hypothesis`, `--evidence`, `--attempt-kind`, `--reproduction`, `--numerical-dir` and
`--out-dir` make declarations and inputs explicit. Exploratory mode requires the three
checksum-verified caches described above; it never downloads or extracts recordings.

Inspect the registry before proposing another attempt:

```bash
uv run python scripts/run_operational_learning.py --registry-command list
uv run python scripts/run_operational_learning.py --registry-command compare --out-dir reproductions/registry_comparison
uv run python scripts/run_operational_learning.py --registry-command explain --attempt 44e1867afab747ee926510085d49e3f7
```

The completed count execution is `44e1867afab747ee926510085d49e3f7`. Its parent is an
interrupted execution of the same grid, retained rather than erased. Re-running an identical
scientific declaration requires `--reproduction <original-attempt>`; the runner then compares
every saved numerical outcome. A changed implementation also needs an explanation through
`--evidence`. A renamed config or changed hypothesis wording cannot bypass duplicate checks.
For a full reproduction, use the count-model exploratory phase, the matching caches, a
fresh numerical rehearsal, that reproduction reference and a fresh local output directory.
No further candidate search is authorized by these commands.

Registry recovery is deliberately explicit. Inspect the process and partial outputs first;
then use `--registry-command recover --attempt <id> --recovery-reason "<explanation>"`.
Use `--recover-lock` only after verifying the lock owner is no longer running. Incomplete tails
are preserved in a recovery backup, and recovery appends an event instead of hiding the interruption.
`--registry-command migrate` imports historical ledger/provenance records idempotently; missing
historical settings remain explicitly unknown.

Original fitting manifests and prediction bytes stay unchanged. Publication receipts separately
record enhanced reporting, checksum verification and visual review. Local detailed/partial outputs
and checkpoints are under `reproductions/operational_archive_20261005`; no raw recordings are copied.
Scientific implementation fingerprints use executable ASTs and exclude presentation functions;
original source-byte fingerprints are retained alongside them. The count fitting manifest's
configuration fingerprint describes the declared 801-node starting specification, while its
`artifact_hashes["configuration.json"]` verifies the resolved 1,601-node configuration.

## Power analysis and the multi-band draft

The [power analysis](power_analysis.md) reads the tracked saved forecasts only, so it needs no
download. Output directories must be new:

```bash
uv run python scripts/power_analysis.py --out-dir reproductions/power_check
uv run python scripts/power_analysis.py --candidates count --out-dir reproductions/power_count_only
```

The draft [multi-band planning tables](multiband_protocol.md) need the cached block files in
`data/processed/` and fit no model:

```bash
uv run python scripts/multiband_inventory.py --out-dir reproductions/multiband_check
```

## Sparse-regime study

The [sparse-regime study](sparse_regime.md) asks whether learned models help when most
measurements are missing. Its [protocol](technical/sparse_regime_protocol.md) was fixed before the
real run. The first three commands work from a fresh clone; the last two need the block files and
write to the experiment registry:

```bash
uv run python scripts/run_sparse_regime.py --phase power --out-dir reproductions/sparse_power
uv run python scripts/run_sparse_regime.py --phase matched --mode synthetic-rehearsal --out-dir reproductions/sparse_rehearsal
uv run python scripts/run_sparse_regime.py --phase matched --mode report --out-dir results/sparse_regime/run
```

A real run (never repeat one without declaring it) is
`--phase matched --mode exploratory --hypothesis "..."`; it appends to
`results/experiment_registry.jsonl` and refuses an existing output directory unless `--resume`
continues an interrupted one. Synthetic rehearsals are workflow checks, not evidence.

## Rebuilding the cached learning inputs

Several adaptive and operational modes read `reproductions/shared_learning_cache` (about 15 MB,
Git-ignored). Rebuild it from the raw archives (about 32 GB of downloads) with:

```bash
uv run python scripts/extract_learning_features.py
```

To summarize a block file's sessions and busiest blocks before modelling, use
`uv run python scripts/explore_blocks.py data/processed/aerpaw_CC1_blocks5MHz.npz`.

## What works from a fresh clone

| You want to | Works without data? |
|---|---|
| Run the synthetic demo and every `--mode synthetic-rehearsal` | Yes |
| Rebuild reports from saved predictions (`--mode report`) | Yes |
| Rerun the power analysis and the sparse-regime power phase | Yes |
| Run tests, lint, link and claim checks | Yes |
| Rerun an `exploratory` study from recordings | No: needs the processed block files |
| Run `--mode audit` of the adaptive study | No: needs the local, Git-ignored `reproductions/adaptive_archive_initial` |
| Use `--cache reproductions/shared_learning_cache` | No: rebuild it first (above) |

## Check the code

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_links.py
uv run python scripts/check_claims.py --check-table
```

Tests use generated data and check causal timing, selection, fallback behavior, saved metrics
and run protections. See the [technical reference](technical/index.md) for provenance limitations
and the pending CC2 protocol. Source/layout changes invalidate earlier local freeze packages;
a fresh matching package is required before future independent archival or CC2 access.
