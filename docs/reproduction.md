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

## Check the code

```bash
uv run pytest
uv run ruff check .
```

Tests use generated data and check causal timing, selection, fallback behavior, saved metrics
and run protections. See the [technical reference](technical/index.md) for provenance limitations
and the pending CC2 protocol. Source/layout changes invalidate earlier local freeze packages;
a fresh matching package is required before future independent archival or CC2 access.
