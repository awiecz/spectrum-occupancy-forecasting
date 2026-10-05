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

## Reproduce the measured-data experiments

This requires the CC1 and LW1 archives: approximately **32 GB of downloads**, plus processed
files and working space. Recordings are excluded from Git. To understand the findings without
downloading data, use the saved outputs linked from the [results overview](results.md).

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
