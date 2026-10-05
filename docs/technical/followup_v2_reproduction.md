# Reproducing the follow-up

The executable specification is `configs/followup_v2.yaml`. CC1/LW1 are reused exploratory
measurements. The [report](followup_v2_results.md) presents results; the
[protocol](followup_v2_protocol.md) defines calibration, selection, uncertainty and pending CC2 confirmation.

## Run locally

From the repository root:

```powershell
uv sync --locked
uv run python scripts/run_followup.py --mode synthetic-rehearsal --out reproductions/rehearsal_new
uv run python scripts/run_followup.py --mode exploratory --out reproductions/exploratory_new
uv run python scripts/run_followup.py --mode report --out reproductions/exploratory_new
```

The real-data command expects `data/processed/aerpaw_CC1_blocks5MHz.npz` and
`aerpaw_LW1_blocks5MHz.npz`; `--cc1-npz` and `--lw1-npz` override those paths. Raw data are
not redistributed. See the [data workflow](reproduction.md) for downloads and extraction.
Every experiment directory must be new. Local reproductions are Git-ignored; the repository
keeps one reviewed result set at `results/followup_v2/exploratory/`.

Synthetic rehearsal runs the primary task and input degradations; synthetic scores are not
research evidence. Report mode verifies saved input checksums and regenerates tables/figures
without RF access or refitting, recording the report-builder fingerprint in the manifest.

## Retained history mixture

`configs/history_mixture.yaml` preserves the declared three-regime experiment. All real-data
results remain exploratory reuse. The [development report](forecast_development.md) explains
retrospective selection, failed advancement and the complete experiment ledger.

```powershell
uv run python scripts/run_history_mixture.py --mode synthetic-rehearsal --out reproductions/mixture_rehearsal_new
uv run python scripts/run_history_mixture.py --mode exploratory --out reproductions/mixture_new
uv run python scripts/run_history_mixture.py --mode report --out reproductions/mixture_new
```

The input overrides are the same `--cc1-npz` and `--lw1-npz` arguments. The optional
`--study history-mixture` must agree with the configuration; no other study or confirmation
mode is accepted. Outputs must be fresh and live under `reproductions/` inside this checkout.
The reviewed result set is `results/followup_v2/history_mixture/`; avoid rebuilding it in place.
To regenerate its reports without fitting, first make a fresh local copy:

```powershell
if (Test-Path -LiteralPath reproductions/mixture_report_new) { throw "Choose a new output directory" }
Copy-Item -LiteralPath results/followup_v2/history_mixture -Destination reproductions/mixture_report_new -Recurse
uv run python scripts/run_history_mixture.py --mode report --out reproductions/mixture_report_new
```

Report mode preserves fitting provenance and appends report-builder provenance to the local
copy. The original predictions and selections stay unchanged. The public ledger identifies
archived methods; the cleaned runner reproduces only the retained mixture experiment.

## Saved artifacts

| File | Purpose |
|---|---|
| `config.yaml`, `manifest.json` | Specification, fitting fingerprint, Git/input hashes, calibration, fold status and artifact checksums |
| `predictions.parquet` | Per-origin labels, target endpoints/counts, forecasts, configurations, scenarios and seeds |
| `validation_predictions.parquet`, `selections.json` | Earlier predictions, every candidate score and selected configurations |
| `operating_thresholds.csv`, `events.csv` | Frozen thresholds, calibration sufficiency and realized rates with denominators |
| `summary.json`, `summary.csv`, `daily.csv` | Pooled and daily metrics with specified reference comparisons |
| `bootstrap.csv`, `dependence.csv`, `oracle.csv` | Block sensitivities/counts, temporal dependence and descriptive oracle points |
| `figures/` | Seven figure families, each with PNG, PDF and plotted-data CSV |

Detailed selections have one authoritative JSON file rather than being repeated in the
manifest. Intermediate `_parts/` checkpoints are local only. Masks are paired across methods;
losses are averaged over masks before temporal resampling. Reused validation scores are
selection diagnostics, not independent generalization estimates.

## Freeze and gate CC2

Confirmation remains pending; this finishing pass does not create or upload a package.
Source consolidation changes the current fingerprint, so earlier local packages no longer
match this checkout. If confirmation is resumed later, create and independently archive a
**fresh** package before any CC2 access:

```powershell
uv run python scripts/freeze_followup.py --out reproductions/followup_v2_freeze.zip
```

The package includes its own source/configuration/protocol hash manifest. Creation refuses
an existing file and does not upload anything. Before CC2 access, independently archive the
exact `followup_v2_freeze.zip` in a published immutable GitHub release or published Zenodo
record. The current verifier needs public unauthenticated archive access. It fetches the
actual package and compares its bytes with the current frozen files; a local timestamp,
claimed hash or mutable release is insufficient.

```powershell
uv run python scripts/run_followup.py --mode confirmation --verify-gate-only --archive-url YOUR_ARCHIVE_API_URL
```

Only after verification, download CC2 from Dryad, obtain its published SHA-256, verify and
extract it with the existing workflow. Retain the extraction record. Then run once:

```powershell
uv run python scripts/run_followup.py --mode confirmation --archive-url YOUR_ARCHIVE_API_URL --cc2-npz data/processed/aerpaw_CC2_blocks5MHz.npz --cc2-raw-archive PATH_TO_CC2_ARCHIVE --cc2-expected-sha256 PUBLISHED_SHA256 --cc2-checksum-source PUBLISHED_CHECKSUM_HTTPS_URL --out results/followup_v2/confirmation_CC2
```

The runner records expected/actual archive hashes, the investigator-supplied published source
reference and verification time. A mismatch fails before NPZ access. The persistent one-shot
lock precedes CC2 reads and remains after failure; do not remove it for another attempt.
Package changes after archival require a disclosed, separately versioned analysis.
Archival verification cannot prove the investigator's no-prior-inspection disclosure.
