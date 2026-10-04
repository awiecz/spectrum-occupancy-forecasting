# Running the separate v2 study

V2 leaves the historical experiment entry points and outputs intact. The executable specification
is `configs/followup_v2.yaml`. Real-data runs reuse previously inspected CC1/LW1 data and are
exploratory. CC2 confirmation is blocked until independent archival verification succeeds.

From the repository root, with the locked environment installed:

```powershell
uv sync --locked
uv run python scripts/run_followup.py --mode synthetic-rehearsal --out reproductions/followup_v2/rehearsal_new
uv run python scripts/run_followup.py --mode exploratory --out results/followup_v2/exploratory_new
uv run python scripts/run_followup.py --mode report --out results/followup_v2/exploratory_new
```

The exploratory command uses the existing `data/processed/aerpaw_CC1_blocks5MHz.npz` and
`aerpaw_LW1_blocks5MHz.npz` products. Optional `--cc1-npz` and `--lw1-npz` select other paths.
Every experiment output directory must be new. Report mode checks the saved inputs' checksums
and regenerates reports without reading RF recordings or fitting models. It records report
regeneration in the manifest. Reports produced by changed code require disclosure.

The synthetic rehearsal executes the primary task and all degradation scenarios on generated
data. The complete staged task list is exercised in the real exploratory run; synthetic tests
also check the task definitions, lead handling and feature allowlists.

## Files and comparisons

`manifest.json` records source fingerprint, Git state, input hashes, configuration hash, node
calibrations, fold status and final artifact checksums. `config.yaml` is the run's specification.
`predictions.parquet` holds every evaluated origin, actual target endpoints, labels, sweep count,
method choice and mask seed. `validation_predictions.parquet` stores the earlier selected-method
validation predictions. `selections.json` retains every candidate's pooled validation score,
selected configurations and frozen operating thresholds. `events.csv` separately reports those
thresholds, validation class counts and evaluation rates/denominators; missing calibrated rates
mean insufficient earlier evidence, not zero error.

`summary.csv` and `daily.csv` contain pooled and fold-specific scores. `bootstrap.csv` holds
paired temporal-block interval sensitivities and block counts. `dependence.csv` contains
within-contiguous-segment autocorrelations. `oracle.csv` is explicitly descriptive and uses
evaluation-label operating points. Six figure families each have PNG, PDF and plotted-data CSV
outputs. None requires refitting. A seventh lead-skill figure supplements the lead-time RMSE
curve. Intermediate `_parts/` files are excluded from Git.

Selected history and learned models are selected independently by earlier validation loss;
residual learners are conditional on the earlier selected history. Reused validation scores are
selection diagnostics, not independent estimates of generalization. The selected learned row
is a fold-varying strategy, not a single retrospectively chosen best model. Family rows report
their validation-selected configuration. Unsuccessful configurations remain in the selection
records rather than disappearing from the study. `operating_thresholds.csv` independently
exports the frozen calibration records for every task and fold.

Outer origins are spaced 15 minutes apart. Inner validation origins use the same common
maximum-lead restriction as outer folds. Training origins are spaced three minutes apart.
The longest decay has finite support of five time constants (ten hours), making the embargo
an actual bound. Missing history falls back toward the earlier training mean. No future label
or simulated loss changes the target. Masks are paired across methods; losses, rather than
predictions, are averaged over the five masks before temporal resampling.

## Private confirmation package

Only after exploratory review, freeze the completed protocol and executable sources:

```powershell
uv run python scripts/freeze_followup.py --out reproductions/followup_v2_freeze.zip
```

This creates a package locally; it does not upload anything. Keep the repository and package
private until you choose to archive it. The gate currently supports a published immutable
GitHub release API reference or a published Zenodo record API reference containing the exact
`followup_v2_freeze.zip` package. Verification retrieves the actual archive payload and compares
protocol, configuration and every frozen source file with the current checkout. A local clock,
claimed checksum, mutable release or unrelated archive URL is insufficient. Public archive
access is required by the current unauthenticated verifier.

After independently archiving the package, verify the gate before downloading CC2:

```powershell
uv run python scripts/run_followup.py --mode confirmation --verify-gate-only --archive-url YOUR_ARCHIVE_API_URL
```

Then download the CC2 archive from the cited Dryad deposit, verify its published checksum and
extract it using the existing extraction workflow. Do not run exploratory scripts on CC2.
Run the frozen pipeline once:

```powershell
uv run python scripts/run_followup.py --mode confirmation --archive-url YOUR_ARCHIVE_API_URL --cc2-npz data/processed/aerpaw_CC2_blocks5MHz.npz --out results/followup_v2/confirmation_CC2
```

The persistent `CC2_CONFIRMATION_LOCK.json` prevents reruns even after a failed attempt.
Failures need an explicitly versioned amendment, not removal of the lock. Gate verification
does not itself prove that the investigator never inspected CC2; preserve the investigator's
disclosure and archive chronology. Changing frozen source/configuration/protocol invalidates
the archived package. See `followup_v2_protocol.md` for the frozen interpretation rules.
