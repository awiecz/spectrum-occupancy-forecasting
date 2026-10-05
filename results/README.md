# Saved results: what each folder holds

Everything here was produced by scripts in `scripts/` and is committed so that findings can be
checked without downloading recordings. Recordings and local scratch runs are not included.
Terms are explained in the [glossary](../docs/glossary.md).

## Current studies

| Folder | What it holds | Explained in |
|---|---|---|
| `data_study/` | Measurement inventory, busy-fraction labels, patterns and sensitivity tables, and their figures | [Data study](../docs/data_study.md) |
| `followup_v2/exploratory/` | Chronological model selection, leads, label offsets, feature ablations and simulated-degradation scenarios | [Technical results](../docs/technical/followup_v2_results.md) |
| `followup_v2/history_mixture/` | The learned history mixture, its comparisons and its experiment ledger | [Development results](../docs/technical/forecast_development.md) |
| `adaptive_learning/` | The 145-setting adaptive search, the reserved later-session assessment and audits | [Adaptive forecasting](../docs/adaptive_learning.md) |
| `residual_state_space/` | Residual diagnosis and the level/trend state-space corrections | [Residual study](../docs/residual_state_space.md) |
| `operational_learning/` | Continuous validation, controller comparison, the activity-count model and the registry views | [Operational learning](../docs/operational_learning.md) |
| `sparse_regime/` | Power step, the matched-training run (predictions, gains, intervals, robustness, figures) and the manifest of one interrupted first execution | [Sparse-regime study](../docs/sparse_regime.md) |
| `power_analysis/` | What the saved forecasts could and could not detect (tables and one figure) | [Power analysis](../docs/power_analysis.md) |
| `multiband/` | Planning tables for the draft multi-band protocol; **no model was fitted** | [Draft protocol](../docs/multiband_protocol.md) |
| `experiment_registry.jsonl` | Append-only log of declared experiments and their outcomes | [Operational learning](../docs/operational_learning.md#evidence-and-experiment-history) |

## Original single-band study

These files come from the project's first analyses and are described in
[`docs/technical/`](../docs/technical/index.md).

| Files | Explained in |
|---|---|
| `dev_window_forecast.json` and its `_x1.5` / `_x6` sensitivity runs | [Development](../docs/technical/dev_results.md) |
| `test_window_forecast.json`, `figures_test/` | [Test results](../docs/technical/test_results.md) |
| `node_LW1_window_forecast.json`, `figures_lw1/` | [Rural-site results](../docs/technical/lw1_results.md) |
| `transfer_results.json`, `figures_transfer/` | [Transfer results](../docs/technical/transfer_results.md) |
| `sanity_checks.json` and `*_log.txt` | [Measurement checks](../docs/technical/sanity_checks.md) |
| `figures/` | Three development figures from the first urban-site run; not embedded in a current page |

## Files to leave alone

Several files are fingerprinted by manifests or by registry entries, so editing or deleting them
would break verification:

- `*_RUN_LOCK.json`: records that the original one-shot test, rural-site and transfer runs happened.
- Every `source_snapshot.zip` and `*_source_snapshot.zip`: byte-for-byte copies of the code that
  produced each study. They look redundant, but each manifest and registry row records their hash.
- `experiment_registry.jsonl` and the `manifest.json` and `configuration.json` files: provenance records.
- Everything under `followup_v2/`, `data_study/`, `adaptive_learning/`, `residual_state_space/` and
  `operational_learning/`: `.gitattributes` keeps the exact bytes (including Windows line endings)
  so checksums survive any platform.

The duplicate-looking `figures_*` folders hold figures cited by the original technical pages.
Nothing here was produced by the CC2 recording, which remains unopened.
