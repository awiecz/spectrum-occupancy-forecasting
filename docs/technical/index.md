# Technical reference

Start with the [data study](../data_study.md), [results overview](../results.md) or
[reproduction guide](../reproduction.md).
These documents support one project; they record distinct analyses rather than competing
versions of the repository.

## Methods and evidence

| Analysis | Detailed results | Saved evidence |
|---|---|---|
| First-day-calibrated descriptive data study | [Illustrated study](../data_study.md) | [Source tables, summary and figures](../../results/data_study/) |
| Chronological model selection, leads and sensitivity | [Study results](followup_v2_results.md) | [Predictions, selections and figures](../../results/followup_v2/exploratory/) |
| History mixture and adaptive development | [Development results](forecast_development.md) | [Mixture outputs and experiment ledger](../../results/followup_v2/history_mixture/) |
| Local adaptation and prior transfer outcome | [Illustrated results](../adaptive_learning.md) | [Predictions, selections and audit](../../results/adaptive_learning/) |
| Registered operational validation, controllers and activity counts | [Illustrated results](../operational_learning.md) | [Reviewed evidence and registry tables](../../results/operational_learning/) |
| Original urban-site development | [Development](dev_results.md) | [Original outputs](../../results/) |
| Original urban-site test | [Test results](test_results.md) | [Protocol](test_protocol.md) |
| Original rural-site evaluation | [Rural-site results](lw1_results.md) | [Protocol](lw1_protocol.md) |
| Cross-site transfer | [Transfer results](transfer_results.md) | [Protocol](transfer_protocol.md) |

[Measurement checks](sanity_checks.md), [original-study commands](reproduction.md),
[study artifact/command reference](followup_v2_reproduction.md), and
[data attribution](../../DATA_ATTRIBUTION.md) provide supporting details.
Planning-only notes and unused dataset investigations are retained locally rather than
published as additional project guides. Detailed unsuccessful development runs are archived
locally; the public ledger records their outcomes and identifiers.

## Provenance and pending evaluation

The [CC2 protocol](followup_v2_protocol.md) remains scientifically unchanged and unarchived.
CC2 has not been inspected. Confirmation must pass the independent-archive gate before data
access; it is not needed to reproduce this exploratory project.

Historical protocol dates are project records, not independently verified registration.
Their locks used absolute-path fingerprints. Current portable hashes and matching predictions
do not reconstruct an exact historical source snapshot or prove no prior data inspection.
The original rural-site channel selection overlaps its first scored fold, and historical
ROC operating points used evaluation-label thresholds. The chronological study separately
uses earlier calibration/validation thresholds and reports insufficient calibration.

Historical figure annotations and protocols may name their former `docs/<filename>` locations;
those supporting documents now live here. Saved numerical artifacts are unchanged.
Source/layout changes require a fresh matching package before future independent archival.
