# Repository guide

Start with the [three-page research note](research_note/research_note.pdf) and
[README](../README.md). The [v2 report](followup_v2_results.md) contains the current
exploratory findings and validation record.

## Current follow-up

- [Results](followup_v2_results.md): lead times, fair model selection, label/feature sensitivity,
  degraded evidence and uncertainty.
- [Reproduction](followup_v2_reproduction.md): commands, artifacts and archival gate.
- [CC2 protocol](followup_v2_protocol.md): prepared privately; confirmation awaits independent timestamping.
- [Saved outputs](../results/followup_v2/exploratory/): one canonical result set, including predictions,
  selections, provenance, summary tables and figures.

## Original study

These are distinct experiments, not alternate copies of v2. Their protocols, numerical
results and locks remain intact.

| Experiment | Protocol | Findings |
|---|---|---|
| CC1 development | Exploratory | [Development](dev_results.md) |
| CC1 held-out test | [Protocol](test_protocol.md) | [Results](test_results.md) |
| LW1 evaluation | [Protocol](lw1_protocol.md) | [Results](lw1_results.md) |
| Cross-node transfer | [Protocol](transfer_protocol.md) | [Results](transfer_results.md) |

[Original-study commands](reproduction.md), [measurement checks](sanity_checks.md),
[dataset investigation](dataset_investigation.md) and [original plan](original_plan.md)
provide the supporting record. NIST data were unavailable, so AERPAW input sparsity is
simulated. The target changed from individual sweeps to window fractions during development.

## Protocols and historical locks

Historical protocol dates are project records, not independently verified registration.
Their locks use absolute-path fingerprints and cannot be reproduced at a different checkout
location; current hashes normalize repository-relative paths and line endings. Later code
changes are not a recovered snapshot of the exact historical evaluation source.

LW1 channel selection overlaps its first scored fold, and historical event operating points
use evaluation-label oracle thresholds. Those limitations remain disclosed in the results.
V2 uses earlier validation thresholds and a separate independent-archive requirement for CC2.

## Layout

`src/` contains the library; `scripts/` provides experiment and plotting commands; `configs/`
contains the original and follow-up specifications; `tests/` uses synthetic data. `results/`
contains the original study and one v2 result set. Raw recordings and local reproductions
are ignored by Git. New experiments cannot overwrite existing run directories.
