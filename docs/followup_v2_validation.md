# V2 implementation and audit checks

Validation performed on 2026-10-04 using the locked environment on Windows.

- `uv run ruff check .`: passed.
- `uv run pytest -o addopts='' -q`: 119 tests passed, including 24 v2 tests.
- Final-code synthetic rehearsal: `reproductions/followup_v2/rehearsal_boundfix`, completed. Generated
  RF inputs are not committed and synthetic scores are not research evidence.
- Exploratory study: CC1 five folds and LW1 three folds; all 13 staged tasks completed.
  The prediction product contains 275,268 rows, including scenario/method/mask repetitions.
  These repetitions are not counted as independent origins.
- Independently recomputed all 412 pooled and 1,648 daily MSE/MAE rows from saved Parquet
  predictions; agreement within floating-point tolerance (`rtol=1e-12`, `atol=1e-14`).
- Verified label equality at each task/node/fold/origin across all methods and input scenarios.
- Verified all 34 completed-run artifact checksums, including the seven figure families.
- Visually inspected predictions, lead RMSE, lead skill, label sensitivity, ablations,
  degradation and bootstrap-sensitivity figures. Their plotted-data CSVs accompany them.
- Historical LW1 sensitivity reproduces all stored pooled RMSEs exactly. Git reports no
  modifications to historical source modules, experiment configurations, protocols, results,
  plots or locks. The README/index and research-note source/PDF were intentionally updated.
- Research note: local `pdflatex -interaction=nonstopmode -halt-on-error research_note.tex`
  succeeded, three pages, no overfull boxes. All three rendered pages reviewed. The built-in
  editor compiler returned a platform-directory error; the existing local compiler supplied
  the verified PDF.
- Frozen package created locally at `reproductions/followup_v2_boundfix/followup_v2_freeze.zip`; its actual contents
  verify against the current protocol, configuration and source hashes. The independent
  archive gate is **not satisfied**. The package was not uploaded and CC2 was not accessed.

The original run and its report-builder provenance remain preserved. The corrected pipeline
was rerun into `results/followup_v2/exploratory_20261004_boundfix/`; all 275,268 prediction rows
and all candidate scores, selections and thresholds exactly match the original run. Its fitting
fingerprint now covers the final bounded-state implementation and reporting code. New tests
check long-gap state/age expiry, invariance to expired observations, raw checksum provenance,
and a mismatched archive failing before derived-data access while retaining its lock.
The refreshed [freeze manifest](followup_v2_freeze_manifest.json) records the corrected package;
the earlier local package is superseded and never independently archived.

The experiment runner refuses existing output directories and protects the future confirmation
with a persistent one-shot lock. Tests cover those controls and actual archive-byte verification,
alongside future-data corruption, first-day calibration isolation, causal lead/window boundaries,
coverage, training embargoes, deterministic selection, feature exclusion, tied/degenerate event
thresholds, saved-prediction round trips, paired masks and insufficient temporal blocks.
