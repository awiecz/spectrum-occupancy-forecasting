# Reproducing the study

Run commands from the repository root. Use `uv sync --locked` first; CI tests Python 3.11 and 3.12.
A download-free workflow demonstration is available with `uv run python scripts/run_synthetic_demo.py`.
Synthetic outputs are labeled and cannot be used as evidence for the real-data study.

## 1. Download and verify

Download `ResultsCC1Feb2022_SigMF.zip` (about 19 GB) and `ResultsLW1Feb2022_SigMF.zip`
(about 12.5 GB) from [the official Dryad release](https://doi.org/10.5061/dryad.hmgqnk9zn)
into `data/raw/aerpaw/`. Browser downloads were used for the original study because scripted
requests were blocked. Compare the complete SHA-256 outputs with the release's published hashes:

```bash
uv run python scripts/sha256.py data/raw/aerpaw/ResultsCC1Feb2022_SigMF.zip
uv run python scripts/sha256.py data/raw/aerpaw/ResultsLW1Feb2022_SigMF.zip
```

The recorded CC1 hash is `c2b1e6d9172f4738b11d78d874c2ca2e3d4232330a99d2b278eb1e05b642f747`.
The historical LW1 protocol stores only an abbreviation (`595e7ea4...11401951`); obtain the complete
expected value from Dryad rather than treating that abbreviation as a checksum.

## 2. Extract the block products

```bash
uv run python scripts/extract_aerpaw_blocks.py data/raw/aerpaw/ResultsCC1Feb2022_SigMF.zip
uv run python scripts/extract_aerpaw_blocks.py data/raw/aerpaw/ResultsLW1Feb2022_SigMF.zip
```

This writes `data/processed/aerpaw_CC1_blocks5MHz.npz` and `aerpaw_LW1_blocks5MHz.npz`.
Extraction is resumable; `--workers 2` reduces parallelism if resources are limited.

## 3. Run independent reproductions

Committed `results/` files and locks are the historical record. The default one-shot commands
refuse to run because those locks already exist. Use **new output and lock paths** to reproduce
without overwriting them; the commands below work in PowerShell and ordinary POSIX shells.

```bash
uv run python scripts/run_window_dev.py data/processed/aerpaw_CC1_blocks5MHz.npz --out reproductions/dev_window_forecast.json
uv run python scripts/run_test_once.py data/processed/aerpaw_CC1_blocks5MHz.npz --out reproductions/test_window_forecast.json --lock reproductions/TEST_RUN_LOCK.json
uv run python scripts/run_second_node.py data/processed/aerpaw_LW1_blocks5MHz.npz --out reproductions/node_LW1_window_forecast.json --lock reproductions/NODE_LW1_RUN_LOCK.json
uv run python scripts/run_transfer.py data/processed/aerpaw_CC1_blocks5MHz.npz data/processed/aerpaw_LW1_blocks5MHz.npz --out reproductions/transfer_results.json --lock reproductions/TRANSFER_RUN_LOCK.json
```

Transfer checks its in-node RMSEs against the committed CC1-development and LW1 results, with a
`1e-6` tolerance. A failed check requires investigation; do not overwrite the historical results.
Each reproduction lock also refuses a second run. For another independent attempt, choose a new
output/lock pair. The `--acknowledge-rerun` option instead records a repeat in that reproduction's
lock; report any such repeats. Reproduction is distinct from a new unseen-data evaluation.

## 4. Render the figures

```bash
uv run python scripts/make_figures.py data/processed/aerpaw_CC1_blocks5MHz.npz reproductions/dev_window_forecast.json --out reproductions/figures_dev --region dev
uv run python scripts/make_figures.py data/processed/aerpaw_CC1_blocks5MHz.npz reproductions/test_window_forecast.json --out reproductions/figures_test --region test
uv run python scripts/make_figures.py data/processed/aerpaw_LW1_blocks5MHz.npz reproductions/node_LW1_window_forecast.json --out reproductions/figures_lw1 --region lw1
uv run python scripts/make_figures.py data/processed/aerpaw_CC1_blocks5MHz.npz reproductions/transfer_results.json --out reproductions/figures_transfer --region transfer
```

Plots include PNG/PDF versions and CSV tables. Figure 1 uses only CC1 development data.
The transfer plotting command ignores the required NPZ argument and reads its result JSON only.

## 5. Reproduce the measurement audit

```bash
uv run python scripts/sanity_checks.py --out reproductions/sanity_checks.json
```

This uses both extracted products and the committed study result JSONs for the evidence-age table;
it does not recompute predictions. Its descriptive CC1 controls use only the development region.

## Interpreting agreement

The current code includes a documentation clarification and portable fingerprint scheme introduced
after the original runs. Numerical forecasting behavior is unchanged by those edits. Historical
fingerprints used absolute paths and cannot be compared across checkout locations; new locks use
`root-relative-v2`, normalize line endings, and record Git state. Neither a matching fingerprint nor
a local lock establishes independently timestamped registration or prevents evaluation by other code.

Compare metrics and tables rather than expecting byte-identical JSON or image files. Original locks,
results and protocols remain unchanged. See [the historical-lock notes](index.md#protocols-and-historical-locks),
[LW1 selection overlap](lw1_results.md#channel-selection-overlap), and
[CC1 event-threshold caveats](test_results.md#event-threshold-interpretation).
