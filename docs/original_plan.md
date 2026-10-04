> **HISTORICAL DOCUMENT.** This is the plan approved at the start of the project (2026-10-03). It was
> **superseded by events**: NIST's file server was unavailable throughout, so the study moved to AERPAW
> data; the target changed from next-sweep occupancy to window busy fraction after development
> results; a second node, transfer analysis and self-audit were added; the planned `rfsi` command-line
> tool was dropped in favour of plain scripts. See [index.md](index.md) for what actually happened and
> where things are.

# Plan: Spectrum Occupancy Prediction under Sparse RF Monitoring

## Context

Public research repo on spectrum-occupancy forecasting. The goal is **an honest, reproducible study on real public RF measurements**, not a demo. The repo is empty (one initial commit, `main`). Environment: Windows 11, Python 3.12, no scientific packages yet, RTX 5060 (not needed), 211 GB free disk, MiKTeX installed (usable for the note).

Decisions already made with you: **NIST primary / AERPAW fallback; no hard deadline; classical models only (no GRU for now); `uv` + `pyproject`, MIT licence, public repo.**

This document is the approval gate. Nothing below has been implemented; no large data has been downloaded.

## 1. Dataset investigation: findings so far

| | **NIST COVID-19 spectrum occupancy** (recommended) | **AERPAW fixed nodes CC1/CC2/LW1** (fallback / transfer) |
|---|---|---|
| Source | NIST PDR, DOI `10.18434/mds2-2553`, Kuester et al.; companion report NIST TN-2240 (2022) and "Radio Bursts in Residences During COVID-19" | Dryad DOI `10.5061/dryad.hmgqnk9zn` (Raouf et al., published 2026-03-27); described in arXiv 2510.08752 |
| Licence | NIST open licence (attribution) | CC0-1.0 |
| Measurement | **Calibrated 4 MHz band power (dBm)**, 0.5 ms apertures; 21 centre frequencies: LTE 700-900 MHz, 2.4 GHz ISM (2412/2437/2462), 5 GHz U-NII, a quiet reference band | Wideband sweeps 87 MHz-6 GHz, 98,868 bins, USRP B205mini, SigMF float32; calibration/gain not documented in the metadata |
| Time structure | 13 residential sites + hospital, Jun 2020-Sep 2021. Per site: shuffled-order frequency sweeps, **105 ms valid dwell per frequency, 77% dead time** (so each channel is revisited irregularly, seconds-to-tens-of-seconds apart); sites paused intermittently | 1 continuous week per node, regular ~15 s sweeps |
| Labels | None supplied as ground truth; occupancy is *derived* by threshold (TN-2240 uses -60 dBm and notes weak-signal sensitivity); denoised power products provided | None |
| Size | Total 243 GB, but per-site tarballs are 0.1-24 GB (raw) / 0.1-24 GB (denoised); **occupancy durations 3 MB-512 MB/site, histograms 0.5-90 MB/site** | 52 GB (19.1 + 20.7 + 12.5 GB zips) |
| Fit to question | Excellent: native sparse/irregular observation, real outages, 2.4 GHz prevalence ~49% of dwells (balanced target), 15-month span for chronological splits | Regular sampling is easy to forecast and subsample, but 1 week limits splits, and 15 s resolution is coarse |

Key empirical facts (from the NIST paper): 2.4 GHz channels have the highest activity (mean 48.8% of dwell windows contain at least one occupancy event); cellular uplinks ~1.9%; 5 GHz ~9.7%. Occupancy events are mostly <=5 ms.

**Recommended first experiment channel:** a 2.4 GHz ISM channel (2437 MHz) at one mid-size site, chosen in Milestone 0 by coverage (not by size alone). Candidate small/mid sites: 02, 12, 06 (1.7-2.5 GB raw each); site 03 (8.6 GB) as an alternative.

### Unverified items (the first task of implementation)
1. **Download reliability:** `data.nist.gov` metadata endpoints respond, but the file-download endpoint timed out from this machine for README/CSV files (connection opens, no bytes). Plan: retry with resumable download (curl `-C -` / browser fallback), verify against the published `.sha256` files. If it stays unreachable, switch to AERPAW LW1/CC1 with a ranged read of one day out of the zip.
2. **Internal schemas** of the tarballs and `occupancy durations/*.csv.gz` (columns, timestamp format, whether non-occupied dwells appear) are not yet seen; the README (7 KB) is the first thing to read. The design below adapts to what is found.
3. NIST landing page shows version 1.16.0 vs API metadata 1.0.0: record the exact file checksums in a manifest instead of trusting the version string.

## 2. Scientific design (first experiment)

**Question:** how does short-term occupancy forecast skill degrade as observation age grows and as measurements are lost?

- **Unit of observation:** one *dwell* = 105 ms of 210 contiguous 0.5 ms band-power bins at a (site, channel) with a wall-clock start time.
- **Occupancy definition:** bin is occupied if denoised power > noise-floor + X dB (X in {3, 6, 10}; noise floor estimated from **training period only**); also report with the fixed -60 dBm reference from TN-2240. Dwell label `y = 1` if it contains >=1 occupancy event (events must not touch the dwell edge, as in the paper). Secondary target: dwell duty cycle (fraction of occupied bins). Threshold sensitivity is a reported result, not a hidden choice.
- **Prediction task:** at each target dwell at time `t`, predict `y_t` from observations of the *same and other channels* with timestamps `<= t - lag`. Horizon is **real time** (seconds), reported by buckets of observation age (e.g. <10 s, 10-30 s, 30-60 s, 1-5 min, >5 min). Since revisit times are irregular, age is a first-class variable, which makes the provenance/age of evidence a central theme.
- **Features (only what is available at prediction time):** last valid state and its age, EWMA of recent occupancy with time decay, counts over last k minutes, recent power summaries (linear-power mean, never dB mean), last-seen state of other channels with their ages, availability mask, hour-of-day/day-of-week only if the span supports it.
- **Methods (all classical):**
  1. Global-rate (climatology) and hour-of-week-rate prior (train-fit)
  2. Persistence (last valid state, with age)
  3. Time-decayed rolling-history estimator, with training-rate fallback when no history exists
  4. Age-conditioned empirical persistence P(occupied | last state, age bucket): a cheap Markov-style baseline
  5. Logistic regression and gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`; add LightGBM only if needed)
  - Deferred: GRU / hidden-Markov models (stretch only after the above show a clear gap).
- **Splits / leakage:** chronological train / dev / test per site with an **embargo >= longest lookback window**; scalers, noise floors, thresholds, imputation fit on train only; dev for model selection; test touched once for the final number; session/outage boundaries respected. Transfer test to a **second site** (and later AERPAW) only if data permit.
- **Robustness (one factor at a time, then combined):** random dwell dropout (0/25/50/75/90%); contiguous outages (1 min, 15 min, 1 h, 6 h); forced observation delay; noise only if defensible (raise effective noise floor). Degradation is applied **only to the predictor-visible stream**; target labels are always from the intact stream; missing data is a mask + age, never 0 power or "idle".
- **Metrics:** confusion matrix, precision/recall for busy, **false-idle rate (denominator = truly occupied targets)**, balanced accuracy and F1 with prevalence, Brier score, reliability diagram / ECE, runtime per prediction; **moving-block bootstrap CIs (by day)** to respect temporal dependence.
- **Claim discipline:** "detected activity at one indoor sensor", not transmitter activity, interference elsewhere, or permission to transmit. A negative result (heuristics tie ML) is acceptable and will be reported.

## 3. Repository design

```
rf-spectrum-intelligence/
├── README.md                      # hero figure, 5-command reproduce, claims / non-claims
├── pyproject.toml, uv.lock        # uv; Python >=3.11
├── LICENSE (MIT), CITATION.cff, DATA_ATTRIBUTION.md
├── configs/                       # YAML: dataset subset, occupancy defs, splits, experiments
├── docs/ dataset_investigation.md, execution_plan.md, research_note/ (LaTeX or md -> PDF)
├── src/rf_spectrum_intelligence/
│   ├── data/        # fetch (resumable + sha256 manifest), NIST readers, processed parquet schema
│   ├── occupancy/   # noise floor, thresholding, burst/idle stats
│   ├── features/    # age-aware feature builder, masks
│   ├── models/      # baselines, sklearn models
│   ├── evaluation/  # splits+embargo, metrics, calibration, block bootstrap
│   ├── robustness/  # degradation operators (dropout, outage, delay, noise)
│   └── viz/         # the three required figures
├── scripts/ / CLI (`rfsi fetch|process|characterize|run|report`)
├── tests/           # units, timestamps, windowing, leakage, degradation-never-touches-targets
├── tests/fixtures/  # tiny real excerpt (<1 MB, licence permitting) or synthetic generator
├── results/         # small JSON/CSV + figures; raw data gitignored (data/ outside VCS)
└── .github/workflows/ci.yml       # ruff + pytest on the fixture
```
Dependencies (minimal): numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib, pyyaml, pytest, ruff; optional jupyter. CPU only; processed per-site parquet; expected disk 2-10 GB (well within 211 GB).

## 4. Milestones (no hard deadline; roughly 6-8 part-time weeks)

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | **Verify data access** (README, occupancy-durations + histograms for all sites ~0.5 GB total, schemas, coverage per site/channel) -> `docs/dataset_investigation.md`, pick site + channel, confirm or revise the question | Recommended site/channel justified with coverage plots; fallback decision recorded |
| M1 | Scaffold (uv, ruff, pytest, CI, licence), fetch + checksum manifest, ingestion into the processed schema | `rfsi fetch/process` reproducible; unit/timestamp tests green |
| M2 | Characterisation: power plots, time/frequency view, gap stats, occupancy fraction, burst/idle durations, autocorrelation, threshold sensitivity | **Figure 1** (measurement characterisation) |
| M3 | Splits + embargo, metrics, leakage tests; baselines 1-4 | **Figure 2** (baseline comparison) incl. false-idle and calibration |
| M4 | Logistic regression + GBT, same protocol | Dev-only model selection; test run once |
| M5 | Robustness grid (single factor then combined) | **Figure 3** (degradation vs age / dropout / outage) |
| M6 | Second-site transfer test (and AERPAW subset if cheap) | Reported only if data truly comparable |
| M7 | Research note (1-3 pp) + README polish + open questions | Third party reproduces central result from README |

## 5. Risks

- NIST file host unreachable/slow -> resumable downloads, browser fallback, AERPAW ranged-zip fallback.
- Tarball internals larger/messier than expected -> use denoised time series for one site first; derive from occupancy-durations files if sufficient.
- Residential indoor site -> strongly site-specific; no generalisation claims beyond the tested site.
- Irregular sampling and 77% dead time complicate "horizon" -> age buckets handle this explicitly.
- Weak signals lost to threshold -> sensitivity analysis and explicit limitation.
- 2020-21 data (pre-5G, COVID usage patterns) -> state it as a limitation.

## 6. Verification

- `uv run pytest` (units incl. dBm->linear averaging, timestamp monotonicity, window/embargo no-overlap, degradation leaves targets untouched), `uv run ruff check`, CI green on the fixture.
- End-to-end smoke: `rfsi fetch -> process -> run configs/exp1.yaml -> report` on the chosen site regenerates the three figures and `results/*.json` deterministically (fixed seeds).
- Manual: check README commands on a clean clone.

## 7. What I need approved

1. The first experiment framing (dwell-level occupancy, forecast skill vs observation age, classical methods).
2. M0 downloads (~0.5 GB of CSVs, then 2-10 GB of one or two sites) and the fallback rule above.
3. Repo name `rf-spectrum-intelligence` (current folder is "RF Spectrum Intelligence") and creating a working branch before the initial scaffold commit.
