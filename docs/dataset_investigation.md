# Dataset investigation (M0) — status: PARTIAL

Last updated: 2026-10-03. Everything marked **[verified]** was read from an official source in this investigation; **[unverified]** items must be checked before any modelling claim is made.

## 1. Candidates

### 1.1 NIST — Distributed Sensing of Radio Spectrum Occupancy Amid COVID-19 (primary)

| Item | Finding |
|---|---|
| Publisher / authors | NIST Communications Technology Laboratory; Kuester, Lu, Gu et al. [verified] |
| Identifier | DOI `10.18434/mds2-2553`, landing page <https://data.nist.gov/pdr/lps/ark:/88434/mds2-2553> [verified] |
| Version | API metadata reports `1.0.0` (modified 2022-04-20, first issued 2022-07-19); the rendered landing page title shows `1.16.0`. Pin by file checksum, not version string. [verified discrepancy] |
| Licence | NIST open licence, <https://www.nist.gov/open/license> (attribution) [verified] |
| Companion papers | NIST TN-2240, <https://doi.org/10.6028/NIST.TN.2240> (151 pp); Kuester, Lu, Gu, "Radio Bursts in Residences During COVID-19" [verified, read in part] |
| Instrument | Custom 2-channel swept SDR; RX1 700-900 MHz, RX2 2-6 GHz; 12-bit; 4 MHz resolution bandwidth; OIP3 -12.4 dBm, input compression -20.9 dBm at 701.5 MHz [verified] |
| Frequencies | 21 centre frequencies: LTE FDD 700-900 MHz (e.g. 701.5, 709, 731.5, 739, 751, 782, 821.3, 842.5, 866.3, 887.5), 2.4 GHz ISM (2412, 2437, 2462), U-NII (5170-5795), and a quiet reference band (2695 MHz) [verified] |
| Detector | RMS band power per 0.5 ms aperture, 2000 samples at 8 MS/s, filtered to 4 MHz; frequency-dependent calibration factor applied (laboratory calibration) -> **dBm** [verified, TN-2240 §2.1.1] |
| Time structure | Sweeps through all frequencies in **freshly shuffled order**; per frequency a dwell of which the **last 105 ms (210 bins)** are recorded (first 195 ms discarded for settling); **dead time averages 77%**; sensors were paused intermittently by their hosts [verified] |
| Coverage | 13 residential sites (indoor, Boulder CO area) + hospital spot measurements; 2020-06-09 to 2021-09-22; number of active sites declines over time [verified] |
| Labels | **None supplied as ground truth.** TN-2240 defines occupancy by a power threshold (-60 dBm in the report, with sensitivity to threshold); the release provides denoised power (receiver average additive noise removed) and derived occupancy-duration tables [verified] |
| Network profiling | Commercial-instrument LTE/WLAN decode for 2 residential sites + hospital; qualitative only; not needed for this study [verified] |
| Mean fraction of dwells with >=1 occupancy event | cellular uplink ~1.9%, 5 GHz U-NII ~9.7%, 2.4 GHz ISM ~48.8% (paper's threshold) [verified] |

**Files (from API metadata, 167 components, 243.5 GB total):**

| Product | Per-site size range | Notes |
|---|---|---|
| `time series/NN.tar.gz` (raw band power) | 0.01-20.2 GB | sites `00`-`12` + `hospital`; 109 GB total |
| `time series denoised/NN.tar.gz` | 0.01-23.8 GB | 131 GB total |
| `occupancy durations/NN.csv.gz` | 15 KB-512 MB | 2.3 GB total |
| `histograms/NN.csv.gz` | 59 KB-90 MB | 0.56 GB total |
| `network profiling/...csv.gz` | KB-7 MB | 18 MB total |
| `README.txt`, `source-code.zip` | 7 KB, 6 MB | |

Smallest tarballs (raw / denoised): site 08 (0.10 / 0.12 GB), 04 (0.27 / 0.34 GB), 12 (1.85 / 2.27 GB), 02 (1.72 / 2.27 GB), 06 (2.45 / 2.92 GB), 03 (8.6 / 10.4 GB). Every file has a published `.sha256` sibling.

**Access status: BLOCKED (transient?).** `data.nist.gov` metadata endpoints (`/od/id/...`, `/rmm/...`) respond in ~1-2 s, but every `/od/ds/...` file download (README, CSVs, tarballs) hangs until Cloudflare returns **HTTP 524 "origin timed out"** (confirmed from curl, PowerShell and the in-app browser on 2026-10-03; the response is a 16-byte error body). This is a server-side fault, not a local network or User-Agent issue. A background retry loop (every ~4 min for ~1 h) is running for the three smallest files.

**[unverified]:** column definitions and timestamp format of `occupancy durations` and the tarball contents; whether non-occupied dwells are represented (needed to build dwell-level labels); per-site coverage/continuity; per-channel noise floor values.

### 1.2 AERPAW fixed nodes CC1 / CC2 / LW1 (fallback, later transfer test)

| Item | Finding |
|---|---|
| Source | Dryad DOI `10.5061/dryad.hmgqnk9zn`, Raouf, Maeng, Ozdemir, Guvenc, Sichitiu; published 2026-03-27; primary article doi:10.1109/COMSNETS56262.2023.10041314 [verified] |
| Licence | CC0-1.0 / public domain [verified] |
| Content | 3 nodes (CC1, CC2 urban; LW1 rural), February 2022, ~1 continuous week each; USRP B205mini; sweep 87 MHz-6 GHz; **98,868 bins per sweep**; one SigMF `.sigmf-data` (float32 LE, **dBm**) + `.sigmf-meta` pair per sweep; timestamp in file name, **America/New_York**; README states no missing values per sweep [verified from README] |
| Sweep period | ~15 s per sweep (arXiv 2510.08752 summary) [verified via search summary; confirm from meta timestamps] |
| Files | `ResultsCC1Feb2022_SigMF.zip` 19.08 GB, `...CC2...zip` 20.66 GB, `...LW1...zip` 12.50 GB; SHA-256 values published by the Dryad API; individual sweeps are ~395 KB so a zip can be read by ranged requests / central directory |
| Access | Dryad API download needs a bearer token (401); scripted web downloads return 403 (bot protection). Browser download works only by hand; the full zip is 12-20 GB. [verified] |
| Caveats | Calibration / gain settings not documented in the metadata; sweeps are across a huge band, so per-channel dwell is short and unspecified; only 1 week limits chronological splits [unverified calibration] |

## 2. Preliminary comparison

| Criterion | NIST | AERPAW |
|---|---|---|
| Documented calibration & units | Yes, dBm with lab calibration, noise denoising | dBm stated, calibration undocumented |
| Native sparse / irregular observation | Yes (shuffled sweeps, 77% dead time, host pauses) | No (regular sweeps) |
| Span for chronological splits | ~15 months (site dependent) | 1 week |
| Prediction target quality | 2.4 GHz ~49% dwell prevalence | Unknown per channel |
| Access today | Origin down (524) | Works only via manual browser download of large zips |
| Subset practicality | Per-site tarballs 0.1-24 GB, CSV products small | Ranged zip reads possible but the zip must be fetched by browser/token |

**Recommendation (unchanged):** NIST primary, one mid-size site, 2.4 GHz channel (e.g. 2437 MHz), decided after reading the README and coverage tables. AERPAW fallback/transfer.

## 3. Decision triggers

- If NIST file downloads recover: read README + `occupancy durations` + `histograms` (~0.5 GB), finalise site/channel and schema, then proceed to M1.
- If NIST stays down: either (a) wait / retry on a schedule, or (b) switch the first experiment to AERPAW LW1 or CC1 via a manual browser download (12-20 GB) and subsample one day. Requires the project owner's go-ahead since it changes the first experiment (regular 15 s sampling, 1 week).

## 4. AERPAW reader (`data/aerpaw.py`) — implemented, verified on synthetic files only

Built while NIST was down. Reads sweeps straight from a node directory or `.zip` (members are read
on demand, selected by time from the file names, so a one-day subset of a 19 GB zip never needs
full extraction), extracts per-channel band statistics, and emits the canonical dwell table.

Documented facts it relies on (README): `results_YYYYMMDD_HHMMSS.sigmf-{data,meta}`, local time
`America/New_York`, 98,868 little-endian float32 values in dBm.

**Verified from the dataset's own `convert_SigMF_to_mat.py` (supplied by the project owner, 2026-10-03):**
`global["core:datatype"]` must be `rf32_le`; the frequency axis is stored explicitly per file in
`global["dataset:frequency_axis_MHz"]`; the payload length must equal the axis length. The reader uses
exactly these keys, checks the axis is identical across sweeps, and only falls back (annotation
edges, then an assumed 87-6000 MHz axis, flagged in `axis_source`) when the key is absent.

**Still unverified (needs one real file):** the actual bin count / axis endpoints, the actual sweep
period (README/paper imply ~15 s), per-hop dwell time (a nominal 1 ms snapshot is used) and absolute
calibration.

Snapshot data has no sub-dwell time bins, so the canonical `duty` column equals `float(occ_any)` for
AERPAW; results from the two datasets must not be compared on duty.

**Access:** Dryad blocks scripted downloads (403) and fronts file URLs with an interactive bot
check, so the zip has to be downloaded by hand in a browser (one node, 12-20 GB). Verify the file
against the SHA-256 published by the Dryad API (see section 1.2) before use.

## 5. Verified facts from the real CC1 archive (inspected 2026-10-03)

File: `ResultsCC1Feb2022_SigMF.zip`, 19,084,799,397 bytes, **SHA-256 matches Dryad**
(`c2b1e6d9…b642f747`). Deflate-compressed; contains macOS debris (`__MACOSX/._results_*`, 163-byte
stubs) which the reader skips.

| Item | Observed |
|---|---|
| Sweeps | 32,529 real pairs (each payload 395,472 B = 98,868 float32) |
| Axis | `global["dataset:frequency_axis_MHz"]`, 87.16-6019.18 MHz, 60.06 kHz steps, strictly increasing; same in every file sampled |
| Metadata | `core:datatype rf32_le`; `dataset:site "CC1"`; description says data were *converted from MATLAB*; `annotations[].core:label` reads `LW1` in this CC1 archive (wrong; ignored); `captures[0].core:datetime` has 1 s resolution with -05:00 offset |
| Sweep period | median 19 s (17-20 s jitter; timestamps are whole seconds), **not 15 s** |
| Sessions | main session 2022-02-08 17:50 -> 02-15 17:50 UTC: **168.0 h, 32,018 sweeps, no gaps > 5 min**; then a 9.2-day gap; tail session 02-24 22:18 -> 02-25 00:59 UTC (2.7 h, 511 sweeps) |
| Power units | dBm per 60 kHz bin; noise floor about -133 dBm (2.4 GHz), -137 dBm (5 GHz); LTE downlink bands -83..-98 dBm. This is per-bin power, NOT NIST-style 4 MHz band power: absolute values are not comparable across datasets |
| 5 GHz U-NII | completely idle all week (flat floor, 0.28 dB std): not usable as a target |
| Cellular 0.7-2.2 GHz | occupied nearly continuously (trivial target) |
| 2.4 GHz Wi-Fi (ch 1/6/11, 20 MHz) | band-mean excess over the 5th-percentile floor: +3 dB prevalence 12-20%, +6 dB 2-5%, +10 dB <1%; strong diurnal pattern (ch6: 7% overnight, 27% afternoon); lag-1 autocorrelation 0.16-0.21 decaying to ~0.08-0.10 at 300 sweeps (~1.6 h) |

**Derived products** (git-ignored, reproducible): `scripts/extract_aerpaw_blocks.py` reduces the
archive to 5 MHz block mean/max power per sweep in ~50 s (16 CPUs); `scripts/explore_blocks.py` and
`scripts/run_dev_baselines.py` use it.

### First dev-split findings (train 18,834 / dev 4,426 sweeps; test 8,005 held out, untouched)

Channel 2437 MHz, busy = band mean > train-period floor + 3 dB (train prevalence 0.189, **dev 0.090**):

| Model | Brier skill vs climatology | AUC |
|---|---|---|
| persistence | -0.74 | 0.51 |
| hour-of-day prior | -0.06 | 0.53 |
| age-conditioned persistence | +0.02 | 0.51 |
| decayed history / logreg / gradient boosting | +0.11 / +0.11 / +0.11 | 0.58 / 0.58 / 0.59 |

At +6 dB there is essentially no skill. Observations: (1) single-sweep labels behave like noisy
draws from a slowly varying activity rate, so persistence is worse than climatology and
calibrated rate estimators win; (2) dev is a weekend day and quieter than train (0.09 vs 0.19), so a
weekday-fitted time-of-day prior hurts: one week cannot separate diurnal from weekday/weekend effects;
(3) under clean-trained evaluation the logistic model extrapolates badly when evidence is delayed or
dropped (age far outside the training range), while trees clip. These are preliminary dev numbers,
produced before any design freeze; the target definition may change after seeing them (see below).

**Design implication:** a longer-window target (busy fraction of the next 5-60 min) is likely better
posed than a single 19 s snapshot; the handoff allows it (section 4C). Any such change is a
post-hoc decision made after viewing dev results and must be reported as such.
