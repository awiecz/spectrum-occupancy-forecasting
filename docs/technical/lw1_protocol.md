# Pre-registered protocol: second node (LW1, rural)

Written 2026-10-03, **before the LW1 archive was downloaded or inspected.** Purpose: check whether
the CC1 findings (urban node) hold on an independent node, with the design **frozen** (no tuning).
`scripts/run_second_node.py` fingerprints code + this document, writes a lock file before loading
any LW1 data, and refuses a second run (re-runs need an explicit flag, are counted and must be
disclosed).

## Design (all inherited from the frozen CC1 design, `configs/frozen_experiment.yaml`)

* Same features, models, hyper-parameters, windows W in {300, 900, 3600} s, busy = band-mean power >
  node-specific noise floor + 3 dB (floor = 5th percentile of the training period's band mean, per
  channel), same embargo (7200 s + W), same degradation scenarios (5 random masks), same metrics.
* Evaluation code change since the CC1 test run: `run_fold` now accepts `channels` and `target_mhz`
  arguments (defaults unchanged; re-running the development experiment gave 0 numeric differences).
* **Target channel (rule fixed now, applied to LW1's first 60 % only):** among Wi-Fi channels 1, 6, 11
  (2412 / 2437 / 2462 MHz, 20 MHz) pick the one with the highest busy prevalence at +3 dB, using the
  noise floor from that same first 60 % of the main session. If the best prevalence is below 5 %, the
  target is declared **degenerate**: only the descriptive selection table is reported, no models.
* **Folds:** the main session (longest gap-free recording) is evaluated with expanding-window
  rolling-origin folds of 1 day each, starting at day 2 (training = everything before
  `fold_start - embargo`), including a final partial fold if it is at least half a day. There is no
  separate dev/test split on LW1: the design is frozen, so all folds are confirmation data. Nothing
  may be changed after seeing results; any change is a new, disclosed analysis.

## Hypotheses (predictions from CC1 evidence)

* **N1.** At W = 900, no learned model (ridge, gradient boosting, either "2 h + correction") beats the
  best of the three simple histories (*Last 30 min*, *Last 2 h*, *decayed history*, chosen per
  pooled RMSE) by more than 0.10 in pooled MSE skill relative to that best simple history.
* **N2.** Gradient boosting trained on clean data has negative MSE skill vs *Last 2 h* (pooled,
  W = 900), as on CC1 development (-0.66) and test (-0.76).
* **N3.** Clean-trained ridge is the least robust of {*Last 2 h*, decayed history, ridge, gradient
  boosting}: in at least 3 of the 4 scenarios {99 % dropout, evidence 10 min late, 30 min late,
  60 min late} its MSE skill vs the training mean is more than 0.5 below that of *Last 2 h*.
* **N4.** The rural node is quieter: pooled mean busy fraction at W = 900 on LW1 is below CC1's
  training-region mean (0.17-0.20).

Exploratory, no prediction: whether the hour-of-day prior is above or below zero R², and which
memory length is best.

## Reporting rules

All models, windows and scenarios are reported regardless of outcome, with the number of windows and
events beside every rate. If the target is degenerate, that is the result. A cross-node transfer
analysis (train on CC1, apply to LW1) is *not* part of this protocol; it would be a separate,
disclosed follow-up. Limits as before: one week, one band, simulated sparsity, uncalibrated per-bin
power, detected activity is not transmitter activity or permission to transmit.

## Addendum A (written after the archive was downloaded and its *structure* inspected, before any
## LW1 activity label, prevalence or model score was computed)

* Archive `ResultsLW1Feb2022_SigMF.zip`: SHA-256 matches Dryad (`595e7ea4…11401951`); 21,617 sweeps;
  98,868 bins; the frequency axis is identical to CC1's; `dataset:site` = `LW1`.
* **Sweep period is about 30 s on LW1 (median), not 19 s as on CC1.** The window-coverage filter
  (a window is kept if it holds at least 80 % of the expected sweeps) previously used 19 s as the
  expected spacing; it now uses the node's own median sweep period. For CC1 that median is exactly
  19 s: re-running the CC1 development experiment after the change gave 0 numeric differences. This is a
  data-quality parameter, not a design change.
* **Main session** (longest gap-free recording; session 0): 2022-02-08 17:57 to 02-13 11:55 UTC,
  114.0 h = 4.75 days, 13,620 sweeps. Two other recordings exist (a single stray sweep on
  02-20 and a 44.6 h session 02-20 19:04 to 02-22 15:42 UTC); they are **not used** in this protocol.
* With a 4.75-day session the fold rule yields three folds: [2, 3) d, [3, 4) d and a partial
  [4, 4.75) d (0.75 d >= 0.5 d), with expanding training windows, exactly as the CC1 development
  folds.
* `scripts/extract_aerpaw_blocks.py` now names its output from the archive file name (with a
  warning if the metadata disagrees); the metadata of this archive agrees (`LW1`).
