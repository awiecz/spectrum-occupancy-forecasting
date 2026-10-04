# Pre-registered protocol for the held-out test evaluation

Written 2026-10-03, **before the test region (the last 25 % of the CC1 main session, about
1.77 days, Sunday evening to Monday) was read by any script or figure.** The exact design is frozen
in `configs/frozen_experiment.yaml`; `scripts/run_test_once.py` fingerprints (SHA-256) the source
code, the config and this document, writes a lock file before loading any test data, and refuses a
second run. A re-run (e.g. after a crash) is possible only with an explicit flag, is counted in
`results/TEST_RUN_LOCK.json`, and must be disclosed.

## What is evaluated

* Data: AERPAW node CC1, main session, Wi-Fi channel 2437 MHz (20 MHz band), February 2022.
* Target: busy fraction of the next W seconds, W in {300, 900, 3600}; busy = band-mean power above the
  training-period noise floor + 3 dB. Primary window: **W = 900 s**.
* Training: everything before `test_start - (7200 s look-back + W)`; this now includes the former
  development folds. Noise floors are refit on that training period. No model is selected,
  tuned or dropped using test data; **all** models and windows are reported whatever the outcome.
* Test windows: non-overlapping windows covering the test region. The test region is one block of
  contiguous time, so there is a single evaluation fold (about 170 windows at W = 900, 42 at 3600).

## Hypotheses and predictions (from development evidence only)

* **H1 (main).** No learned model (ridge, gradient boosting, or either "last 2 h + correction") beats
  the *Last 2 h* moving average by more than 0.10 in MSE skill at W = 900 (`mss_vs_ref <= +0.10` for
  each). Evidence that learning adds value requires a learned model with `mss_vs_ref > +0.10` *and* a
  95 % block-bootstrap interval for it that excludes 0. I expect H1 to hold.
* **H2.** The hour-of-day prior has negative within-region R² (worse than the test region's own
  mean) at every window length.
* **H3 (robustness, W = 900).** Under 90 % random sweep dropout the *Last 2 h* average keeps MSE
  skill vs climatology above 0.30, and under evidence delayed by 10 to 60 minutes the clean-trained
  ridge is worse than *Last 2 h* by more than 0.5 in `mss_vs_ref`. Under 99 % dropout gradient
  boosting (clean- or degradation-trained) stays above 0 skill vs climatology while *Last 2 h* and
  ridge fall below 0.
* **H4 (level shift).** The test region's mean busy fraction (W = 900) is higher than that of the last
  development day (0.096), because the test region includes a working day (Monday). If this fails,
  the weekday/weekend explanation of the level drift is not supported.

## Primary and secondary metrics

Primary: RMSE and within-region R² at W = 900, and `mss_vs_ref` (skill relative to *Last 2 h*).
Secondary: MSE skill vs the training mean (reported with the caveat that level drift biases that
baseline), calibration slope, event metrics for "high-activity window" (busy fraction >= 0.1 / 0.2 /
0.3) including false-idle at equal false-busy rates (10 % and 20 %) and AUC at theta = 0.2,
robustness tables (random-mask average over 5 seeds, with spread) and block-bootstrap intervals
(2 h blocks, 1000 resamples).

## Interpretation rules

* Results that contradict the development pattern are reported as they are; the design is not
  re-tuned and no metric is swapped after seeing test numbers.
* Event-based false-idle numbers are interpretable only if the test region contains enough
  high-activity windows; the number of event windows is reported next to every rate.
* Limits that stand regardless of outcome: one node, one week, one band, simulated (not native)
  sparsity, uncalibrated per-bin power, detected activity at one sensor is not transmitter activity
  or permission to transmit.
