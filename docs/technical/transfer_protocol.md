# Pre-registered protocol: cross-node transfer (CC1 to LW1)

Written 2026-10-03, **before any transfer result existed.** `scripts/run_transfer.py` fingerprints code
plus this document, writes a lock (`results/TRANSFER_RUN_LOCK.json`) before computing anything, and
refuses a second run (a re-run needs an explicit flag, is counted and must be disclosed).

**Question.** Does a model fitted on one node forecast the busy fraction of the next window on a
different node as well as a model fitted on that node's own earlier data, and does adding the other
node's data help? (The in-node results are in `docs/lw1_results.md` and `docs/dev_results.md`.)

## Honest status of the evaluation data

* The LW1 folds were already scored in the in-node LW1 run, and the CC1 development folds in the
  development work. This analysis **re-uses those windows for a different question**; they are not
  untouched data. Protection comes from a fixed design and predictions registered now, not from
  unseen data. The **CC1 test region stays out of this analysis entirely** (its rows are removed
  before any computation).
* In-node arms are recomputed with the same code and must reproduce the stored in-node numbers (check
  below).

## Design (everything else inherited from `configs/frozen_experiment.yaml`)

* Same features, models, hyper-parameters, windows W in {300, 900, 3600} s (primary W = 900),
  busy = band-mean power > floor + 3 dB, same embargo (7200 s + W), same rolling-origin one-day folds
  as each target node's own protocol (LW1: [2,3), [3,4), [4,4.75) days; CC1 development: [2,3),
  [3,4), [4,5.23) days).
* **Arms**, for each target node and fold (training data = windows with origin <= `fold_start -
  (7200 s + W)`, for *every* node used; the same cutoff for source and target, so a source node's
  simultaneous recordings never leak into a fold):
  1. **in-node:** train on the target node only.
  2. **transfer:** train on the source node only (its own noise floor and labels); the target is
     labelled with the *target's own* noise floor from its pre-cutoff, unlabelled power (so "transfer"
     means transferring the model, with per-node floor calibration).
  3. **joint:** train on source and target together.
  4. **transfer-absolute (exploratory):** as transfer, but the target is labelled with the source node's
     absolute dBm thresholds (no target-node calibration at all).
* Primary direction: **CC1 -> LW1**. Secondary, exploratory (no hypotheses): LW1 -> CC1 on CC1's
  development folds.
* Both nodes use the same time zone (local Raleigh time, `America/New_York`), channel 2437 MHz as
  target and 2412/2437/2462 MHz as feature channels.
* **Metric.** Pooled mean squared error over the three folds. Transfer penalty = MSE(arm) /
  MSE(in-node), with a 95 % paired block-bootstrap interval (2 h blocks, 1000 resamples) on the
  per-window squared errors. Also reported: RMSE, within-fold R^2, MSE skill relative to the best
  simple history, and the same table for W = 300 and 3600.

## Consistency checks (abort and disclose if they fail)

* The in-node LW1 arm reproduces `results/node_LW1_window_forecast.json`, and the in-node CC1 arm
  reproduces `results/dev_window_forecast.json`, RMSE for every non-augmented model within 1e-6 (floating-point summation order).
* A rehearsal with identical source and target data (transfer must equal in-node exactly, joint must
  not crash) is run before the real analysis.

## Hypotheses (predictions, all at W = 900, CC1 -> LW1)

* **T1.** The three history averages (*Last 30 min*, *Last 2 h*, *decayed history*) are essentially
  training-free: their transfer penalty lies in [0.95, 1.10].
* **T2.** Gradient boosting and ridge (trained on raw features) lose from transfer: penalty >= 1.10
  for each.
* **T3.** The residual models transfer better than their raw counterparts: penalty(ridge + correction)
  < penalty(ridge) and penalty(GB + correction) < penalty(GB).
* **T4.** Adding the other node does not help: the joint-training penalty (MSE joint / MSE in-node) is
  >= 0.95 for each of ridge, gradient boosting and both correction models.

Exploratory, no prediction: transfer-absolute vs transfer, LW1 -> CC1, and the W = 300 / 3600 tables.

## Reporting rules

All arms, models, windows and both directions are reported regardless of outcome, with the number
of windows beside every figure. No design change after seeing results. Limits as before: one week per
node, one band, one channel, simulated sparsity, uncalibrated per-bin power, detected activity is not
transmitter activity or permission to transmit; two nodes cannot support general claims about
transfer across sites.
