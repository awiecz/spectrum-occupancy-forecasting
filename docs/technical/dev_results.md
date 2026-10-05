# Development-region results (PRELIMINARY — no test data used)

Data: AERPAW node CC1, main recording session (7.0 days, 32,018 sweeps, ~19 s apart), the 2437 MHz (Wi-Fi
channel 6) 20 MHz band; activity there is consistent with Wi-Fi but was not decoded (see
`sanity_checks.md`). Reproduce: `scripts/extract_aerpaw_blocks.py`, then
`scripts/run_window_dev.py data/processed/aerpaw_CC1_blocks5MHz.npz` (raw numbers in
`results/dev_window_forecast.json`).

> **Update:** the held-out test region has since been evaluated once (see
> [test_results.md](test_results.md)). The headline development conclusion below, that a 2-hour moving
> average matches the learned models, **did not fully replicate**: on the test region the level
> rises, shorter memories and ridge scored better on point estimates, and gradient boosting was
> unstable. Treat this file as development evidence only.

## Target and protocol

* Busy sweep = channel band-mean power > noise floor + 3 dB (floor = 5th percentile of the training
  period's band mean; refitted per fold).
* Target = busy **fraction** of sweeps in the next window `[o, o+W)`, W = 5, 15 and 60 min, forecast
  at origin `o` from observations that ended before `o` (minus any delay). Labels come from the
  intact stream; degradations touch inputs only.
* 3 rolling-origin folds over the development region (first 5.23 days): train on everything before
  `dev_start - (look-back + W)`, score one day of non-overlapping windows. Look-back = 7200 s.
* Fold dev days are progressively quieter than training: mean busy fraction 0.21 / 0.13 / 0.10 against
  a training mean of ~0.20 throughout. **A fixed climatology is therefore biased by construction;**
  skill against it overstates what a model learned. Two extra yardsticks are reported:
  skill vs the strong naive reference `recent_7200s` (busy fraction over the previous 2 h) and R²
  within each fold (variation around the fold's own mean).

## What was decided after seeing earlier dev scores (post-hoc, disclosed)

1. Move from next-sweep binary targets to window-fraction targets (single-sweep labels were nearly
   unpredictable: persistence scored worse than climatology).
2. Move from one dev day to rolling folds, add the `recent_7200s` reference and within-fold R², and
   report event metrics at several fixed cut-offs (the single train-derived cut-off was degenerate
   on the quiet dev day).

## Results, W = 15 min (pooled over 3 folds, 307 windows)

| Method | RMSE | MSE skill vs climatology | MSE skill vs 2 h recent | within-fold R² |
|---|---|---|---|---|
| climatology | 0.121 | 0 | -2.37 | -0.64 |
| hour-of-day prior | 0.138 | -0.30 | -3.39 | -1.14 |
| recent 5 min | 0.106 | +0.23 | -1.60 | -0.27 |
| recent 30 min | 0.066 | +0.70 | -0.01 | +0.51 |
| **recent 2 h** | **0.066** | **+0.70** | 0 | **+0.51** |
| decayed (tau 15 min) | 0.067 | +0.69 | -0.04 | +0.49 |
| ridge | 0.092 | +0.42 | -0.97 | +0.04 |
| gradient boosting | 0.085 | +0.51 | -0.66 | +0.19 |

W = 5 min and W = 60 min give the same ordering (2 h moving average best or tied). 95 % block
bootstrap (2 h blocks) intervals for the skill vs climatology: recent 2 h [+0.61, +0.79], gradient
boosting [+0.30, +0.67], ridge [+0.21, +0.58].

**Reading:** a plain 2-hour moving average of recent busy fraction is as good as anything tried;
the learned models do *not* beat it (they were fitted on earlier, busier days and the level drifted
down). Short-memory methods (5-min persistence) are poor because the labels are noisy; hour-of-day
priors hurt because one week confounds time-of-day with weekday/weekend and the level trend.

## Robustness (W = 15 min, clean-trained models; skill vs climatology)

| Perturbation of predictor inputs | recent 2 h | decayed | ridge | gradient boosting | GB trained with degradation |
|---|---|---|---|---|---|
| none | +0.70 | +0.69 | +0.42 | +0.51 | +0.26 |
| 50 % of sweeps dropped | +0.68 | +0.60 | +0.39 | +0.46 | +0.26 |
| 90 % dropped | +0.54 | +0.22 | +0.13 | +0.39 | +0.21 |
| 99 % dropped (~1 sweep / 32 min) | **-1.40** | -0.25 | **-9.78** | +0.33 | +0.22 |
| 25 % of time in 15-min outages | +0.69 | +0.65 | -0.05 | +0.50 | +0.29 |
| 25 % of time in 60-min outages | +0.64 | +0.62 | **-2.16** | +0.50 | +0.31 |
| evidence delayed 10 min | +0.69 | +0.70 | **-1.02** | +0.51 | +0.29 |
| delayed 1 h | +0.60 | +0.49 | -1.06 | +0.50 | +0.28 |
| delayed 2 h | +0.48 | +0.02 | +0.03 | +0.46 | +0.27 |

* The long moving average degrades gracefully until evidence becomes extremely sparse; the decayed
  estimator (tau = 15 min) loses its evidence once it is >2 h old.
* The clean-trained ridge model is unstable (extrapolates when input ages leave the training range);
  gradient boosting clips and is steadier; degradation-aware training makes both stable but costs
  clean-data accuracy.
* **Event false-idle** (high-activity window = busy fraction >= theta; theta fixed at 0.1/0.2/0.3,
  dev prevalence 0.59/0.22/0.08): at theta 0.2 the 2 h average misses 22 % of high-activity windows
  (forecast < 0.2); gradient boosting 18 %, ridge 13 %, but the learned models are biased upward
  (+0.03..0.05), which trades false-idle for false-busy. Equal-false-busy comparisons are still to do.

## Limits that matter

* One node, one week, one band; the level drifts across the week, so folds are not exchangeable and
  there are only three of them. Nothing here supports claims about other sites, seasons or bands.
* "Busy" = detected power above a train-derived threshold at a sweep snapshot (~19 s apart); it is
  not transmitter activity and says nothing about permission to transmit. The 3 dB threshold is a
  choice; sensitivity (X = 1.5/6 dB) has not yet been run on window targets.
* AERPAW sweeps are regular, so sparsity here is *simulated* by degrading predictor inputs; it is not
  native sparsity (that was the reason NIST was the first choice).
* Channel-level power is per 60 kHz bin, uncalibrated: only relative levels are meaningful.

## Next (not yet done)

Residual/relative-level learning (predict deviation from the 2 h average), threshold sensitivity,
equal-false-busy event analysis, the three required figures, then freeze the design and evaluate the
held-out test region exactly once; second node (LW1) as a transfer check.
