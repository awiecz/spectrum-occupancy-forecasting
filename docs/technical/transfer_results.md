# Cross-node transfer results (CC1 <-> LW1)

Protocol: [transfer_protocol.md](transfer_protocol.md), written before any transfer result existed. Raw
numbers: `results/transfer_results.json`; console log: `results/transfer_run_log.txt`; figure and data
table: `results/figures_transfer/fig4_transfer.png` / `.csv`.

## Protocol adherence

* One run: `results/TRANSFER_RUN_LOCK.json` (`reruns: 0`, fingerprint `53c330ce…51cfb`, identical to
  the rehearsal's). A rehearsal with the same table as both nodes gave transfer = in-node exactly
  (all penalties 1.00).
* **Consistency checks passed with a difference of exactly 0:** the recomputed in-node arms reproduce
  the stored LW1 results and the stored CC1 development results (max |RMSE difference| = 0.00e+00 for
  every non-augmented model and window).
* Causality across nodes: every node's training data stops at the same cutoff
  (`fold_start - (7200 s + W)`), so a source node's simultaneous recordings never leak into a fold; a
  unit test corrupts the source node after the cutoff and checks predictions do not move. The CC1 test
  region was removed before any computation.
* The evaluation windows were **already scored** in the in-node LW1 run and the CC1 development work;
  this analysis re-uses them for a different question. Protection comes from the fixed design and
  registered predictions, not from unseen data.

## Primary result: train on CC1, forecast LW1 (W = 15 min, 261 windows, three folds)

Penalty = pooled MSE of the arm / pooled MSE of training on LW1 only (1 = no penalty), with 95 %
paired block-bootstrap intervals.

| Model | RMSE in-node | RMSE trained on CC1 | Transfer penalty [95 % CI] | Joint (CC1 + LW1) penalty [95 % CI] |
|---|---|---|---|---|
| train mean (climatology) | 0.101 | 0.157 | 2.43 [1.78, 3.09] | 1.55 |
| hour-of-day prior | 0.100 | 0.212 | 4.54 [2.76, 7.13] | 1.96 |
| last 30 min / last 2 h | 0.073 / 0.081 | same | 1.00 [1.00, 1.00] | 1.00 |
| decayed history | 0.075 | 0.075 | 1.00 [1.00, 1.01] | 1.00 |
| ridge | 0.074 | 0.142 | 3.74 [2.02, 7.24] | 1.03 [0.94, 1.12] |
| gradient boosting | 0.078 | 0.126 | 2.64 [1.61, 4.78] | 0.94 [0.85, 1.03] |
| last 2 h + ridge correction | 0.075 | 0.104 | 1.93 [1.35, 3.12] | 1.05 [0.99, 1.13] |
| last 2 h + GB correction | 0.076 | 0.088 | 1.34 [1.13, 1.83] | 1.07 [0.98, 1.20] |

The same ordering holds at 5 min (ridge 2.49, boosting 2.31, corrections 1.60 and 1.06) and 60 min
(ridge 8.21 [3.90, 16.8], boosting 2.64, corrections 1.48 and 1.34; only 63 windows).

## Registered hypotheses (W = 900, CC1 -> LW1)

| | Prediction | Outcome |
|---|---|---|
| **T1** | History averages are training-free: penalty in [0.95, 1.10] | **Supported** (1.00, 1.00, 1.003) |
| **T2** | Ridge and boosting lose from transfer: penalty >= 1.10 | **Supported** (3.74 and 2.64, intervals above 1) |
| **T3** | Residual models transfer better than their raw counterparts | **Supported** (1.93 < 3.74; 1.34 < 2.64; same at 5 and 60 min) |
| **T4** | Adding the other node does not help: joint penalty >= 0.95 for all four learned models | **Rejected** by its criterion: boosting 0.94 [0.85, 1.03] (0.87 [0.82, 0.95] at 5 min, 0.82 at 60 min); ridge 1.03, corrections 1.05 and 1.07 |

T4 failed narrowly and only for boosting, and the interval at 15 min includes 1; the point estimate
is consistent across windows, so the registered claim "no gain for any learned model" does not stand.

## What the transfer penalty looks like

* **History averages carry no node-specific parameters** and transfer unchanged; the penalty sits
  entirely in the models, the hour-of-day table and the train mean, which encode the *source node's*
  activity level and daily profile (CC1 training mean about 0.20; LW1's own training mean about
  0.11-0.13, and its evaluation folds 0.04-0.12).
* Transferred ridge and boosting over-predict (mean bias +0.07 vs +0.02 and +0.03 in-node);
  bias accounts for roughly a third to a half of their excess MSE (from pooled RMSE and bias), the rest
  is a mis-specified response. Residual-on-level models, whose forecast is anchored to the target's own
  recent level, lose much less (bias +0.02 for the GB correction).
* **Pooling was approximately neutral or beneficial here.** At 15 minutes, joint-training penalties
  for the four learned models ranged from 0.94 to 1.07, with intervals including 1. Boosting had
  a lower point estimate; the other three had slightly higher ones. This limited comparison does
  not establish a generally safe pooling strategy.

## Exploratory: train on LW1, forecast CC1's development folds (W = 900, 307 windows)

No hypotheses were registered for this direction. Transfer penalties: ridge 4.20 [3.15, 5.79],
last 2 h + ridge correction 6.27 [4.46, 8.79] (so T3's pattern does **not** generalise: the ridge
correction is worse than raw ridge here), boosting 0.98 [0.65, 1.52], last 2 h + GB correction
1.10 [0.93, 1.31], train mean 0.75 [0.44, 1.06] (LW1's lower activity level happens to be closer
to CC1's declining development days than CC1's own training mean). Joint training helped the learned
models on CC1's development folds (ridge 0.52, boosting 0.68, hour-of-day 0.50; point estimates, see
the JSON for intervals). Linear models transfer badly in both directions, boosting well in the reverse
direction; I do not know why. One candidate is that LW1's inactive channels 1 and 11 leave the
cross-channel features of a LW1-trained ridge unconstrained, which would break on CC1 where they are
active; this was not tested.

## Calibration sensitivity (exploratory, W = 900)

Labelling the target node with the *source node's absolute dBm thresholds* (no per-node floor
calibration) changes the task completely: LW1's busy fraction per fold falls from [0.118, 0.044, 0.053]
to [0.017, 0.003, 0.002], and CC1's rises from [0.206, 0.126, 0.096] to [0.408, 0.308, 0.251]. The two
nodes' noise floors differ by only about 1 dB, yet "busy" prevalence changes by a factor of 2-25. The
RMSE of that arm is therefore not comparable with the others (different labels) and no penalty is
reported for it. The practical lesson: with uncalibrated per-bin power, a busy/idle label is
meaningful only relative to a node's own noise floor, and occupancy defined by an absolute threshold
cannot be compared across nodes.

## Limits

Two nodes in the same region and week (same time zone, similar 2.4 GHz environment), one band and one
channel, three overlapping folds per direction, many models and windows compared (single intervals are
not confirmatory), in-node models trained on only 2-4 days. Sparsity is simulated, power is
uncalibrated, and "busy" is detected power above a threshold at one sensor, not transmitter activity or
permission to transmit. Nothing here supports general claims about transfer across sites.


## Additional interpretation limits

The LW1 target-channel selection period overlaps its first scored fold, including the target
windows reused here; see [the selection-overlap note](lw1_results.md#channel-selection-overlap).
Any equal-false-busy metrics in the result JSON are evaluation-label oracle ROC comparisons,
not frozen operating thresholds. Historical registration and fingerprint limits are described
in [the documentation index](../index.md#protocols-and-historical-locks).
