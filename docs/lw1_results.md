# Second node (LW1, rural): confirmation of the frozen design

Protocol: [lw1_protocol.md](lw1_protocol.md) (including Addendum A), written before the archive was
downloaded. Raw numbers: `results/node_LW1_window_forecast.json`; figures: `results/figures_lw1/`;
console log: `results/node_LW1_run_log.txt`.

## Protocol adherence

* One run: `results/NODE_LW1_RUN_LOCK.json` shows `reruns: 0` and fingerprint `7afc149a…d7301`.
* Archive checked against Dryad's SHA-256 before use. Structure (sessions, sweep period, metadata,
  frequency axis) was inspected before any activity label, prevalence or score was computed.
* Code changes since the CC1 test run, each verified to leave all CC1 development results
  numerically identical (0 differences): `run_fold` takes `channels` / `target_mhz`; the window-coverage
  filter uses the node's own median sweep period (30 s on LW1, 19 s on CC1; a hard-coded 19 s would have
  discarded every LW1 window); plotting code. These are recorded in the protocol addendum.
* Unlike CC1 there is **no separate held-out split on LW1**: the design was frozen on CC1, so all
  three LW1 folds ([2, 3), [3, 4), [4, 4.75) days of the 4.75-day main session; 261 windows at
  W = 900) are confirmation data. The folds share training data (expanding windows) and are not
  independent; one node, one week, one band.

## What the data looked like

LW1 is much quieter than CC1. Channel selection (first 60 % of the session): channel 6 (2437 MHz)
prevalence 0.128 at +3 dB, channels 1 and 11 only 0.003 each, so the target was **not degenerate** and
channel 6 was selected by the pre-registered rule. Pooled mean busy fraction at W = 900 is 0.073
(CC1: 0.17-0.20 in training, 0.19 on its test region); fold means 0.118, 0.044, 0.053.

## Scoring of the registered hypotheses

| | Registered claim | Outcome |
|---|---|---|
| **N1** | No learned model beats the best simple history (here *Last 30 min*) by more than 0.10 in pooled MSE skill (W = 900) | **Supported.** Relative to *Last 30 min*: ridge -0.01, gradient boosting -0.12, ridge+correction -0.06, GB+correction -0.09, degradation-trained ridge -0.08 and GB -0.37. |
| **N2** | Clean-trained gradient boosting has negative MSE skill vs *Last 2 h* (as on CC1 dev -0.66 and test -0.76) | **Rejected.** +0.10 [-0.34, +0.27] (degradation-trained: -0.11). |
| **N3** | Clean-trained ridge is the least robust of {*Last 2 h*, decayed, ridge, GB}: > 0.5 below *Last 2 h* in at least 3 of 4 scenarios (99 % dropout; 10, 30, 60 min late) | **Supported 4 of 4**, and ridge is the worst of the four in each (skill vs train mean: -12.2 / -1.8 / -10.5 / -29.9, against *Last 2 h* -2.2 / +0.3 / +0.2 / +0.1). |
| **N4** | LW1 is quieter than CC1 (pooled mean busy fraction below CC1's 0.17-0.20) | **Supported** (0.073). |

## W = 15 min (261 windows, three folds pooled)

| Method | RMSE | R² within fold | MSE skill vs *Last 2 h* [95 % block-bootstrap CI] |
|---|---|---|---|
| train mean | 0.101 | -0.48 | -0.52 |
| hour-of-day prior | 0.099 | -0.45 | -0.49 |
| last 5 min | 0.110 | -0.78 | -0.84 |
| last 30 min | **0.073** | +0.22 | +0.19 [0.00, +0.40] |
| last 2 h (reference) | 0.081 | +0.03 | 0 |
| decayed history | 0.075 | +0.18 | +0.15 [-0.05, +0.40] |
| ridge | 0.074 | +0.21 | +0.18 [-0.07, +0.37] |
| gradient boosting | 0.077 | +0.12 | +0.10 [-0.34, +0.27] |
| last 2 h + ridge correction | 0.075 | +0.18 | +0.15 |
| last 2 h + GB correction | 0.076 | +0.15 | +0.12 |
| ridge, degradation-trained | 0.076 | +0.16 | +0.13 |
| GB, degradation-trained | 0.086 | -0.07 | -0.11 |

At 5 minutes (789 windows) the 30-minute and decayed memories beat the 2 h average with intervals
excluding 0 (+0.15 [0.03, 0.26] and +0.19 [0.03, 0.29]); at 15 minutes the interval just touches 0, and at
60 minutes (63 windows) nothing is resolved. Within-fold R² is low everywhere (best +0.24 at 5 min;
negative at 60 min): within a day the busy share on this node is largely unexplained by recent history.

Robustness (W = 900, MSE skill vs train mean; random-mask averages), none / 90 % dropout / 99 % /
60-min outages / 10 min late / 60 min late:
*Last 2 h* 0.34 / 0.12 / -2.15 / 0.28 / 0.29 / 0.09; decayed 0.45 / 0.07 / -0.06 / 0.40 / 0.44 / 0.17;
ridge 0.46 / -0.20 / -12.22 / -1.62 / -1.78 / -29.87; gradient boosting 0.41 / 0.29 / 0.23 / 0.39 / 0.42 /
0.26.

Event metrics are not interpretable here: only 10 % of windows reach a busy fraction of 0.2 and 4 %
reach 0.3 (a few dozen and about ten windows).

## Reading across the two nodes (and CC1's development vs test regions)

* **Replicated on both nodes:** learned models give at most a marginal gain over a tuned history
  average and never clearly beat the best simple one; a clean-trained ridge model is unstable under
  delayed or dropped evidence while history averages degrade gracefully until evidence is extremely
  sparse; the hour-of-day prior does not help reliably (R² -1.14 CC1 dev, +0.04 CC1 test, -0.45 LW1).
* **Not stable:** the best memory length. A 2 h average was best on CC1's development days but is
  beaten by 30-minute or decayed memories on CC1's test region and on LW1.
* **Gradient boosting** was clearly worse than the simple histories on CC1 in both regions (RMSE 0.085
  vs 0.066 on the development days; 0.126 vs 0.095 on the test region) but not on LW1 (0.077 vs
  0.073-0.081). A level-shift explanation does not hold up: the level also moved a lot on LW1
  (fold means 0.118, 0.044, 0.053) and on CC1's development days (drifting down). I have no tested
  explanation for why boosting suffers on CC1 only.
* Limits unchanged: one week per node, one band, simulated rather than native sparsity, uncalibrated
  per-bin power; "busy" is detected power above a threshold at one sensor, not transmitter activity
  or permission to transmit. No claim is made about other sites, seasons or bands.
