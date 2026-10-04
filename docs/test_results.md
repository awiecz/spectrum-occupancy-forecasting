# Held-out test results (evaluated once)

Protocol: [test_protocol.md](test_protocol.md), written before the test region was read. Frozen
design: `configs/frozen_experiment.yaml`. Raw numbers: `results/test_window_forecast.json`;
figures: `results/figures_test/` (each with a CSV of the plotted values).

## Protocol adherence (read this first)

* **One evaluation.** `results/TEST_RUN_LOCK.json` records `reruns: 0`, the code fingerprint
  (`5e0cc61d…a2cb66`, SHA-256 over `src/`, `scripts/run_test_once.py`, the config and the protocol)
  and the results hash. The same fingerprint was produced by a rehearsal that ran the whole script on
  *development-only* data (lock and output in a scratch folder) before the real run.
* The test region is the last 25 % of the CC1 main session (5.23 to 7.00 days after the session start; about 1.77 days,
  from Sunday 13 Feb ~18:20 to Tuesday 15 Feb 12:50 local, including the whole of Monday). Training used everything before `test_start - (7200 s + W)`, so the
  former development days were in the training set. Noise floors were refit on that period.
* At the time of this results write-up, post-run edits were only to plotting code (`viz/figures.py`, `scripts/make_figures.py`: titles,
  axis limits, label placement) and regenerated development figures; no evaluation code changed.
  The lock's fingerprint differed from that state by plotting code only. The current source also
  includes later portability and documentation changes; see [historical-lock notes](index.md#protocols-and-historical-locks).
* Before the run, one *registered prediction* (H3) was corrected to match development evidence; the
  corrected text is the one in the protocol file.
* Many models x windows x metrics are reported and compared, so single intervals below should not be
  read as confirmatory after multiplicity.

## Scoring of the registered hypotheses

| | Registered claim | Outcome |
|---|---|---|
| **H1** | No learned model beats *Last 2 h* by more than 0.10 MSE skill at W = 900 (evidence of value needs a model above +0.10 with a CI excluding 0) | **Rejected by its criterion.** Ridge: +0.24 [+0.01, +0.33]; ridge + correction: +0.17 [+0.02, +0.23]. Not at W = 300 or 3600 (CIs include 0), and gradient boosting is much worse than the reference (-0.76; degradation-trained -1.08). The non-learned *decayed history* is best on point estimates (+0.33, CI [-0.15, +0.62]). |
| **H2** | Hour-of-day prior has negative within-region R² at every window | **Rejected.** R² = +0.04 / +0.04 / +0.05 (5 / 15 / 60 min): slightly positive, the test region does have a time-of-day pattern the prior partly catches. |
| **H3** | Robustness (W = 900): *Last 2 h* keeps skill > 0.30 at 90 % dropout; clean-trained ridge is > 0.5 worse than *Last 2 h* under 10-60 min delay; at 99 % dropout only gradient boosting stays above 0 | **Supported on every part** (0.59; ridge -5.9 / -6.2 / -4.4 vs *Last 2 h* -0.2 / -0.5 / -0.9 in skill vs the clean reference; at 99 %: gradient boosting +0.14 / +0.12, *Last 2 h* -1.23, ridge -10.5). |
| **H4** | Test-region mean busy fraction exceeds the last development day's | **Supported.** 0.191 vs 0.096 (training mean 0.167). |

## W = 15 min (169 non-overlapping windows)

| Method | RMSE | R² (test region) | MSE skill vs *Last 2 h* [95 % block-bootstrap CI] |
|---|---|---|---|
| train mean | 0.170 | -0.02 | -2.20 |
| hour-of-day prior | 0.165 | +0.04 | -2.02 |
| last 5 min | 0.105 | +0.61 | -0.23 |
| last 30 min | 0.080 | +0.77 | +0.29 [-0.22, +0.61] |
| last 2 h (reference) | 0.095 | +0.68 | 0 |
| decayed history (15 min) | **0.078** | **+0.79** | +0.33 [-0.15, +0.62] |
| ridge | 0.083 | +0.76 | +0.24 [+0.01, +0.33] |
| gradient boosting | 0.126 | +0.44 | -0.76 [-1.31, +0.04] |
| last 2 h + ridge correction | 0.087 | +0.73 | +0.17 [+0.02, +0.23] |
| last 2 h + GB correction | 0.095 | +0.68 | +0.01 [-0.07, +0.15] |
| ridge, degradation-trained | 0.091 | +0.71 | +0.09 [-0.06, +0.18] |
| GB, degradation-trained | 0.137 | +0.34 | -1.08 [-1.68, -0.13] |

W = 5 and 60 min: the history averages and ridge cluster near RMSE 0.10-0.12 (5 min) and 0.07-0.09
(60 min); gradient boosting is again worse than the 2 h reference (-0.03 and -0.43); there are only
42 windows at 60 min, so those intervals are wide.

## Event-threshold interpretation

The equal-false-busy rates reported below are **oracle ROC comparisons**: for each model,
`event_tradeoff` uses the evaluated region's non-event labels to choose its score threshold.
The thresholds were not selected on development data and then frozen for this test region.
These numbers describe separation at empirical ROC operating points, not an achieved deployment
false-busy budget or a prospective false-idle guarantee. Discrete scores and ties also mean the
quantile rule need not attain exactly 10% or 20% false-busy. A deployable evaluation would freeze
thresholds on earlier data and report both realized rates on later data as a separate analysis.

## What changed relative to development

* **The development conclusion "a 2-hour moving average is as good as anything" did not replicate.**
  On the test region the level rises (Monday), a 2 h average lags it, and shorter memories (30 min,
  decayed history) and ridge score better on point estimates. The best memory length depends on how
  fast the level moves; no single window was best in both regions.
* **Gradient boosting was clearly worse than the simple histories on CC1** (here and on the
  development days). The test region coincided with a doubling of activity, but boosting was also
  worse on the development days (level drifting down) and fine on LW1, so a level-shift explanation
  is not supported; the cause is untested. The ridge model is better here but, trained on clean
  data, it is the least robust model under delayed or dropped evidence.
* **Event view** (high-activity window = busy fraction >= 0.2; 33 % of windows, 56 events):
  at equal false-busy rates of 20 %, false-idle is 0.09 for gradient boosting and the
  "2 h + correction" models, 0.11 for ridge, 0.15 for *Last 2 h* and 0.22 for decayed history; at a
  10 % false-busy budget the history averages and learned models miss 42-58 % of events (hour-of-day
  prior 78 %). AUC is 0.85-0.89 for them, 0.77 for last-5-min and 0.71 for the hour-of-day prior. At theta = 0.3 (24 events) the miss rates are 39-65 %: too few events for a ranking.
* Robustness (W = 900, MSE skill vs train mean; random-mask averages): none / 90 % dropout / 99 %
  dropout / 60-min outages / evidence 10 min late / 60 min late / 120 min late:
  *Last 2 h* 0.69 / 0.59 / -1.23 / 0.69 / 0.64 / 0.40 / 0.14;
  decayed 0.79 / 0.51 / -0.05 / 0.78 / 0.75 / 0.37 / 0.01;
  ridge 0.76 / 0.37 / -10.46 / -2.37 / -1.16 / -0.67 / -22.57;
  gradient boosting 0.45 / 0.29 / +0.14 / 0.43 / 0.42 / 0.39 / 0.26.

## Honest bottom line

On real AERPAW data the forecast of the busy share of the channel-6 band (consistent with Wi-Fi, not decoded) over the next 5-60 minutes is
dominated by tracking the slowly varying activity level; learned models give at most a marginal
gain over a tuned moving average and can be much worse, the preferred memory length is not stable
across days, and robustness differs sharply between methods (the simple averages degrade gracefully
until evidence is extremely sparse; the ridge model is unstable; boosted trees hold up under sparsity
but were clearly worse than simple histories on CC1 in clean conditions). All of this is one node, one week, one band, with simulated sparsity and
uncalibrated per-bin power, and "busy" means detected power above a threshold at one sensor, not
transmitter activity or permission to transmit.

## Caveats added after a self-audit (see [sanity_checks.md](sanity_checks.md))

* The busy signal is real and channel-shaped (negative control, drift and spectral-shape checks), but
  it was not decoded, so "Wi-Fi" is an inference.
* The test region holds about **33 effective independent windows** at W = 15 min (169 nominal; AR(1)
  estimate), and the models were trained on only a few days. Rankings between close methods are
  unresolved, the reported intervals are probably too narrow, and "learned models do not clearly beat
  tuned averages" says little about what more training data would do.
* The most transferable findings are the evidence-age curve and the calibration sensitivity, not the
  model ranking.
