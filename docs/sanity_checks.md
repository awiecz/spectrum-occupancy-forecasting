# Sanity checks on the measurements and the headline numbers

Reproduce with `uv run python scripts/sanity_checks.py --out reproductions/sanity_checks.json`
(writes a separate reproduction; original console
output in `results/sanity_checks_log.txt`). The descriptive controls use whole-period noise floors, no
forecasting, and the CC1 *development* region only (the held-out CC1 test region is never read by this
script). Section 5 reads stored result files and computes nothing new on any test data.

**Terminology.** The pre-registered protocol documents (`test_protocol.md`, `lw1_protocol.md`,
`transfer_protocol.md`) say "Wi-Fi channel 6". They are frozen (their text is fingerprinted into the
lock files) and were not edited. What was actually measured is **power in the 2437 MHz +- 10 MHz
band**: the activity there is *consistent with* a Wi-Fi channel (checks 1-3 below) but nothing was
decoded, so it is not verified to be Wi-Fi. Elsewhere in the repository the wording is "2.4 GHz
channel-6 band".

## 1-3. Is the "busy" signal real, and channel-shaped?

| Check | CC1 (dev region) | LW1 |
|---|---|---|
| **Negative control:** busy prevalence in five idle 5 GHz bands (same rule, floor + 3 dB) | <= 0.0001 | <= 0.0006 |
| Target band prevalence (same rule) | 0.166 | 0.097 |
| **Receiver drift:** hourly-median span of the idle-band level | 0.10 dB | 0.12 dB |
| Target band quiet-state (10th percentile) level span over the day | 0.45 dB | 0.13 dB |
| **Spectral shape:** busy-minus-idle excess inside 2430-2445 MHz | +3.5 dB | +2.7 dB |
| same, outside the channel (2455-2490 MHz) | +0.3 dB | +0.0 dB |

The busy rule does not fire on noise, the receiver does not drift anywhere near the 3 dB threshold,
and the busy state is a bump localised on the channel-6 band that vanishes away from it, as a ~20 MHz
emitter would produce, not a wideband artefact or a frequency hopper. This does not identify the
emitter and there is no ground-truth label.

## 4. How much independent data is there?

Autocorrelation of the 15-minute busy fractions at lags 15 / 30 / 60 / 120 min: CC1
[0.67, 0.59, 0.60, 0.47]; LW1 [0.57, 0.55, 0.41, 0.23]. An AR(1) approximation (`n (1-rho)/(1+rho)`
with the 15-minute lag) turns the nominal window counts into roughly **60 effective windows for CC1
development (307 nominal), 33 for the CC1 test region (169), and 72 for LW1 (261)**. The 2 h
block bootstrap used for the intervals still leaves residual dependence between blocks (autocorrelation
0.47 on CC1 at 2 h), so the reported intervals are probably **too narrow** and rankings between
close methods (for example *Last 30 min* vs *decayed history* vs ridge) are not resolved. In
addition, the learned models were trained on only 2-5 days, so "learned models do not clearly beat
tuned moving averages" is a statement about these small training sets, not about what more data
could do.

## 5. The value of old evidence

MSE skill vs the train mean at W = 15 min when the newest usable sweep is `d` minutes old (stored
results; the delay scenarios are deterministic), and the delay at which
skill falls to half of its no-delay value (linear interpolation on this grid, "inf" = never within
120 min):

| Result set | Estimator | d = 0 | 1 | 10 | 30 | 60 | 120 | Half-life |
|---|---|---|---|---|---|---|---|---|
| CC1 dev | decayed history (15 min) | +0.69 | +0.69 | +0.70 | +0.64 | +0.49 | +0.02 | 78 min |
| CC1 dev | last 2 h average | +0.70 | +0.70 | +0.69 | +0.65 | +0.60 | +0.48 | > 120 min |
| CC1 dev | gradient boosting | +0.51 | +0.51 | +0.51 | +0.49 | +0.50 | +0.46 | > 120 min |
| CC1 test | decayed history | +0.79 | +0.78 | +0.75 | +0.63 | +0.37 | +0.01 | 57 min |
| CC1 test | last 2 h average | +0.69 | +0.68 | +0.64 | +0.54 | +0.40 | +0.14 | 72 min |
| CC1 test | gradient boosting | +0.45 | +0.44 | +0.42 | +0.40 | +0.39 | +0.26 | > 120 min |
| LW1 | decayed history | +0.45 | +0.43 | +0.44 | +0.26 | +0.17 | +0.01 | 44 min |
| LW1 | last 2 h average | +0.34 | +0.34 | +0.29 | +0.18 | +0.09 | +0.04 | 32 min |
| LW1 | gradient boosting | +0.41 | +0.45 | +0.42 | +0.31 | +0.26 | +0.30 | > 120 min |

For a short-memory estimator, evidence loses half of its value in roughly **45-80 minutes** on both
nodes; long-memory averages and boosting (which also uses time of day) degrade more slowly, and the
half-life depends on the estimator and the day. The point is that in these data the useful life of a
measurement is of the order of an hour, not seconds or days. This is one band at one sensor in one
week; it is a measured description, not a general law.
