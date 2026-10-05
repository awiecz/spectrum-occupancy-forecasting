# Sparse-regime study: protocol

Prepared 2026-10-06, **before any matched-training model was fitted**. Exploratory: all
recordings were inspected in earlier work, and CC2 is untouched. This protocol and
[`configs/sparse_regime.yaml`](../../configs/sparse_regime.yaml) are authoritative together.
Terms are explained in the [glossary](../glossary.md).

## Question

When a forecaster sees only a sparse, stale or delayed view of the channel, can a learned model
that is trained and selected for that same degraded view cut squared error by a **large** amount
against the best simple baseline that is selected the same way?

Earlier studies trained and selected models on complete input and then scored them under
simulated degradation. Matched training and selection is a different, declared question.

## What was seen before this protocol

From saved exploratory predictions (no new fitting), at 99% dropout:

- Plain two-hour history is a poor baseline there: its squared error is 0.054 (urban) and 0.032
  (rural), worse than always predicting the average (0.015 and 0.010).
- Shrinking history toward the average, with the weight fitted on the scored windows, scored
  0.0098 (urban) and 0.0089 (rural), below every unmatched learned model at the urban site.
- At the rural site, boosting on the later folds (0.0028) beat that shrinkage proxy (0.0046).

These observations motivate the strong baselines below. They do not describe matched models,
none of which has been fitted, so no outcome of this study has been seen.

## Rule amendment (disclosed)

The power step ([`results/sparse_regime/power/`](../../results/sparse_regime/power/)) first
used this rule: *a primary cell whose typical minimum detectable gain exceeds 50% is
descriptive, and if every primary cell is descriptive the confirmatory claim is abandoned.*
The noise came from the six **unmatched** learned families. Against the strongest simple
baseline it gave 109% (urban) and 82% (rural later folds), so the original rule **triggered**.

Those families are far worse than the baseline in this regime, which inflates the noise of any
gain estimate and probably overstates it for matched models. Before fitting any matched model, the
rule was amended with the owner's agreement. The amended noise proxy scales each cell's
baseline noise by the candidate-to-baseline noise ratio seen in the earlier clean-input
candidates, and uses no sparse-regime learned result. It gives a typical minimum detectable gain of
about 13% in both primary cells, so the 20% minimum margin binds (δ = 0.20) and nothing is
demoted. The original result is kept in `margin.json`.

The amended proxy assumes matched models resemble the baseline as closely as clean-input candidates
did. If they do not, true noise is larger. The decision below therefore uses the *observed*
bootstrap lower bound, never the planning figure. If a primary cell's noise turns out too
large, the label is simply not "supported".

## Estimand

For each recording and scenario, `S = 1 − MSE(selected learned) / MSE(H*)` on identical windows
and identical masks.

- **H\*** is the strongest simple baseline, chosen on matched validation among: fixed two-hour
  history; the best of the eight history candidates (moving averages and decayed averages at
  5/15/30/120 minutes); and **shrunk history** `m + λ(h − m)`, where `h` is fixed history, `m`
  the earlier training mean and `λ` a single least-squares weight fitted on the same matched
  validation windows (clipped to [0, 1]). The baseline thus gets the benefit of a fitted weight.
- **Selected learned** is the best on matched validation of ridge (4 penalties), boosting
  (depth 2/3 × leaf 20/40) and boosting on the residual of selected history (same grid): the
  frozen v2 grid, which the configuration loader enforces.
- Also reported: each family, fixed history, selected history, shrunk history, climatology, and
  boosting refit on five feature subsets at the primary scenario (no new search).

## Data and scenarios

- CC1 and LW1, first session, with the same calibration, daily folds, ten-hour embargo and two
  six-hour validation blocks as the v2 study.
- **Primary scenario: 99% dropout** (about one sweep in a hundred visible). Secondary and descriptive: 50%
  and 90% dropout, a one-hour outage covering 25% of time, and one- and two-hour delay. Clean
  input is the control.
- **Matched views.** Training, validation and scoring features come from the same degradation
  through independent random masks (training seeds 101–103, validation seeds 201–203, scoring
  seeds 21–25; the exploratory study used 1–5). Labels are never degraded.
- **LW1 later session is not run.** The v2 rules need at least two and a half days, and that
  session has 1.9. The rural primary cell is therefore LW1's **later folds** (fold index above 0).
  Its first fold overlaps old channel selection and is reported separately.

## Intervals

Learned: boosted quantile regression at levels 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95. Comparators:
selected learned and H\* with signed quantiles of earlier validation errors added to their point
forecasts, and the empirical quantiles of earlier outcomes. Intervals run for clean input and 99%
dropout. Reported: mean pinball loss, the Winkler score and coverage of the central 80%
interval, and its width. A CRPS approximation was dropped as an unneeded extra degree of freedom.
Intervals are descriptive and cannot rescue the primary claim.

## Inference

Window losses are averaged over scoring masks before comparing. Whole blocks of 32 consecutive
windows (eight hours; 16 where eight-hour blocks number under four) are resampled 2,000 times
(seed 0), giving a one-sided 95% lower bound for `S`. Block-length sensitivity, absolute-error
gain, gain without the worst 10% of windows, leave-one-day-out stability and the best block's
share of the improvement are reported for the primary cells.

## Decision rule

**Large effect supported** only if all hold at 99% dropout, with δ = 0.20:

1. Urban (all folds): lower bound of `S` against H\* is at least δ.
2. Rural (later folds): lower bound is at least δ.
3. Rural all folds: `S` is at least δ/2.
4. Absolute-error gain is positive in both primary cells.
5. No single block supplies more than half of the total improvement in either.

**Honest negative** if either primary lower bound is at or below zero, or the two primary
estimates have opposite signs. **Directional only** otherwise: improvement is likely but not
shown to be large. Small clean-input gains are never claimed, and secondary scenarios,
intervals and ablations cannot change the label. The label is computed by code
(`sparse_reporting.decide`), not chosen afterwards.

## Pre-written outcomes

- **Supported:** "With only about one sweep in a hundred visible, a learned model reduced
  error by at least 20% against the best simple baseline at both sites." Scope sentence applies.
- **Directional only:** "Learned models probably improve on the best simple baseline when input is
  sparse, but the data do not show the improvement is large."
- **Honest negative:** "Learned models did not reliably beat a matched, well-tuned simple baseline
  when input was sparse; the large gains seen earlier came from comparing against plain history."

Scope sentence for every document: three recordings at two sites, February 2022; sparsity is
simulated; data were reused from earlier exploratory work; CC2 is the only independent test and
has not been run.

## Registration and completion

Declare the attempt in the registry with this protocol's hash and the configuration before the
real run, then record `running`, `findings` and `completed`. Parameter changes need a parent
attempt and evidence. Completion needs tests and lint to pass, saved predictions to reproduce every
table without refitting, every figure reviewed, and every headline traced to saved rows.
Publish negative and inconclusive outcomes with equal prominence.

## Limits

Three recordings at two sites, with few independent blocks at the rural site (about 4.7 eight-hour
blocks in the primary cell). Simulated sparsity differs from real sensor failure. The strongest baseline
includes a fitted weight, and the power proxy rests on a clean-input noise ratio.
