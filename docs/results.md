# Results overview

**Recent activity predicts future activity, but the tested learned models do not establish a reliable advantage over simple averages.**

Start with the [data study](data_study.md) to see what was measured, how activity labels are
constructed and why history can be informative.

## What is being predicted?

The main task predicts the fraction of measurements classified as busy in a 15-minute period
starting 15 minutes after the forecast. A forecast made at noon concerns 12:15–12:30.
“Busy” means received power above a calibrated detection threshold, rather than decoded Wi-Fi
traffic or a direct measurement of transmitter duty cycle.

Measurements come from two AERPAW sites in February 2022: CC1 (urban, seven days) and LW1
(rural, 4.75 days). Forecasts use earlier observations. Separate tests simulate missing
observations, outages and delay.

## Does learning improve the forecasts?

In the chronological model-selection study, the trained strategy had **8.7% higher squared
error at CC1 and 5.5% higher error at LW1** than independently selected recent history.
Earlier validation did not reliably choose a model that performed well later.

Further exploratory development found a learned mixture of eight history forecasts.
On the common evaluation folds used for that development comparison:

| Comparison | Urban site (CC1) | Rural site (LW1) |
|---|---:|---:|
| Error reduction against validation-selected history | 1.35% | 6.80% |
| Error increase against a fixed two-hour average | 7.63% | 1.87% |
| Evaluated forecast periods | 351 | 151 |

Here “error” means mean squared error: larger mistakes receive more weight. Lower error is
better. Both comparisons use the same mixture predictions, but different baselines. The
fixed two-hour average wins at both sites. The mixture misses the declared requirement
of at least 5% improvement over selected history at each site, so it does not advance.

The chronological study and mixture table use different evaluation populations; their
percentages are not a direct before/after comparison. All three tested mixture training
regimes and a ledger of unsuccessful experiments remain available.

Sources: [chronological-study results](technical/followup_v2_results.md),
[development results](technical/forecast_development.md) and
[saved mixture comparison](../results/followup_v2/history_mixture/common_summary.csv).

## What else did we learn?

**Forecasts depend on recent evidence.** In the original study, one short-memory forecast lost
about half its predictive skill when its newest usable measurement was 45–80 minutes old.
Longer averages retained useful information longer.

**Activity labels depend on calibration.** Applying a different site's absolute threshold
changed the measured busy share substantially. Comparing sites requires a consistent,
well-understood detection rule.

**Rare activity changes cause large mistakes.** At the urban site, the worst 10% of windows
account for roughly 60% of squared error: 58% for fixed two-hour history across all 439
first-session windows ([loss structure](../results/power_analysis/loss_structure.csv)), and
62–64% for history and the learned mixture on the 351 common development windows
([saved mixture predictions](../results/followup_v2/history_mixture/predictions.parquet)).
Tested transition models did not solve this problem.

## How strong is the evidence?

The earlier finer-input/shared-training study found no qualifying improvement.
Its globally selected challenger (`history_fine_900s_12h` in the
[adaptive experiment ledger](../results/adaptive_learning/experiment_ledger.csv))
increased clean MSE by 1.99% urban and 4.36% rural against fixed two-hour history.
After a local freeze, a single reserved later rural session showed
160.45% higher error ([saved assessment](../results/adaptive_learning/references/later_assessment/summary.csv)). That assessment is complete and unfavorable; it is separate from the
still-pending CC2 confirmation. That frozen assessment did not tune on its outcomes; the
subsequent [adaptive study](adaptive_learning.md) explicitly reuses this session as development.

The adaptive search tests 145 declared candidates on identical origins: 439 urban, 239 rural
first-session and 81 later rural forecasts. No strategy meets the robust 5%-on-each-recording
promotion rule. Adaptive aggregation approximately matches fixed two-hour history and gains
0.30% on first-session rural data. This is a small exploratory gain, not learned superiority.

The [targeted residual follow-up](residual_state_space.md) found retrospective candidate
headroom and persistent errors. Its declared level/trend study nevertheless increased urban
MSE by 0.42%/0.22% against two-hour history and selected history-only on both rural recordings.
Neither family qualifies; controller degradation penalties are reported separately from learning.

The [operational study](operational_learning.md), declared in the append-only registry, then
aligned validation with continuous updates and isolated controller effects. Its count model
retained fixed history on urban and later rural data, while increasing first-session rural MSE
by 22.68% against fixed history and 45.68% against selected history. No new method qualifies.
The registry preserves declarations, historical outcomes and the interrupted implementation run.

The [power analysis](power_analysis.md) then asked how much these recordings could have
revealed. A true 5% improvement would almost never have been confirmed on all three
recordings; only gains of roughly 20–30% or more would typically be detected. The negative
results therefore rule out large learned gains but not small ones. A
[draft multi-band protocol](multiband_protocol.md) examines whether pooling many frequency
bands could help; it has not been run, and its planning numbers show that bands recorded at
the same site and time fail together, so pooling helps less than hoped.

The [sparse-regime study](sparse_regime.md) asked what happens when only about one sweep in a
hundred is visible and models are trained and selected for that view. Learned models beat plain
history by a wide margin (72.5% lower urban error), but a plain history pulled toward the
average captured most of that gain. Against that stronger baseline the urban gain was 13.3% with
a lower bound of −33.5%, and the rural later folds showed −1.6%. The pre-declared rule returned
an **honest negative**. Intervals and other scenarios were descriptive only.

The recordings are short, nearby forecasts are correlated, and the data were repeatedly
inspected during development. The mixture's apparent improvements are exploratory; its
uncertainty estimates do not account for the full adaptive search. Rural-site uncertainty
is especially weak because few independent temporal blocks are available.

There is no independently confirmed learned-model superiority. CC2, the unused third site,
remains uninspected; evaluation requires a verified, independently timestamped protocol.
The present results concern these recordings and this channel.

For supporting tables, measurement checks, unsuccessful outcomes and protocols, use the
[technical reference](technical/index.md).
