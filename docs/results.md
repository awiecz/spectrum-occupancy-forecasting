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

**Rare activity changes cause large mistakes.** In retrospective diagnosis, the worst 10%
of urban-site forecasts accounted for about 64% of squared error. Tested transition models
did not solve this problem.

## How strong is the evidence?

The recordings are short, nearby forecasts are correlated, and the data were repeatedly
inspected during development. The mixture's apparent improvements are exploratory; its
uncertainty estimates do not account for the full adaptive search. Rural-site uncertainty
is especially weak because few independent temporal blocks are available.

There is no independently confirmed learned-model superiority. CC2, the unused third site,
remains uninspected; evaluation requires a verified, independently timestamped protocol.
The present results concern these recordings and this channel.

For supporting tables, measurement checks, unsuccessful outcomes and protocols, use the
[technical reference](technical/index.md).
