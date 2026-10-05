# Glossary

Plain-language meanings of the terms used across this project. Terms are grouped by theme;
each links back to where the idea matters most.

## The data

**AERPAW.** The Aerial Experimentation and Research Platform for Advanced Wireless, a wireless
research testbed run with North Carolina State University. Its public spectrum-monitoring
recordings are the only data this project uses. See [data attribution](../DATA_ATTRIBUTION.md).

**Sweep.** One pass of the receiver across many radio frequencies, saving how strong the
received signal was at each. Think of a snapshot of the airwaves. A sweep does not decode
messages or identify who is transmitting. In these recordings a sweep arrives roughly every
19 seconds (urban) or 20–30 seconds (rural).

**2437 MHz / channel 6.** The frequency, in megahertz, at the centre of the band studied. It is
conventionally associated with Wi-Fi channel 6. We analyse a band about 20 MHz wide around it.

**CC1, LW1, CC2.** Fixed monitoring receivers in the same measurement campaign (February 2022).
**CC1** is the urban site and **LW1** is the rural site; both are analysed. **CC2** is a third
site that this project has deliberately left unopened, so it can serve as an independent final
test. See [Frozen protocol](#how-we-keep-ourselves-honest).

**dB and dBm.** Decibels are a logarithmic scale for power ratios; dBm is power relative to one
milliwatt. Adding 3 dB roughly doubles power.

**Noise floor.** The level of background radio energy when nothing in the band is transmitting.
Here each site's noise floor is estimated from the fifth percentile of band power during the
**first 24 hours** of its recording, then held fixed.

**Busy.** A sweep counts as busy when the band's power exceeds the noise floor by more than 3 dB.
It means "energy detected above background", not decoded traffic, a specific transmitter or
channel capacity.

**Busy fraction.** The share of sweeps inside a time window that were busy. A busy fraction of
0.30 means 30% of the sweeps in that window exceeded the threshold.

**Window.** A 15-minute stretch of time over which the busy fraction is computed.

**Lead.** How far ahead the forecast looks. The main task uses a 15-minute lead: a forecast
made at noon concerns the window 12:15–12:30.

## Forecasting and scores

**History baseline.** A forecast that simply averages recent busy fractions, for example the
**fixed two-hour history**: "next window will look like the last two hours". It needs no
training and is the benchmark every learned model must beat.

**Climatology.** A forecast that always predicts the long-run average busy fraction.

**MSE and RMSE.** Mean squared error is the average of squared forecast mistakes, so large
mistakes count extra. RMSE is its square root, in the same units as the busy fraction. Lower
is better.

**Skill.** How much better (positive) or worse (negative) a forecast is than a reference,
written as the fraction of the reference's error removed: 0.10 means 10% less squared error.

**Learned model.** A forecast produced by a method that fits its settings to past data, such as
ridge regression or gradient boosting (below), instead of a fixed rule.

**Ridge regression.** A linear model that fits a weighted sum of recent measurements while
discouraging extreme weights.

**Boosting.** A model built from many small decision trees, each correcting the previous ones.
This project uses histogram-based gradient boosting.

**Dropout, outage, delay.** Ways the inputs are made worse in simulation. *Dropout* hides a
random share of sweeps (99% dropout keeps about one sweep in a hundred). An *outage* hides a
continuous stretch. *Delay* makes measurements arrive late. Real receivers did not fail;
the gaps are simulated.

**Quantile forecast, pinball loss, interval coverage.** A quantile forecast states a range
(for example, "90% likely to lie between 0.05 and 0.30") instead of one number. Pinball loss
scores such forecasts; coverage is how often the real value fell inside the stated range.

**Conformal interval.** A way to turn any forecast into an interval by measuring how big its
mistakes were on held-back data.

## Evaluating fairly

**Fold.** One chronological slice of the recording used for testing, with earlier data used for
fitting. Folds never mix future data into the past.

**Embargo.** A gap left between fitting data and test data so that overlapping measurements
cannot leak information from one to the other.

**Leakage.** Any way information from the future reaches a forecast made earlier. It makes
results look better than they could be in real use.

**Causal.** Using only information that had already arrived at forecast time.

**Controller.** In the operational study, the rule that decides when a learned correction is
allowed and when to fall back to history.

**Bootstrap and block bootstrap.** A way to estimate uncertainty by re-sampling the results
many times. The block version re-samples whole stretches of time, because neighbouring
windows are alike and cannot be treated as independent.

**Confidence interval.** A range that conveys how uncertain an estimate is.

**Statistical power.** The chance that a comparison detects a difference that is really there.
Low power means a "no difference" result says little.

**Minimum detectable gain.** The smallest true improvement a comparison would reliably detect.
See the [power analysis](power_analysis.md).

**Effective sample size.** How many independent observations the data are worth once
clustering and a few very large errors are taken into account.

## How we keep ourselves honest

**Exploratory vs confirmatory.** *Exploratory* results come from data that were inspected while
methods were developed; they generate hypotheses. *Confirmatory* results come from a test
declared in advance on data nobody had looked at. This project has exploratory results only.

**Registry.** An append-only log of every declared experiment, its settings and outcome,
including failures. It is a local record, not an external timestamp.

**Frozen protocol.** A written plan, with its exact code, locked before a test is run so it
cannot be adjusted afterward. The [CC2 protocol](technical/followup_v2_protocol.md) is frozen
but has not been run.

**One-shot gate.** A software lock that allows the final CC2 test to run once, and only after
the frozen package has been independently archived with a trustworthy timestamp.

## Status labels used in this repository

- **Exploratory:** completed analyses on inspected data.
- **Declared in the registry:** logged before running, with a local fingerprint. Not
  independently timestamped.
- **Frozen and pending:** locked but not yet run (the CC2 confirmation).
- **Draft:** a plan that has not been run (the multi-band protocol).
