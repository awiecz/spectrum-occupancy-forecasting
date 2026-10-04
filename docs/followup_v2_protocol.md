# Follow-up v2: frozen CC2 confirmation protocol

Prepared privately on 2026-10-04, after synthetic rehearsal, CC1/LW1 exploratory evaluation,
prediction-file metric checks and visual review of all seven figure families. **Not independently
registered yet. CC2 confirmation is pending and must not run.** The investigator's stated
disclosure is that CC2 activity has not been inspected. Do not download, summarize, plot or
score CC2 until the package is independently timestamped and the executable gate verifies it.

This protocol and `configs/followup_v2.yaml` are authoritative together with the exact frozen
source package. `scripts/freeze_followup.py` creates `followup_v2_freeze.zip`, containing both,
all executable Python sources, entry points, dependency specification/lock and cryptographic
manifest. The package's hashes are generated mechanically, not handwritten into this document.
The verifier compares the remotely archived bytes and current frozen files. Local Git dates
alone do not satisfy the gate. No archive exists at preparation; the repository stays private.

## Question and primary estimand

Does the **earlier-validation-selected learned strategy** lower prospective MSE relative to an
independently earlier-selected history strategy, for a 15-minute busy-fraction target starting
15 minutes ahead at 2437 MHz, under clean inputs? The target is a fraction of observed sweeps
crossing a noise-relative detector threshold, not continuous transmitter duty cycle.

CC1/LW1 v2 are exploratory reuse; the historical evaluations retain their original protocols,
locks and outputs. CC2 adds site evidence from the same measurement campaign. It cannot
establish seasonal generalization. No data-driven channel substitution is permitted.

## Calibration, origins and separation

- Compute each channel's fifth-percentile band-mean noise floor and the target channel's median
  nominal sweep-start interval using sweeps ending in the node's first 24 hours only. Freeze
  these for all later folds. Primary labels use floor +3 dB; target coverage must be ≥80% of
  the calibrated expected sweep count. Targets use half-open `[start,end)` sweep-start windows.
- Start evaluation at recording start plus two days. Use expanding daily folds; retain a final
  partial fold only when ≥12 hours. Every target must finish inside its fold. Record excluded
  and insufficient-data folds; never replace them with a different region or channel.
- Use 15-minute outer-origin spacing and a common grid restricted by the maximum planned
  lead (two hours) plus target width. Within this fixed recording grid, coverage exclusions
  may remove different origins at different leads; disclose this rather than filling labels.
- Predictor observations must end by origin minus simulated delay. Decayed histories have
  finite support of five time constants; maximum clean feature look-back is ten hours.
  Fitting targets finish by outer fold start minus this bound. Apply the same embargo at each
  inner validation block. Training origins are spaced three minutes apart.
- Use the last twelve hours before the eligible outer fitting cutoff for two chronological
  six-hour validation blocks. Restrict their origin grids by the same maximum lead. Calibration
  must precede validation. Fit imputation/scaling only on each relevant training period.

## Selection and full reporting

Candidate order is the order encoded by the frozen pipeline; exact ties choose the first.
History candidates are moving averages of 5/15/30/120 minutes, followed by exponential histories
with those time constants. Select history by pooled validation MSE. Missing histories use the
earlier training mean; decayed histories include a fixed prior pseudo-count of one.

Learned families are ridge, histogram gradient boosting, and residual versions of each on the
selected history. Ridge penalties are 1/10/100/1000. Boosting uses depth 2/3 crossed with minimum
leaf size 20/40. Other settings and random seeds are frozen in YAML. Select each family by pooled
validation MSE, then the learned-family strategy by those same earlier scores. Residual tuning
is conditional on selected history. Reused validation scores are selection diagnostics.

Report climatology, fixed two-hour history, selected history, each selected learned family and
the selected learned strategy, along with **every candidate's validation score**. Do not replace
the primary strategy with a family that performs better after evaluation. No positive outcome
is required for completion.

## Staged secondary analyses

Run the frozen 13 clean tasks: the primary task; other leads 0/30/60/120 minutes; other label
offsets +2/+4/+6 dB; and five non-full feature variants. Primary is reused for the full-feature,
+3 dB, 15-minute-lead setting, avoiding duplicate fits. Repeat validation selection within each
task. Features use explicit allowlists: history; history plus power; history plus age/counts;
history plus calendar; history plus cross-channel state; all features.

Only the primary task receives frozen dropout probabilities 0.5/0.9/0.99, nominal 25%-span
outages of 15/60 minutes, and delays of 1/10/30/60/120 minutes. Use five paired stochastic masks,
seeds 1–5, and deterministic seed zero for clean/delay scenarios. Freeze clean-selected
configurations. Compare clean-trained ridge/boosting and their existing augmentation designs:
clean plus 0.5/0.9 dropout, 10/60-minute delay and a 60-minute outage, seeds 10–14. Do not retune
on degraded evaluation labels. Labels always remain intact. Compare against history degraded
by the **same** scenario and seed; clean history is a separate diagnostic reference.

LW1 first/later-fold overlap sensitivity is retrospective and applies to historical LW1 data
only. It is not a new CC2 confirmation task. Forecast prediction intervals are deferred.

## Metrics, uncertainty and event decisions

Save per-origin prediction Parquet files and earlier validation predictions, selection records,
operating thresholds, node calibration and provenance. All summaries and figures derive from
these saved files without refitting. Report daily and pooled RMSE, MAE, within-fold R², skill
against selected and fixed history, and paired MSE differences. Include prevalence, sweep and
window counts, degenerate labels and missing outcomes.

Examine error and paired-loss-difference autocorrelation only within contiguous fold segments.
Pair model/reference origins and masks. Average losses over masks before temporal inference.
Bootstrap whole temporal blocks with 2,000 resamples (seed zero), separately per node, using
2/4/8/12-hour blocks. No block crosses folds or observation gaps. Eight hours is primary.
Report block counts; suppress intervals below four blocks and flag counts below ten as weak.
Partial terminal blocks are retained and weighted through their observation counts. Sensitivity
does not cure short-record inference, selection uncertainty or label-definition uncertainty.

For busy fraction ≥0.2, choose the lowest earlier validation score threshold meeting false-busy
budgets 10%/20%, including entire tied-score groups. Require ≥10 positive and ≥20 negative
validation windows. Otherwise mark calibration insufficient and make no calibrated operating-
rate claim. Freeze thresholds before evaluation. Report realized false-busy/false-idle rates
and denominators for every scenario/mask, including zero-event cases. Budget attainment in
validation is not a guarantee in evaluation. Oracle ROC points use evaluation labels and stay
separately labeled descriptive outputs.

## Decision rules and acceptance

The primary reported effect is pooled MSE(selected learned) minus MSE(selected history).
Negative values favor learned prediction. Call the primary comparison supportive of learned
improvement only when the eight-hour paired 95% interval lies wholly below zero and there are
at least ten blocks. Call it supportive of deterioration only when wholly above zero with
at least ten blocks. Otherwise call it inconclusive; fewer than four blocks cannot receive
an interval-based conclusion. Also disclose the fixed-reference comparison and all sensitivities.
Secondary family/lead/label/ablation/scenario findings cannot rescue an unsuccessful primary
comparison and do not carry multiplicity-adjusted confirmatory claims.

Completion requires original tests/lint and new causality tests to pass, original artifacts to
remain unchanged, saved predictions to reproduce summary metrics, published figures to be
visually reviewed, and every headline claim to name a comparison and trace to saved rows.
Publish failures, degenerate labels, insufficient calibration/data/blocks and inconclusive or
small improvements. A failed scientific hypothesis does not constitute a failed implementation.

## Archival gate and one-shot execution

After the exact package is independently timestamped, run `--verify-gate-only` against its
immutable GitHub release API or published Zenodo record API. Only after successful verification
download CC2, verify the source deposit's published archive checksum, extract the node product,
and run `--mode confirmation` once with the same archive reference. Preserve the raw archive
checksum verification in the run record alongside the derived input hash. The software's
one-shot lock is created before CC2 input access and retained after failure.

Evaluate every eligible later CC2 fold with the frozen settings. Do not inspect alternate bands,
change offsets/leads, extend training searches or tune acceptance thresholds after seeing CC2.
Changes after archival require a separately versioned analysis and explicit disclosure. Never
delete the confirmation lock to obtain another attempt. Archive verification establishes package
integrity and a server timestamp; the investigator's no-prior-inspection claim requires its
own disclosure. Commands and supported archive restrictions are in `followup_v2_reproduction.md`.
