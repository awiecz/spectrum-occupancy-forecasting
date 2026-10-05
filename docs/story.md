# The AERPAW deep dive, in plain language

This is a project about one question and one dataset. The dataset is a set of public radio
recordings from the AERPAW research testbed. The question is whether a computer can learn to
predict how busy a Wi-Fi channel will be a little while from now, and whether that learned
prediction is any better than a simple rule of thumb. Words in *italics* are explained in the
[glossary](glossary.md).

**The short answer so far.** With complete recent measurements, a simple rule of thumb is very hard
to beat. Many learned models were tried and none reliably did better. The recordings are short,
though, so they could only rule out *large* improvements, not small ones. When most measurements
are missing, learned models fix the way plain history collapses, but so does a one-line fix, and
the learned models' extra benefit was not shown (Chapter 6).

## Chapter 1: What was recorded

Imagine standing in a field with a radio that sweeps across many frequencies every 20 seconds or so
and writes down how strong the signal is at each one. Two such receivers, one in an urban setting
(**CC1**) and one in a rural setting (**LW1**), ran for several days in February 2022. We look at
the frequencies around 2437 MHz, the middle of Wi-Fi channel 6.

A receiver cannot read messages. It only measures energy. So we call a *sweep* **busy** when the
energy is more than 3 dB above the quiet background level seen in that recording's first day. The
**busy fraction** of a 15-minute *window* is the share of its sweeps that were busy. The task: at
noon, predict the busy fraction for 12:15 to 12:30.

> **Tried:** define "busy" and the forecasting task. **Happened:** a clear, measurable target.
> **Learned:** "busy" means energy detected, not traffic decoded, and the threshold matters (next).

## Chapter 2: What the data look like

Activity tends to persist. In the urban recording a window's busy fraction correlates 0.75 with
the window 15 minutes later. That persistence is what makes forecasting possible at all.

The definition of busy matters a great deal. In the rural recording, nudging the threshold from
+2 dB to +4 dB drops the busy share from 30.15% to 0.70%. Any comparison between sites needs a
consistent, understood rule. See the [data study](data_study.md) for the figures.

> **Tried:** look before modelling. **Happened:** persistence, occasional jumps, and a threshold
> that changes the picture. **Learned:** recent history carries real information.

## Chapter 3: The benchmark

The simplest forecast is "the next 15 minutes will look like the last two hours". It needs no
training. In the urban recording its squared error is 0.0061, against 0.0151 for always predicting
the long-run average. Every learned model in this project had to beat that rule of thumb.

> **Tried:** a no-training benchmark. **Happened:** it is strong. **Learned:** the bar for a
> learned model is high.

## Chapter 4: Many attempts to beat it

Each attempt below asked a different version of the same question. All were
*exploratory*: they were run on the same short recordings.

| What was tried | What happened |
|---|---|
| Models chosen on earlier data (ridge, boosting) | The selected learned strategy had 8.7% more error (urban) and 5.5% more (rural) than independently selected history |
| A learned mixture of eight history forecasts | 1.35% and 6.80% less error than selected history, but 7.63% and 1.87% more than the fixed two-hour average |
| Training on both sites with finer inputs | 1.99% (urban) and 4.36% (rural) more error than the fixed average; on a reserved later rural session, 160.45% more |
| Adapting locally as measurements arrive (145 declared settings) | The best was a +0.30% gain on first-session rural data; none met the declared 5%-on-every-recording rule |
| Correcting leftover errors with simple state models | Urban error went up by 0.42% and 0.22% |
| A model of five-minute activity counts | First-session rural error went up by 22.68% against the fixed average |

Along the way, a log (the *registry*) recorded every declared attempt, including the failures, so
that results could not be quietly cherry-picked.

> **Tried:** six families of learned models. **Happened:** none reliably beat the rule of thumb.
> **Learned:** with complete input and these data, the benchmark stands.

## Chapter 5: Could we have known?

A fair worry about any negative result is whether the experiment could have detected a win. The
[power analysis](power_analysis.md) checks. With only a few days of recordings, a model that truly
cut error by 5% would be confirmed on all three recordings only 3.5% of the time even under
optimistic assumptions, and 0.3% of the time under typical ones. Only improvements of roughly 20%
to 30% would typically be detected.

So the negative results rule out large gains. They say much less about small ones.

> **Tried:** measure what the data could reveal. **Happened:** a small true gain would almost
> never have been confirmed. **Learned:** "no improvement found" is weaker than it sounds.

## Chapter 6: What if most measurements are missing?

Real monitors fail: sensors drop out, links break, data arrive late. So the project also asked what
happens when the forecaster sees only a patchy view. We simulated that by hiding sweeps, down to
about one in a hundred, while still scoring against the complete recording. See the
[sparse-regime chapter](sparse_regime.md) for the full study.

The early results looked promising. Several learned models beat plain two-hour history by a wide
margin once most data were hidden. Then we asked a tougher question: what if the comparison is
against the *best* simple rule, and what if the learned model is trained and selected for the
patchy view? We wrote the test down first, then ran it.

At the urban site, plain history's squared error was 0.0459 with only a hundredth of the data. A
simple fix, pulling history toward the long-run average, brought the strongest simple baseline to
0.0146. The learned model trained for the patchy view reached 0.0126. That is a 72.5% improvement
over plain history, but only 13.3% over the stronger baseline, and with this little data we cannot
tell that apart from zero. At the rural site the gain was −1.6%. The test's verdict was **honest
negative**: learned models did not reliably beat a well-tuned simple baseline.

> **Tried:** train and select learned models for sparse input, and compare them against the best
> simple baseline. **Happened:** they fixed plain history's collapse, and so did shrinking it
> toward the average. **Learned:** most of the earlier "gain" came from a weak comparison, and
> training models for the patchy view helped them (error 0.0374 became 0.0126) without proving they
> are better than a one-line fix.

## Chapter 7: What this does and does not mean

- **Does mean:** with complete recent measurements, a two-hour average is a strong forecaster of
  Wi-Fi channel activity in these recordings, and the learned models tried here did not beat it.
- **Also means:** with most measurements missing, a learned model trained for that situation
  clearly beat plain history, but it did not reliably beat a simple baseline that pulls history
  toward the average.
- **Does not mean:** that no learned model could ever help, or that small gains do not exist.
- **Limits:** three recordings at two sites in February 2022; sparsity is simulated; data were
  reused from earlier exploratory work; and CC2, the one recording held back as an independent
  test, has not been run.
- **What would settle it:** many more days of recordings from more sites, with a test declared in
  advance. The [power analysis](power_analysis.md) estimates how much.

Every number on this page is checked against the saved result files by
[`scripts/check_claims.py`](../scripts/check_claims.py); see the
[claim-to-evidence table](claims.md).
