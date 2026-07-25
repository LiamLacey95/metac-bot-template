# What this bot does differently, and why

Forked from `Metaculus/metac-bot-template`. `template_bot.py` is Metaculus's original, kept
unmodified as the baseline being measured against.

## The scoring rule, since it drives everything

Summer FutureEval and MiniBench both use **spot peer score**: `100 x (your log score - the mean
log score of every other bot)`, on your last forecast before the question closes. Unforecast
questions score **0**, not negative.

Two consequences that matter more than any prompt:

1. **The crowd term is outside your control**, so maximising expected peer score is exactly
   maximising your own expected log score. The rule is essentially proper — predict your true
   belief. It is *not* a reason to deviate from the consensus for its own sake.
2. **Prize money is split in proportion to (sum of peer scores) squared.** Because
   `E[S²] = E[S]² + Var(S)`, variance is paid for at fixed expected score, and *coverage*
   compounds: every question you miss is a zero dragging a quadratic. Never missing a question is
   worth more than any clever aggregation.

## Changes, in descending order of expected value

### 1. Cross-model ensemble

The template draws N samples from **one** model at temperature. Those samples share a bias, so
averaging them cancels decode noise and nothing else — the ensemble is one opinion repeated.

This rotates samples round-robin across three model families (`ENSEMBLE_MODELS`). Errors are then
partly independent, which is the only condition under which extremising the aggregate is a
correction rather than bias amplification. With a single model configured, extremisation is
automatically disabled.

### 2. Logit-mean aggregation with a tail cap

Template: median of the probabilities. Here: mean in logit space, optional extremisation, clipped
to `[0.02, 0.98]`.

The cap is the one calibration intervention that separated winners in the Fall 2025 survey. It
insures against a catastrophic misread without touching the mid-range.

### 3. Disagreement triggers research, it does not blend

Sample spread is real information, but the useful response is to go find the missing fact rather
than to average the ignorance. `needs_more_research()` exposes the signal; high-disagreement
questions are flagged in the logs.

## What I removed, and why — the useful part of this document

The first version of this bot shrank every binary forecast toward a prior of 0.35, by an amount
driven by sample spread. It was wrong in four separate ways, and all four are worth stating
because they are easy mistakes to make again:

1. **The prize rule taxes caution.** Prize is quadratic in summed peer score, so variance is paid
   for. A standing haircut buys insurance the payout formula actively penalises.
2. **It could never extremise.** The trust weight was capped below 1, so the layer pulled toward
   0.35 *even when all samples agreed perfectly*. That is a directional bet on the question set,
   not a calibration correction.
3. **The prior was wrong for the questions.** "Most 'will X happen' questions resolve No" is
   folklore; Metaculus's own MiniBench analysis found that the nothing-ever-happens prior baked
   into template bots actively hurt on auto-generated questions.
4. **The evidence assumed its conclusion.** The old test simulated a forecaster that was
   overconfident *by construction*, then reported that a shrinkage layer helped. That is not
   evidence about this bot; it is arithmetic about the assumption.

The honest version of the idea is **Platt scaling** — a slope and bias in logit space, *fitted*
from this bot's own resolved forecasts. Metaculus validated it on real AIB data as the best of
five calibration methods, and ships an implementation. It needs ~200 resolved forecasts to fit,
which this bot does not have yet. A fixed prior with a fixed trust weight is Platt scaling with
both parameters guessed, which is strictly worse than not doing it.

## Testing

```bash
python test_calibration.py
```

Property tests, no dependencies, no network. They check the things that must hold by
construction: unanimous samples survive untouched, aggregation is symmetric under complement
(a fixed prior breaks this, which is how the old bug would have been caught), extremisation moves
away from 0.5 in both directions, and clipping bounds hold.

The accompanying comparison reports logit-mean against the template's median across calibration
regimes rather than picking the flattering one. The two are close — this layer is not where the
points are, and the file says so.

## What is deliberately not done yet

- **Numeric and multiple-choice** fall through to the template untouched. This is where bots bleed
  the most points against human forecasters, so it is the highest-value remaining work. Proven
  open-source pipelines exist and porting one is a change that should land on its own.
- **Fitted Platt scaling**, once there are enough resolved forecasts of this bot's own.
- **A second research source.** Breadth of search correlates with score more than any single
  provider does.
