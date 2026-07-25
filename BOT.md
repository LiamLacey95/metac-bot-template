# Calibrated aggregation — what this bot does differently

Forked from `Metaculus/metac-bot-template`. The research path and prompts are deliberately left
close to the template. Exactly one thing is changed, so that any movement in the score can be
attributed to it.

## The change

The template samples N forecasts per question and takes the **median of the probabilities**.

This takes the **mean in logit space** and then **shrinks it toward a base rate by an amount set
by how much the samples disagreed**.

```python
logits    = [logit(p) for p in samples]
mean      = sum(logits) / len(logits)
spread    = stdev(logits)
agreement = 1 / (1 + (spread / spread_scale) ** 2)   # tight samples -> ~1, scattered -> ~0
weight    = global_trust * agreement
final     = prior_logit + weight * (mean - prior_logit)
```

## Why

The tournament scores with a log score, which punishes confident mistakes far harder than it
rewards confident hits. Three things follow:

1. **Language models are overconfident.** They report 5% and 95% more often than the world
   obliges. The median of five overconfident samples is still overconfident — averaging in
   probability space does not fix a systematic bias, it just smooths it.

2. **Disagreement is information, and the median throws it away.** Five runs landing on
   0.88–0.92 is a different epistemic state from five runs scattered across 0.15–0.90, yet both
   can produce the same median. In the second case the model does not know, and a confident
   number is a fiction with a straight face. Spread separates these, and it is free — the
   samples already exist, so this costs no extra model calls.

3. **The anchor is not 0.5.** Most "will X happen by date Y" questions resolve No. The default
   prior here is 0.35 and it is the first parameter that should be fitted against resolved
   questions rather than assumed.

## Evidence, including where it loses

`python test_calibration.py` simulates forecasters at varying overconfidence `c` and noise, then
compares log scores against the template's median. The simulation is built so it *can* return a
negative answer, and at `c = 1.0` it does.

| Forecaster | Delta vs median (nats) |
| --- | --- |
| Perfectly calibrated, low noise (`c=1.0`, σ=0.5) | **−0.0019** |
| Calibrated but noisy (`c=1.0`, σ=2.0) | +0.0435 |
| Mildly overconfident (`c=1.3`) | +0.0086 → +0.0500 |
| Strongly overconfident (`c=2.5`) | +0.0625 → **+0.1393** |

The case for shipping it is the asymmetry: the cost when the assumption is wrong is about
**−0.002 nats**; the gain when it is right is up to **+0.14**. That is a favourable bet by
roughly two orders of magnitude, and it is the shape of bet a log score rewards.

**This is a simulation, not evidence about real language models.** It shows the method works
*if* LLM forecasts are overconfident, which is documented elsewhere but not measured here. The
honest next step is fitting `prior`, `global_trust` and `spread_scale` against resolved
questions, and MiniBench's two-week cycle is the right instrument for that.

## Scope

Binary questions only. Numeric and multiple-choice aggregation fall through to the framework
untouched — they are a different problem with a different fix, and one calibration story
stretched across all three is how a regression ships that nobody can attribute.
