"""Aggregation layer for a Metaculus AIB bot.

## What the scoring rule actually rewards

The tournament scores with **spot peer score**: 100 x (your log score - the mean log score of
every other bot), taken on your last forecast before the question closes. The crowd term is
outside your control, so maximising expected peer score is exactly maximising your own expected
log score. The rule is essentially proper: predict your true belief.

The part that changes the design is the *prize* rule, not the score rule. Prize money is split in
proportion to **(sum of peer scores) squared, when positive**. Since E[S^2] = E[S]^2 + Var(S),
variance is paid for at fixed expected score. A systematically cautious bot caps its upside
superlinearly. Insurance against catastrophe is worth buying at the tails; a standing haircut
across the whole mid-range is not.

## What this does

    final_logit = extremise * mean(logit(p_i))
    clip to [logit(0.02), logit(0.98)]

That is the whole binary layer. Deliberately absent:

- **No blend toward a fixed prior.** An earlier version shrank every forecast toward 0.35 on the
  theory that most "will X happen by date Y" questions resolve No. That is a standing directional
  bet on the question set rather than a calibration correction, it fires even when all samples
  agree perfectly, and Metaculus's own MiniBench analysis found the "nothing ever happens" prior
  baked into template bots actively hurt on auto-generated questions. Removed.
- **No extremisation by default.** Extremising is right when samples carry partly independent
  information, which is true across model families and false for N temperature-resamples of one
  model sharing one research dossier. `EXTREMISE_CROSS_MODEL` is the value to use once the
  ensemble genuinely spans families; until then 1.0 is the honest setting.

## What spread is for

Disagreement between samples is real information, but blending it toward a constant was the wrong
use. Scattered samples mean the ensemble is missing a *fact*, and the right response is to go find
the fact rather than to average the ignorance. `disagreement()` exposes the signal for that
purpose; see `needs_more_research()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Metaculus accepts 0.001-0.999. The template clips at [0.01, 0.99]. A slightly tighter cap is
# the one calibration intervention the Fall 2025 winners' survey singled out (r=+0.48 with
# winning): it buys insurance against a catastrophic misread without touching the mid-range.
MIN_P = 0.02
MAX_P = 0.98

# Only defensible once samples span different model families, where errors are partly
# independent. Applied to same-model resamples it just amplifies one model's shared bias.
EXTREMISE_CROSS_MODEL = 1.3


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def log_score(p: float, outcome: int) -> float:
    """Natural-log score of a single forecast. Higher is better; 0 is perfection."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p) if outcome == 1 else math.log(1.0 - p)


@dataclass
class AggregationConfig:
    extremise: float = 1.0
    """1.0 = report the ensemble's belief unchanged. Raise toward EXTREMISE_CROSS_MODEL only
    when samples come from different model families."""

    min_p: float = MIN_P
    max_p: float = MAX_P

    research_trigger_spread: float = 1.5
    """Logit-space standard deviation above which the samples disagree enough that the ensemble
    is probably missing a fact. Samples of 0.25 and 0.75 sit about 2.2 logits apart."""


def aggregate(samples: list[float], config: AggregationConfig | None = None) -> float:
    """Combine forecast samples into one probability.

    Mean-of-logits rather than median-of-probabilities: it uses every sample instead of the
    middle one, and it is the space in which probabilistic evidence adds.
    """
    cfg = config or AggregationConfig()
    if not samples:
        raise ValueError("aggregate() needs at least one sample")

    logits = [logit(p) for p in samples]
    mean_logit = sum(logits) / len(logits)
    adjusted = cfg.extremise * mean_logit

    return min(max(sigmoid(adjusted), cfg.min_p), cfg.max_p)


def disagreement(samples: list[float]) -> float:
    """Standard deviation of the samples in logit space. 0.0 for a single sample."""
    if len(samples) < 2:
        return 0.0
    logits = [logit(p) for p in samples]
    mean_logit = sum(logits) / len(logits)
    var = sum((x - mean_logit) ** 2 for x in logits) / (len(logits) - 1)
    return math.sqrt(var)


def needs_more_research(samples: list[float], config: AggregationConfig | None = None) -> bool:
    """True when the ensemble disagrees enough to be worth another research pass.

    This is what the spread signal is actually for. Scattered samples usually mean a specific
    fact is missing, not that the answer is genuinely 50/50.
    """
    cfg = config or AggregationConfig()
    return disagreement(samples) > cfg.research_trigger_spread


def median(samples: list[float]) -> float:
    """The template's aggregator, kept as the baseline being measured against."""
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    p = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return min(max(p, MIN_P), MAX_P)
