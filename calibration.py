"""Calibration layer for a Metaculus AIB bot.

The tournament scores with a log score, which punishes confident mistakes far harder than it
rewards confident hits. Being right about the *direction* is worth little if the probability is
wrong; that is the exact failure that sank my last competition entry, where ranking transferred
across the distribution shift and calibration did not.

The template bot samples N independent forecasts per question and takes the median. Two facts
pull in opposite directions there:

  1. An individual LLM forecast is overconfident - it says 5% and 95% far more often than the
     world resolves that way.
  2. The median of several forecasts is underconfident, because averaging pulls toward the middle.

Which effect dominates is not fixed: it depends on whether the samples agree. When five runs all
say 0.90, the ensemble carries real information and should not be softened much. When they say
0.15/0.40/0.60/0.85/0.90, the model does not know, and the median of 0.60 is a fiction with a
confident face on it.

So the shrinkage is driven by the observed spread rather than by a constant. That is the whole
idea, and it is cheap: it needs no extra model calls, only the samples the bot already produced.

Everything here operates in logit space, because that is the space in which probabilistic
evidence adds and in which "halfway between 0.9 and 0.99" means something sensible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Metaculus clamps submissions to [0.01, 0.99]. Going to the boundary costs 4.6 nats when wrong,
# which is a bad trade on any question that is not close to certain.
MIN_P = 0.01
MAX_P = 0.99


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
class CalibrationConfig:
    """Defaults are deliberately mild. Every one of these should be tuned against resolved
    questions once there are enough of them; until then, erring toward the prior is the cheaper
    mistake under a log score."""

    prior: float = 0.35
    """Base rate to shrink toward. Metaculus binary questions resolve YES well under half the
    time - most 'will X happen by date Y' questions do not happen - so 0.5 is the wrong anchor.
    This is the single most important number here and the first thing to fit on real data."""

    global_trust: float = 0.85
    """Systematic haircut applied even when the samples agree perfectly, to counter the
    overconfidence that survives ensembling."""

    spread_scale: float = 1.20
    """Logit-space spread at which the ensemble is treated as carrying about half its nominal
    weight. Roughly: samples of 0.30 and 0.70 differ by ~1.7 logits."""

    max_abs_logit: float = 4.0
    """Hard ceiling on final confidence, ~1.8% to ~98.2%. A bot with no private information
    should not be more certain than this."""


def aggregate(samples: list[float], config: CalibrationConfig | None = None) -> float:
    """Combine independent forecast samples into one calibrated probability.

    Mean-of-logits rather than median-of-probabilities: it is the aggregator that treats evidence
    additively, and unlike the median it actually uses every sample. The median throws away the
    disagreement, which is the signal this whole module is built on.
    """
    cfg = config or CalibrationConfig()
    if not samples:
        raise ValueError("aggregate() needs at least one sample")

    logits = [logit(p) for p in samples]
    mean_logit = sum(logits) / len(logits)

    if len(logits) > 1:
        var = sum((x - mean_logit) ** 2 for x in logits) / (len(logits) - 1)
        spread = math.sqrt(var)
    else:
        # A single sample carries no disagreement information, so it gets no benefit of the
        # doubt: treat it as if it sat at the half-weight spread.
        spread = cfg.spread_scale

    # Agreement -> trust. Disagreement -> fall back toward the base rate.
    agreement = 1.0 / (1.0 + (spread / cfg.spread_scale) ** 2)
    weight = cfg.global_trust * agreement

    prior_logit = logit(cfg.prior)
    shrunk = prior_logit + weight * (mean_logit - prior_logit)
    shrunk = max(-cfg.max_abs_logit, min(cfg.max_abs_logit, shrunk))

    return min(max(sigmoid(shrunk), MIN_P), MAX_P)


def median(samples: list[float]) -> float:
    """The template's aggregator, kept here as the thing to beat."""
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    p = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return min(max(p, MIN_P), MAX_P)
