"""Property tests for the aggregation layer. Run: python test_calibration.py

The previous version of this file ran a simulation that assumed the forecaster was overconfident
and then reported that a shrinkage layer helped. That assumed its conclusion. Overconfidence is
the thing that would need measuring, not asserting, and it can only be measured against resolved
forecasts from this bot.

So this file tests properties that must hold by construction, plus one comparison against the
template's median that makes no assumption about which direction the model is miscalibrated.
"""

import math
import random

from calibration import (
    MAX_P,
    MIN_P,
    AggregationConfig,
    aggregate,
    disagreement,
    log_score,
    logit,
    median,
    needs_more_research,
    sigmoid,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def test_is_plain_logit_mean() -> None:
    samples = [0.2, 0.5, 0.8]
    expected = sigmoid(sum(logit(p) for p in samples) / 3)
    assert approx(aggregate(samples), expected), "default aggregate must be the logit mean"


def test_no_pull_toward_any_prior() -> None:
    """The bug that motivated the rewrite: agreeing samples must survive untouched."""
    for p in (0.90, 0.75, 0.60, 0.40, 0.25, 0.10):
        out = aggregate([p] * 7)
        assert approx(out, p, 1e-6), f"unanimous {p} came back as {out} - something is pulling it"


def test_symmetry() -> None:
    """Aggregating the complements must give the complement. A fixed prior breaks this, which is
    exactly how a standing directional bet shows up."""
    samples = [0.15, 0.35, 0.62, 0.71]
    flipped = [1 - p for p in samples]
    assert approx(aggregate(samples), 1 - aggregate(flipped), 1e-9)


def test_clipping() -> None:
    assert aggregate([0.9999] * 5) == MAX_P
    assert aggregate([0.0001] * 5) == MIN_P
    assert MIN_P == 0.02 and MAX_P == 0.98


def test_extremise_pushes_away_from_half() -> None:
    samples = [0.70, 0.75, 0.80]
    plain = aggregate(samples)
    hot = aggregate(samples, AggregationConfig(extremise=1.3))
    assert hot > plain > 0.5, "extremisation must move a >50% belief further from 0.5"
    # and symmetrically below 0.5
    low = [1 - p for p in samples]
    assert aggregate(low, AggregationConfig(extremise=1.3)) < aggregate(low) < 0.5


def test_disagreement_signal() -> None:
    assert approx(disagreement([0.5]), 0.0)
    assert approx(disagreement([0.6, 0.6, 0.6]), 0.0)
    tight = disagreement([0.70, 0.72, 0.74])
    wide = disagreement([0.15, 0.55, 0.90])
    assert wide > tight
    assert not needs_more_research([0.70, 0.72, 0.74])
    assert needs_more_research([0.10, 0.50, 0.92]), "a 0.10-0.92 spread must trigger more research"


def test_single_sample_is_passed_through() -> None:
    assert approx(aggregate([0.63]), 0.63, 1e-6)


def compare_against_median(c: float, noise: float, n: int = 4000, seed: int = 42):
    """Log score of logit-mean vs median. `c` is the forecaster's calibration slope: >1 is
    overconfident, <1 underconfident, 1 calibrated. Reported across the range rather than
    assumed."""
    rng = random.Random(seed)
    tot_med = tot_agg = 0.0
    for _ in range(n):
        q = rng.betavariate(1.6, 3.0)
        outcome = 1 if rng.random() < q else 0
        samples = [sigmoid(c * logit(q) + rng.gauss(0, noise)) for _ in range(7)]
        tot_med += log_score(median(samples), outcome)
        tot_agg += log_score(aggregate(samples), outcome)
    return tot_med / n, tot_agg / n


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} property tests passed.\n")

    print("Logit-mean vs the template's median, across calibration regimes.")
    print("No assumption is made about which regime this bot is in - that needs resolved data.\n")
    print(f"{'slope c':>8} {'noise':>6} {'median':>9} {'logit-mean':>11} {'delta':>9}")
    print("-" * 46)
    for c in (0.8, 1.0, 1.3, 2.0):
        for noise in (0.5, 1.5):
            m, a = compare_against_median(c, noise)
            print(f"{c:>8.1f} {noise:>6.1f} {m:>9.4f} {a:>11.4f} {a - m:>+9.4f}")

    print("\nThe two aggregators are close, as they should be - this layer is not where the")
    print("points are. The real edges this season are coverage, model choice, and the numeric")
    print("pipeline. See BOT.md.")


if __name__ == "__main__":
    main()
