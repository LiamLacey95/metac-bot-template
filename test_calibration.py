"""Does the calibration layer actually beat the template's median? Run: python test_calibration.py

The simulation is deliberately built so it CAN return a negative answer. A forecaster is modelled
as reporting sigmoid(c * logit(q) + noise), where q is the true probability:

    c > 1  overconfident - pushes forecasts toward 0 and 1 (the documented LLM failure)
    c = 1  calibrated
    c < 1  underconfident

If shrinkage only ever helped, the simulation would be rigged. It should lose at c <= 1, and the
grid below reports that rather than hiding it.
"""

import math
import random

from calibration import CalibrationConfig, aggregate, log_score, logit, median, sigmoid

N_QUESTIONS = 4000
N_SAMPLES = 5
SEED = 42


def simulate(c: float, noise: float, n_samples: int = N_SAMPLES, seed: int = SEED):
    """Returns (mean log score of median, mean log score of aggregate). Higher is better."""
    rng = random.Random(seed)
    total_median = 0.0
    total_agg = 0.0

    for _ in range(N_QUESTIONS):
        # True probabilities spread across the range, with a base rate near 0.35 - most
        # "will X happen" questions resolve No.
        q = rng.betavariate(1.6, 3.0)
        outcome = 1 if rng.random() < q else 0

        samples = [sigmoid(c * logit(q) + rng.gauss(0, noise)) for _ in range(n_samples)]

        total_median += log_score(median(samples), outcome)
        total_agg += log_score(aggregate(samples), outcome)

    return total_median / N_QUESTIONS, total_agg / N_QUESTIONS


def base_rate_check(seed: int = SEED) -> float:
    rng = random.Random(seed)
    return sum(rng.betavariate(1.6, 3.0) for _ in range(20000)) / 20000


def main() -> None:
    print(f"Simulated base rate: {base_rate_check():.3f}  (config prior: {CalibrationConfig().prior})")
    print(f"{N_QUESTIONS} questions, {N_SAMPLES} samples each, log score in nats (higher = better)\n")

    header = f"{'overconf c':>10} {'noise':>6} {'median':>9} {'calibrated':>11} {'delta':>9}   verdict"
    print(header)
    print("-" * len(header))

    results = {}
    for c in (1.0, 1.3, 1.6, 2.0, 2.5):
        for noise in (0.5, 1.0, 2.0):
            m, a = simulate(c, noise)
            delta = a - m
            results[(c, noise)] = delta
            verdict = "better" if delta > 0.001 else ("worse" if delta < -0.001 else "~same")
            print(f"{c:>10.1f} {noise:>6.1f} {m:>9.4f} {a:>11.4f} {delta:>+9.4f}   {verdict}")

    print()

    # The claim being tested: the layer earns its place where LLMs actually live - overconfident
    # and noisy. It is allowed to lose elsewhere, and the assertions say so explicitly.
    overconfident = [d for (c, _), d in results.items() if c >= 1.6]
    assert all(d > 0 for d in overconfident), (
        "calibration must beat the median wherever the forecaster is meaningfully overconfident"
    )

    calibrated_low_noise = results[(1.0, 0.5)]
    print(f"Honest limitation: against a perfectly calibrated, low-noise forecaster the layer "
          f"scores {calibrated_low_noise:+.4f} nats.")
    if calibrated_low_noise < 0:
        print("  It costs something there, as it should - shrinking a correct forecast is pure loss.")
        print("  The bet is that no LLM is in that regime. That is checkable against resolved")
        print("  questions and is the first thing to verify with real data.")

    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\nLargest gain: {best[1]:+.4f} nats at overconfidence c={best[0][0]}, noise={best[0][1]}")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
