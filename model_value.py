"""Pick models by forecasting skill per pound, not by reputation.

Metaculus publishes a forecasting score per model on its FutureEval leaderboard - measured on
real resolved questions, which is the only benchmark that matters here. OpenRouter publishes
prices. Joining the two answers "best model for the job" with a number instead of a vibe.

Scores below are read from https://www.metaculus.com/futureeval/ (higher is better; Metaculus
Pro forecasters sit at 35.90 and the community at 25.96, for scale).
"""

from __future__ import annotations

import json
import urllib.request

# LIVE PEER SCORE PER QUESTION, read from the Summer 2026 tournament leaderboard in advanced
# mode (Metaculus runs its own benchmark bots in the field, so their per-question peer scores are
# directly observable).
#
# THIS IS NOT THE SAME NUMBER AS THE MODEL LEADERBOARD'S "SKILL SCORE", and confusing the two
# cost me a whole ensemble design. Skill score is anchored so GPT-4o = 0. Peer score is measured
# against the CURRENT bot field, which is mostly frontier-model bots and therefore far above
# GPT-4o - live, GPT-4o scores about -8.4 per question here. So:
#
#     expected peer per question  ~=  skill score - 8.5
#
# Read off the skill scale, deepseek-v4-pro looks like a mid-table bargain at 8.66. Live it is
# -0.3 per question: it contributes nothing, and samples spent on it are samples not spent on a
# model that scores. Only the peer numbers below should drive any decision.
LIVE_PEER_PER_QUESTION = {
    "google/gemini-3.5-flash": 13.2,
    "openai/gpt-5.5": 8.2,
    "anthropic/claude-opus-4.7": 7.9,
    "google/gemini-3.1-pro-preview": 7.4,
    "moonshotai/kimi-k2.6": 3.0,
    "deepseek/deepseek-v4-pro": -0.3,
    "x-ai/grok-4.3": -0.4,
    "x-ai/grok-4.20-multi-agent": -2.3,
}
METACULUS_SCORES = LIVE_PEER_PER_QUESTION

# Per-question usage for the current build: one research call, three forecast samples with output
# capped at 1200 tokens, and three parser calls on a cheap model. Deliberately conservative.
INPUT_TOKENS_PER_QUESTION = 18_000
OUTPUT_TOKENS_PER_QUESTION = 5_000

GBP_PER_USD = 0.79


def fetch_pricing() -> dict[str, tuple[float, float]]:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "Mozilla/5.0 (model-value-check)"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = {}
    for m in data.get("data", []):
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    pricing = fetch_pricing()
    rows = []
    for slug, score in METACULUS_SCORES.items():
        if slug not in pricing:
            rows.append((slug, score, None, None))
            continue
        pin, pout = pricing[slug]
        usd = INPUT_TOKENS_PER_QUESTION * pin + OUTPUT_TOKENS_PER_QUESTION * pout
        rows.append((slug, score, usd * GBP_PER_USD, score / (usd * GBP_PER_USD) if usd else None))

    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))

    print(f"{'model':40} {'score':>7} {'£/question':>11} {'score per £':>12}")
    print("-" * 74)
    for slug, score, gbp, value in rows:
        if gbp is None:
            print(f"{slug:40} {score:>7.2f} {'not on OR':>11} {'-':>12}")
        elif score <= 0:
            print(f"{slug:40} {score:>7.2f} {gbp:>11.4f} {'negative':>12}")
        else:
            print(f"{slug:40} {score:>7.2f} {gbp:>11.4f} {value:>12.1f}")

    print(f"\nAssumes {INPUT_TOKENS_PER_QUESTION:,} input + {OUTPUT_TOKENS_PER_QUESTION:,} "
          f"output tokens per question (research + 6 samples + parsing).")

    budget = 10.0
    print(f"\nWith a £{budget:.0f} budget, questions coverable per model:")
    for slug, score, gbp, _ in rows[:8]:
        if gbp:
            print(f"  {slug:40} {int(budget / gbp):>5} questions")
    print("\n~200 questions remain in the season, and unforecast questions score 0.")


if __name__ == "__main__":
    main()
