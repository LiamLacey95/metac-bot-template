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

# Metaculus FutureEval forecasting scores, captured 2026-07-25.
METACULUS_SCORES = {
    "google/gemini-3.1-pro-preview": 19.84,
    "x-ai/grok-4.20-multi-agent": 14.99,
    "anthropic/claude-opus-4.7": 14.62,
    "openai/gpt-5.5": 14.06,
    "openai/gpt-5.1": 12.14,
    "openai/gpt-5": 11.49,
    "moonshotai/kimi-k2.6": 11.39,
    "x-ai/grok-4.3": 11.35,
    "anthropic/claude-sonnet-4.5": 9.17,
    "z-ai/glm-5": 8.66,
    "deepseek/deepseek-v4-pro": 8.66,
    "anthropic/claude-opus-4.5": 7.75,
    "google/gemini-3-flash-preview": 7.32,
    "x-ai/grok-4.20": 6.13,
    # Negative scores. Cheap is not the same as good value.
    "minimax/minimax-m2.5": -4.74,
    "qwen/qwen3-max-thinking": -1.88,
}

# Rough per-question usage for this bot: one research call plus six forecast samples and their
# parsing. Deliberately conservative - better to over-estimate the season's cost.
INPUT_TOKENS_PER_QUESTION = 28_000
OUTPUT_TOKENS_PER_QUESTION = 9_000

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
