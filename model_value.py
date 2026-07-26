"""Pick models by measured forecasting skill per pound, not by reputation.

Metaculus publishes a forecasting score per model on its FutureEval leaderboard, measured on real
resolved questions - the only benchmark that matters here. OpenRouter publishes prices. Joining
the two answers "best model for the job" with a number instead of a vibe.

    python model_value.py

TWO NUMBERS THAT ARE EASY TO CONFUSE, AND CONFUSING THEM COST THIS BOT AN ENSEMBLE:

- The leaderboard's **skill score** is anchored so GPT-4o = 0.
- The tournament pays on **peer score**, measured against the current bot field, which is mostly
  frontier-model bots and therefore far above GPT-4o. Live, GPT-4o scores about -8.4 per question.

So `expected peer per question ~= skill score - 8.5`, and a model has to clear roughly 8.5 skill
just to break even against the field. Read off the skill scale, DeepSeek V4 Pro looks like a
mid-table bargain at 8.66; live it is about -0.3 and contributes nothing.

WHAT "HIGH" MEANS. Metaculus benchmarks several models twice, once at default reasoning and once
at high (`... High`). The gap is large and consistent - Grok 4.20 goes 6.13 to 14.99, GPT 5.1 goes
4.64 to 12.14, Kimi K2 goes 0.97 to 5.64, DeepSeek v3.1 goes 2.37 to 5.41. Reasoning effort is not
a cost dial with a small quality penalty attached; on this task it is worth more than most model
upgrades. Weigh that against the fact that thinking is billed as output tokens.
"""

from __future__ import annotations

import json
import urllib.request

# Skill scores read from https://www.metaculus.com/futureeval/leaderboard/ on 2026-07-26, joined
# to the OpenRouter slug that actually serves each model. `effort` records which leaderboard row
# this is - the same slug appears twice where Metaculus benchmarked both reasoning settings.
#
# For scale: Metaculus Pro Forecasters 35.90, Metaculus Community 25.96, GPT-4o 0.00.
#
# 77 models are scored there. The eight-model table this file used to hold was not a shortlist, it
# was everything I had bothered to look up - and it did not contain the model the bot was running.
MEASURED = [
    # (leaderboard name,       openrouter slug,                  skill, effort)
    ("Gemini 3.1 Pro High",    "google/gemini-3.1-pro-preview",  19.84, "high"),
    ("Gemini 3.1 Pro",         "google/gemini-3.1-pro-preview",  19.03, "default"),
    ("Grok 4.20 Multi-Agent",  "x-ai/grok-4.20-multi-agent",     14.99, "default"),
    ("Claude Opus 4.7 High",   "anthropic/claude-opus-4.7",      14.62, "high"),
    ("GPT-5.5 High",           "openai/gpt-5.5",                 14.06, "high"),
    ("Kimi K2.6",              "moonshotai/kimi-k2.6",           11.39, "default"),
    ("Grok 4.3 High",          "x-ai/grok-4.3",                  11.35, "high"),
    ("Gemini 3 Pro",           "google/gemini-3-pro",            10.57, "default"),
    ("Claude Sonnet 4.5",      "anthropic/claude-sonnet-4.5",     9.17, "default"),
    ("GLM-5",                  "z-ai/glm-5",                      8.66, "default"),
    ("DeepSeek V4 Pro High",   "deepseek/deepseek-v4-pro",        8.66, "high"),
    ("Qwen 3.6 Plus",          "qwen/qwen-3.6-plus",              7.53, "default"),
    ("Gemini 3 Flash",         "google/gemini-3-flash",           7.32, "default"),
    ("Grok 4.20",              "x-ai/grok-4.20",                  6.13, "default"),
    ("GLM 5.1",                "z-ai/glm-5.1",                    5.71, "default"),
    ("Gemini 3.1 Flash Lite",  "google/gemini-3.1-flash-lite",    3.99, "default"),
    ("Gemma 4",                "google/gemma-4-31b-it",           1.72, "default"),
    ("Qwen3 Max Thinking",     "qwen/qwen3-max-thinking",        -1.88, "default"),
    ("MiniMax M2.5",           "minimax/minimax-m2.5",           -4.74, "default"),
]

# Models this bot has run that Metaculus has NOT benchmarked. Not "untested so probably fine" -
# unmeasured, on a leaderboard whose spread runs from +19.84 to -18.82. google/gemini-3.5-flash
# was released 2026-05-19, earlier than several models that ARE scored, so its absence from the
# leaderboard is not a recency artefact.
UNMEASURED = ["google/gemini-3.5-flash", "google/gemini-3.5-flash-lite", "perplexity/sonar"]

GPT4O_LIVE_PEER = -8.4  # what a skill score of 0 is worth against the current bot field

# Per-question usage: one research call, two forecast samples, two parser calls. Output includes
# thinking, which is why the high-effort rows cost more on the same slug.
INPUT_TOKENS_PER_QUESTION = 18_000
OUTPUT_TOKENS_DEFAULT = 4_000
OUTPUT_TOKENS_HIGH = 12_000

GBP_PER_USD = 0.79
BUDGET_GBP = 4.20  # what is actually left, not a round number


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
    for name, slug, skill, effort in MEASURED:
        peer = skill + GPT4O_LIVE_PEER
        if slug not in pricing:
            rows.append((name, effort, skill, peer, None, None))
            continue
        pin, pout = pricing[slug]
        out_tokens = OUTPUT_TOKENS_HIGH if effort == "high" else OUTPUT_TOKENS_DEFAULT
        gbp = (INPUT_TOKENS_PER_QUESTION * pin + out_tokens * pout) * GBP_PER_USD
        rows.append((name, effort, skill, peer, gbp, (peer / gbp) if gbp and peer > 0 else None))

    rows.sort(key=lambda r: (r[5] is None, -(r[5] or 0)))

    header = (f"{'model':24} {'effort':8} {'skill':>6} {'peer/q':>7} "
              f"{'£/q':>8} {'peer per £':>11} {'£4.20 buys':>11}")
    print(header)
    print("-" * len(header))
    for name, effort, skill, peer, gbp, value in rows:
        if gbp is None:
            print(f"{name:24} {effort:8} {skill:>6.2f} {peer:>7.2f} "
                  f"{'not on OR':>8} {'-':>11} {'-':>11}")
        elif value is None:
            print(f"{name:24} {effort:8} {skill:>6.2f} {peer:>7.2f} "
                  f"{gbp:>8.4f} {'scores <=0':>11} {'-':>11}")
        else:
            print(f"{name:24} {effort:8} {skill:>6.2f} {peer:>7.2f} "
                  f"{gbp:>8.4f} {value:>11.1f} {int(BUDGET_GBP / gbp):>9} q")

    print("\nUNMEASURED - no Metaculus score exists for these, at any price:")
    for slug in UNMEASURED:
        pin, pout = pricing.get(slug, (None, None))
        price = f"in ${pin * 1e6:.2f}/M  out ${pout * 1e6:.2f}/M" if pin is not None else "not on OR"
        print(f"  {slug:34} {price}")

    print(
        "\nPeer per pound is the right sort order only while the budget binds, and it flatters\n"
        "cheap models. Prize share is proportional to the SQUARE of summed peer score, so a model\n"
        "that scores twice as well on half as many questions is not a wash - it is ahead. Coverage\n"
        "wins when peer scores are close together; quality wins when they are not."
    )


if __name__ == "__main__":
    main()
