"""Score candidate models against each other on questions this tournament has already resolved.

Metaculus's model leaderboard is the best evidence available, and it is three months stale - the
newest model on it is GPT-5.5 (24 April 2026), while OpenRouter carries 53 newer ones including
Kimi K3, Claude Opus 5, GPT-5.6 and Grok 4.5. Picking between those from reputation is exactly
the mistake that put an unmeasured model at the top of this bot's table. So measure them here.

    python bakeoff.py --questions 10 --models x-ai/grok-4.5,openai/gpt-5.6-luna
    python bakeoff.py --dry-run            # print the cost estimate and stop

WHAT THIS MEASURES, AND WHAT IT DOES NOT

Each model gets the question text, background, resolution criteria and fine print - and NO
research. That is deliberate. Every one of these questions has already resolved, so a live search
returns the answer, and a leaked result scores every model near-perfectly and hides the difference
between them. The cost is that this measures reasoning from a fixed context rather than the full
pipeline, so treat it as a screen rather than a ranking: it reliably catches a model that is bad
at this task, and it cannot separate two models a point or two apart. Metaculus needs ~181
forecasts per model for a confidence interval still ~7 points wide; ten will not beat that.

Two scores are reported. The log score is absolute accuracy. The peer score mirrors what the
tournament actually pays on - each model against the mean of the others on the same question -
which is the number that decides prize money.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import dotenv

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import BinaryPrediction, GeneralLlm, structure_output  # noqa: E402

dotenv.load_dotenv()
logging.basicConfig(level=logging.ERROR)

TOURNAMENT = 33022
PARSER_MODEL = "openrouter/deepseek/deepseek-v4-pro"

# Only models cheap enough to actually RUN for a season. Evaluating one that cannot be afforded in
# production answers a question nobody asked - Opus 5 at $25/M output and GPT-5.6 Sol at $30/M
# would each cost more per question than the whole ensemble does now.
DEFAULT_CANDIDATES = [
    "x-ai/grok-4.20-multi-agent",  # the measured baseline: 14.99 skill, the best scored per pound
    "x-ai/grok-4.5",
    "openai/gpt-5.6-luna",
    "z-ai/glm-5.2",
]

PROMPT = """You are a professional forecaster.

Question: {title}

Background:
{background}

Resolution criteria:
{criteria}

{fine_print}

You have no search tool. Reason from what you know and from the text above.

Before answering you write:
(a) The status quo outcome if nothing changed.
(b) A brief scenario that results in a No outcome.
(c) A brief scenario that results in a Yes outcome.

Good forecasters put extra weight on the status quo outcome, since the world changes slowly most
of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def _get(url: str, tries: int = 5) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {os.environ['METACULUS_TOKEN']}",
            "User-Agent": "Mozilla/5.0 (bakeoff)",
        },
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Metaculus rate-limits the detail endpoint hard enough that fetching twenty
            # questions trips it. Back off rather than half-populating the sample.
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_resolved(limit: int) -> list[dict]:
    """Resolved binary questions from this tournament - the exact distribution being forecast."""
    listing = _get(
        "https://www.metaculus.com/api/posts/?"
        + urllib.parse.urlencode(
            {"tournaments": TOURNAMENT, "statuses": "resolved", "limit": 100}
        )
    ).get("results", [])

    out = []
    for p in listing:
        if ((p.get("question") or {}).get("type")) != "binary":
            continue
        # The list endpoint returns resolution: null on every post. It is only populated on the
        # detail endpoint, so each candidate costs one extra (free) call.
        q = (_get(f"https://www.metaculus.com/api/posts/{p['id']}/").get("question")) or {}
        time.sleep(1.0)  # the detail endpoint 429s well before twenty sequential calls
        if q.get("resolution") not in ("yes", "no"):
            continue
        out.append(
            {
                "id": p["id"],
                "title": p["title"],
                "background": q.get("description") or "",
                "criteria": q.get("resolution_criteria") or "",
                "fine_print": q.get("fine_print") or "",
                "outcome": 1.0 if q["resolution"] == "yes" else 0.0,
            }
        )
        if len(out) >= limit:
            break
    return out


async def forecast(model: str, question: dict) -> float | None:
    llm = GeneralLlm(model=f"openrouter/{model}", temperature=0.3, timeout=180, allowed_tries=2)
    try:
        reasoning = await llm.invoke(
            PROMPT.format(
                title=question["title"],
                background=question["background"][:4000],
                criteria=question["criteria"],
                fine_print=question["fine_print"][:1000],
            )
        )
        parsed: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=PARSER_MODEL, num_validation_samples=1
        )
        return max(0.01, min(0.99, parsed.prediction_in_decimal))
    except Exception as e:  # noqa: BLE001 - one model failing must not abort the comparison
        print(f"    {model} failed on {question['id']}: {type(e).__name__}")
        return None


def log_score(p: float, outcome: float) -> float:
    """Natural-log score. 0 is a perfect call, more negative is worse; -0.693 is a coin flip."""
    return math.log(p if outcome else 1 - p)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--models", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--dry-run", action="store_true", help="estimate cost and stop")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    questions = fetch_resolved(args.questions)
    print(f"{len(questions)} resolved binary questions, {len(models)} models "
          f"= {len(questions) * len(models)} forecast calls\n")

    if args.dry_run:
        pricing = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    "https://openrouter.ai/api/v1/models",
                    headers={"User-Agent": "Mozilla/5.0 (bakeoff)"},
                ),
                timeout=45,
            ).read()
        )
        price = {m["id"]: m.get("pricing", {}) for m in pricing.get("data", [])}
        total = 0.0
        for m in models:
            p = price.get(m, {})
            est = len(questions) * (3_000 * float(p.get("prompt", 0)) +
                                    4_000 * float(p.get("completion", 0)))
            total += est
            print(f"  {m:34} ~${est:.3f}")
        print(f"\n  estimated total ~${total:.2f} (plus parsing, roughly $0.01)")
        return 0

    # predictions[model][question_id]
    predictions: dict[str, dict[int, float]] = {m: {} for m in models}
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['title'][:72]}  -> {'YES' if q['outcome'] else 'NO'}")
        results = await asyncio.gather(*(forecast(m, q) for m in models))
        for m, p in zip(models, results):
            if p is not None:
                predictions[m][q["id"]] = p
        print("    " + "  ".join(
            f"{m.split('/')[-1]}={predictions[m].get(q['id'], float('nan')):.2f}" for m in models
        ))

    # Peer score: each model against the mean of the OTHERS on the same question, which is what
    # the tournament pays on. Only questions every model answered are comparable.
    common = set.intersection(*(set(predictions[m]) for m in models)) if models else set()
    outcome = {q["id"]: q["outcome"] for q in questions}

    print(f"\n{'model':34} {'n':>4} {'log score':>10} {'peer':>8}")
    print("-" * 60)
    rows = []
    for m in models:
        answered = predictions[m]
        if not answered:
            continue
        mean_log = sum(log_score(p, outcome[i]) for i, p in answered.items()) / len(answered)
        peers = []
        for qid in common:
            others = [log_score(predictions[o][qid], outcome[qid]) for o in models if o != m]
            if others:
                peers.append(100 * (log_score(predictions[m][qid], outcome[qid])
                                    - sum(others) / len(others)))
        rows.append((m, len(answered), mean_log, sum(peers) / len(peers) if peers else 0.0))

    for m, n, ls, peer in sorted(rows, key=lambda r: -r[3]):
        print(f"{m:34} {n:>4} {ls:>10.4f} {peer:>+8.1f}")

    print(f"\nPeer measured on the {len(common)} questions every model answered. A screen, not a\n"
          f"ranking - see the module docstring before acting on a gap of a point or two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
