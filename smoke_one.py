"""Forecast a single question and report whether it parsed. Verification that costs pennies.

A full pass over the bot-testing-area tournament is ~20 questions and ~$0.60-0.87 of credit. Most
of the bugs worth catching - a truncated rationale, a parser that cannot extract a schema, a dead
model slug - show up identically on one question. Use this to confirm a fix, and spend the full
pass only on the final check before enabling the schedule.

    python smoke_one.py binary
    python smoke_one.py numeric
    python smoke_one.py multiple_choice
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import dotenv

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from main import ENSEMBLE_MODELS, CalibratedBot, build_llms  # noqa: E402

dotenv.load_dotenv()
logging.basicConfig(level=logging.WARNING)

QUESTION_TYPES = {
    "binary": BinaryQuestion,
    "numeric": NumericQuestion,
    "multiple_choice": MultipleChoiceQuestion,
}


async def main(kind: str) -> int:
    questions = MetaculusClient().get_all_open_questions_from_tournament("bot-testing-area")
    wanted = QUESTION_TYPES[kind]
    match = next((q for q in questions if isinstance(q, wanted)), None)
    if match is None:
        print(f"No open {wanted.__name__} in the sandbox tournament - nothing to smoke test.")
        return 2

    bot = CalibratedBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # one sample: this checks plumbing, not accuracy
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
        ensemble_models=ENSEMBLE_MODELS[:1],
        llms=build_llms(free=False),
    )

    print(f"Question: {match.question_text}\n{match.page_url}\n")
    reports = await bot.forecast_questions([match], return_exceptions=True)
    report = reports[0]

    if isinstance(report, BaseException):
        print(f"FAIL: {type(report).__name__}: {report}")
        return 1

    reasoning = (report.explanation or "").strip()
    print(f"Prediction: {str(report.prediction)[:400]}")
    # The failure this catches is a rationale cut off before its answer, so the END is the part
    # worth printing - a truncated one stops mid-sentence instead of on a probability.
    print(f"\nRationale ends: ...{reasoning[-200:]!r}")

    # A prompt change is only real if the model actually answers it. The base-rate step is the
    # best-evidenced thing in the Metaculus bot-maker survey and also the easiest to add and then
    # never check, so this asserts the rationale really contains a reference class rather than
    # trusting that a longer instruction was obeyed.
    lowered = reasoning.lower()
    has_base_rate = any(k in lowered for k in ("base rate", "reference class"))
    print(f"\nbase-rate step present: {'YES' if has_base_rate else 'NO - prompt was ignored'}")
    if not has_base_rate:
        print("  (rationale head)\n  " + reasoning[:500].replace("\n", "\n  "))
    return 0 if has_base_rate else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(QUESTION_TYPES), nargs="?", default="binary")
    raise SystemExit(asyncio.run(main(parser.parse_args().kind)))
