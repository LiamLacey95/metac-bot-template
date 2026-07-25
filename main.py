"""Forecasting bot for the Metaculus AI Benchmark.

Built on Metaculus's template. Changes from it, in descending order of how much they matter:

1. **Cross-model ensemble.** The template samples N forecasts from one model at temperature. Those
   samples share a bias, so averaging them cancels decode noise and nothing else. This rotates
   samples across model families, where errors are partly independent - which is also the only
   condition under which extremising the aggregate is defensible rather than bias amplification.

2. **Logit-mean aggregation with a tail cap** instead of median-of-probabilities. See
   calibration.py; the reasoning is in BOT.md.

3. **Disagreement is logged**, not blended. Scattered samples mean a missing fact, and the
   intended response is another research pass rather than averaging the ignorance.

Numeric questions now elicit 13 percentiles including tail anchors, which the template's six
cannot express. Multiple-choice still falls through to the template.
"""

import argparse
import asyncio
import itertools
import logging
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
)

from calibration import (  # noqa: E402
    EXTREMISE_CROSS_MODEL,
    AggregationConfig,
    aggregate,
    disagreement,
    needs_more_research,
)
from template_bot import SummerTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# READ THIS BEFORE CHANGING THE MODEL LIST.
#
# FutureEval publishes two different numbers and they are easy to confuse. The "skill score" on
# the model leaderboard is anchored so that GPT-4o = 0. The tournament pays on PEER score, which
# is measured against the current bot field - and that field is mostly frontier-model bots, so it
# sits far above GPT-4o. Live in this tournament, GPT-4o scores about -8.4 per question, so a
# skill score converts to expected peer score by subtracting roughly 8.5.
#
# That correction inverts the ranking a naive reading gives:
#
#   model                     skill    live peer/q
#   gemini-3.5-flash              -         +13.2   <- best in the tournament, and cheap
#   gpt-5.5-high                  -          +8.2
#   claude-opus-4.7-high      14.62          +7.9
#   gemini-3.1-pro-high       19.84          +7.4
#   kimi-k2.6                 11.39          ~+3
#   deepseek-v4-pro            8.66          -0.3   <- contributes nothing
#   grok-4.20-multi-agent     14.99          -2.3   <- actively negative
#
# The previous version of this list was half deepseek and included models with negative live
# peer scores, chosen from skill scores as though they were peer scores. Extra samples from a
# zero-peer model do not add coverage value; they dilute the ones that work.
#
# There is also no coverage-versus-quality trade to make, because the best live model is nearly
# the cheapest strong one. So: run it on everything.
ENSEMBLE_MODELS = [
    "openrouter/google/gemini-3.5-flash",
    "openrouter/google/gemini-3.5-flash",
    "openrouter/google/gemini-3.5-flash",
]
# Slugs are verified against OpenRouter's public model list by check_models.py. Run it after any
# edit here: a wrong slug does not fail loudly, it silently removes one family from the ensemble
# and every forecast is quietly worse.

# Output caps. Flash bills thinking as output at the higher rate, so this is the dominant cost
# lever - but the two question types need different room.
#
# A binary answer needs a short rationale and one probability. A numeric answer needs the same
# rationale PLUS thirteen percentile lines, and capping both at 1200 truncated every numeric
# response mid-list: the run failed with `ValidationError: NumericDistribution` on every numeric
# question and produced no forecasts at all. Caught by smoke-testing the rebuild rather than by
# reading it.
MAX_BINARY_TOKENS = 1200
MAX_NUMERIC_TOKENS = 2400

# Zero-cost models, for plumbing checks only. Not competitive, and rate-limited to 429s under
# any sustained load. The route to frontier models at no cost is Metaculus's sponsored-credit
# programme, not OpenRouter's free tier.
FREE_ENSEMBLE_MODELS = [
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/openai/gpt-oss-20b:free",
]


def build_llms(free: bool) -> dict:
    """Pin every model role explicitly.

    Not optional: the framework's default researcher is `gpt-4o-search-preview-2025-03-11`, which
    OpenAI has deprecated. Left unset, every research call fails with a 404 and the bot produces
    no forecasts at all - it does not degrade, it stops. Found by smoke-testing against the
    bot-testing-area tournament, which is exactly what that sandbox is for.
    """
    if free:
        worker = "openrouter/google/gemma-4-31b-it:free"
        return {
            "default": GeneralLlm(model=worker, temperature=0.3, timeout=120, allowed_tries=2),
            "researcher": GeneralLlm(model=worker, temperature=0.1, timeout=120, allowed_tries=2),
            "summarizer": worker,
            "parser": worker,
        }
    return {
        # Numeric and multiple-choice run on the same model as binary, deliberately. Metaculus's
        # own analysis puts the human-versus-bot gap WIDEST on non-binary questions, and they are
        # roughly 40% of the set - so that is where a weak model bleeds the most peer score.
        # Downgrading them to save money would be cutting into the deepest wound.
        "default": GeneralLlm(
            model=ENSEMBLE_MODELS[0],
            temperature=0.3,
            timeout=120,
            allowed_tries=2,
            max_tokens=MAX_NUMERIC_TOKENS,
        ),
        # Plain sonar, not sonar-pro: this is a search call, and the extra reasoning tier is not
        # what makes it useful.
        "researcher": GeneralLlm(
            model="openrouter/perplexity/sonar", temperature=0.1, timeout=180, allowed_tries=2
        ),
        # Parsing is extraction from text that a schema already constrains. Model quality is
        # irrelevant here, and this was the single largest pure waste in the original build.
        "summarizer": "openrouter/xiaomi/mimo-v2.5",
        "parser": "openrouter/xiaomi/mimo-v2.5",
    }


class CalibratedBot(SummerTemplateBot2026):
    """Template bot with a cross-model ensemble and logit-mean aggregation."""

    # The template parses each forecast twice and compares, which turns 6 forecast calls into 12
    # parser calls and makes parsing - not forecasting - the dominant cost. Measured at
    # $0.20/question against a budget that allows $0.05. One parse, on a cheap model, with the
    # structured-output schema already constraining the result.
    _structure_output_validation_samples = 1

    def __init__(self, *args, ensemble_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = ensemble_models or []
        self._model_cycle = itertools.cycle(self.ensemble_models) if self.ensemble_models else None
        # Extremising only earns its place once samples span DIFFERENT families, where errors are
        # partly independent. Count distinct models, not list length: the list repeats a model to
        # weight it, so three entries of one model is still one opinion sampled three times, and
        # extremising it would amplify a shared bias rather than correct anything.
        distinct_families = {m.rsplit("/", 1)[0] for m in self.ensemble_models}
        self.aggregation = AggregationConfig(
            extremise=EXTREMISE_CROSS_MODEL if len(distinct_families) > 1 else 1.0
        )

    def _forecasting_llm(self):
        """Round-robin across the configured families; fall back to the framework's default."""
        if self._model_cycle is None:
            return self.get_llm("default", "llm")
        return GeneralLlm(
            model=next(self._model_cycle),
            temperature=0.3,
            timeout=90,
            allowed_tries=2,
            # Flash bills thinking tokens as output, which is the dominant cost line. The
            # rationale only needs to be long enough to reach a probability.
            max_tokens=MAX_BINARY_TOKENS,
        )

    async def _binary_prompt_to_forecast(self, question, prompt):
        # Same body as the template's, with the model chosen per call rather than fixed.
        from forecasting_tools import BinaryPrediction, ReasonedPrediction, structure_output

        llm = self._forecasting_llm()
        reasoning = await llm.invoke(prompt)
        parsed: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        # Clip only at the edges the API rejects; the real cap is applied once, at aggregation.
        decimal_pred = max(0.001, min(0.999, parsed.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question, research: str):
        """Ask for tail anchors, which the template's six percentiles cannot express.

        The template elicits 10/20/40/60/80/90. That distribution has no vocabulary for "the
        outcome might land outside the displayed range" - the extremes it can state are P10 and
        P90, so everything beyond them is whatever the CDF builder extrapolates. Under a log
        score a distribution with tails that are too thin is punished savagely on exactly the
        questions that surprise you, which are the ones that move a leaderboard.

        Adding 1 / 2.5 / 5 and 95 / 97.5 / 99 lets a forecaster put real mass past an open bound
        and say so explicitly. This mirrors the 13-percentile elicitation used by the
        open-source bots that place well on numeric questions.
        """
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(
            question
        )
        from forecasting_tools import clean_indents
        from datetime import datetime

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Give the answer in the units requested (e.g. whether you write 1,000,000 or 1 million).
            - Never use scientific notation.
            - Values must increase monotonically: percentile 1 is the smallest, percentile 99 the largest.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}

            On the tails, which decide this question's score more than the middle does:
            - Set the 90/10 interval wide. Good forecasters are humble about unknown unknowns.
            - The 1st and 99th percentiles are where you put genuine surprise. Do not set them
              just outside your 10/90 - they should cover scenarios you consider unlikely but
              possible, including ones you have not specifically imagined.
            - If a bound is open and you believe the outcome may fall beyond it, place the
              relevant percentiles PAST that bound rather than piling them up against it.
              Bunching percentiles at an edge claims certainty you do not have.

            The last thing you write is your final answer as:
            "
            Percentile 1: XX
            Percentile 2.5: XX
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 97.5: XX
            Percentile 99: XX
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if not isinstance(question, BinaryQuestion):
            return await super()._aggregate_predictions(predictions, question)

        samples = [float(p) for p in predictions if isinstance(p, (int, float))]
        if len(samples) != len(predictions):
            logger.warning(
                "Non-float binary predictions on %s; using the framework aggregator.",
                question.page_url,
            )
            return await super()._aggregate_predictions(predictions, question)

        result = aggregate(samples, self.aggregation)
        spread = disagreement(samples)

        logger.info(
            "%s | samples=%s | spread=%.2f | aggregate=%.3f%s",
            question.page_url,
            [round(s, 3) for s in samples],
            spread,
            result,
            "  <- HIGH DISAGREEMENT, ensemble is likely missing a fact"
            if needs_more_research(samples, self.aggregation)
            else "",
        )
        return result  # type: ignore[return-value]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Disable the cross-model ensemble (and extremisation with it).",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Use zero-cost models. For smoke tests only - they are not competitive.",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    bot = CalibratedBot(
        research_reports_per_question=1,
        # Three, not six. Extra samples of one model buy maybe half a peer point while doubling
        # the bill, and coverage is worth far more than sample count while the budget binds.
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        ensemble_models=(
            []
            if args.single_model
            else (FREE_ENSEMBLE_MODELS if args.free else ENSEMBLE_MODELS)
        ),
        llms=build_llms(free=args.free),
    )
    if args.free:
        logger.warning(
            "Running on zero-cost models. These are for verifying plumbing, not for competing: "
            "the free tier scores near zero on Metaculus's own model leaderboard."
        )

    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/summer-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-summer-2025/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        minibench = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
        forecast_reports = seasonal + minibench
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )
    else:
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
