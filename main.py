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

Numeric and multiple-choice fall through to the template.

A 13-percentile elicitation with tail anchors was tried, blamed for a run in which every numeric
question failed with `ValidationError: NumericDistribution`, and reverted. That diagnosis was
wrong. Comparing the failing logs against the last passing run showed the percentile list had
completed fine there; what broke the run was two other changes made in the same window - a 1200
token cap that cut the rationale off before its answer, and a parser model that could not extract
a schema. Both are fixed above. The percentile change is worth re-trying now that it is not
carrying the blame for something else.
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

# Output cap. Flash bills thinking as output at the higher rate, so this looks like the obvious
# cost lever - and squeezing it was the single most expensive mistake in this build.
#
# At 1200 the reasoning was cut mid-sentence before it reached the probability line. The parser
# then correctly reported no forecast in the text, and the question scored nothing. A truncated
# call still bills for every token it generated, so a tight cap does not save money: it pays full
# price for a guaranteed zero. Two full sandbox passes and $1.06 went on this.
#
# The cap has to cover THINKING as well as the answer, because Flash spends both out of the same
# budget and bills them at the same rate. That is why 4000 still truncated: a question that
# provoked a long deliberation had nothing left over to write the answer with, so the length of
# the visible output depended on how hard the model happened to think.
#
# So this is a runaway guard and NOT the cost lever. Squeezing it does not save money, it buys
# truncated answers at full price.
MAX_FORECAST_TOKENS = 16000

# The actual cost lever. Flash is not cheap in absolute terms - $9.00/M output against
# deepseek's $0.87 - and it bills thinking at that output rate, so an unbounded thinking budget
# is the one line item that can quietly drain the season's credit. It is still the best value in
# the field (232 peer points per pound against 150 for the next best; run model_value.py), so the
# answer is to bound its thinking rather than to move off it.
#
# "low" rather than off: these are judgement questions, and the reasoning is what is being paid
# for. Measure with a bracketed smoke_one run before changing this in either direction.
REASONING_EFFORT = "low"

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
            max_tokens=MAX_FORECAST_TOKENS,
            reasoning_effort=REASONING_EFFORT,
        ),
        # Plain sonar, not sonar-pro: this is a search call, and the extra reasoning tier is not
        # what makes it useful.
        "researcher": GeneralLlm(
            model="openrouter/perplexity/sonar", temperature=0.1, timeout=180, allowed_tries=2
        ),
        # Parsing is extraction from text that a schema already constrains, so this is the right
        # place to spend nothing - but "cheap" is not the same as "any cheap model". Swapping this
        # to xiaomi/mimo-v2.5 to shave a fraction of a cent produced 293 parse failures in one
        # pass: it answered `<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>` on text that plainly held a
        # forecast. deepseek-v4-pro is a poor forecaster (-0.3 live peer) and a reliable
        # extractor, which is exactly the job. Zero parse failures over a full pass.
        "summarizer": "openrouter/deepseek/deepseek-v4-pro",
        "parser": "openrouter/deepseek/deepseek-v4-pro",
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
            max_tokens=MAX_FORECAST_TOKENS,
            reasoning_effort=REASONING_EFFORT,
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
        # Two, not six. The samples all come from one model at temperature, so averaging them
        # cancels decode noise and nothing else - the third sample buys a fraction of a peer
        # point. Coverage buys much more: an unforecast question scores exactly 0, and the prize
        # pool pays on the SQUARE of summed peer score, so a question skipped for want of credit
        # is the most expensive thing that can happen. Two samples is ~50% more questions covered.
        # Raise this the day Metaculus's sponsored credits land and the budget stops binding.
        predictions_per_research_report=2,
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
