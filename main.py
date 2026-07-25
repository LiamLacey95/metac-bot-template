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

Deliberately unchanged: numeric and multiple-choice handling falls through to the template. Those
are where the field bleeds most points, so they are the next thing to fix - but porting someone
else's proven pipeline is a separate change that should land on its own.
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

# Three families, so the errors are not the same errors. All reachable on the sponsored
# OpenRouter credits. Order matters only in that the rotation is round-robin.
# Six samples per question, weighted by forecasting-skill-per-pound rather than by reputation.
# Scores are Metaculus's own FutureEval measurements on resolved questions; prices are
# OpenRouter's. model_value.py recomputes the whole table.
#
#   gemini-3.1-pro   19.84  £0.130/q   the best forecaster available, one sample for its judgement
#   kimi-k2.6        11.39  £0.034/q   strong and cheap; two samples
#   deepseek-v4-pro   8.66  £0.016/q   best value on the board; three samples
#
# Roughly £0.04 per question, so a £10 budget covers ~245 questions against the ~200 remaining.
# An all-Gemini ensemble would score higher per question and cover only 77 of them - and since
# unforecast questions score 0 while prize money is quadratic in the summed score, 200 questions
# at a lower edge beats 77 at a higher one.
#
# Deliberately excluded despite being cheap: MiniMax M2.5 (-4.74) and Qwen3 Max Thinking (-1.88)
# score NEGATIVE on Metaculus's forecasting leaderboard. Cheap and wrong is not value.
ENSEMBLE_MODELS = [
    "openrouter/google/gemini-3.1-pro-preview",
    "openrouter/moonshotai/kimi-k2.6",
    "openrouter/moonshotai/kimi-k2.6",
    "openrouter/deepseek/deepseek-v4-pro",
    "openrouter/deepseek/deepseek-v4-pro",
    "openrouter/deepseek/deepseek-v4-pro",
]
# Slugs are verified against OpenRouter's public model list by check_models.py. Run it after any
# edit here: a wrong slug does not fail loudly, it silently removes one family from the ensemble
# and every forecast is quietly worse.

# Zero-cost models, for smoke tests and plumbing checks only. They are NOT competitive: on
# Metaculus's own model leaderboard the free tier scores near zero (Gemma 4: 1.72,
# GPT-OSS-120B: -0.72) against ~14-20 for the frontier models above. Running a season on these
# would be paying nothing for nothing. The route to frontier models at no cost is Metaculus's
# sponsored-credit programme, not OpenRouter's free tier.
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
        # Numeric and multiple-choice questions use this one; only binary rotates the ensemble.
        "default": GeneralLlm(
            model="openrouter/moonshotai/kimi-k2.6", temperature=0.3, timeout=120, allowed_tries=2
        ),
        # Sonar searches the live web, which is what the deprecated default was there to do.
        # AskNews is the other sponsored option and is worth adding as a second source later:
        # search breadth correlates with score more than any single provider does.
        "researcher": GeneralLlm(
            model="openrouter/perplexity/sonar", temperature=0.1, timeout=180, allowed_tries=2
        ),
        # Summarising and parsing are mechanical. Paying frontier prices for them is waste that
        # comes straight out of question coverage.
        "summarizer": "openrouter/deepseek/deepseek-v4-pro",
        "parser": "openrouter/deepseek/deepseek-v4-pro",
    }


class CalibratedBot(SummerTemplateBot2026):
    """Template bot with a cross-model ensemble and logit-mean aggregation."""

    def __init__(self, *args, ensemble_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = ensemble_models or []
        self._model_cycle = itertools.cycle(self.ensemble_models) if self.ensemble_models else None
        # Extremising only earns its place once samples span families. With a single model the
        # samples share one bias and extremising would amplify it.
        self.aggregation = AggregationConfig(
            extremise=EXTREMISE_CROSS_MODEL if len(self.ensemble_models) > 1 else 1.0
        )

    def _forecasting_llm(self):
        """Round-robin across the configured families; fall back to the framework's default."""
        if self._model_cycle is None:
            return self.get_llm("default", "llm")
        return GeneralLlm(model=next(self._model_cycle), temperature=0.3, timeout=90, allowed_tries=2)

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
        predictions_per_research_report=6,  # two passes through three families
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
