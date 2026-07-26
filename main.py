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
# THERE ARE TWO METACULUS LEADERBOARDS AND ONLY ONE OF THEM PREDICTS ANYTHING HERE.
#
# 1. The MODEL leaderboard (metaculus.com/futureeval/leaderboard/) publishes a "skill score"
#    anchored so GPT-4o = 0. It is measured on a different question set, and as of 2026-07-26 its
#    newest entry is GPT-5.5 from 24 April.
# 2. The TOURNAMENT leaderboard, with the "advanced" toggle on, shows Metaculus's own benchmark
#    bots competing in THIS season on THESE questions. Divide Total Score by Questions and you
#    have the live peer score per question - which is the number prize money is computed from.
#
# The two disagree violently, and only the second one is the payout:
#
#   benchmark bot (live, 56-57 questions)      peer/q     model-leaderboard skill
#   metac-gemini-3-5-flash+asknews             +13.48     not listed at all
#   metac-gpt-5-5-instant+asknews              +11.24     13.84
#   metac-gpt-5-5-high+asknews                  +8.30     14.06
#   metac-claude-opus-4-7-high+asknews          +7.87     14.62
#   metac-claude-opus-4-8-high+asknews          +6.74     not listed
#   metac-gemini-3-1-pro-high+asknews           +7.40     19.84  <- top of the model leaderboard
#   metac-gemini-3-1-pro+asknews                +3.45     19.03
#   metac-deepseek-v4-pro-high+asknews          -0.30      8.66
#   metac-grok-4-3-high+asknews                 -0.39     11.35
#   metac-kimi-k2-6+asknews                     -1.74     11.39
#   metac-grok-4-20-multi-agent+asknews         -2.26     14.99
#
# gemini-3.1-pro-high tops the model leaderboard at 19.84 and earns +7.40 live. grok-4.20
# multi-agent sits fifth at 14.99 and is NEGATIVE live. Ranking by skill score picks a model that
# loses points. This file briefly did exactly that, on the reasoning that gemini-3.5-flash "had no
# measured score" - it has the best one in the tournament, on the only leaderboard that pays.
#
# So: read the tournament leaderboard in advanced mode. It is also the only source that covers
# recent models at all - claude-opus-4-8, claude-sonnet-4-6, claude-fable-5-high, minimax-m3 and
# glm-5-2 all appear there and on no model leaderboard. Kimi K3, Opus 5, GPT-5.6 and Grok 4.5 are
# not benchmarked anywhere yet, so switching to one would be a guess, and the guesses on this page
# have a poor record.
# CURRENT SETTING IS A DELIBERATE BET ON UNMEASURED MODELS, made by Liam, against the table above.
# Nothing released since 24 April is benchmarked anywhere - not on the model leaderboard, not as a
# tournament benchmark bot - so Kimi K3, GPT-5.6 and Opus 5 have no forecasting score at all. The
# case for them is the trend line on FutureEval's own chart, which slopes up with release date;
# the case against is that the same chart has MiniMax M2.5 at -4.74 and Qwen3 Max Thinking at
# -1.88, both recent. Recency raises the ceiling and guarantees nothing.
#
# Two families rather than two copies of one, which is also the only condition under which the
# extremisation step below is defensible.
#
# THE COST. GPT-5.6 Sol is $5/M in and $30/M out; Kimi K3 is $3/M and $15/M. Together, at two
# samples on low reasoning, that is roughly £0.23 a question against gemini-3.5-flash's £0.05 -
# about 22 questions of coverage on the credit left, against ~105, with ~200 still to come before
# the season closes on 6 September. Prize share is proportional to the SQUARE of summed peer
# score, so coverage that small needs these models to be dramatically better, not marginally.
# If the sponsored credits arrive, this is the right list. If they do not, revert to
# gemini-3.5-flash and take the coverage.
ENSEMBLE_MODELS = [
    "openrouter/openai/gpt-5.6-sol",
    "openrouter/moonshotai/kimi-k3",
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

# "low" because thinking bills as output and GPT-5.6 Sol charges $30/M for it, so an unbounded
# thinking budget on this list would cost more per question than the entire previous ensemble.
# Measured on the model it replaced, low effort cut cost 3.7x for an answer that reached the same
# conclusion by the same route.
#
# The counter-evidence is on the model leaderboard, where the high-effort row of the same model is
# consistently and substantially better - Grok 4.20 goes 6.13 to 14.99, GPT 5.1 goes 4.64 to
# 12.14, Kimi K2 goes 0.97 to 5.64. On a bigger budget this should be "high", and the fact that it
# is not is a budget decision rather than a forecasting one.
REASONING_EFFORT = "low"

# Omitted entirely rather than passed as None, so the request carries the provider's own default
# instead of an explicit null the provider may or may not interpret the same way.
EFFORT_KWARG = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}

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
            **EFFORT_KWARG,
        ),
        # THE RESEARCH LAYER IS WORTH MORE THAN THE MODEL. Metaculus runs the same forecasting
        # model behind several different research providers in this tournament, which isolates the
        # research variable exactly. Live peer score per question, all on deepseek-r1:
        #
        #   + exa-online       +1.27
        #   + asknews          -2.88
        #   + NO RESEARCH      -6.99
        #   + exa-answer       -8.11
        #   + sonar           -15.48   <- what this bot used
        #   + sonar-pro       -16.03
        #
        # Perplexity Sonar is 8.5 points per question WORSE THAN DOING NO RESEARCH, and 16.8
        # below the best option. Nothing else on the board - not the model, not the aggregation,
        # not the sample count - moves the score by anything like that much.
        #
        # `:online` is OpenRouter's Exa-backed web search plugin, which is the same search layer
        # behind the winning row. It bills per result on top of the model's own tokens.
        # Pinned to a cheap model deliberately, NOT to ENSEMBLE_MODELS[0]. Research is retrieval
        # and summarisation; the judgement happens in the forecast call. Running it on Sol at
        # $30/M output would roughly double the bill for the part of the pipeline where the
        # provider matters more than the model.
        "researcher": GeneralLlm(
            model="openrouter/google/gemini-3.5-flash:online",
            temperature=0.1,
            timeout=180,
            allowed_tries=2,
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
            **EFFORT_KWARG,
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
