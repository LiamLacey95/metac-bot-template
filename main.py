"""Calibrated forecasting bot for the Metaculus AI Benchmark.

Built on Metaculus's own template. The prompts and research path are left close to the template
on purpose: the change being tested is the aggregation step, and changing several things at once
would make the result unreadable.

What is different from the template:

  The template samples N forecasts per question and takes the MEDIAN of the probabilities.
  This takes the mean in LOGIT space and then shrinks it toward a base rate, by an amount that
  depends on how much the samples disagreed with each other.

Why. The tournament is scored with a log score, which punishes confident mistakes much harder
than it rewards confident hits. Language models are documented as overconfident - they say 5%
and 95% more often than the world obliges - so the median of five overconfident samples is still
overconfident. But the ensemble is not uniformly untrustworthy: when five independent runs agree
closely the aggregate carries real information, and when they scatter from 0.15 to 0.90 the
median of 0.60 is a fiction wearing a confident face. Spread is the signal that separates those
two cases, it is free (the samples already exist), and the template discards it.

See calibration.py for the mechanics and test_calibration.py for the evidence, including the
regime where this LOSES.
"""

import argparse
import asyncio
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
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
)

from calibration import CalibrationConfig, aggregate  # noqa: E402
from template_bot import SummerTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class CalibratedBot(SummerTemplateBot2026):
    """The template bot with the aggregation step replaced."""

    calibration = CalibrationConfig()

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        # Only binary questions are handled here. Numeric and multiple-choice aggregation is a
        # different problem with a different fix, and pretending one calibration story covers all
        # three is how you ship a regression you cannot attribute.
        if not isinstance(question, BinaryQuestion):
            return await super()._aggregate_predictions(predictions, question)

        samples = [float(p) for p in predictions if isinstance(p, (int, float))]
        if len(samples) != len(predictions):
            logger.warning(
                "Binary question %s produced non-float predictions; falling back to the "
                "framework aggregator.",
                question.page_url,
            )
            return await super()._aggregate_predictions(predictions, question)

        calibrated = aggregate(samples, self.calibration)
        baseline = await super()._aggregate_predictions(predictions, question)

        logger.info(
            "Calibrated %s: samples=%s framework=%.3f calibrated=%.3f",
            question.page_url,
            [round(s, 3) for s in samples],
            float(baseline),  # type: ignore[arg-type]
            calibrated,
        )
        return calibrated  # type: ignore[return-value]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the calibrated forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    bot = CalibratedBot(
        research_reports_per_question=1,
        # More samples than the template's 5: the whole method reads the spread between them,
        # and a spread estimated from 5 points is itself noisy.
        predictions_per_research_report=7,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
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
