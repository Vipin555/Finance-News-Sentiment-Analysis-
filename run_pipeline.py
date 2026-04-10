"""
Unified Pipeline — Forex Factory Scraper + Sentiment Analyzer
═══════════════════════════════════════════════════════════════
Run with ONE command:
    .venv\Scripts\python.exe run_pipeline.py

What it does (in order):
  1. SCRAPE  — Launches Playwright, scrapes ForexFactory calendar + event details,
               stores raw data into Supabase (ff_calendar_events + ff_event_details).
  2. ANALYZE — Loads FinBERT, runs 3-layer sentiment analysis on today's events,
               stores actionable BULLISH/BEARISH signals into ff_sentiment_results.

CLI flags:
  --skip-scrape       Skip the scraping phase (useful if data is already fresh)
  --skip-analysis     Skip the analysis phase (useful to only update raw data)
  --reset-db          Truncate scraper tables before re-scraping
  --force-details     Re-scrape event detail overlays even if cached
  --days-ahead N      How many days ahead to scrape (default: 5)
  --loop              Keep running in a loop (scrape → analyze → sleep → repeat)
  --interval N        Sleep N minutes between loop iterations (default: 30)
"""

import argparse
import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Scrape
# ─────────────────────────────────────────────────────────────────────────────

async def run_scrape(
    *,
    reset_db: bool = False,
    force_details: bool = False,
    days_ahead: int = 5,
    mode: str = "week",
) -> bool:
    """Run the ForexFactory scraper. Returns True on success."""
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   PHASE 1: SCRAPING FOREX FACTORY    ║")
    logger.info("╚══════════════════════════════════════╝")
    t0 = time.perf_counter()

    try:
        # Import here so FinBERT doesn't wait for Playwright install
        from full_forexfactory_scrape import scrape_calendar_and_details

        rows = await scrape_calendar_and_details(
            reset_db=reset_db,
            force_details=force_details,
            days_ahead=days_ahead,
            mode=mode,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "Scraping complete — %d rows stored in %.1fs", len(rows), elapsed
        )
        return True

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Scraping FAILED after %.1fs: %s", elapsed, exc, exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Analyze
# ─────────────────────────────────────────────────────────────────────────────

async def run_analysis() -> bool:
    """Run sentiment analysis on today's events. Returns True on success."""
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  PHASE 2: SENTIMENT ANALYSIS (NLP)   ║")
    logger.info("╚══════════════════════════════════════╝")
    t0 = time.perf_counter()

    try:
        from nlp_engine import finbert
        from analyzer import analyze_today

        # Load model only once (cached on subsequent calls)
        if not finbert._model:
            logger.info("Loading FinBERT model…")
            finbert.load()
            logger.info("FinBERT ready.")

        results = await analyze_today()
        elapsed = time.perf_counter() - t0
        stored = sum(
            1 for r in results
            if r.get("label") != "NEUTRAL"
        )
        logger.info(
            "Analysis complete — %d events processed, %d signals stored in %.1fs",
            len(results), stored, elapsed,
        )
        return True

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Analysis FAILED after %.1fs: %s", elapsed, exc, exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    *,
    skip_scrape: bool = False,
    skip_analysis: bool = False,
    reset_db: bool = False,
    force_details: bool = False,
    days_ahead: int = 5,
    mode: str = "week",
    loop: bool = False,
    interval_minutes: int = 30,
):
    """Full pipeline: Scrape → Analyze (optionally in a loop)."""

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Forex Sentiment Pipeline starting up")
    logger.info("  Scrape: %s  |  Analyze: %s  |  Loop: %s",
                "ON" if not skip_scrape else "SKIP",
                "ON" if not skip_analysis else "SKIP",
                f"every {interval_minutes}m" if loop else "OFF")
    logger.info("═══════════════════════════════════════════════════")

    iteration = 0
    while True:
        iteration += 1
        if loop:
            logger.info("── Iteration %d ──", iteration)

        # Phase 1
        if not skip_scrape:
            scrape_ok = await run_scrape(
                reset_db=(reset_db and iteration == 1),  # only reset on 1st run
                force_details=force_details,
                days_ahead=days_ahead,
                mode=mode,
            )
            if not scrape_ok:
                logger.warning("Scrape failed — analysis will use stale data.")

        # Phase 2
        if not skip_analysis:
            await run_analysis()

        if not loop:
            break

        logger.info("Sleeping for %d minutes...", interval_minutes)
        await asyncio.sleep(interval_minutes * 60)

    # Cleanup
    try:
        from database import close_pool
        await close_pool()
    except Exception:
        pass

    logger.info("Pipeline finished.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified Forex Factory Scraper + Sentiment Analysis Pipeline"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip the scraping phase (only run analysis)",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip the analysis phase (only run scraping)",
    )
    parser.add_argument(
        "--reset-db", action="store_true",
        help="Truncate scraper tables before re-scraping",
    )
    parser.add_argument(
        "--force-details", action="store_true",
        help="Re-scrape event detail overlays even if cached",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=5,
        help="Extract events from today through N days ahead (default: 5)",
    )
    parser.add_argument(
        "--mode", choices=["week", "day"], default="week",
        help="Scraper calendar mode (default: week)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Keep running in a loop (scrape → analyze → sleep → repeat)",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Minutes between loop iterations (default: 30)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run_pipeline(
                skip_scrape=args.skip_scrape,
                skip_analysis=args.skip_analysis,
                reset_db=args.reset_db,
                force_details=args.force_details,
                days_ahead=args.days_ahead,
                mode=args.mode,
                loop=args.loop,
                interval_minutes=args.interval,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
