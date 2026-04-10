"""
Analyzer orchestrator script – Forex Factory sentiment analyzer.

Startup
───────
  1. FinBERT model loaded into memory
  2. PostgreSQL NOTIFY listener started as background task
  3. Periodically re-analyzes today's events and stores only actionable signals
     (BULLISH or BEARISH with confidence >= 0.5) to PostgreSQL.
"""
import asyncio
import logging

from analyzer import analyze_today
from config import settings
from database import close_pool
from listener import start_listener, stop_listener
from nlp_engine import finbert

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main_loop():
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=== Sentiment Analyzer starting up ===")
    logger.info("Loading FinBERT model (%s)…", settings.FINBERT_MODEL)
    finbert.load()
    logger.info("FinBERT ready.")

    logger.info("Starting PostgreSQL NOTIFY listener…")
    start_listener()

    try:
        while True:
            # Run smart analysis – only stores BULLISH/BEARISH with confidence >= 0.5
            logger.info("Running scheduled analyze_today()...")
            await analyze_today()

            logger.info("Sleeping for 5 minutes...")
            await asyncio.sleep(300)

    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("An error occurred in main loop: %s", exc, exc_info=True)
    finally:
        # ── Shutdown ──────────────────────────────────────────────────────────────
        logger.info("=== Sentiment Analyzer shutting down ===")
        await stop_listener()
        await close_pool()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
