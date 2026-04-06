"""
FastAPI application – Forex Factory sentiment analyzer.

Startup
───────
  1. FinBERT model loaded into memory
  2. PostgreSQL NOTIFY listener started as background task

Endpoints
─────────
  GET  /health
       Returns model + device info; use as k8s liveness probe.

  POST /analyze/{calendar_event_id}
       Run the full 3-layer analysis for a single calendar row.
       Idempotent – safe to call multiple times (upserts result).

  POST /analyze/today
       Analyze all events for today's calendar date.
       Smart: skips events already analyzed unless a new actual value
       has landed (stale result). Recommended primary trigger.

  POST /analyze/batch
       Analyze a list of calendar_event_ids.
       Batches ≤ 10 return synchronously; larger batches are
       dispatched to a background task and return 202 immediately.

  POST /analyze/recent
       Analyze all unanalyzed events from today (primary) then the last
       N calendar days (fallback). Useful for backfilling after a restart.

  GET  /signal
       Current 4-hour aggregated XAUUSD signal.

  GET  /signal/{window_hours}
       Aggregated signal for a custom window (1–168 hours).

  GET  /signal/all
       All configured windows (1h, 4h, 24h) in one response.

  POST /webhook/new-event
       Lightweight webhook for scrapers: POST {"calendar_event_id": 42}
       to trigger async analysis. Faster than NOTIFY for external scrapers.
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analyzer import analyze_event, analyze_recent, analyze_today, get_aggregated_signal
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
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=== Sentiment Analyzer starting up ===")
    logger.info("Loading FinBERT model (%s)…", settings.FINBERT_MODEL)
    finbert.load()
    logger.info("FinBERT ready.")

    logger.info("Starting PostgreSQL NOTIFY listener…")
    start_listener()

    yield  # ← app is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("=== Sentiment Analyzer shutting down ===")
    await stop_listener()
    await close_pool()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title        = "FF Sentiment Analyzer",
    description  = "XAUUSD (Gold/USD) sentiment from Forex Factory economic calendar",
    version      = "1.0.0",
    lifespan     = lifespan,
    docs_url     = "/docs",
    redoc_url    = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    calendar_event_ids: List[int] = Field(
        ..., min_length=1, max_length=500, description="IDs from ff_calendar_events.id"
    )


class RecentRequest(BaseModel):
    hours: int = Field(default=24, ge=1, le=168, description="Look-back window in hours")


class WebhookPayload(BaseModel):
    calendar_event_id: int
    event_id:          Optional[str] = None


class SignalResponse(BaseModel):
    xauusd_signal:     float
    label:             str
    confidence:        float
    window_hours:      int
    event_count:       int
    high_impact_count: int
    top_contributors:  list
    computed_at:       str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    """Liveness / readiness probe."""
    return {
        "status":  "ok",
        "model":   settings.FINBERT_MODEL,
        "device":  settings.DEVICE,
        "version": "1.0.0",
    }


@app.post("/analyze/batch", tags=["analysis"])
async def analyze_batch_endpoint(
    payload:           BatchRequest,
    background_tasks:  BackgroundTasks,
):
    """
    Analyze multiple events.
    - ≤ 10 IDs   → synchronous, results returned immediately
    - > 10 IDs   → dispatched to background, returns 202 with count
    """
    ids = payload.calendar_event_ids

    if len(ids) > 10:
        async def _run_batch():
            for cid in ids:
                await analyze_event(cid)

        background_tasks.add_task(_run_batch)
        return {"status": "processing", "count": len(ids), "message": "Queued in background"}

    results = []
    for cid in ids:
        r = await analyze_event(cid)
        if r:
            results.append(r)

    return {"status": "done", "count": len(results), "results": results}


@app.post("/analyze/today", tags=["analysis"])
async def analyze_today_endpoint(background_tasks: BackgroundTasks):
    """
    Analyze all economic events scheduled for **today's calendar date**.

    Smart behavior:
    - Events with no sentiment result yet → analyzed immediately.
    - Events whose result is stale AND now have an actual value → re-analyzed
      (picks up post-release data).
    - Events already analyzed and not stale → skipped.

    This is the recommended endpoint to call on a schedule (e.g. every 30 min).
    """
    results = await analyze_today()
    return {
        "status":  "done",
        "date":    "today",
        "count":   len(results),
        "results": results,
    }


@app.post("/analyze/recent", tags=["analysis"])
async def analyze_recent_endpoint(payload: RecentRequest):
    """
    Analyze all unanalyzed events.

    Primary look-up: events with ``calendar_date = CURRENT_DATE`` (today).
    Fallback: events within the last N calendar days (derived from ``hours``).
    Last resort: events with no calendar_date, scraped within the last N hours.

    Use ``/analyze/today`` for smarter day-aware analysis (re-runs stale results).
    """
    results = await analyze_recent(payload.hours)
    return {"count": len(results), "hours": payload.hours, "results": results}


@app.post("/analyze/{calendar_event_id}", tags=["analysis"])
async def analyze_single(calendar_event_id: int):
    """
    Full 3-layer analysis for a single ff_calendar_events row.
    Returns the persisted sentiment result.
    """
    result = await analyze_event(calendar_event_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Event {calendar_event_id} not found in ff_calendar_enriched",
        )
    return result


@app.get("/signal", response_model=SignalResponse, tags=["signal"])
async def get_signal_default():
    """
    Current XAUUSD signal using a 4-hour look-back window.
    This is the main endpoint for the trading system.
    """
    signal = await get_aggregated_signal(4)
    return SignalResponse(
        xauusd_signal     = signal.xauusd_signal,
        label             = signal.label,
        confidence        = signal.confidence,
        window_hours      = signal.window_hours,
        event_count       = signal.event_count,
        high_impact_count = signal.high_impact_count,
        top_contributors  = signal.contributing_events,
        computed_at       = signal.computed_at.isoformat(),
    )


@app.get("/signal/all", tags=["signal"])
async def get_all_signals():
    """
    Returns signals for all configured windows (default: 1h, 4h, 24h) in one call.
    """
    out = {}
    for hours in settings.SIGNAL_WINDOWS_HOURS:
        sig = await get_aggregated_signal(hours)
        out[f"{hours}h"] = {
            "xauusd_signal":     sig.xauusd_signal,
            "label":             sig.label,
            "confidence":        sig.confidence,
            "event_count":       sig.event_count,
            "high_impact_count": sig.high_impact_count,
            "computed_at":       sig.computed_at.isoformat(),
        }
    return out


@app.get("/signal/{window_hours}", response_model=SignalResponse, tags=["signal"])
async def get_signal_window(window_hours: int):
    """
    XAUUSD signal for a custom look-back window (1–168 hours).
    """
    if not 1 <= window_hours <= 168:
        raise HTTPException(
            status_code=400, detail="window_hours must be between 1 and 168"
        )
    signal = await get_aggregated_signal(window_hours)
    return SignalResponse(
        xauusd_signal     = signal.xauusd_signal,
        label             = signal.label,
        confidence        = signal.confidence,
        window_hours      = signal.window_hours,
        event_count       = signal.event_count,
        high_impact_count = signal.high_impact_count,
        top_contributors  = signal.contributing_events,
        computed_at       = signal.computed_at.isoformat(),
    )


@app.post("/webhook/new-event", tags=["ingest"])
async def webhook_new_event(
    payload:          WebhookPayload,
    background_tasks: BackgroundTasks,
):
    """
    Webhook for external scrapers that can't use PostgreSQL NOTIFY.
    POST {"calendar_event_id": 42} to trigger async analysis.
    Returns immediately (analysis runs in the background).
    """
    background_tasks.add_task(analyze_event, payload.calendar_event_id)
    return {
        "status":             "queued",
        "calendar_event_id":  payload.calendar_event_id,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = settings.HOST,
        port    = settings.PORT,
        reload  = False,
        workers = 1,  # Keep at 1: FinBERT model is not fork-safe
    )
