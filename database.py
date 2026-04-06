"""
Async PostgreSQL helpers using asyncpg.

All queries target the existing schema plus the new ff_sentiment_results table
created by migrations/001_sentiment.sql and the calendar_date columns from
002_calendar_date.sql.

Key design decisions after the schema update
─────────────────────────────────────────────
• Filtering is now done on ``calendar_date`` (the actual event date) rather
  than ``scraped_at`` (when the scraper ran).  This ensures that events are
  processed relative to *when they occur*, not when they were ingested.
• ``fetch_sentiment_window`` exposes both ``calendar_date`` and the raw
  ``time`` text so the signal aggregator can reconstruct the event's scheduled
  datetime for precise exponential time-decay.
• Two new helpers support the /analyze/today endpoint:
    - fetch_todays_events         – all today's events (analyzed or not)
    - fetch_events_by_date_range  – multi-day backfill
"""
import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database connection pool created.")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed.")


# ── Read helpers ───────────────────────────────────────────────────────────────

async def fetch_enriched_event_by_calendar_id(
    calendar_id: int,
) -> Optional[Dict[str, Any]]:
    """Fetch one row from ff_calendar_enriched by the calendar primary key."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ff_calendar_enriched WHERE id = $1",
            calendar_id,
        )
        return dict(row) if row else None


async def fetch_enriched_events_by_event_id(
    event_id: str,
) -> List[Dict[str, Any]]:
    """
    All calendar rows for an event_id (same recurring event across scrape runs).
    Ordered newest first so the caller can take the latest reading.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM ff_calendar_enriched
            WHERE event_id = $1
            ORDER BY scraped_at DESC
            """,
            event_id,
        )
        return [dict(r) for r in rows]


async def fetch_recent_enriched_events(hours: int = 24) -> List[Dict[str, Any]]:
    """
    Unanalyzed enriched events, prioritising *today's* calendar events.

    Strategy (in order):
      1. Return every unanalyzed event where calendar_date = CURRENT_DATE.
      2. If calendar_date is NULL for all rows (old data), fall back to a
         date-range of the last N calendar days (derived from ``hours``).

    The ``hours`` parameter is retained for API compatibility; it now acts as
    a fallback window expressed in calendar days (hours ÷ 24, rounded up).

    Only currencies that drive XAUUSD are included.
    """
    pool = await get_pool()
    fallback_days = max(1, (hours + 23) // 24)   # ceil(hours / 24)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.*
            FROM ff_calendar_enriched e
            WHERE NOT EXISTS (
                SELECT 1 FROM ff_sentiment_results sr
                WHERE sr.calendar_event_id = e.id
            )
              AND (
                    e.currency IN ('USD','EUR','GBP','JPY','CHF','AUD','CAD','NZD','CNY','All')
                    OR e.currency IS NULL
              )
              AND (
                    -- Prefer events with an explicit calendar_date of today
                    e.calendar_date = CURRENT_DATE
                    -- Fall back: calendar_date within the last N days
                    OR (
                        e.calendar_date IS NOT NULL
                        AND e.calendar_date >= CURRENT_DATE - ($1::int || ' days')::INTERVAL
                        AND e.calendar_date <= CURRENT_DATE
                    )
                    -- Last resort: calendar_date is NULL, use scraped_at window
                    OR (
                        e.calendar_date IS NULL
                        AND e.scraped_at >= NOW() - ($2::text || ' hours')::INTERVAL
                    )
              )
            ORDER BY
                -- Today first, then by time text so the earliest event of the day runs first
                e.calendar_date DESC NULLS LAST,
                e."time" ASC NULLS LAST
            """,
            fallback_days,
            str(hours),
        )
        return [dict(r) for r in rows]


async def fetch_todays_events() -> List[Dict[str, Any]]:
    """
    All enriched events for today's calendar date (analyzed or not).

    Used by the /analyze/today endpoint to decide which events need a fresh
    sentiment pass (new events or events with stale results after actual release).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                e.*,
                sr.computed_at   AS sentiment_computed_at,
                sr.label         AS existing_label,
                sr.xauusd_signal AS existing_signal
            FROM ff_calendar_enriched e
            LEFT JOIN ff_sentiment_results sr ON sr.calendar_event_id = e.id
            WHERE e.calendar_date = CURRENT_DATE
              AND (
                    -- Include standard forex currencies + global events ('All' = OPEC, Fed, IMF etc.)
                    e.currency IN ('USD','EUR','GBP','JPY','CHF','AUD','CAD','NZD','CNY','All')
                    OR e.currency IS NULL
              )
            ORDER BY e."time" ASC NULLS LAST
            """,
        )
        return [dict(r) for r in rows]


async def fetch_events_by_date_range(
    date_from: str,
    date_to: str,
) -> List[Dict[str, Any]]:
    """
    All unanalyzed enriched events between two calendar dates (inclusive).
    Dates should be ISO strings: 'YYYY-MM-DD'.

    Useful for backfill after a restart or when the scraper delivers late data.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.*
            FROM ff_calendar_enriched e
            WHERE NOT EXISTS (
                SELECT 1 FROM ff_sentiment_results sr
                WHERE sr.calendar_event_id = e.id
            )
              AND e.calendar_date BETWEEN $1::date AND $2::date
              AND (
                    e.currency IN ('USD','EUR','GBP','JPY','CHF','AUD','CAD','NZD','CNY','All')
                    OR e.currency IS NULL
              )
            ORDER BY e.calendar_date ASC, e."time" ASC NULLS LAST
            """,
            date_from,
            date_to,
        )
        return [dict(r) for r in rows]


async def fetch_sentiment_window(hours: int) -> List[Dict[str, Any]]:
    """
    Fetch persisted sentiment results for the current time window.

    Strategy:
      • Primary filter: events whose calendar_date is today AND whose
        computed_at is within the last N hours (re-analysis of today's events).
      • Secondary: if calendar_date is NULL, fall back to computed_at filter
        so older data is still returned.

    Exposes calendar_date and the raw ``time`` text alongside computed_at so
    the signal aggregator can build the event's scheduled datetime for precise
    exponential time-decay (instead of using scraped_at).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                sr.*,
                ce.scraped_at,
                ce.calendar_date,
                ce."time"           AS event_time_text,
                ce.calendar_date_label
            FROM ff_sentiment_results  sr
            JOIN ff_calendar_events    ce ON sr.calendar_event_id = ce.id
            WHERE (
                -- Today's events that have been computed recently
                (ce.calendar_date = CURRENT_DATE
                 AND sr.computed_at >= NOW() - ($1::text || ' hours')::INTERVAL)
                -- Or any event without a calendar_date (legacy data) computed recently
                OR (ce.calendar_date IS NULL
                    AND sr.computed_at >= NOW() - ($1::text || ' hours')::INTERVAL)
            )
            ORDER BY sr.computed_at DESC
            """,
            str(hours),
        )
        return [dict(r) for r in rows]


# ── Write helpers ──────────────────────────────────────────────────────────────

async def upsert_sentiment_result(result: Dict[str, Any]) -> None:
    """
    Insert a sentiment result, or update it if it already exists for the
    same calendar_event_id (unique constraint).
    """
    pool = await get_pool()

    # reasoning is stored as JSONB; ensure it's a string
    reasoning = result.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning = json.dumps(reasoning)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ff_sentiment_results (
                calendar_event_id, event_id, currency, impact, event_name,
                deviation_score, deviation_confidence,
                nlp_positive, nlp_negative, nlp_neutral, nlp_score,
                composite_score, xauusd_signal,
                label, confidence, reasoning
            ) VALUES (
                $1,  $2,  $3,  $4,  $5,
                $6,  $7,
                $8,  $9,  $10, $11,
                $12, $13,
                $14, $15, $16::jsonb
            )
            ON CONFLICT (calendar_event_id) DO UPDATE SET
                deviation_score      = EXCLUDED.deviation_score,
                deviation_confidence = EXCLUDED.deviation_confidence,
                nlp_positive         = EXCLUDED.nlp_positive,
                nlp_negative         = EXCLUDED.nlp_negative,
                nlp_neutral          = EXCLUDED.nlp_neutral,
                nlp_score            = EXCLUDED.nlp_score,
                composite_score      = EXCLUDED.composite_score,
                xauusd_signal        = EXCLUDED.xauusd_signal,
                label                = EXCLUDED.label,
                confidence           = EXCLUDED.confidence,
                reasoning            = EXCLUDED.reasoning::jsonb,
                computed_at          = NOW()
            """,
            result["calendar_event_id"],
            result.get("event_id"),
            result.get("currency"),
            result.get("impact"),
            result.get("event_name"),
            result["deviation_score"],
            result["deviation_confidence"],
            result["nlp_positive"],
            result["nlp_negative"],
            result["nlp_neutral"],
            result["nlp_score"],
            result["composite_score"],
            result["xauusd_signal"],
            result["label"],
            result["confidence"],
            reasoning,
        )
