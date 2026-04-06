"""
PostgreSQL LISTEN / NOTIFY listener.

When the scraper inserts a new row into ff_calendar_events it should fire:

    NOTIFY new_calendar_event, '<id>';

The trigger in migrations/001_sentiment.sql does this automatically.
This module listens on that channel and calls analyze_event() in the background.

Features
────────
- Reconnects automatically if the DB connection drops
- Cancels cleanly on FastAPI shutdown
- Non-blocking: notifications are handled as async tasks
"""
import asyncio
import logging
from typing import Optional

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

_listener_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Notification handler
# ---------------------------------------------------------------------------

async def _handle_notification(
    conn: asyncpg.Connection,
    pid:     int,
    channel: str,
    payload: str,
) -> None:
    """
    Called by asyncpg when a NOTIFY arrives on the channel.

    payload is expected to be the calendar_event_id as a plain integer string.
    Import is deferred inside the function to avoid circular imports at module load.
    """
    logger.debug("NOTIFY on '%s': payload=%r", channel, payload)

    # Lazy import to avoid circular dependency at module level
    from analyzer import analyze_event  # noqa: PLC0415

    try:
        calendar_event_id = int(payload.strip())
    except (ValueError, AttributeError):
        logger.error("Invalid NOTIFY payload (expected int): %r", payload)
        return

    # Fire as an independent task so the listener loop is never blocked
    asyncio.create_task(_safe_analyze(calendar_event_id))


async def _safe_analyze(calendar_event_id: int) -> None:
    """Wrapper that catches all exceptions so a failing analysis never kills the loop."""
    try:
        from analyzer import analyze_event  # noqa: PLC0415
        result = await analyze_event(calendar_event_id)
        if result:
            logger.info(
                "Auto-analyzed event %d → label=%s signal=%.4f",
                calendar_event_id,
                result["label"],
                result["xauusd_signal"],
            )
        else:
            logger.warning("analyze_event returned None for id=%d", calendar_event_id)
    except Exception as exc:
        logger.error(
            "Error auto-analyzing event %d: %s", calendar_event_id, exc, exc_info=True
        )


# ---------------------------------------------------------------------------
# Listener loop with reconnect
# ---------------------------------------------------------------------------

async def _listener_loop() -> None:
    """
    Main loop: connect → listen → wait → reconnect on failure.
    Exits cleanly when cancelled.
    """
    while True:
        conn: Optional[asyncpg.Connection] = None
        try:
            conn = await asyncpg.connect(settings.DATABASE_URL)
            await conn.add_listener(settings.PG_NOTIFY_CHANNEL, _handle_notification)
            logger.info(
                "Listening on PostgreSQL channel '%s'", settings.PG_NOTIFY_CHANNEL
            )

            # Idle loop – we just need to stay alive and let asyncpg dispatch notifs
            while not conn.is_closed():
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Listener task cancelled – shutting down gracefully.")
            break

        except Exception as exc:
            logger.error("Listener error: %s – reconnecting in 5 s", exc)
            await asyncio.sleep(5)

        finally:
            if conn and not conn.is_closed():
                try:
                    await conn.remove_listener(
                        settings.PG_NOTIFY_CHANNEL, _handle_notification
                    )
                    await conn.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Public API (called from FastAPI lifespan)
# ---------------------------------------------------------------------------

def start_listener() -> asyncio.Task:
    """Spawn the listener as an asyncio background task."""
    global _listener_task
    loop = asyncio.get_running_loop()
    _listener_task = loop.create_task(_listener_loop(), name="pg_notify_listener")
    return _listener_task


async def stop_listener() -> None:
    """Cancel and await the listener task."""
    global _listener_task
    if _listener_task and not _listener_task.done():
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
    _listener_task = None
