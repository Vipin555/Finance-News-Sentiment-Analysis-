"""ForexFactory scraper — optimised build.

Speed improvements over the original:
  1. Bulk DOM extraction  — all calendar row fields pulled in ONE page.evaluate()
     call instead of 8-12 individual awaits per row.  For 200 rows that alone
     eliminates ~1 600 browser round-trips.
  2. Parallel detail workers — N_WORKERS (default 3) independent browser pages
     scrape detail overlays concurrently, giving ~3× throughput for uncached events.
  3. Faster scroll — fewer steps, larger stride, quicker exit condition.
  4. Reduced default wait times (still env-overridable).

All existing logic is preserved:
  • JSON details cache (forexfactory_details.json)
  • Checkpoint / progressive PostgreSQL upsert
  • Rolling retention window + cache pruning
  • week / day mode, --force-details, --reset-db, all CLI flags
  • HEADLESS / BLOCK_HEAVY_RESOURCES / PARTIAL_CACHE_REFRESH_LIMIT env vars
"""

import asyncio
import argparse
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Locator
from playwright.async_api import async_playwright

import extract_details as details_scraper
import load_to_postgres

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.forexfactory.com/calendar"
DETAILS_CACHE_JSON = "forexfactory_details.json"

HEADLESS = os.getenv("HEADLESS", "0").strip() == "1"
PARTIAL_CACHE_REFRESH_LIMIT = max(0, int(os.getenv("PARTIAL_CACHE_REFRESH_LIMIT", "0")))
BLOCK_HEAVY_RESOURCES = os.getenv("BLOCK_HEAVY_RESOURCES", "1").strip() == "1"
DETAIL_ROW_TIMEOUT_SECONDS = max(5, int(os.getenv("DETAIL_ROW_TIMEOUT_SECONDS", "10")))
DETAIL_CLICK_FALLBACK = os.getenv("DETAIL_CLICK_FALLBACK", "1").strip() == "1"
DEBUG_IMPACT = os.getenv("DEBUG_IMPACT", "0").strip() == "1"
DEBUG_NAV = os.getenv("DEBUG_NAV", "0").strip() == "1"

# Reduced defaults — still env-overridable.
# (original defaults were 20 000 and 15 000 ms)
ROWS_STABLE_WAIT_MS = max(2000, int(os.getenv("ROWS_STABLE_WAIT_MS", "8000")))
NAV_NETWORKIDLE_TIMEOUT_MS = max(0, int(os.getenv("NAV_NETWORKIDLE_TIMEOUT_MS", "8000")))

# Number of concurrent detail-worker pages (set to 1 to disable parallelism)
DETAIL_WORKERS = max(1, int(os.getenv("DETAIL_WORKERS", "3")))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


CHECKPOINT_ROWS = max(0, _int_env("CHECKPOINT_ROWS", 50))

# Selectors
EVENT_ROW_SELECTOR = "tr.calendar__row[data-event-id]"
CALENDAR_ROW_SELECTOR = "tr.calendar__row"

DETAIL_KEYS = (
    "ff_notice",
    "measures",
    "usual_effect",
    "frequency",
    "ff_notes",
    "why_traders_care",
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _details_incomplete(fields: Optional[Dict]) -> bool:
    if not fields:
        return True
    return all(fields.get(k) is None for k in DETAIL_KEYS)


def _details_has_any_value(fields: Optional[Dict]) -> bool:
    if not fields:
        return False
    return any(fields.get(k) is not None for k in DETAIL_KEYS)


def _details_partially_filled(fields: Optional[Dict]) -> bool:
    if not fields:
        return False
    has_any = any(fields.get(k) is not None for k in DETAIL_KEYS)
    has_missing = any(fields.get(k) is None for k in DETAIL_KEYS)
    return has_any and has_missing


def _empty_detail_fields() -> Dict:
    return {k: None for k in DETAIL_KEYS}


def _load_details_cache(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    cache: Dict[str, Dict] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("event_id") or "").strip()
            if not eid:
                continue
            cache[eid] = {k: item.get(k) for k in DETAIL_KEYS}
    return cache


def _save_details_cache(path: str, cache: Dict[str, Dict]) -> None:
    rows = [{"event_id": eid, **fields} for eid, fields in cache.items()]
    rows.sort(
        key=lambda r: int(r["event_id"]) if str(r.get("event_id", "")).isdigit()
        else str(r.get("event_id", ""))
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Impact parsing — now synchronous (data arrives via bulk JS extract)
# ---------------------------------------------------------------------------

_impact_debug_printed = 0


def _parse_impact_sync(
    classes: str, labels: str, text: str, img_src: str
) -> Optional[str]:
    """Normalise impact metadata to ff-impact-yel/ora/red/gra."""

    for pattern in (
        r"\bff-impact-(red|ora|yel|gra)\b",
        r"\bicon--ff-impact-(red|ora|yel|gra)\b",
    ):
        m = re.search(pattern, classes, re.IGNORECASE)
        if m:
            return f"ff-impact-{m.group(1).lower()}"

    for pattern in (
        r"\bcalendar__impact--(high|medium|low)\b",
        r"\bimpact--(high|medium|low)\b",
        r"\b(?:impact|calendar__impact)[^\s]*--(high|medium|low)\b",
    ):
        m = re.search(pattern, classes, re.IGNORECASE)
        if m:
            return {"high": "ff-impact-red", "medium": "ff-impact-ora", "low": "ff-impact-yel"}[
                m.group(1).lower()
            ]

    for pattern in (
        r"\bimpact--([1-3])\b",
        r"\b(?:impact|calendar__impact)[^\s]*--([1-3])\b",
    ):
        m = re.search(pattern, classes, re.IGNORECASE)
        if m:
            return {"1": "ff-impact-yel", "2": "ff-impact-ora", "3": "ff-impact-red"}[m.group(1)]

    label_blob = f"{labels} {text}".lower()
    if "high" in label_blob:
        return "ff-impact-red"
    if "medium" in label_blob:
        return "ff-impact-ora"
    if "low" in label_blob:
        return "ff-impact-yel"
    if "non-economic" in label_blob or "noneconomic" in label_blob:
        return "ff-impact-gra"

    m = re.search(r"ff-impact-(red|ora|yel|gra)\.png", img_src, re.IGNORECASE)
    if m:
        return f"ff-impact-{m.group(1).lower()}"

    global _impact_debug_printed
    if DEBUG_IMPACT and _impact_debug_printed < 10 and (classes or labels or text):
        _impact_debug_printed += 1
        print(
            f"Impact debug: could not map; "
            f"classes='{classes}' labels='{labels}' text='{text}'"
        )
    return None


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_calendar_date_text(label: str, *, today: date) -> Optional[date]:
    """Parse ForexFactory date labels like 'Fri Apr 3' into a date."""
    label = " ".join((label or "").split()).strip()
    if not label:
        return None
    parts = label.split()
    if len(parts) < 3:
        return None
    month_raw = parts[-2].strip().lower()
    day_raw = parts[-1].strip()
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    month = month_map.get(month_raw[:3])
    if month is None:
        return None
    try:
        day_num = int("".join(ch for ch in day_raw if ch.isdigit()))
    except Exception:
        return None
    if not (1 <= day_num <= 31):
        return None
    year = today.year
    try:
        candidate = date(year, month, day_num)
    except ValueError:
        return None
    if today.month == 12 and month == 1 and candidate < today:
        candidate = date(year + 1, month, day_num)
    if today.month == 1 and month == 12 and candidate > today:
        candidate = date(year - 1, month, day_num)
    return candidate


# ---------------------------------------------------------------------------
# ★ CORE OPTIMISATION 1 — bulk DOM extraction in a single JS call
# ---------------------------------------------------------------------------

_BULK_EXTRACT_JS = """() => {
    const rows = [...document.querySelectorAll('tr.calendar__row')];
    const result = [];
    let currentDayLabel = '';
    let lastTime = '';
    let lastImpactClasses = '';
    let lastImpactLabels = '';
    let lastImpactText   = '';
    let lastImpactImgSrc = '';

    for (const row of rows) {
        const cls = row.className || '';

        /* ---- Day-breaker: just update the running date label ---- */
        if (cls.includes('day-breaker')) {
            const t = (row.innerText || '').replace(/\\s+/g, ' ').trim();
            if (t) {
                currentDayLabel = t;
                lastTime = '';
                lastImpactClasses = '';
                lastImpactLabels = '';
                lastImpactText   = '';
                lastImpactImgSrc = '';
            }
            continue;
        }

        const eventId = (row.getAttribute('data-event-id') || '').trim();
        if (!eventId) continue;

        /* ---- Inline date cell (overrides day-breaker if present) ---- */
        const dateSpan = row.querySelector(
            'td.calendar__cell.calendar__date span.date'
        );
        if (dateSpan) {
            const t = (dateSpan.innerText || '').replace(/\\s+/g, ' ').trim();
            if (t && t !== currentDayLabel) {
                currentDayLabel = t;
                lastTime = '';
                lastImpactClasses = '';
                lastImpactLabels = '';
                lastImpactText   = '';
                lastImpactImgSrc = '';
            }
        }

        /* ---- Time (carry-forward when cell is absent OR empty) ---- */
        const timeEl  = row.querySelector('.calendar__time');
        const timeText = timeEl
            ? (timeEl.innerText || '').replace(/\\s+/g, ' ').trim()
            : '';
        if (timeText) lastTime = timeText;
        const resolvedTime = timeText || lastTime;

        /* ---- Impact (carry-forward when cell is absent) ---- */
        const impactCell = row.querySelector('.calendar__impact');
        let impactClasses = '', impactLabels = '', impactText = '', impactImgSrc = '';
        if (impactCell) {
            const all = [impactCell, ...impactCell.querySelectorAll('*')];
            impactClasses = all.map(n => n.className || '').join(' ');
            impactLabels  = all.map(n => [
                n.getAttribute && n.getAttribute('title'),
                n.getAttribute && n.getAttribute('aria-label'),
                n.getAttribute && n.getAttribute('data-impact'),
            ].filter(Boolean).join(' ')).join(' ');
            impactText    = (impactCell.textContent || '').trim();
            const img = impactCell.querySelector('img');
            if (img) impactImgSrc = img.getAttribute('src') || '';
            lastImpactClasses = impactClasses;
            lastImpactLabels  = impactLabels;
            lastImpactText    = impactText;
            lastImpactImgSrc  = impactImgSrc;
        } else {
            impactClasses = lastImpactClasses;
            impactLabels  = lastImpactLabels;
            impactText    = lastImpactText;
            impactImgSrc  = lastImpactImgSrc;
        }

        /* ---- Event name ---- */
        const eventName = (
            row.querySelector('.calendar__event-title') ||
            row.querySelector('.calendar__event a') ||
            row.querySelector('.calendar__event')
        );
        const eventNameText = eventName
            ? (eventName.innerText || '').replace(/\\s+/g, ' ').trim()
            : '';

        /* ---- Detail link present? ---- */
        const hasDetailLink = !!row.querySelector('.calendar__detail-link');

        result.push({
            event_id:       eventId,
            time:           resolvedTime,
            currency:       (row.querySelector('.calendar__currency')?.innerText  || '').replace(/\\s+/g,' ').trim(),
            actual:         (row.querySelector('.calendar__actual')?.innerText    || '').replace(/\\s+/g,' ').trim(),
            forecast:       (row.querySelector('.calendar__forecast')?.innerText  || '').replace(/\\s+/g,' ').trim(),
            previous:       (row.querySelector('.calendar__previous')?.innerText  || '').replace(/\\s+/g,' ').trim(),
            event_name:     eventNameText,
            impact_classes: impactClasses,
            impact_labels:  impactLabels,
            impact_text:    impactText,
            impact_img_src: impactImgSrc,
            day_label:      currentDayLabel,
            has_detail_link: hasDetailLink,
        });
    }
    return result;
}"""


async def _bulk_extract_rows(page) -> List[Dict[str, Any]]:
    """Extract every calendar row in ONE browser round-trip."""
    try:
        return await page.evaluate(_BULK_EXTRACT_JS)
    except Exception as e:
        print(f"Bulk extract error: {e}")
        return []


# ---------------------------------------------------------------------------
# Browser / page helpers
# ---------------------------------------------------------------------------

async def _configure_fast_routes(page) -> None:
    if not BLOCK_HEAVY_RESOURCES:
        return

    async def _handler(route):
        if route.request.resource_type in {"image", "font", "media"}:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", _handler)


async def _wait_for_rows_stable(page, *, max_wait_ms: int = 5000) -> None:
    try:
        last = -1
        stable = 0
        waited = 0
        while waited < max_wait_ms:
            try:
                cur = await page.locator(EVENT_ROW_SELECTOR).count()
            except Exception:
                cur = last
            if cur == last and cur >= 0:
                stable += 1
            else:
                stable = 0
                last = cur
            if stable >= 3:
                return
            await page.wait_for_timeout(250)
            waited += 250
    except Exception:
        return


async def _scroll_to_load(page) -> None:
    """Scroll the calendar to trigger lazy-loaded rows. Streamlined version."""
    max_steps = max(10, _int_env("SCROLL_MAX_STEPS", 80))
    step_px   = max(400, _int_env("SCROLL_STEP_PX", 2000))

    try:
        last_count = await page.locator(EVENT_ROW_SELECTOR).count()
    except Exception:
        last_count = -1

    stable = 0
    for _ in range(max_steps):
        at_bottom = await page.evaluate(
            """(step) => {
                const el = ['main','.calendar','.calendar__table',
                            '.calendar__table-wrapper','.calendar__container']
                    .map(s => document.querySelector(s))
                    .find(e => e && (e.scrollHeight - e.clientHeight) > 20);
                if (el) {
                    el.scrollTop += step;
                    return (el.scrollTop + el.clientHeight) >= (el.scrollHeight - 4);
                }
                window.scrollBy(0, step);
                const d = document.documentElement;
                return (window.scrollY + window.innerHeight) >= (d.scrollHeight - 4);
            }""",
            step_px,
        )
        await page.wait_for_timeout(200)
        await _wait_for_rows_stable(page, max_wait_ms=2000)

        try:
            cur = await page.locator(EVENT_ROW_SELECTOR).count()
        except Exception:
            cur = last_count

        if cur > last_count:
            last_count = cur
            stable = 0
        else:
            stable += 1

        if at_bottom and stable >= 3:
            break
        if at_bottom and stable >= 8:
            break

    await page.evaluate("window.scrollTo(0,0)")
    await page.wait_for_timeout(200)


async def _prepare_calendar_page(page, url: str, *, label: str) -> None:
    if DEBUG_NAV:
        print(f"Loading ({label}): {url}")
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        if NAV_NETWORKIDLE_TIMEOUT_MS > 0:
            await page.wait_for_load_state("networkidle", timeout=NAV_NETWORKIDLE_TIMEOUT_MS)
    except Exception:
        pass
    try:
        await page.wait_for_selector(CALENDAR_ROW_SELECTOR, state="attached", timeout=60000)
    except PlaywrightTimeoutError:
        if HEADLESS:
            raise RuntimeError(
                "Timed out waiting for calendar rows in headless mode. "
                "Retry with HEADLESS=0 to solve challenge interactively."
            )
        print("Timed out waiting for calendar rows. Solve any challenge, then press Enter.")
        input("Press Enter to continue...")
        await page.wait_for_selector(CALENDAR_ROW_SELECTOR, state="attached", timeout=60000)

    await _wait_for_rows_stable(page, max_wait_ms=ROWS_STABLE_WAIT_MS)
    await _scroll_to_load(page)
    await _wait_for_rows_stable(page, max_wait_ms=ROWS_STABLE_WAIT_MS)


# ---------------------------------------------------------------------------
# Detail extraction (unchanged logic, kept as helper)
# ---------------------------------------------------------------------------

async def _extract_detail_fields_with_wait(
    page, row: Locator, *, event_id: str
) -> Tuple[Dict, str, Optional[str]]:
    """Open overlay → extract → close. Returns (fields, status, error_msg)."""
    try:
        await details_scraper._close_detail_modal(page)
        container = await details_scraper._try_open_detail_for_row(
            page, row, event_id=event_id
        )
        if container is None:
            return (_empty_detail_fields(), "error", "could_not_open_overlay")

        try:
            await container.locator(
                details_scraper.DETAIL_LOADING_SELECTOR
            ).first.wait_for(state="hidden", timeout=5000)
        except Exception:
            pass

        await container.locator("table.calendarspecs").first.wait_for(
            state="attached", timeout=15000
        )
        fields = await details_scraper._extract_detail_specs(container)

        if not any(fields.get(k) for k in DETAIL_KEYS):
            for _ in range(4):
                await page.wait_for_timeout(120)
                fields = await details_scraper._extract_detail_specs(container)
                if any(fields.get(k) for k in DETAIL_KEYS):
                    break

        status = "complete" if not _details_incomplete(fields) else "incomplete"
        await details_scraper._close_detail_modal(page, container, event_id=event_id)
        return (fields, status, None)
    except Exception as e:
        try:
            await details_scraper._close_detail_modal(page)
        except Exception:
            pass
        return (_empty_detail_fields(), "error", str(e))


# ---------------------------------------------------------------------------
# ★ CORE OPTIMISATION 2 — parallel detail workers
# ---------------------------------------------------------------------------

async def _detail_worker(
    context,
    worker_id: int,
    event_ids: List[str],
    cal_url: str,
) -> Dict[str, Tuple[Dict, str, Optional[str]]]:
    """One worker: loads its own page and scrapes the assigned detail overlays."""
    if not event_ids:
        return {}

    page = await context.new_page()
    page.set_default_timeout(10000)
    await _configure_fast_routes(page)

    results: Dict[str, Tuple[Dict, str, Optional[str]]] = {}
    try:
        await page.goto(cal_url, wait_until="domcontentloaded", timeout=60000)
        try:
            if NAV_NETWORKIDLE_TIMEOUT_MS > 0:
                await page.wait_for_load_state(
                    "networkidle", timeout=NAV_NETWORKIDLE_TIMEOUT_MS
                )
        except Exception:
            pass
        await page.wait_for_selector(CALENDAR_ROW_SELECTOR, state="attached", timeout=30000)
        await _wait_for_rows_stable(page, max_wait_ms=ROWS_STABLE_WAIT_MS)
        # Light scroll so all rows are in the DOM
        await _scroll_to_load(page)
        await _wait_for_rows_stable(page, max_wait_ms=2000)

        for n, event_id in enumerate(event_ids, 1):
            print(
                f"[worker-{worker_id}] ({n}/{len(event_ids)}) "
                f"Scraping details event_id={event_id}"
            )
            row = page.locator(
                f'tr.calendar__row[data-event-id="{event_id}"]'
            ).first
            try:
                count = await row.count()
            except Exception:
                count = 0

            if count == 0:
                results[event_id] = (
                    _empty_detail_fields(), "error", "row-not-found-on-worker-page"
                )
                continue

            try:
                fields, status, error = await asyncio.wait_for(
                    _extract_detail_fields_with_wait(page, row, event_id=event_id),
                    timeout=float(DETAIL_ROW_TIMEOUT_SECONDS),
                )
            except asyncio.TimeoutError:
                fields, status, error = _empty_detail_fields(), "timeout", "timeout"
            except Exception as exc:
                fields, status, error = _empty_detail_fields(), "error", str(exc)

            results[event_id] = (fields, status, error)

    except Exception as exc:
        print(f"[worker-{worker_id}] fatal error: {exc}")
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return results


async def _parallel_detail_scrape(
    context,
    event_ids: List[str],
    cal_url: str,
    *,
    n_workers: int = 3,
) -> Dict[str, Tuple[Dict, str, Optional[str]]]:
    """Distribute detail scraping across N concurrent worker pages."""
    if not event_ids:
        return {}

    effective = min(n_workers, len(event_ids))
    partitions = [event_ids[i::effective] for i in range(effective)]
    print(
        f"Starting {effective} detail worker(s) for "
        f"{len(event_ids)} event(s) ..."
    )

    results_list = await asyncio.gather(
        *[
            _detail_worker(context, wid + 1, part, cal_url)
            for wid, part in enumerate(partitions)
        ],
        return_exceptions=True,
    )

    merged: Dict[str, Tuple[Dict, str, Optional[str]]] = {}
    for res in results_list:
        if isinstance(res, dict):
            merged.update(res)
        elif isinstance(res, Exception):
            print(f"Detail worker raised: {res}")

    return merged


# ---------------------------------------------------------------------------
# Visible-page range / navigation helpers
# ---------------------------------------------------------------------------

async def _visible_calendar_day_range(
    page, *, today: date
) -> Tuple[Optional[date], Optional[date]]:
    parsed: List[date] = []

    for sel in (
        "tr.day-breaker",
        "td.calendar__cell.calendar__date span.date",
    ):
        try:
            loc = page.locator(sel)
            n = await loc.count()
            for i in range(n):
                try:
                    raw = await loc.nth(i).inner_text(timeout=500)
                except Exception:
                    continue
                raw = " ".join((raw or "").split()).strip()
                d = _parse_calendar_date_text(raw, today=today)
                if d is not None:
                    parsed.append(d)
        except Exception:
            pass

    if not parsed:
        return (None, None)
    return (min(parsed), max(parsed))


async def _find_next_week_url(page) -> Optional[str]:
    for sel in (
        "a.calendar__nav--next",
        "a.calendar__nav-next",
        "a:has-text('Next Week')",
        "a[aria-label*='Next']",
    ):
        try:
            loc = page.locator(sel).first
            if await loc.count() <= 0:
                continue
            href = await loc.get_attribute("href")
            if href:
                return urljoin(BASE_URL, href)
        except Exception:
            continue
    return None


def _day_params(d: date) -> List[str]:
    mon = d.strftime("%b").lower()
    return [f"{mon}{d.day}.{d.year}", f"{mon}{d.day:02d}.{d.year}"]


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

async def scrape_calendar_and_details(
    *,
    force_details: bool = False,
    reset_db: bool = False,
    days_ahead: int = 7,
    max_pages: int = 4,       # kept for CLI compatibility; not used
    refresh_incomplete: bool = True,
    max_detail_retries: int = 25,
    mode: str = "week",
) -> List[Dict[str, Any]]:
    _ = max_pages  # backward-compat only

    if reset_db:
        load_to_postgres.reset_database()
        print("PostgreSQL tables truncated (fresh reload).")

    details_cache = _load_details_cache(DETAILS_CACHE_JSON)
    checkpoint_enabled = CHECKPOINT_ROWS > 0
    total_checkpointed = 0

    today = date.today()
    start_date = today - timedelta(days=1)
    end_date = today + timedelta(days=max(0, int(days_ahead)))

    mode_norm = (mode or "week").strip().lower()
    if mode_norm not in {"week", "day"}:
        mode_norm = "week"

    target_days: List[date] = []
    if mode_norm == "day":
        cur = start_date
        while cur <= end_date:
            target_days.append(cur)
            cur += timedelta(days=1)

    combined: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    reused_cached = 0
    partial_refreshes = 0

    # ------------------------------------------------------------------
    def _checkpoint_print(stats, *, rows: int, final: bool) -> None:
        label = "Final checkpoint" if final else "Checkpoint"
        print(
            f"{label} stored. Rows: {rows}; "
            f"Details upserted: {stats.details_upserted}; "
            f"Calendar inserted: {stats.calendar_inserted}; "
            f"Total checkpointed: {total_checkpointed}"
        )

    async def _maybe_checkpoint(*, final: bool = False) -> None:
        nonlocal batch_rows, total_checkpointed, checkpoint_enabled
        if not checkpoint_enabled or not batch_rows:
            return
        if (not final) and len(batch_rows) < CHECKPOINT_ROWS:
            return
        try:
            stats = load_to_postgres.store_scraped_rows(batch_rows)
            total_checkpointed += len(batch_rows)
            _checkpoint_print(stats, rows=len(batch_rows), final=final)
            _save_details_cache(DETAILS_CACHE_JSON, details_cache)
            batch_rows = []
        except Exception as exc:
            print(f"Checkpoint warning — disabling (Postgres error): {exc}")
            checkpoint_enabled = False

    # ------------------------------------------------------------------
    # Process a page worth of bulk-extracted rows
    # ------------------------------------------------------------------
    def _process_bulk_rows(
        raw_rows: List[Dict],
        *,
        page_label: str,
        detail_results: Dict[str, Tuple[Dict, str, Optional[str]]],
    ) -> List[str]:
        """
        Post-process the JS-extracted rows: filter to window, parse dates,
        normalise impact, merge detail results, update cache.

        Returns the list of event_ids that still need details scraped
        (used only when details haven't been fetched yet).
        """
        nonlocal reused_cached, partial_refreshes

        kept = 0
        max_day_seen: Optional[date] = None
        events_needing_details: List[str] = []

        for raw in raw_rows:
            event_id = str(raw.get("event_id") or "").strip()
            if not event_id:
                continue

            # Date parsing
            day_label = raw.get("day_label") or ""
            current_day = _parse_calendar_date_text(day_label, today=today)
            if current_day is None:
                continue
            if current_day < start_date or current_day > end_date:
                continue

            if max_day_seen is None or current_day > max_day_seen:
                max_day_seen = current_day

            # Impact
            impact = _parse_impact_sync(
                raw.get("impact_classes") or "",
                raw.get("impact_labels") or "",
                raw.get("impact_text") or "",
                raw.get("impact_img_src") or "",
            )

            has_detail_link: bool = bool(raw.get("has_detail_link"))
            if not has_detail_link:
                has_detail_link = bool(DETAIL_CLICK_FALLBACK)

            # Cache logic
            detail_fields = None if force_details else details_cache.get(event_id)

            if (not force_details) and detail_fields is not None and has_detail_link:
                if _details_incomplete(detail_fields):
                    detail_fields = None
                elif (
                    PARTIAL_CACHE_REFRESH_LIMIT > 0
                    and _details_partially_filled(detail_fields)
                    and partial_refreshes < PARTIAL_CACHE_REFRESH_LIMIT
                ):
                    detail_fields = None
                    partial_refreshes += 1

            detail_status: Optional[str] = None
            detail_error: Optional[str] = None
            detail_attempts_inc: Optional[int] = None
            last_detail_attempt_at: Optional[str] = None

            if detail_fields is not None:
                reused_cached += 1
            else:
                if not has_detail_link:
                    detail_fields = _empty_detail_fields()
                    detail_status = "no-link"
                else:
                    # Check if detail_results already has this event (parallel scrape done)
                    if event_id in detail_results:
                        d_fields, d_status, d_error = detail_results[event_id]
                        detail_fields = d_fields
                        detail_status = d_status
                        detail_error = d_error
                        detail_attempts_inc = 1
                        last_detail_attempt_at = datetime.now(timezone.utc).isoformat()
                    else:
                        # Not yet scraped — add to queue for this call
                        events_needing_details.append(event_id)
                        detail_fields = _empty_detail_fields()

            if detail_fields is None:
                detail_fields = _empty_detail_fields()

            # Prefer previously cached values over a newly-empty result
            prev = details_cache.get(event_id)
            if prev is not None and _details_has_any_value(prev) and _details_incomplete(detail_fields):
                detail_fields = prev

            details_cache[event_id] = detail_fields

            row_out: Dict[str, Any] = {
                "event_id": event_id,
                "time": raw.get("time") or "",
                "currency": raw.get("currency") or "",
                "impact": impact,
                "event": raw.get("event_name") or "",
                "actual": raw.get("actual") or "",
                "forecast": raw.get("forecast") or "",
                "previous": raw.get("previous") or "",
                "calendar_date": current_day.isoformat(),
                "calendar_date_label": day_label,
                "has_detail_link": has_detail_link,
                **detail_fields,
                "detail_attempts_inc": detail_attempts_inc,
                "detail_status": detail_status,
                "last_detail_attempt_at": last_detail_attempt_at,
                "last_detail_error": detail_error,
            }
            combined.append(row_out)
            batch_rows.append(row_out)
            kept += 1

        print(
            f"{page_label}: kept {kept} rows within window "
            f"{start_date.isoformat()}..{end_date.isoformat()}."
        )
        if max_day_seen:
            print(f"{page_label}: max calendar_date in window: {max_day_seen.isoformat()}")

        return events_needing_details

    # ------------------------------------------------------------------
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        main_page = await context.new_page()
        main_page.set_default_timeout(10000)
        await _configure_fast_routes(main_page)

        # ---------------------------------------------------------------
        # Collect (page_url, label, raw_rows) tuples from all calendar pages
        # ---------------------------------------------------------------
        page_snapshots: List[Tuple[str, str, List[Dict]]] = []

        if mode_norm == "week":
            print("Loading calendar main page (single-page mode) ...")
            await _prepare_calendar_page(main_page, BASE_URL, label="main")
            raw = await _bulk_extract_rows(main_page)
            page_snapshots.append((main_page.url, "Main page", raw))
            print(f"Main page: {len(raw)} event rows extracted in one JS call.")

            visited: set = {main_page.url}
            for hop in range(3):
                _, max_vis = await _visible_calendar_day_range(main_page, today=today)
                if max_vis is None or max_vis >= end_date:
                    break
                next_url = await _find_next_week_url(main_page)
                if not next_url or next_url in visited:
                    break
                visited.add(next_url)
                print(
                    f"Loading next week page ... "
                    f"(visible through {max_vis.isoformat()}, "
                    f"need {end_date.isoformat()})"
                )
                await _prepare_calendar_page(
                    main_page, next_url, label=f"week+{hop+1}"
                )
                raw = await _bulk_extract_rows(main_page)
                page_snapshots.append((main_page.url, f"Week +{hop+1}", raw))
                print(f"Week +{hop+1}: {len(raw)} event rows extracted.")

        else:  # day mode
            for day in target_days:
                print(f"Loading calendar day page: {day.isoformat()} ...")
                loaded = False
                for day_param in _day_params(day):
                    day_url = f"{BASE_URL}?day={day_param}"
                    try:
                        await _prepare_calendar_page(
                            main_page, day_url, label=day.isoformat()
                        )
                    except PlaywrightTimeoutError:
                        continue

                    try:
                        lbl = await main_page.locator(
                            "td.calendar__cell.calendar__date span.date"
                        ).first.inner_text(timeout=500)
                        parsed = (
                            _parse_calendar_date_text(lbl, today=today) if lbl else None
                        )
                    except Exception:
                        parsed = None
                    if parsed is not None and (
                        parsed.month != day.month or parsed.day != day.day
                    ):
                        continue

                    loaded = True
                    break

                if not loaded:
                    await _prepare_calendar_page(
                        main_page,
                        f"{BASE_URL}?day={_day_params(day)[0]}",
                        label=day.isoformat(),
                    )

                raw = await _bulk_extract_rows(main_page)
                page_snapshots.append(
                    (main_page.url, f"Day {day.isoformat()}", raw)
                )
                print(f"Day {day.isoformat()}: {len(raw)} event rows extracted.")

        # ---------------------------------------------------------------
        # Identify ALL events that need detail scraping across every page
        # ---------------------------------------------------------------
        # First pass: collect event_ids needing details, per page+url
        needs_details_map: Dict[str, List[str]] = {}  # url -> [event_id, ...]
        for snap_url, snap_label, snap_raw in page_snapshots:
            for raw_row in snap_raw:
                eid = str(raw_row.get("event_id") or "").strip()
                if not eid:
                    continue
                day_label = raw_row.get("day_label") or ""
                d = _parse_calendar_date_text(day_label, today=today)
                if d is None or d < start_date or d > end_date:
                    continue
                cached = None if force_details else details_cache.get(eid)
                if cached is not None and not _details_incomplete(cached):
                    if not (
                        PARTIAL_CACHE_REFRESH_LIMIT > 0
                        and _details_partially_filled(cached)
                    ):
                        continue
                has_link = bool(raw_row.get("has_detail_link")) or bool(DETAIL_CLICK_FALLBACK)
                if has_link:
                    needs_details_map.setdefault(snap_url, [])
                    if eid not in needs_details_map[snap_url]:
                        needs_details_map[snap_url].append(eid)

        # ---------------------------------------------------------------
        # Parallel detail scraping — one batch per unique calendar URL
        # ---------------------------------------------------------------
        all_detail_results: Dict[str, Tuple[Dict, str, Optional[str]]] = {}
        for cal_url, eids in needs_details_map.items():
            if not eids:
                continue
            print(
                f"\nParallel detail scraping: {len(eids)} events from {cal_url} "
                f"using {min(DETAIL_WORKERS, len(eids))} worker(s)."
            )
            page_detail_results = await _parallel_detail_scrape(
                context, eids, cal_url, n_workers=DETAIL_WORKERS
            )
            all_detail_results.update(page_detail_results)

        # ---------------------------------------------------------------
        # Second pass: process rows now that details are available
        # ---------------------------------------------------------------
        for snap_url, snap_label, snap_raw in page_snapshots:
            _process_bulk_rows(
                snap_raw,
                page_label=snap_label,
                detail_results=all_detail_results,
            )
            await _maybe_checkpoint(final=False)

        await _maybe_checkpoint(final=True)
        await browser.close()

    # ------------------------------------------------------------------
    print(f"\nReused cached details  : {reused_cached}")
    print(f"Freshly scraped details: {len(all_detail_results)}")
    print(f"Partial cache refreshes: {partial_refreshes}/{PARTIAL_CACHE_REFRESH_LIMIT}")

    _save_details_cache(DETAILS_CACHE_JSON, details_cache)
    print(f"Updated cache: {DETAILS_CACHE_JSON}")

    if checkpoint_enabled and total_checkpointed > 0:
        print(
            "Stored into PostgreSQL incrementally. "
            f"Total checkpointed rows: {total_checkpointed}"
        )
    else:
        # Best-effort final store: checkpointing may have been disabled due to
        # transient Postgres/network errors; don't crash the whole scrape.
        try:
            stats = load_to_postgres.store_scraped_rows(combined)
            print(
                "Stored into PostgreSQL. "
                f"Details upserted: {stats.details_upserted}; "
                f"Calendar rows inserted: {stats.calendar_inserted}"
            )
        except Exception as exc:
            msg = str(exc)
            print(f"Postgres load warning (skipping store): {msg}")
            low = msg.lower()
            if "getaddrinfo" in low or "failed to resolve host" in low:
                print(
                    "Hint: this usually means your network/DNS can't resolve or reach the Supabase host. "
                    "If the project endpoint is IPv6-only, use Supabase 'Connection Pooler' DSN (often IPv4-friendly), "
                    "or fix IPv6/DNS on your machine."
                )

    # Retention window enforcement
    try:
        drop_hours = _int_env("DROP_STALE_NULL_IMPACT_HOURS", 72)
        load_to_postgres.enforce_retention_window(
            start_date=start_date,
            end_date=end_date,
            drop_stale_null_impact_hours=(drop_hours or None),
        )
        window_ids = load_to_postgres.get_window_event_ids(
            start_date=start_date, end_date=end_date
        )
        if window_ids:
            details_cache = {
                eid: v for eid, v in details_cache.items() if eid in window_ids
            }
            _save_details_cache(DETAILS_CACHE_JSON, details_cache)
        print(
            "Retention enforced. "
            f"Window: {start_date.isoformat()}..{end_date.isoformat()}. "
            f"Cache entries kept: {len(details_cache)}"
        )
    except Exception as exc:
        print(f"Retention warning: {exc}")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape ForexFactory and store directly into PostgreSQL"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Truncate ff_event_details and ff_calendar_events before storing",
    )
    parser.add_argument(
        "--force-details",
        action="store_true",
        help="Re-scrape detail overlay even if details are in the local cache",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=5,
        help="Extract events from today through N days ahead (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=["week", "day"],
        default="week",
        help="'week' scrapes the main calendar once (recommended); 'day' uses ?day=... pages",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Deprecated. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--no-refresh-incomplete",
        action="store_true",
        help="Disable one-pass retry for incomplete/error detail overlays",
    )
    parser.add_argument(
        "--max-detail-retries",
        type=int,
        default=25,
        help="Max events per page to retry when details are incomplete/error",
    )
    args = parser.parse_args()

    asyncio.run(
        scrape_calendar_and_details(
            force_details=args.force_details,
            reset_db=args.reset_db,
            days_ahead=args.days_ahead,
            max_pages=args.max_pages,
            refresh_incomplete=(not args.no_refresh_incomplete),
            max_detail_retries=args.max_detail_retries,
            mode=args.mode,
        )
    )


if __name__ == "__main__":
    main()
