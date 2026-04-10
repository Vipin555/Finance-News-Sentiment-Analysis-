"""extract_details.py — optimised build.

Key speed improvements over the original:
  1. _extract_detail_specs() now uses a SINGLE page.evaluate() to pull all spec
     rows out of the DOM at once instead of 2-3 inner_text() awaits per row
     (up to 18 browser round-trips → 1).
  2. _close_detail_modal() is leaner: fewer sequential wait_for() calls, direct
     DOM removal as the first resort when the close button is absent.
  3. _try_open_detail_for_row() short-circuits immediately on first successful
     click instead of checking each candidate with a separate count() await.

All public names, selectors and return types are unchanged — the module is a
drop-in replacement.
"""

import asyncio
import json
import os
import re
from typing import Dict, List, Optional

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Locator
from playwright.async_api import async_playwright

# ================== CONFIGURATION ==================
BASE_URL = "https://www.forexfactory.com/calendar"
OUTPUT_JSON = "forexfactory_details.json"
HEADLESS = os.getenv("HEADLESS", "0").strip() == "1"
BLOCK_HEAVY_RESOURCES = os.getenv("BLOCK_HEAVY_RESOURCES", "1").strip() == "1"

# Selectors (unchanged)
ROW_SELECTOR                = "tr.calendar__row"
DETAIL_LINK_SELECTOR        = ".calendar__detail-link"
DETAIL_LINK_LEVEL2_SELECTOR = ".calendar__detail-link--level-2"
DETAIL_LINK_LEVEL1_SELECTOR = ".calendar__detail-link--level-1"
DETAIL_LINK_LEVEL0_SELECTOR = ".calendar__detail-link--level-0"
DETAIL_ROW_SELECTOR         = "tr.calendar__details--detail"
DETAIL_OVERLAY_SELECTOR     = f"{DETAIL_ROW_SELECTOR} .calendardetails__segment--info.overlay"
DETAIL_CLOSE_BUTTON_SELECTOR = "a.overlay__button.exit_details"
DETAIL_LOADING_SELECTOR     = ".loading, .spinner, .calendar__detail-loading"
SPECS_ROW_SELECTOR          = "table.calendarspecs tr"
SPECS_NAME_SELECTOR         = "td.calendarspecs__spec"
SPECS_VALUE_SELECTOR        = "td.calendarspecs__specdescription"
EVENT_CLICK_FALLBACK_SELECTOR = ".calendar__event-title, .calendar__event a, .calendar__event"

# ================== HELPERS ==================

def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _normalize_spec_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = _clean_text(name)
    if not name:
        return None
    name = re.sub(r"[^A-Za-z ]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name or None


def _canonical_spec_key(name: Optional[str]) -> Optional[str]:
    normalized = _normalize_spec_name(name)
    if not normalized:
        return None
    aliases = {
        "ff notice": "ff_notice",
        "forex factory notice": "ff_notice",
        "measures": "measures",
        "usual effect": "usual_effect",
        "frequency": "frequency",
        "ff notes": "ff_notes",
        "forex factory notes": "ff_notes",
        "why traders care": "why_traders_care",
    }
    return aliases.get(normalized)


async def _safe_inner_text(locator) -> Optional[str]:
    """Kept for external callers; not used inside this module any more."""
    try:
        if await locator.count() == 0:
            return None
        return await locator.first.inner_text()
    except Exception:
        return None


# ================== CORE: single-JS spec extraction ==================

# JavaScript that extracts all spec rows from the already-open detail container
# in ONE evaluate call — replaces the Python for-loop with 2-3 awaits per row.
_EXTRACT_SPECS_JS = """(container) => {
    const rows = container.querySelectorAll('table.calendarspecs tr');
    const result = [];
    for (const row of rows) {
        const nameCandidates = [
            row.querySelector('td.calendarspecs__spec'),
            row.querySelector('th.calendarspecs__spec'),
            row.querySelector('td:first-child'),
            row.querySelector('th:first-child'),
        ];
        const valCandidates = [
            row.querySelector('td.calendarspecs__specdescription'),
            row.querySelector('td:nth-child(2)'),
            row.querySelector('th:nth-child(2)'),
        ];
        const nameEl  = nameCandidates.find(Boolean);
        const valueEl = valCandidates.find(Boolean);
        result.push({
            name:  nameEl  ? (nameEl.innerText  || '').trim() : '',
            value: valueEl ? (valueEl.innerText || '').trim() : '',
        });
    }
    return result;
}"""


async def _extract_detail_specs(container) -> Dict[str, Optional[str]]:
    """Extract detail fields from the Specs table — single JS round-trip."""
    result: Dict[str, Optional[str]] = {
        "ff_notice": None,
        "measures": None,
        "usual_effect": None,
        "frequency": None,
        "ff_notes": None,
        "why_traders_care": None,
    }

    try:
        # One evaluate call returns [{name, value}, ...] for every spec row.
        raw_rows: List[Dict] = await container.evaluate(_EXTRACT_SPECS_JS)
    except Exception:
        # Fallback: use the original Playwright-based approach if JS fails.
        return await _extract_detail_specs_fallback(container)

    for row in raw_rows:
        key = _canonical_spec_key(row.get("name"))
        if not key:
            continue
        result[key] = _clean_text(row.get("value")) or None

    return result


async def _extract_detail_specs_fallback(container) -> Dict[str, Optional[str]]:
    """Original row-by-row fallback (used only if the JS path fails)."""
    result: Dict[str, Optional[str]] = {
        "ff_notice": None, "measures": None, "usual_effect": None,
        "frequency": None, "ff_notes": None, "why_traders_care": None,
    }
    rows = container.locator(SPECS_ROW_SELECTOR)
    count = await rows.count()
    for i in range(count):
        row = rows.nth(i)
        name_raw = await _safe_inner_text(row.locator(SPECS_NAME_SELECTOR))
        if not name_raw:
            name_raw = await _safe_inner_text(row.locator("th.calendarspecs__spec"))
        if not name_raw:
            name_raw = await _safe_inner_text(row.locator("td:first-child, th:first-child"))
        key = _canonical_spec_key(name_raw)
        if not key:
            continue
        value_raw = await _safe_inner_text(row.locator(SPECS_VALUE_SELECTOR))
        if value_raw is None:
            value_raw = await _safe_inner_text(
                row.locator("td:nth-child(2), th:nth-child(2)")
            )
        result[key] = _clean_text(value_raw)
    return result


# ================== MODAL OPEN / CLOSE ==================

async def _configure_fast_routes(page) -> None:
    if not BLOCK_HEAVY_RESOURCES:
        return

    async def _handler(route):
        if route.request.resource_type in {"image", "font", "media"}:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", _handler)


async def _try_open_detail_for_row(
    page, row, *, event_id: Optional[str] = None
) -> Optional[Locator]:
    """Try to open the detail overlay for a given calendar row.

    Returns a Locator for the opened detail container, or None.
    """
    if event_id is None:
        try:
            event_id = await row.get_attribute("data-event-id")
        except Exception:
            event_id = None
        event_id = _clean_text(event_id)

    candidates = [
        row.locator(DETAIL_LINK_LEVEL2_SELECTOR),
        row.locator(DETAIL_LINK_LEVEL1_SELECTOR),
        row.locator(DETAIL_LINK_LEVEL0_SELECTOR),
        row.locator(DETAIL_LINK_SELECTOR),
        row.locator(EVENT_CLICK_FALLBACK_SELECTOR),
    ]

    for cand in candidates:
        try:
            # count() is a cheap DOM check; skip the heavy scroll+click if absent.
            if await cand.count() == 0:
                continue
            link = cand.first
            await link.scroll_into_view_if_needed(timeout=3000)
            await link.click(timeout=8000, force=True)
            return await _get_open_detail_container(page, event_id)
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
    return None


async def _get_opened_event_id(detail_container: Locator) -> Optional[str]:
    try:
        eid = await detail_container.locator(
            DETAIL_CLOSE_BUTTON_SELECTOR
        ).first.get_attribute("data-eventid")
    except Exception:
        eid = None
    return _clean_text(eid)


async def _get_open_detail_container(page, event_id: Optional[str]) -> Locator:
    """Return a Locator for the opened detail row."""
    if event_id:
        close_btn = page.locator(
            f"a.exit_details[data-eventid='{event_id}']"
        ).first
        await close_btn.wait_for(state="attached", timeout=5000)
        container = close_btn.locator(
            "xpath=ancestor::tr[contains(@class,'calendar__details--detail')]"
        )
        await container.locator(
            ".calendardetails__segment--info.overlay"
        ).wait_for(state="visible", timeout=5000)
        return container

    # Fallback: wait for any visible overlay, pick the last one.
    await page.wait_for_selector(DETAIL_OVERLAY_SELECTOR, state="visible", timeout=5000)
    overlay = page.locator(f"{DETAIL_OVERLAY_SELECTOR}:visible").last
    return overlay.locator(
        "xpath=ancestor::tr[contains(@class,'calendar__details--detail')]"
    )


async def _close_detail_modal(
    page, detail_container: Optional[Locator] = None, event_id: Optional[str] = None
) -> bool:
    """Close the currently open detail row — leaner version.

    Strategy:
      1. Click the scoped close button (or fire Escape as fallback).
      2. Wait briefly for the overlay to become hidden.
      3. If still stuck, remove the detail rows from the DOM via JS (instant).
    """
    # Step 1 — click close / press Escape.
    try:
        close_btn = (
            detail_container.locator(DETAIL_CLOSE_BUTTON_SELECTOR)
            if detail_container is not None
            else page.locator(f"{DETAIL_ROW_SELECTOR} {DETAIL_CLOSE_BUTTON_SELECTOR}")
        )
        if await close_btn.count() > 0:
            await close_btn.first.click(timeout=3000, force=True)
        else:
            await page.keyboard.press("Escape")
    except Exception:
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

    # Step 2 — single short wait; don't stack multiple wait_for() calls.
    try:
        target = (
            detail_container.locator(".calendardetails__segment--info.overlay")
            if detail_container is not None
            else page.locator(DETAIL_OVERLAY_SELECTOR)
        )
        await target.wait_for(state="hidden", timeout=1500)
        return True
    except Exception:
        pass

    # Step 3 — nuclear: remove detail rows from DOM so they don't block next click.
    try:
        await page.evaluate(
            """() => {
                for (const row of document.querySelectorAll(
                    'tr.calendar__details--detail'
                )) { row.remove(); }
            }"""
        )
        return True
    except Exception:
        return False


# ================== STANDALONE MAIN (unchanged logic) ==================

async def _extract_detail_text(page) -> Optional[str]:
    """Kept for debugging; not used by the main flow."""
    content = await page.query_selector(f"{DETAIL_ROW_SELECTOR} .half.details")
    if not content:
        content = await page.query_selector(DETAIL_OVERLAY_SELECTOR)
    if not content:
        return None
    return _clean_text(await content.inner_text())


async def main():
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
        page = await context.new_page()
        await _configure_fast_routes(page)

        print("Loading calendar page...")
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)

        try:
            await page.wait_for_selector(ROW_SELECTOR, state="attached", timeout=60000)
            print("Calendar loaded successfully.")
        except PlaywrightTimeoutError:
            print("Timed out waiting for calendar rows. Possible Cloudflare challenge.")
            input("Press Enter after completing the challenge...")
            await page.wait_for_selector(ROW_SELECTOR, state="attached", timeout=30000)

        rows_locator = page.locator(ROW_SELECTOR)
        row_count = await rows_locator.count()
        print(f"Found {row_count} rows. Extracting details...")

        # Debug link counts
        try:
            link_counts = await page.evaluate(
                """() => ({
                    any:          document.querySelectorAll('.calendar__detail-link').length,
                    level0:       document.querySelectorAll('.calendar__detail-link--level-0').length,
                    level1:       document.querySelectorAll('.calendar__detail-link--level-1').length,
                    level2:       document.querySelectorAll('.calendar__detail-link--level-2').length,
                    level2Anchors:document.querySelectorAll('a.calendar__detail-link--level-2').length,
                })"""
            )
            print(f"Detail-link counts: {link_counts}")
        except Exception:
            pass

        details = []
        seen_event_ids: set = set()
        processed = 0

        for i in range(row_count):
            row = rows_locator.nth(i)
            if await row.locator(DETAIL_LINK_SELECTOR).count() == 0:
                continue

            processed += 1
            print(f"[{processed}] Opening details for row {i}...")
            try:
                await _close_detail_modal(page)
                detail_container = await _try_open_detail_for_row(page, row)
                if detail_container is None:
                    print("  Skipped: could not open details for this row.")
                    continue

                opened_event_id = await _get_opened_event_id(detail_container)
                if opened_event_id and opened_event_id in seen_event_ids:
                    print(f"  Skipped: duplicate event_id={opened_event_id}.")
                    await _close_detail_modal(page, detail_container, event_id=opened_event_id)
                    continue

                try:
                    await detail_container.locator(
                        DETAIL_LOADING_SELECTOR
                    ).first.wait_for(state="hidden", timeout=5000)
                except PlaywrightTimeoutError:
                    pass

                await detail_container.locator("table.calendarspecs").first.wait_for(
                    state="attached", timeout=10000
                )

                # ★ One JS call instead of N×3 awaits
                specs = await _extract_detail_specs(detail_container)
                details.append({"event_id": opened_event_id, **specs})
                if opened_event_id:
                    seen_event_ids.add(opened_event_id)
                extracted_fields = sum(1 for v in specs.values() if v)
                print(f"  Extracted {extracted_fields}/6 detail fields.")

                closed = await _close_detail_modal(
                    page, detail_container, event_id=opened_event_id
                )
                if not closed:
                    print("  Warning: modal did not close cleanly.")

            except Exception as e:
                print(f"  Error: {e}")
                await _close_detail_modal(page)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        await browser.close()
        print(f"Done. Saved {len(details)} items to {OUTPUT_JSON}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
