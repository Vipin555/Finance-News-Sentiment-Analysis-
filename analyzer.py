"""
Analyzer orchestrator.

Wires all three layers together:
  Layer 1 → deviation_scorer   (quantitative actual vs forecast)
  Layer 2 → nlp_engine (FinBERT)
  Layer 3 → composite + aggregation

Public API
──────────
  analyze_event(calendar_event_id)  → single-event result dict
  analyze_recent(hours)             → batch analysis of recent events
  get_aggregated_signal(hours)      → aggregated AggregatedSignal for a window
"""
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from config import settings
from database import (
    fetch_enriched_event_by_calendar_id,
    fetch_recent_enriched_events,
    fetch_todays_events,
    upsert_sentiment_result,
)

from deviation_scorer import DeviationResult, compute_deviation_score
from nlp_engine import NLPResult, finbert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impact_weight(impact: Optional[str]) -> float:
    """
    Handles both plain ("red") and prefixed ("ff-impact-red", "ff-impact-yel",
    "ff-impact-grn") formats that Forex Factory scrapers may produce.
    """
    s = (impact or "").lower().strip()
    if s.startswith("ff-impact-"):
        s = s[len("ff-impact-"):]
    abbrev = {"yel": "yellow", "grn": "green", "org": "yellow"}
    s = abbrev.get(s, s)
    return settings.IMPACT_WEIGHTS.get(s, settings.IMPACT_WEIGHTS["unknown"])


def _composite_and_confidence(
    dev:    DeviationResult,
    nlp:    NLPResult,
) -> tuple[float, float]:
    """
    Combine deviation + NLP scores into a composite XAUUSD signal.

    Weight allocation is dynamic:
      - If we have an actual value with good data  → configured 60/40 split
      - If we only have actual vs previous (noisier)→ shift toward NLP
      - If there is no actual at all                → NLP only (upcoming event)
    """
    conf = dev.confidence  # 0.9 / 0.6 / 0.1 / 0.0

    if conf >= 0.85:
        d_w = settings.DEVIATION_WEIGHT      # 0.60
        n_w = settings.NLP_WEIGHT            # 0.40
    elif conf >= 0.5:
        # Partial data: scale deviation weight down
        d_w = settings.DEVIATION_WEIGHT * conf
        n_w = 1.0 - d_w
    else:
        # No usable deviation: rely entirely on NLP
        d_w, n_w = 0.0, 1.0

    composite = dev.xauusd_score * d_w + nlp.score * n_w

    # Confidence reflects data quality and NLP certainty
    # NLP certainty = how far from random (0.333 each)
    nlp_certainty = max(nlp.positive, nlp.negative, nlp.neutral) - 0.333
    nlp_certainty = max(0.0, nlp_certainty) / 0.667   # normalise to [0, 1]
    final_conf = d_w * conf + n_w * nlp_certainty
    final_conf = min(1.0, final_conf)

    return composite, final_conf


def _label(signal: float) -> str:
    if signal >= settings.BULLISH_THRESHOLD:
        return "BULLISH"
    if signal <= settings.BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _build_result(
    event:     Dict[str, Any],
    dev:       DeviationResult,
    nlp:       NLPResult,
    composite: float,
    conf:      float,
) -> Dict[str, Any]:
    """Assemble the full result dict ready for DB insertion."""
    signal = max(-1.0, min(1.0, composite))
    lbl    = _label(signal)

    reasoning = json.dumps({
        "deviation": {
            "raw_score":            round(dev.raw_score, 4),
            "xauusd_score":         round(dev.xauusd_score, 4),
            "confidence":           round(dev.confidence, 4),
            "actual":               dev.actual_parsed,
            "forecast":             dev.forecast_parsed,
            "previous":             dev.previous_parsed,
            "direction_multiplier": dev.direction_multiplier,
            "note":                 dev.reasoning,
        },
        "nlp": {
            "positive":    round(nlp.positive, 4),
            "negative":    round(nlp.negative, 4),
            "neutral":     round(nlp.neutral, 4),
            "score":       round(nlp.score, 4),
            "dominant":    nlp.dominant,
            "text_tokens": nlp.token_count,
        },
        "composite": {
            "score":          round(composite, 4),
            "xauusd_signal":  round(signal, 4),
            "label":          lbl,
            "impact_weight":  _impact_weight(event.get("impact")),
        },
    })

    return {
        "calendar_event_id":    event["id"],
        "event_id":             event.get("event_id"),
        "currency":             event.get("currency"),
        "impact":               event.get("impact"),
        "event_name":           event.get("event_name"),
        "event_date":           event.get("calendar_date"),
        "event_time":           event.get("time"),
        "pre_release":          not bool(event.get("actual")),  # True = actual not yet published
        # ── Deviation layer ──
        "deviation_score":      round(dev.xauusd_score, 4),
        "deviation_confidence": round(dev.confidence, 4),
        # ── NLP layer ────────
        "nlp_positive":         round(nlp.positive, 4),
        "nlp_negative":         round(nlp.negative, 4),
        "nlp_neutral":          round(nlp.neutral, 4),
        "nlp_score":            round(nlp.score, 4),
        # ── Composite ────────
        "composite_score":      round(composite, 4),
        "xauusd_signal":        round(signal, 4),
        "label":                lbl,
        "confidence":           round(conf, 4),
        "reasoning":            reasoning,
    }


# ---------------------------------------------------------------------------
# Storage gate (two-phase: pre-release vs post-release)
# ---------------------------------------------------------------------------

def _nlp_certainty(nlp: NLPResult) -> float:
    """How far FinBERT is from a uniform 1/3 distribution (0 = random, 1 = certain)."""
    raw = max(nlp.positive, nlp.negative, nlp.neutral) - 0.333
    return max(0.0, raw) / 0.667


def _impact_key(impact: Optional[str]) -> str:
    """Normalise impact string to config key."""
    s = (impact or "").lower().strip()
    if s.startswith("ff-impact-"):
        s = s[len("ff-impact-"):]
    return {"yel": "yellow", "grn": "green", "org": "orange"}.get(s, s)


def _should_store(result: Dict[str, Any], nlp: NLPResult) -> tuple[bool, str]:
    """
    Two-phase gate:

    POST-release (actual data present):
      • Label must be BULLISH or BEARISH (signal >= ±0.08)
      • Combined confidence must be >= POST_RELEASE_MIN_CONFIDENCE (0.45)
      → Merges deviation + NLP, so we trust this signal strongly.

    PRE-release (no actual yet):
      • Label must be BULLISH or BEARISH
      • NLP certainty must clear an impact-weighted bar:
          red >= 0.10, orange >= 0.13, yellow >= 0.20, green/unknown = never
      → Pre-release signals are leading indicators; stored for high-impact
        events where even NLP-only direction has trading relevance.
    """
    if result["label"] == "NEUTRAL":
        return False, "NEUTRAL signal"

    if not result["pre_release"]:
        # POST-release
        if result["confidence"] < settings.POST_RELEASE_MIN_CONFIDENCE:
            return False, f"post-release conf {result['confidence']:.4f} < {settings.POST_RELEASE_MIN_CONFIDENCE}"
        return True, "post-release BULLISH/BEARISH"
    else:
        # PRE-release: gate on impact-weighted NLP certainty
        imp_key = _impact_key(result.get("impact"))
        threshold = settings.PRE_RELEASE_NLP_THRESHOLDS.get(imp_key, 0.99)
        cert = _nlp_certainty(nlp)
        if cert < threshold:
            return False, f"pre-release NLP cert {cert:.4f} < {threshold} ({imp_key} impact)"
        return True, f"pre-release {imp_key}-impact BULLISH/BEARISH"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyze_event(calendar_event_id: int) -> Optional[Dict[str, Any]]:
    """
    Full 3-layer analysis for a single calendar row.
    Persists the result to ff_sentiment_results and returns the result dict.
    """
    event = await fetch_enriched_event_by_calendar_id(calendar_event_id)
    if not event:
        logger.warning("No event found for calendar_event_id=%d", calendar_event_id)
        return None

    dev = compute_deviation_score(
        actual       = event.get("actual"),
        forecast     = event.get("forecast"),
        previous     = event.get("previous"),
        usual_effect = event.get("usual_effect"),
        currency     = event.get("currency"),
        event_name   = event.get("event_name"),
    )

    nlp = finbert.analyze(
        event_name       = event.get("event_name"),
        why_traders_care = event.get("why_traders_care"),
        usual_effect     = event.get("usual_effect"),
        ff_notes         = event.get("ff_notes"),
        measures         = event.get("measures"),
        ff_notice        = event.get("ff_notice"),
    )

    composite, conf = _composite_and_confidence(dev, nlp)
    result = _build_result(event, dev, nlp, composite, conf)

    store, reason = _should_store(result, nlp)
    if store:
        try:
            await upsert_sentiment_result(result)
            logger.info(
                "Stored %s signal for '%s' [%s] conf=%.4f pre_release=%s",
                result["label"], result["event_name"], reason,
                result["confidence"], result["pre_release"]
            )
        except Exception as exc:
            logger.error("Failed to save sentiment result for id=%d: %s", calendar_event_id, exc)
    else:
        logger.debug("Skipped '%s': %s", result.get("event_name"), reason)

    return result


async def analyze_recent(hours: int = 24) -> List[Dict[str, Any]]:
    """
    Batch-analyze all events scraped in the last N hours.

    Uses analyze_batch on FinBERT for efficiency (single forward pass per batch).
    """
    events = await fetch_recent_enriched_events(hours)
    if not events:
        logger.info("No recent events to analyze (last %dh).", hours)
        return []

    # Batch NLP inference
    nlp_results = finbert.analyze_batch(events)
    results: List[Dict[str, Any]] = []

    for event, nlp in zip(events, nlp_results):
        dev = compute_deviation_score(
            actual       = event.get("actual"),
            forecast     = event.get("forecast"),
            previous     = event.get("previous"),
            usual_effect = event.get("usual_effect"),
            currency     = event.get("currency"),
            event_name   = event.get("event_name"),
        )
        composite, conf = _composite_and_confidence(dev, nlp)
        result = _build_result(event, dev, nlp, composite, conf)

        store, reason = _should_store(result, nlp)
        if store:
            try:
                await upsert_sentiment_result(result)
            except Exception as exc:
                logger.error("Failed to save result for id=%s: %s", event.get("id"), exc)
        else:
            logger.debug("Skipped '%s': %s", result.get("event_name"), reason)

        results.append(result)

    stored = sum(1 for r in results if _should_store(r, finbert.analyze(
        event_name=r.get("event_name")))[0])
    logger.info("Analyzed %d events (last %dh).", len(results), hours)
    return results


async def analyze_today() -> List[Dict[str, Any]]:
    """
    Smart analysis for today's economic events (calendar_date = today).

    - Skips events already analyzed recently (fresher than STALE_RESULT_HOURS).
    - Re-analyzes events whose result is stale AND now have an actual value.
    - Runs NLP in batch for efficiency.
    """
    import datetime as dt
    from datetime import timezone

    events = await fetch_todays_events()
    if not events:
        logger.info("No forex-relevant events for today.")
        return []

    stale_cutoff = dt.datetime.now(timezone.utc) - dt.timedelta(
        hours=settings.STALE_RESULT_HOURS
    )

    to_analyze: List[Dict[str, Any]] = []
    for ev in events:
        last_analyzed = ev.get("sentiment_computed_at")
        has_actual = bool(ev.get("actual"))

        if last_analyzed is None:
            # Never analyzed before
            to_analyze.append(ev)
        elif last_analyzed < stale_cutoff and has_actual:
            # Stale result AND actual is now available (post-release upgrade)
            to_analyze.append(ev)
        elif has_actual and ev.get("existing_label") is not None:
            # Was analyzed pre-release; actual just arrived → force re-analysis
            # so deviation score can now contribute
            # Guard: only re-analyze if result was stored pre-release
            # (we check deviation_confidence would have been 0 at that time via pre_release flag on result)
            # Simple heuristic: re-analyze if last analysis was within STALE_RESULT_HOURS*4
            upgrade_cutoff = dt.datetime.now(timezone.utc) - dt.timedelta(
                hours=settings.STALE_RESULT_HOURS * 4
            )
            if last_analyzed > upgrade_cutoff:
                logger.debug(
                    "Re-analyzing pre-release event id=%s (%s) — actual data now available",
                    ev.get("id"), ev.get("event_name"),
                )
                to_analyze.append(ev)
        else:
            logger.debug(
                "Skipping fresh result for event id=%s (%s)",
                ev.get("id"), ev.get("event_name"),
            )

    if not to_analyze:
        logger.info("All today's events have fresh results. Nothing to re-analyze.")
        return []

    nlp_results = finbert.analyze_batch(to_analyze)
    results: List[Dict[str, Any]] = []
    stored_count = 0

    for event, nlp in zip(to_analyze, nlp_results):
        dev = compute_deviation_score(
            actual       = event.get("actual"),
            forecast     = event.get("forecast"),
            previous     = event.get("previous"),
            usual_effect = event.get("usual_effect"),
            currency     = event.get("currency"),
            event_name   = event.get("event_name"),
        )
        composite, conf = _composite_and_confidence(dev, nlp)
        result = _build_result(event, dev, nlp, composite, conf)

        store, reason = _should_store(result, nlp)
        if store:
            try:
                await upsert_sentiment_result(result)
                stored_count += 1
                logger.info(
                    "Stored %s signal for '%s' [%s] conf=%.4f pre_release=%s",
                    result["label"], result["event_name"], reason,
                    result["confidence"], result["pre_release"]
                )
            except Exception as exc:
                logger.error("Failed to save result for id=%s: %s", event.get("id"), exc)
        else:
            logger.debug("Skipped '%s': %s", result.get("event_name"), reason)

        results.append(result)

    logger.info(
        "analyze_today: processed %d / %d events, stored %d actionable signals.",
        len(results), len(events), stored_count
    )
    return results