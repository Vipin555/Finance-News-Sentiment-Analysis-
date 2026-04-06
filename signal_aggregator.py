"""
Time-window signal aggregator (Layer 3 – aggregation step).

Takes a list of persisted sentiment rows and returns a single
XAUUSD directional signal using:
  - Impact weighting   (red=3, yellow=2, green=1)
  - Exponential time decay (configurable half-life)
  - Confidence weighting (higher-confidence events count more)

Output: AggregatedSignal with label BULLISH / BEARISH / NEUTRAL
"""
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    window_hours:       int
    xauusd_signal:      float   # weighted average ∈ [-1, +1]
    label:              str     # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence:         float   # 0–1; higher = more events + higher certainty
    event_count:        int
    high_impact_count:  int     # number of red-impact events in window
    contributing_events: List[Dict[str, Any]] = field(default_factory=list)
    computed_at:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impact_weight(impact: Optional[str]) -> float:
    """
    Resolve impact string to a weight.

    Handles both plain ("red") and prefixed ("ff-impact-red", "ff-impact-yel",
    "ff-impact-grn") formats that Forex Factory scrapers may produce.
    """
    s = (impact or "").lower().strip()
    if s.startswith("ff-impact-"):
        s = s[len("ff-impact-"):]
    # Map FF abbreviations to canonical config keys
    abbrev = {"yel": "yellow", "grn": "green", "org": "yellow"}
    s = abbrev.get(s, s)
    return settings.IMPACT_WEIGHTS.get(s, settings.IMPACT_WEIGHTS["unknown"])


def _time_decay(scraped_at: datetime, now: datetime) -> float:
    """
    Exponential decay: w = exp(-λ·t)
    λ = ln(2) / half_life_hours
    An event at t=0 (just scraped) → w=1.0
    An event at t=half_life_hours  → w=0.5
    """
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (now - scraped_at).total_seconds() / 3600)
    lam   = math.log(2) / settings.TIME_DECAY_HALF_LIFE_HOURS
    return math.exp(-lam * age_h)


def _to_datetime(value: Any) -> Optional[datetime]:
    """Coerce string / datetime / None to a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def aggregate_signals(
    sentiment_rows: List[Dict[str, Any]],
    window_hours: int,
) -> AggregatedSignal:
    """
    Aggregate a list of sentiment result rows into a single XAUUSD signal.

    Each row is expected to contain (at minimum):
      xauusd_signal  – float  – the individual event gold signal
      impact         – str    – "red" / "yellow" / "green"
      confidence     – float  – individual event confidence
      scraped_at     – datetime or ISO string
      event_name     – str    – for the contributing_events list
      currency       – str
    """
    now = datetime.now(timezone.utc)

    if not sentiment_rows:
        return AggregatedSignal(
            window_hours=window_hours,
            xauusd_signal=0.0,
            label="NEUTRAL",
            confidence=0.0,
            event_count=0,
            high_impact_count=0,
        )

    total_weight     = 0.0
    weighted_signal  = 0.0
    sum_confidence   = 0.0
    high_impact_count = 0
    contributing: List[Dict[str, Any]] = []

    for row in sentiment_rows:
        signal     = float(row.get("xauusd_signal") or 0.0)
        impact     = str(row.get("impact") or "unknown")
        confidence = float(row.get("confidence") or 0.5)
        scraped_at = _to_datetime(row.get("scraped_at")) or now

        # Composite weight: decay × impact × confidence
        decay    = _time_decay(scraped_at, now)
        imp_w    = _impact_weight(impact)
        weight   = decay * imp_w * confidence

        weighted_signal += signal * weight
        total_weight    += weight
        sum_confidence  += confidence * imp_w   # impact-boosted confidence sum

        impact_clean = impact.lower().replace("ff-impact-", "")
        if impact_clean == "red":
            high_impact_count += 1

        contributing.append({
            "event_name": row.get("event_name"),
            "currency":   row.get("currency"),
            "impact":     impact,
            "signal":     round(signal, 4),
            "weight":     round(weight, 4),
            "decay":      round(decay, 4),
        })

    # ── Final signal ──────────────────────────────────────────────────────────
    final_signal = weighted_signal / total_weight if total_weight > 0 else 0.0
    final_signal = max(-1.0, min(1.0, final_signal))

    # ── Confidence: blend of avg event quality and window depth ───────────────
    n           = len(sentiment_rows)
    avg_conf    = sum_confidence / n
    # count_boost asymptotes to 1 as n → ∞ (5 events ≈ 0.63, 10 ≈ 0.86)
    count_boost = 1.0 - math.exp(-n / 5.0)
    final_conf  = min(1.0, avg_conf * count_boost)

    # ── Label ─────────────────────────────────────────────────────────────────
    if final_signal >= settings.BULLISH_THRESHOLD:
        label = "BULLISH"
    elif final_signal <= settings.BEARISH_THRESHOLD:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    # Sort contributors by weight descending; keep top 10
    contributing.sort(key=lambda x: x["weight"], reverse=True)

    return AggregatedSignal(
        window_hours        = window_hours,
        xauusd_signal       = round(final_signal, 4),
        label               = label,
        confidence          = round(final_conf, 4),
        event_count         = n,
        high_impact_count   = high_impact_count,
        contributing_events = contributing[:10],
        computed_at         = now,
    )