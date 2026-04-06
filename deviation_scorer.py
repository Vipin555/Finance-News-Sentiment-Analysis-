"""
Quantitative deviation scorer (Layer 1).

Computes how much the actual reading beat or missed the forecast, then
maps that deviation through the currency-to-XAUUSD direction table to
produce a gold-directional score in [-1, +1].

Pipeline
--------
1. Parse FF number strings  →  float (handles K/M/B, %, pipes, <> prefixes)
2. Compute raw % deviation  →  (actual - forecast) / |forecast|
3. Normalise with tanh      →  smooth [-1, +1] even for extreme deviations
4. Apply usual_effect dir   →  flip sign if "less than forecast is good"
5. Check event-type override→  inflation events push gold up; hawkish USD down
6. Map currency → XAUUSD   →  USD bullish = gold bearish (×-1); EUR/GBP ×+0.4…
"""
import math
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XAUUSD direction per currency
# Positive  →  currency bullish tends to be gold bullish  (USD is the key inverse)
# Negative  →  currency bullish tends to be gold bearish
# ---------------------------------------------------------------------------
CURRENCY_GOLD_DIRECTION: dict = {
    "USD": -1.0,   # Strong USD  → bearish gold (core inverse relationship)
    "EUR":  0.45,  # Strong EUR  → weak USD     → mildly bullish gold
    "GBP":  0.40,
    "AUD":  0.35,  # AUD/CAD are risk-on; also correlated with commodities
    "CAD":  0.30,
    "NZD":  0.30,
    "JPY":  0.25,  # JPY is itself a safe-haven; strengthening ≈ risk-off ≈ gold up
    "CHF":  0.25,  # Same for CHF
    "CNY":  0.20,
    "ALL":  0.00,  # Global/multi-currency events: no clear direction
}

# Events where a high actual reading means INFLATION → bullish gold
# (regardless of which currency field says)
INFLATION_KEYWORDS = frozenset({
    "cpi", "ppi", "inflation", "core inflation", "price index",
    "consumer price", "producer price", "pce deflator",
    "core pce", "retail price", "import price",
})

# Events that represent hawkish USD policy  → bearish gold
# These override even if currency != USD
USD_HAWKISH_KEYWORDS = frozenset({
    "nonfarm payroll", "non-farm payroll", "nfp",
    "fed funds rate", "fomc", "rate decision", "interest rate decision",
    "rate hike", "rate statement",
    "durable goods", "gdp",  # Strong GDP = less dovish Fed
})

# Events that represent fear / risk-off  → bullish gold
RISK_OFF_KEYWORDS = frozenset({
    "initial jobless claims", "jobless claims", "unemployment rate",
    "manufacturing pmi",  # contraction signals risk-off
})


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class DeviationResult:
    raw_score: float           # currency-directional score [-1, +1]
    xauusd_score: float        # gold-directional score [-1, +1]
    confidence: float          # 0 = no data, 0.6 = vs previous, 0.9 = vs forecast
    actual_parsed: Optional[float]
    forecast_parsed: Optional[float]
    previous_parsed: Optional[float]
    direction_multiplier: float
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ff_number(value: Optional[str]) -> Tuple[Optional[float], bool]:
    """
    Parse a Forex Factory numeric string into (float, is_percentage).

    Handled formats
    ───────────────
    "256K"     → 256_000
    "3.2%"     → 3.2, is_pct=True
    "-1.5B"    → -1_500_000_000
    "0.50|0.25"→ 0.50  (pipe = revised|previous; we take the first)
    "<-0.1"    → -0.1  (inequality prefix stripped)
    "TBD" / "" → None
    """
    if not value:
        return None, False

    raw = value.strip()
    if not raw or raw.upper() in ("TBD", "N/A", "-", "—", "NULL", "NONE"):
        return None, False

    # Pipe-separated → take the first (most recent / revised) value
    if "|" in raw:
        raw = raw.split("|")[0].strip()

    # Strip inequality / approximation prefixes
    raw = raw.lstrip("<>≤≥~≈ ")

    is_pct = "%" in raw
    raw = raw.replace("%", "").replace(",", "").strip()

    multiplier = 1.0
    suffix = raw[-1].upper() if raw else ""
    if suffix == "K":
        multiplier, raw = 1_000, raw[:-1]
    elif suffix == "M":
        multiplier, raw = 1_000_000, raw[:-1]
    elif suffix == "B":
        multiplier, raw = 1_000_000_000, raw[:-1]
    elif suffix == "T":
        multiplier, raw = 1_000_000_000_000, raw[:-1]

    try:
        return float(raw) * multiplier, is_pct
    except ValueError:
        logger.debug("Cannot parse FF number: %r", value)
        return None, False


def _parse_usual_effect(usual_effect: Optional[str]) -> int:
    """
    Returns:
        +1  if "Actual greater than Forecast is good for currency"
        -1  if "Actual less than Forecast is good for currency"
         0  if unknown / missing
    """
    if not usual_effect:
        return 0
    text = usual_effect.lower()
    # Explicit direction words
    if any(w in text for w in ("greater", "higher", "more", "above")):
        return +1
    if any(w in text for w in ("less", "lower", "fewer", "below")):
        return -1
    return 0


def _event_type_override(
    event_name: Optional[str],
) -> Optional[float]:
    """
    Check if the event name triggers a gold-direction override.

    Returns a direction multiplier (+1 or -1) if an override applies,
    None if we should fall through to the normal currency-direction table.
    """
    if not event_name:
        return None

    name = event_name.lower()

    if any(kw in name for kw in INFLATION_KEYWORDS):
        # Inflation beat → gold bullish (regardless of currency)
        return +1.0

    if any(kw in name for kw in USD_HAWKISH_KEYWORDS):
        # Strong payrolls / Fed hawkish → gold bearish
        return -1.0

    if any(kw in name for kw in RISK_OFF_KEYWORDS):
        # Bad jobs data → risk-off → gold bullish
        return +0.6  # softer override

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_deviation_score(
    actual:       Optional[str],
    forecast:     Optional[str],
    previous:     Optional[str],
    usual_effect: Optional[str],
    currency:     Optional[str],
    event_name:   Optional[str],
) -> DeviationResult:
    """
    Full quantitative deviation pipeline.

    Returns a DeviationResult with both a currency-level score and a
    XAUUSD-directional score.
    """

    # ── 1. Parse numbers ──────────────────────────────────────────────────────
    actual_val,   _ = _parse_ff_number(actual)
    forecast_val, _ = _parse_ff_number(forecast)
    previous_val, _ = _parse_ff_number(previous)

    # ── 2. Raw percentage deviation ──────────────────────────────────────────
    raw_deviation = 0.0
    confidence    = 0.0
    comparison    = "none"

    if actual_val is not None and forecast_val is not None:
        denom = abs(forecast_val) if forecast_val != 0 else 1.0
        raw_deviation = (actual_val - forecast_val) / denom
        confidence    = 0.9
        comparison    = "actual_vs_forecast"

    elif actual_val is not None and previous_val is not None:
        denom = abs(previous_val) if previous_val != 0 else 1.0
        raw_deviation = (actual_val - previous_val) / denom
        confidence    = 0.6
        comparison    = "actual_vs_previous"

    elif actual_val is not None:
        # Actual present but no baseline → signal magnitude unknown
        raw_deviation = 0.0
        confidence    = 0.1
        comparison    = "actual_only"

    else:
        # Upcoming event: no actual yet → deviation layer is silent
        return DeviationResult(
            raw_score=0.0,
            xauusd_score=0.0,
            confidence=0.0,
            actual_parsed=None,
            forecast_parsed=forecast_val,
            previous_parsed=previous_val,
            direction_multiplier=0.0,
            reasoning="No actual value – upcoming event",
        )

    # ── 3. Normalise with tanh  ───────────────────────────────────────────────
    # Scale factor 2.0: 50% miss → ≈0.46, 100% → ≈0.76, 200% → ≈0.96
    normalised = math.tanh(raw_deviation * 2.0)

    # ── 4. Apply usual_effect direction ──────────────────────────────────────
    effect_dir = _parse_usual_effect(usual_effect)
    if effect_dir == 0:
        effect_dir = +1  # Default assumption: bigger = better for currency

    currency_score = normalised * effect_dir  # currency-level direction

    # ── 5. Event-type override ────────────────────────────────────────────────
    override = _event_type_override(event_name)

    if override is not None:
        # Inflation / hawkish overrides bypass the currency table
        # and directly set gold direction from the deviation magnitude
        xauusd_score       = abs(currency_score) * override
        direction_multiplier = override
        note = (
            f"Event-type override [{event_name}] → gold_dir={override:+.1f} | "
            f"deviation={raw_deviation:.4f} ({comparison})"
        )
    else:
        # ── 6. Map currency to XAUUSD direction ───────────────────────────────
        cur = (currency or "").upper().strip()
        direction_multiplier = CURRENCY_GOLD_DIRECTION.get(cur, 0.20)
        xauusd_score = currency_score * direction_multiplier
        note = (
            f"Currency [{cur}] gold_dir={direction_multiplier:+.2f} | "
            f"deviation={raw_deviation:.4f} ({comparison}) | "
            f"effect_dir={effect_dir:+d}"
        )

    # ── Clip final values to [-1, +1] ─────────────────────────────────────────
    xauusd_score  = max(-1.0, min(1.0, xauusd_score))
    currency_score = max(-1.0, min(1.0, currency_score))

    return DeviationResult(
        raw_score=currency_score,
        xauusd_score=xauusd_score,
        confidence=confidence,
        actual_parsed=actual_val,
        forecast_parsed=forecast_val,
        previous_parsed=previous_val,
        direction_multiplier=direction_multiplier,
        reasoning=note,
    )
