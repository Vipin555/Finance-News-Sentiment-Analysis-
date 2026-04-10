"""
Central configuration. All values can be overridden via environment variables
or a .env file in the project root.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:5432/forex"
    )

    # ── FinBERT ───────────────────────────────────────────────────────────────
    FINBERT_MODEL: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    # "cuda" to use GPU, "cpu" for CPU-only
    DEVICE: str = os.getenv("DEVICE", "cpu")
    FINBERT_MAX_TOKENS: int = 512
    FINBERT_BATCH_SIZE: int = 16

    # ── Scoring weights (must sum to 1.0) ────────────────────────────────────
    DEVIATION_WEIGHT: float = float(os.getenv("DEVIATION_WEIGHT", "0.60"))
    NLP_WEIGHT: float = float(os.getenv("NLP_WEIGHT", "0.40"))

    # ── Impact multipliers ───────────────────────────────────────────────────
    # red = high-impact (Fed, NFP), yellow = medium, green = low
    IMPACT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "red":     3.0,
        "yellow":  2.0,
        "green":   1.0,
        "unknown": 0.5,
    })

    # ── Time-decay (exponential; half-life in hours) ─────────────────────────
    # After 4 hours an event carries 50% of its original weight
    TIME_DECAY_HALF_LIFE_HOURS: float = float(
        os.getenv("TIME_DECAY_HALF_LIFE_HOURS", "4.0")
    )

    # ── Signal label thresholds ───────────────────────────────────────────────
    # Lowered from ±0.15 → ±0.08 to capture moderate but real directional bias
    BULLISH_THRESHOLD: float = 0.08
    BEARISH_THRESHOLD: float = -0.08

    # ── Storage filters (real-world two-phase approach) ───────────────────────
    # POST-release: event has actual data → use combined deviation+NLP confidence
    POST_RELEASE_MIN_CONFIDENCE: float = float(
        os.getenv("POST_RELEASE_MIN_CONFIDENCE", "0.45")
    )
    # PRE-release: event not yet released → use NLP certainty gated by impact
    # Only store directional signals where the impact justifies the uncertainty
    PRE_RELEASE_NLP_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "red":     0.10,   # High impact (NFP, CPI, Fed) – low bar
        "orange":  0.13,   # Medium-high
        "yellow":  0.20,   # Medium – needs stronger NLP signal
        "green":   0.99,   # Low impact – never store pre-release
        "unknown": 0.99,
    })

    # ── FastAPI server ────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ── PostgreSQL LISTEN/NOTIFY channel ─────────────────────────────────────
    PG_NOTIFY_CHANNEL: str = os.getenv("PG_NOTIFY_CHANNEL", "new_calendar_event")

    # ── Calendar-date-aware settings ─────────────────────────────────────────
    # Timezone offset of the FF calendar data (hours east of UTC).
    # Used when converting calendar_date + time text → UTC for time-decay.
    # e.g. 5.5 = IST (UTC+5:30), 0 = UTC, -5 = EST
    TIMEZONE_OFFSET_HOURS: float = float(
        os.getenv("TIMEZONE_OFFSET_HOURS", "0.0")
    )

    # How many hours old a sentiment result must be before we re-analyze it.
    # Re-analysis matters when an event's "actual" value arrives post-release.
    STALE_RESULT_HOURS: int = int(os.getenv("STALE_RESULT_HOURS", "1"))


settings = Settings()
