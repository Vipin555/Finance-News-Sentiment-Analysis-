"""
FinBERT NLP engine (Layer 2).

Loads ProsusAI/finbert once at startup.  Exposes:
  - analyze(event_dict)        → NLPResult  (single event)
  - analyze_batch([event_dict])→ [NLPResult] (batch, more efficient)

Output labels from ProsusAI/finbert
────────────────────────────────────
  index 0  → positive
  index 1  → negative
  index 2  → neutral

score = positive_prob - negative_prob  ∈ [-1, +1]
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class NLPResult:
    positive:    float
    negative:    float
    neutral:     float
    score:       float   # positive - negative  ∈ [-1, +1]
    dominant:    str     # "positive" | "negative" | "neutral"
    text_used:   str     # first 200 chars of the input (for debugging)
    token_count: int


# ---------------------------------------------------------------------------
# FinBERT singleton
# ---------------------------------------------------------------------------

class _FinBERTEngine:
    """
    Lazy-loaded FinBERT singleton.
    Call .load() once during app startup; .analyze() / .analyze_batch() thereafter.
    """

    def __init__(self) -> None:
        self._model:     Optional[TFAutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer]                        = None

    # ── Startup ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return  # already loaded

        logger.info("Loading FinBERT model: %s", settings.FINBERT_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(settings.FINBERT_MODEL)
        # NOTE: This loads TensorFlow weights. If the model repo only provides
        # PyTorch weights, this will raise; in that case choose a TF-compatible
        # model or install torch and use from_pt conversion.
        self._model = TFAutoModelForSequenceClassification.from_pretrained(
            settings.FINBERT_MODEL
        )

        # TensorFlow will use GPU automatically if available.
        if settings.DEVICE == "cpu":
            logger.info("FinBERT loaded (TensorFlow). DEVICE=cpu (GPU usage not forced off).")
        else:
            logger.info("FinBERT loaded (TensorFlow).")

    # ── Text construction ──────────────────────────────────────────────────────

    @staticmethod
    def _build_text(
        event_name:      Optional[str],
        why_traders_care: Optional[str],
        usual_effect:    Optional[str],
        ff_notes:        Optional[str],
        measures:        Optional[str],
        ff_notice:       Optional[str],
    ) -> str:
        """
        Concatenate available text fields into a single FinBERT input string.

        Priority: event_name → why_traders_care → usual_effect → ff_notes → measures
        Fields are pipe-separated so the model can distinguish context boundaries.
        Each field is capped at a budget to keep total tokens ≤ 512.
        """
        parts: List[str] = []

        def _clean(s: Optional[str], max_chars: int) -> Optional[str]:
            if not s or not s.strip():
                return None
            return re.sub(r"\s+", " ", s.strip())[:max_chars]

        if t := _clean(event_name, 80):
            parts.append(t)
        if t := _clean(why_traders_care, 300):
            parts.append(t)
        if t := _clean(usual_effect, 120):
            parts.append(t)
        if t := _clean(ff_notes, 200):
            parts.append(t)
        if t := _clean(measures, 150):
            parts.append(t)
        if t := _clean(ff_notice, 100):
            parts.append(t)

        return " | ".join(parts) if parts else (event_name or "")

    # ── Low-level batch inference ──────────────────────────────────────────────

    def _score_texts(
        self, texts: List[str]
    ) -> List[Tuple[float, float, float]]:
        """Returns list of (positive, negative, neutral) probabilities."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("FinBERT not loaded. Call .load() first.")

        results: List[Tuple[float, float, float]] = []

        for i in range(0, len(texts), settings.FINBERT_BATCH_SIZE):
            batch = texts[i : i + settings.FINBERT_BATCH_SIZE]

            # Replace empty texts with a neutral placeholder to avoid tokenizer errors
            batch = [t if t.strip() else "no information available" for t in batch]

            inputs = self._tokenizer(
                batch,
                return_tensors="tf",
                truncation=True,
                max_length=settings.FINBERT_MAX_TOKENS,
                padding=True,
            )

            logits = self._model(**inputs).logits
            probs = tf.nn.softmax(logits, axis=-1).numpy().tolist()

            for p in probs:
                # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
                results.append((float(p[0]), float(p[1]), float(p[2])))

        return results

    # ── Public API ─────────────────────────────────────────────────────────────

    def _make_result(
        self, text: str, probs: Tuple[float, float, float]
    ) -> NLPResult:
        pos, neg, neu = probs
        score    = pos - neg
        dominant = max(
            [("positive", pos), ("negative", neg), ("neutral", neu)],
            key=lambda x: x[1],
        )[0]
        tokens = (
            self._tokenizer.tokenize(text) if self._tokenizer and text else []
        )
        return NLPResult(
            positive    = round(pos, 4),
            negative    = round(neg, 4),
            neutral     = round(neu, 4),
            score       = round(score, 4),
            dominant    = dominant,
            text_used   = text[:200],
            token_count = min(len(tokens), settings.FINBERT_MAX_TOKENS),
        )

    def analyze(
        self,
        event_name:       Optional[str] = None,
        why_traders_care: Optional[str] = None,
        usual_effect:     Optional[str] = None,
        ff_notes:         Optional[str] = None,
        measures:         Optional[str] = None,
        ff_notice:        Optional[str] = None,
    ) -> NLPResult:
        """Analyze a single event. All arguments are optional."""
        text  = self._build_text(
            event_name, why_traders_care, usual_effect, ff_notes, measures, ff_notice
        )
        probs = self._score_texts([text])[0]
        return self._make_result(text, probs)

    def analyze_batch(self, events: List[dict]) -> List[NLPResult]:
        """
        Analyze a list of event dicts from ff_calendar_enriched.
        Keys used: event_name, why_traders_care, usual_effect,
                   ff_notes, measures, ff_notice.
        """
        texts = [
            self._build_text(
                ev.get("event_name"),
                ev.get("why_traders_care"),
                ev.get("usual_effect"),
                ev.get("ff_notes"),
                ev.get("measures"),
                ev.get("ff_notice"),
            )
            for ev in events
        ]
        probs_list = self._score_texts(texts)
        return [
            self._make_result(text, probs)
            for text, probs in zip(texts, probs_list)
        ]


# Module-level singleton – import and use this everywhere
finbert = _FinBERTEngine()
