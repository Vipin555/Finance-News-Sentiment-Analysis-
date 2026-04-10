-- =============================================================================
-- 001_sentiment.sql
-- Run this once against your existing forex database.
-- Idempotent: safe to re-run (uses IF NOT EXISTS / CREATE OR REPLACE).
-- =============================================================================

-- ── 1. Sentiment results table ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.ff_sentiment_results (
    id                   BIGSERIAL   PRIMARY KEY,

    -- FK back to the calendar row that was analysed
    calendar_event_id    BIGINT      UNIQUE
                         REFERENCES public.ff_calendar_events(id) ON DELETE CASCADE,

    -- Denormalised for fast querying without joins
    event_id             TEXT,
    currency             TEXT,
    impact               TEXT,
    event_name           TEXT,

    event_date           DATE,
    event_time           TEXT,
    -- TRUE = analyzed before actual was published (NLP-only leading indicator)
    -- FALSE = actual data was present at analysis time (deviation+NLP combined)
    pre_release          BOOLEAN  DEFAULT TRUE,

    -- ── Deviation layer (Layer 1) ─────────────────────────────────────────────
    deviation_score      FLOAT,   -- gold-directional deviation  ∈ [-1, +1]
    deviation_confidence FLOAT,   -- 0 = no actual, 0.6 = vs prev, 0.9 = vs forecast

    -- ── NLP layer (Layer 2) ───────────────────────────────────────────────────
    nlp_positive         FLOAT,   -- FinBERT softmax probability
    nlp_negative         FLOAT,
    nlp_neutral          FLOAT,
    nlp_score            FLOAT,   -- positive - negative  ∈ [-1, +1]

    -- ── Composite output (Layer 3) ────────────────────────────────────────────
    composite_score      FLOAT,   -- pre-clamp weighted combination
    xauusd_signal        FLOAT,   -- final gold signal  ∈ [-1, +1]
    label                TEXT     CHECK (label IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    confidence           FLOAT,   -- 0–1; higher = more certain

    -- Full reasoning payload for auditability / debugging
    reasoning            JSONB,

    computed_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_sr_event_id    ON public.ff_sentiment_results (event_id);
CREATE INDEX IF NOT EXISTS idx_sr_currency    ON public.ff_sentiment_results (currency);
CREATE INDEX IF NOT EXISTS idx_sr_label       ON public.ff_sentiment_results (label);
CREATE INDEX IF NOT EXISTS idx_sr_computed_at ON public.ff_sentiment_results (computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sr_signal      ON public.ff_sentiment_results (xauusd_signal);


-- ── 2. NOTIFY trigger ─────────────────────────────────────────────────────────
-- Fires after every INSERT on ff_calendar_events and sends the new row's id
-- on the channel the Python listener is watching.

CREATE OR REPLACE FUNCTION public.fn_notify_new_calendar_event()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_notify('new_calendar_event', NEW.id::TEXT);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_notify_new_calendar_event
    ON public.ff_calendar_events;

CREATE TRIGGER trg_notify_new_calendar_event
    AFTER INSERT ON public.ff_calendar_events
    FOR EACH ROW
    EXECUTE FUNCTION public.fn_notify_new_calendar_event();


-- ── 3. Enriched view with sentiment ───────────────────────────────────────────
-- Convenience view for the trading system: everything in one row.

CREATE OR REPLACE VIEW public.ff_full_enriched AS
SELECT
    e.id,
    e.event_id,
    e.time,
    e.currency,
    e.impact,
    e.event_name,
    e.actual,
    e.forecast,
    e.previous,
    e.scraped_at,
    -- Details
    d.why_traders_care,
    d.usual_effect,
    d.ff_notes,
    -- Sentiment
    s.xauusd_signal,
    s.label,
    s.confidence,
    s.deviation_score,
    s.nlp_score,
    s.composite_score,
    s.reasoning        AS sentiment_reasoning,
    s.computed_at      AS sentiment_computed_at
FROM      public.ff_calendar_events     e
LEFT JOIN public.ff_event_details       d USING (event_id)
LEFT JOIN public.ff_sentiment_results   s ON s.calendar_event_id = e.id;
