-- =============================================================================
-- 002_calendar_date.sql
-- Aligns the existing database with the updated schema:
--   • Adds calendar_date (DATE) to ff_calendar_events if not present
--   • Adds calendar_date_label (TEXT) to ff_calendar_events if not present
--   • Creates an index on calendar_date for fast date-filtering
--   • Recreates ff_calendar_enriched and ff_full_enriched views to expose
--     both new columns so the Python application can query by event date.
--
-- Idempotent: safe to re-run.
-- =============================================================================

-- ── 1. Add new columns (safe no-ops if they already exist) ───────────────────

ALTER TABLE public.ff_calendar_events
    ADD COLUMN IF NOT EXISTS calendar_date       DATE,
    ADD COLUMN IF NOT EXISTS calendar_date_label TEXT;


-- ── 2. Index on calendar_date for fast "today's events" queries ──────────────

CREATE INDEX IF NOT EXISTS idx_ff_calendar_date
    ON public.ff_calendar_events USING btree (calendar_date);


-- ── 3. Rebuild ff_calendar_enriched to expose the two new columns ────────────
-- Drop dependent views first to avoid column-rename conflicts, then recreate.

DROP VIEW IF EXISTS public.ff_full_enriched CASCADE;
DROP VIEW IF EXISTS public.ff_calendar_enriched CASCADE;

CREATE VIEW public.ff_calendar_enriched AS
SELECT
    c.id,
    c.event_id,
    c.calendar_date,
    c.calendar_date_label,
    c."time",
    c.currency,
    c.impact,
    c.event_name,
    c.actual,
    c.forecast,
    c.previous,
    d.ff_notice,
    d.measures,
    d.usual_effect,
    d.frequency,
    d.ff_notes,
    d.why_traders_care,
    c.scraped_at
FROM public.ff_calendar_events c
LEFT JOIN public.ff_event_details d USING (event_id);


-- ── 4. Rebuild ff_full_enriched (adds calendar_date + label alongside existing sentiment columns) ──

CREATE VIEW public.ff_full_enriched AS
SELECT
    e.id,
    e.event_id,
    e.calendar_date,
    e.calendar_date_label,
    e."time",
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
FROM      public.ff_calendar_events   e
LEFT JOIN public.ff_event_details     d USING (event_id)
LEFT JOIN public.ff_sentiment_results s ON s.calendar_event_id = e.id;
