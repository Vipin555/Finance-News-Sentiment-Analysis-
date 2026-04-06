# FF Sentiment Analyzer

Real-time XAUUSD (Gold/USD) sentiment engine built on Forex Factory economic calendar data.

Reads from your existing `ff_calendar_events` / `ff_event_details` tables, runs a 3-layer scoring pipeline, and exposes live signals via FastAPI.

---

## Architecture

```
Scraper inserts row          PG NOTIFY trigger        POST /webhook
       │                            │                      │
       └────────────────────────────┴──────────────────────┘
                                    │
                              FastAPI app
                                    │
                          Analyzer orchestrator
                         ┌──────────┴──────────┐
                         │                      │
               Layer 1: Deviation       Layer 2: FinBERT NLP
               (actual vs forecast)     (event text sentiment)
                         │                      │
                         └──────────┬───────────┘
                                    │
                       Layer 3: Composite scorer
                       (60% deviation + 40% NLP)
                       (impact weight × time decay)
                                    │
                       ff_sentiment_results table
                       xauusd_signal ∈ [-1, +1]
```

---

## Quick start

```bash
# 1. Clone / copy the project
cd forex_sentiment

# 2. Install dependencies  (Python 3.11+ recommended)
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env → set DATABASE_URL to your PostgreSQL connection string

# 4. Run the DB migration (once)
psql $DATABASE_URL -f migrations/001_sentiment.sql

# 5. Start the API
python main.py
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:password@localhost:5432/forex` | asyncpg-compatible URL |
| `FINBERT_MODEL` | `ProsusAI/finbert` | HuggingFace model ID (downloaded on first run) |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `DEVIATION_WEIGHT` | `0.60` | Weight of quantitative layer in composite score |
| `NLP_WEIGHT` | `0.40` | Weight of NLP layer in composite score |
| `TIME_DECAY_HALF_LIFE_HOURS` | `4.0` | Hours until an event carries 50% of its weight |
| `HOST` | `0.0.0.0` | FastAPI bind address |
| `PORT` | `8000` | FastAPI port |

---

## API reference

### Trading signal (main endpoint)

```
GET /signal
```

Returns the current 4-hour aggregated XAUUSD signal.  
This is the endpoint your trading system should poll.

```json
{
  "xauusd_signal":     -0.3142,
  "label":             "BEARISH",
  "confidence":        0.74,
  "window_hours":      4,
  "event_count":       6,
  "high_impact_count": 2,
  "top_contributors": [
    {
      "event_name": "Nonfarm Payrolls",
      "currency":   "USD",
      "impact":     "red",
      "signal":     -0.72,
      "weight":     2.68,
      "decay":      0.89
    }
  ],
  "computed_at": "2024-11-01T14:00:00+00:00"
}
```

`xauusd_signal` interpretation:
- `+1.0` = strongly bullish gold
- `0.0`  = no directional bias
- `-1.0` = strongly bearish gold

---

### All windows at once

```
GET /signal/all
```

Returns signals for 1h, 4h, and 24h windows in a single call.

---

### Trigger analysis manually

```
POST /analyze/{calendar_event_id}
```

Run the full pipeline for a single row. Idempotent (upserts result).

```
POST /webhook/new-event
Content-Type: application/json

{"calendar_event_id": 42}
```

Trigger async analysis from an external scraper without NOTIFY.

---

### Backfill recent events

```
POST /analyze/recent
Content-Type: application/json

{"hours": 24}
```

---

## How the 3 layers work

### Layer 1 — Quantitative deviation scorer

1. Parses FF number strings (`256K`, `3.2%`, `0.50|0.25`, `<-0.1`, etc.)
2. Computes `(actual - forecast) / |forecast|`; falls back to `actual vs previous` if no forecast
3. Normalises with `tanh(deviation × 2)` → smooth `[-1, +1]`
4. Applies `usual_effect` direction (flips sign if "less than forecast is good")
5. Checks event-type override table:
   - CPI / PPI / PCE / inflation → gold bullish on a beat
   - NFP / FOMC / GDP → gold bearish on a beat (strong USD)
   - Jobless claims → soft gold bullish on a miss (risk-off)
6. Maps currency → XAUUSD direction (USD = -1.0, EUR = +0.45, JPY/CHF = +0.25 …)

### Layer 2 — FinBERT NLP

Builds a single text from: `event_name | why_traders_care | usual_effect | ff_notes | measures`  
Runs `ProsusAI/finbert` (110M parameter BERT variant trained on financial text).  
Output: `positive`, `negative`, `neutral` probabilities. Score = `positive - negative`.

This layer works even for upcoming events with no `actual` value yet.

### Layer 3 — Composite + aggregation

**Per-event composite:**
```
composite = deviation_score × 0.60  +  nlp_score × 0.40
```
Dynamic weighting: if `deviation.confidence < 0.5` (no actual or actual-only), the deviation weight is reduced and NLP takes over.

**Window aggregation:**
```
weight_i = time_decay(age_i) × impact_weight(impact_i) × confidence_i
signal   = Σ(signal_i × weight_i) / Σ(weight_i)
```
Time decay: `w = exp(-ln(2) / half_life × age_hours)`  
Impact weights: red=3.0, yellow=2.0, green=1.0

---

## Database objects created by migration

| Object | Type | Purpose |
|--------|------|---------|
| `ff_sentiment_results` | Table | Persisted per-event sentiment results |
| `ff_full_enriched` | View | Everything in one row (calendar + details + sentiment) |
| `trg_notify_new_calendar_event` | Trigger | Fires `NOTIFY new_calendar_event, '<id>'` on INSERT |
| `idx_sr_computed_at` | Index | Fast time-window queries |

---

## Scraper integration

### Option A — PostgreSQL NOTIFY (recommended)

The migration installs an `AFTER INSERT` trigger automatically.  
No scraper changes needed — just run the migration and start the analyzer.

### Option B — HTTP webhook

If your scraper runs in a separate network or can't rely on the listener:

```python
import httpx

def after_insert_hook(calendar_event_id: int):
    httpx.post(
        "http://analyzer:8000/webhook/new-event",
        json={"calendar_event_id": calendar_event_id},
    )
```

---

## Running the tests

```bash
# Install test dependency
pip install pytest pytest-asyncio

# Run all unit tests (no DB or model required)
pytest tests/ -v

# Run with real FinBERT model (downloads ~440MB on first run)
FINBERT_INTEGRATION=1 pytest tests/test_nlp_engine.py -v -k integration
```

Test coverage summary:

| File | Tests | Notes |
|------|-------|-------|
| `test_deviation_scorer.py` | 25 | Covers all number formats, direction logic, edge cases |
| `test_signal_aggregator.py` | 14 | Decay, impact weighting, label thresholds |
| `test_nlp_engine.py` | 15 | Mocked model; optional real-model integration test |
| `test_analyzer.py` | 9  | Full pipeline with mocked DB + FinBERT |

---

## Notes on GPU

Set `DEVICE=cuda` in `.env` if you have a CUDA-capable GPU.  
FinBERT fits in ~500 MB of VRAM (FP32). On CPU, inference is ~100–300ms per event.

The server is intentionally single-worker (`workers=1` in `main.py`) because the PyTorch model is not fork-safe. For horizontal scaling, run multiple instances behind a load balancer — each instance holds its own model copy.

---

## Interpreting the signal in your trading system

```python
import httpx

response = httpx.get("http://analyzer:8000/signal").json()

signal     = response["xauusd_signal"]    # float in [-1, +1]
confidence = response["confidence"]        # float in [0, 1]

# Example usage:
if confidence > 0.6 and signal > 0.2:
    # High-confidence bullish: consider long XAUUSD
    pass
elif confidence > 0.6 and signal < -0.2:
    # High-confidence bearish: consider short XAUUSD
    pass
else:
    # Low confidence or neutral: hold / reduce position size
    pass
```

Always combine with technical analysis — this signal is one input, not a standalone strategy.
