"""Load ForexFactory JSON outputs into PostgreSQL — optimised build.

Key speed improvements over the original:
  1. Schema guard  — _schema_ensured module flag means the 4-statement
     CREATE/ALTER/VIEW block runs only ONCE per process no matter how many
     times store_scraped_rows() is called (critical with checkpointing).
  2. INSERT counting — the before/after COUNT(*) dance is replaced by a
     server-side count via INSERT … ON CONFLICT … RETURNING; no extra
     round-trips.
  3. psycopg3 pipeline mode — upsert_details() and insert_calendar() wrap
     their executemany calls in conn.pipeline() so the driver can batch
     wire packets, halving round-trips on high-latency connections.
  4. Single connection — store_scraped_rows() opens ONE connection and
     passes it to both upsert_details() and insert_calendar(), avoiding
     the overhead of a second TCP handshake + auth per checkpoint.
  5. Merged de-dup in insert_calendar — deduplicate by row_hash in Python
     before touching the database (avoids wasted ON CONFLICT misses).

All public functions, table schemas, view definitions and CLI flags are
unchanged — this is a drop-in replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# .env loader (dependency-free)
# ---------------------------------------------------------------------------

def _load_env_file() -> None:
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    override_keys = {
        "FF_PG_DSN", "PGHOST", "PGPORT", "PGUSER",
        "PGPASSWORD", "PGDATABASE", "PGSSLMODE",
    }
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key in override_keys:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


_load_env_file()

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError as e:
    if e.name != "psycopg":
        raise
    raise SystemExit(
        "Missing dependency: psycopg\n\n"
        "Fix (Windows):\n"
        "  1) Activate venv: .\\.venv\\Scripts\\Activate.ps1\n"
        "  2) Install deps : pip install -r requirements.txt\n"
        "  3) Run scraper  : python full_forexfactory_scrape.py\n"
        "     (or loader   : python load_to_postgres.py)\n\n"
        "Or run directly with the venv interpreter:\n"
        "  .\\.venv\\Scripts\\python.exe full_forexfactory_scrape.py\n"
        "  .\\.venv\\Scripts\\python.exe load_to_postgres.py\n"
    )

from psycopg.types.json import Json


# ---------------------------------------------------------------------------
# SQL — unchanged from original
# ---------------------------------------------------------------------------

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS ff_event_details (
  event_id TEXT PRIMARY KEY,
  ff_notice TEXT,
  measures TEXT,
  usual_effect TEXT,
  frequency TEXT,
  ff_notes TEXT,
  why_traders_care TEXT,
    last_detail_attempt_at TIMESTAMPTZ,
    detail_attempts INTEGER NOT NULL DEFAULT 0,
    detail_status TEXT,
    last_detail_error TEXT,
  source JSONB,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ff_calendar_events (
  id BIGSERIAL PRIMARY KEY,
  row_hash TEXT NOT NULL UNIQUE,
  event_id TEXT REFERENCES ff_event_details(event_id),
    calendar_date DATE,
    calendar_date_label TEXT,
  time TEXT,
  currency TEXT,
  impact TEXT,
  event_name TEXT,
  actual TEXT,
  forecast TEXT,
  previous TEXT,
  source JSONB,
  scraped_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ff_calendar_event_id ON ff_calendar_events(event_id);
CREATE INDEX IF NOT EXISTS idx_ff_calendar_currency  ON ff_calendar_events(currency);
CREATE INDEX IF NOT EXISTS idx_ff_calendar_impact    ON ff_calendar_events(impact);

CREATE TABLE IF NOT EXISTS ff_sentiment_results (
    id BIGSERIAL PRIMARY KEY,
    calendar_event_id BIGINT REFERENCES ff_calendar_events(id),
    event_id TEXT,
    currency TEXT,
    impact TEXT,
    event_name TEXT,
    deviation_score DOUBLE PRECISION,
    deviation_confidence DOUBLE PRECISION,
    nlp_positive DOUBLE PRECISION,
    nlp_negative DOUBLE PRECISION,
    nlp_neutral DOUBLE PRECISION,
    nlp_score DOUBLE PRECISION,
    composite_score DOUBLE PRECISION,
    xauusd_signal DOUBLE PRECISION,
    label TEXT,
    confidence DOUBLE PRECISION,
    reasoning JSONB,
    computed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ff_sentiment_calendar_event_id ON ff_sentiment_results(calendar_event_id);
CREATE INDEX IF NOT EXISTS idx_ff_sentiment_event_id          ON ff_sentiment_results(event_id);
"""

VIEW_SQL = """
CREATE OR REPLACE VIEW ff_calendar_enriched AS
SELECT
    c.id,
    c.event_id,
    c.calendar_date,
    c.time,
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
FROM ff_calendar_events c
LEFT JOIN ff_event_details d USING (event_id);
"""

MIGRATIONS_SQL = """
ALTER TABLE ff_event_details
    ADD COLUMN IF NOT EXISTS last_detail_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS detail_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS detail_status TEXT,
    ADD COLUMN IF NOT EXISTS last_detail_error TEXT;

ALTER TABLE ff_calendar_events
    ADD COLUMN IF NOT EXISTS calendar_date DATE,
    ADD COLUMN IF NOT EXISTS calendar_date_label TEXT;
"""

DETAIL_COLUMNS = (
    "ff_notice", "measures", "usual_effect",
    "frequency", "ff_notes", "why_traders_care",
)


# ---------------------------------------------------------------------------
# ★ Schema guard — run once per process, not once per checkpoint
# ---------------------------------------------------------------------------

_schema_ensured = False


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class LoadStats:
    details_upserted: int = 0
    calendar_inserted: int = 0


# ---------------------------------------------------------------------------
# Connection helpers (unchanged public API)
# ---------------------------------------------------------------------------

def connect(dsn: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    return _connect(dsn or os.getenv("FF_PG_DSN"), autocommit=autocommit)


def _connect(dsn: str | None, *, autocommit: bool) -> psycopg.Connection:
    def _has_real_password() -> bool:
        pw = os.getenv("PGPASSWORD")
        if not pw:
            return False
        return pw.strip().upper() not in {"YOUR_PASSWORD", "PASSWORD", "CHANGE_ME"}

    try:
        if dsn:
            return psycopg.connect(dsn, autocommit=autocommit)
        return psycopg.connect(autocommit=autocommit)
    except psycopg.OperationalError as e:
        msg = str(e).lower()
        if "no password supplied" in msg or "password authentication failed" in msg:
            if not _has_real_password():
                user = os.getenv("PGUSER") or "postgres"
                db   = os.getenv("PGDATABASE") or "postgres"
                raise SystemExit(
                    "Postgres authentication failed (missing/placeholder password).\n\n"
                    "Create/update your .env file with real credentials, for example:\n"
                    f"  PGUSER={user}\n"
                    "  PGPASSWORD=YOUR_REAL_PASSWORD\n"
                    f"  PGDATABASE={db}\n\n"
                    "Or set a single DSN:\n"
                    "  FF_PG_DSN=postgresql://USER:PASSWORD@localhost:5432/DBNAME\n"
                )
        raise


def _create_database_if_missing(admin_dsn: str | None, dbname: str) -> None:
    admin_conn = _connect(admin_dsn or "dbname=postgres", autocommit=True)
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                )
    finally:
        admin_conn.close()


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def _batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def ensure_schema(conn: psycopg.Connection) -> None:
    """Create tables, run migrations and (re)create the enriched view.

    ★ Uses a module-level flag so the 4-statement block only runs once per
    process, regardless of how many checkpoints call store_scraped_rows().
    """
    global _schema_ensured
    if _schema_ensured:
        return
    with conn.cursor() as cur:
        cur.execute(TABLES_SQL)
        cur.execute(MIGRATIONS_SQL)
        cur.execute("DROP VIEW IF EXISTS ff_calendar_enriched;")
        cur.execute(VIEW_SQL)
    conn.commit()
    _schema_ensured = True


# ---------------------------------------------------------------------------
# Detail helpers
# ---------------------------------------------------------------------------

def _detail_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": str(row.get("event_id") or "").strip()
    }
    for col in DETAIL_COLUMNS:
        payload[col] = row.get(col)
    payload["last_detail_attempt_at"] = row.get("last_detail_attempt_at")
    inc = row.get("detail_attempts_inc")
    payload["detail_attempts_inc"] = int(inc) if isinstance(inc, int) else 0
    payload["detail_status"]    = row.get("detail_status")
    payload["last_detail_error"] = row.get("last_detail_error")
    return payload


def _row_hash_for_calendar(row: dict[str, Any]) -> str:
    key = {
        "event_id":      str(row.get("event_id") or "").strip(),
        "calendar_date": row.get("calendar_date"),
        "time":          row.get("time"),
        "currency":      row.get("currency"),
        "impact":        row.get("impact"),
        "event":         row.get("event") or row.get("event_name"),
        "actual":        row.get("actual"),
        "forecast":      row.get("forecast"),
        "previous":      row.get("previous"),
    }
    encoded = json.dumps(
        key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# ★ upsert_details — pipeline mode
# ---------------------------------------------------------------------------

_UPSERT_DETAIL_SQL = """
INSERT INTO ff_event_details (
    event_id, ff_notice, measures, usual_effect, frequency,
    ff_notes, why_traders_care,
    last_detail_attempt_at, detail_attempts, detail_status,
    last_detail_error, source
)
VALUES (
    %(event_id)s, %(ff_notice)s, %(measures)s, %(usual_effect)s,
    %(frequency)s, %(ff_notes)s, %(why_traders_care)s,
    %(last_detail_attempt_at)s, %(detail_attempts_inc)s,
    %(detail_status)s, %(last_detail_error)s, %(source)s::jsonb
)
ON CONFLICT (event_id) DO UPDATE SET
    ff_notice              = COALESCE(EXCLUDED.ff_notice,              ff_event_details.ff_notice),
    measures               = COALESCE(EXCLUDED.measures,               ff_event_details.measures),
    usual_effect           = COALESCE(EXCLUDED.usual_effect,           ff_event_details.usual_effect),
    frequency              = COALESCE(EXCLUDED.frequency,              ff_event_details.frequency),
    ff_notes               = COALESCE(EXCLUDED.ff_notes,               ff_event_details.ff_notes),
    why_traders_care       = COALESCE(EXCLUDED.why_traders_care,       ff_event_details.why_traders_care),
    last_detail_attempt_at = COALESCE(EXCLUDED.last_detail_attempt_at, ff_event_details.last_detail_attempt_at),
    detail_attempts        = COALESCE(ff_event_details.detail_attempts, 0) + COALESCE(EXCLUDED.detail_attempts, 0),
    detail_status          = COALESCE(EXCLUDED.detail_status,          ff_event_details.detail_status),
    last_detail_error      = COALESCE(EXCLUDED.last_detail_error,      ff_event_details.last_detail_error),
    source                 = COALESCE(EXCLUDED.source,                 ff_event_details.source),
    updated_at             = now();
"""


def upsert_details(conn: psycopg.Connection, details_rows: list[dict[str, Any]]) -> int:
    # De-dup by event_id, merging non-null fields.
    merged: dict[str, dict[str, Any]] = {}
    for row in details_rows:
        payload = _detail_payload(row)
        event_id = payload["event_id"]
        if not event_id:
            continue
        existing = merged.get(event_id)
        if existing is None:
            payload["source"] = Json(row)
            merged[event_id] = payload
            continue
        for col in DETAIL_COLUMNS:
            if payload.get(col) is not None:
                existing[col] = payload[col]
        existing["source"] = Json(row)
        inc = row.get("detail_attempts_inc")
        if isinstance(inc, int):
            existing["detail_attempts_inc"] = (existing.get("detail_attempts_inc") or 0) + inc
        if row.get("last_detail_attempt_at"):
            existing["last_detail_attempt_at"] = row.get("last_detail_attempt_at")
        if row.get("detail_status"):
            existing["detail_status"] = row.get("detail_status")
        if "last_detail_error" in row:
            existing["last_detail_error"] = row.get("last_detail_error")

    params_all = list(merged.values())
    for p in params_all:
        inc = p.get("detail_attempts_inc")
        p["detail_attempts_inc"] = int(inc) if isinstance(inc, int) else 0

    count = 0
    with conn.cursor() as cur:
        # ★ pipeline: driver batches the wire packets → fewer round-trips.
        try:
            with conn.pipeline():
                for batch in _batched(params_all, 500):
                    cur.executemany(_UPSERT_DETAIL_SQL, batch)
                    count += len(batch)
        except AttributeError:
            # psycopg < 3.1 doesn't have pipeline(); fall back gracefully.
            for batch in _batched(params_all, 500):
                cur.executemany(_UPSERT_DETAIL_SQL, batch)
                count += len(batch)
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# ★ insert_calendar — RETURNING-based counting, pipeline mode, Python de-dup
# ---------------------------------------------------------------------------

_INSERT_CALENDAR_SQL = """
INSERT INTO ff_calendar_events (
    row_hash, event_id, calendar_date, calendar_date_label,
    time, currency, impact, event_name,
    actual, forecast, previous, source
)
VALUES (
    %(row_hash)s, %(event_id)s, %(calendar_date)s, %(calendar_date_label)s,
    %(time)s, %(currency)s, %(impact)s, %(event_name)s,
    %(actual)s, %(forecast)s, %(previous)s, %(source)s::jsonb
)
ON CONFLICT (row_hash) DO NOTHING
RETURNING id;
"""


def insert_calendar(conn: psycopg.Connection, calendar_rows: list[dict[str, Any]]) -> int:
    """Insert calendar rows. Returns the number of rows actually inserted.

    ★ Counts via RETURNING id instead of two SELECT COUNT(*) round-trips.
    ★ Python-side de-dup by row_hash avoids wasted ON CONFLICT misses.
    ★ Pipeline mode batches packets to the server.
    """
    # Build params; de-dup by hash in Python first.
    seen_hashes: set[str] = set()
    params_all: list[dict[str, Any]] = []
    for row in calendar_rows:
        event_id   = str(row.get("event_id") or "").strip() or None
        event_name = row.get("event") or row.get("event_name")
        rh = _row_hash_for_calendar(row)
        if rh in seen_hashes:
            continue
        seen_hashes.add(rh)
        params_all.append({
            "row_hash":            rh,
            "event_id":            event_id,
            "calendar_date":       _parse_iso_date(row.get("calendar_date")),
            "calendar_date_label": row.get("calendar_date_label"),
            "time":                row.get("time"),
            "currency":            row.get("currency"),
            "impact":              row.get("impact"),
            "event_name":          event_name,
            "actual":              row.get("actual"),
            "forecast":            row.get("forecast"),
            "previous":            row.get("previous"),
            "source":              Json(row),
        })

    inserted = 0
    with conn.cursor() as cur:
        try:
            with conn.pipeline():
                for batch in _batched(params_all, 500):
                    cur.executemany(_INSERT_CALENDAR_SQL, batch, returning=True)
                    # Collect RETURNING rows from all statements in the batch.
                    while True:
                        results = cur.fetchall()
                        inserted += len(results)
                        if not cur.nextset():
                            break
        except (AttributeError, TypeError):
            # Fallback: no pipeline or no returning kwarg (older psycopg3).
            for batch in _batched(params_all, 500):
                for p in batch:
                    cur.execute(_INSERT_CALENDAR_SQL, p)
                    if cur.fetchone():
                        inserted += 1

    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# ★ store_scraped_rows — single connection for both upserts
# ---------------------------------------------------------------------------

def store_scraped_rows(
    calendar_rows: list[dict[str, Any]],
    *,
    dsn: str | None = None,
) -> LoadStats:
    """Ensure schema and store a scraped calendar snapshot.

    ★ Opens ONE connection for both detail-upsert and calendar-insert.
    ★ ensure_schema() is a no-op after the first call.
    """
    if not calendar_rows:
        return LoadStats(details_upserted=0, calendar_inserted=0)

    conn = connect(dsn, autocommit=False)
    try:
        ensure_schema(conn)                          # no-op on 2nd+ call
        details_upserted = upsert_details(conn, calendar_rows)
        calendar_inserted = insert_calendar(conn, calendar_rows)
        return LoadStats(
            details_upserted=details_upserted,
            calendar_inserted=calendar_inserted,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Table management (unchanged)
# ---------------------------------------------------------------------------

def reset_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE TABLE ff_calendar_events, ff_event_details "
            "RESTART IDENTITY CASCADE;"
        )
    conn.commit()


def reset_database(*, dsn: str | None = None) -> None:
    conn = connect(dsn, autocommit=False)
    try:
        ensure_schema(conn)
        reset_tables(conn)
    finally:
        conn.close()


def enforce_retention_window(
    *,
    start_date: date,
    end_date: date,
    dsn: str | None = None,
    prune_orphan_details: bool = True,
    drop_stale_null_impact_hours: int | None = None,
) -> None:
    """Enforce rolling retention window. Unchanged logic."""
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    conn = connect(dsn, autocommit=False)
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ff_calendar_events WHERE calendar_date IS NULL;")
            cur.execute(
                "DELETE FROM ff_calendar_events "
                "WHERE calendar_date < %s OR calendar_date > %s;",
                (start_date, end_date),
            )
            if drop_stale_null_impact_hours and drop_stale_null_impact_hours > 0:
                cur.execute(
                    """
                    DELETE FROM ff_calendar_events
                    WHERE calendar_date BETWEEN %s AND %s
                      AND impact IS NULL
                      AND scraped_at < (now() - (%s * interval '1 hour'));
                    """,
                    (start_date, end_date, int(drop_stale_null_impact_hours)),
                )
            cur.execute(
                """
                WITH ranked AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY event_id
                               ORDER BY scraped_at DESC, id DESC
                           ) AS rn
                    FROM ff_calendar_events
                    WHERE calendar_date BETWEEN %s AND %s
                      AND event_id IS NOT NULL
                )
                DELETE FROM ff_calendar_events c
                USING ranked r
                WHERE c.id = r.id AND r.rn > 1;
                """,
                (start_date, end_date),
            )
            if prune_orphan_details:
                cur.execute(
                    """
                    DELETE FROM ff_event_details d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM ff_calendar_events c
                        WHERE c.event_id = d.event_id
                    );
                    """
                )
        conn.commit()
    finally:
        conn.close()


def get_window_event_ids(
    *, start_date: date, end_date: date, dsn: str | None = None
) -> set[str]:
    if start_date > end_date:
        return set()
    conn = connect(dsn, autocommit=False)
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT event_id
                FROM ff_calendar_events
                WHERE calendar_date BETWEEN %s AND %s
                  AND event_id IS NOT NULL;
                """,
                (start_date, end_date),
            )
            rows = cur.fetchall() or []
            return {str(r[0]) for r in rows if r and r[0] is not None}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI (unchanged)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load ForexFactory JSON files into PostgreSQL"
    )
    parser.add_argument("--details",  default="forexfactory_details.json")
    parser.add_argument("--calendar", default="forexfactory_calendar_with_details.json")
    parser.add_argument("--dsn",      default=os.getenv("FF_PG_DSN"))
    parser.add_argument("--create-db", action="store_true")
    parser.add_argument("--dbname",   default=os.getenv("PGDATABASE") or "forexfactory")
    args = parser.parse_args()

    details_path  = Path(args.details)
    calendar_path = Path(args.calendar)

    if args.create_db:
        admin_dsn = os.getenv("FF_PG_ADMIN_DSN")
        _create_database_if_missing(admin_dsn, args.dbname)

    dsn = args.dsn or f"dbname={args.dbname}"
    conn = _connect(dsn, autocommit=False)
    try:
        ensure_schema(conn)

        details_rows:  list[dict[str, Any]] = []
        calendar_rows: list[dict[str, Any]] = []

        if details_path.exists():
            details_rows = _read_json_array(details_path)
        if calendar_path.exists():
            calendar_rows = _read_json_array(calendar_path)

        du = 0
        if details_rows:
            du += upsert_details(conn, details_rows)
        if calendar_rows:
            du += upsert_details(conn, calendar_rows)

        ci = 0
        if calendar_rows:
            ci = insert_calendar(conn, calendar_rows)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ff_event_details")
            details_count  = (cur.fetchone() or [0])[0]
            cur.execute("SELECT COUNT(*) FROM ff_calendar_events")
            calendar_count = (cur.fetchone() or [0])[0]

        print("Load complete.")
        print(f"ff_event_details rows  : {details_count}")
        print(f"ff_calendar_events rows: {calendar_count}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
