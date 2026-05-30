"""SQLite provenance / metadata store.

Tracks, for every series we cache, *where it came from*, *when it was pulled*, its
coverage, and any data-quality flags raised by the cross-check layer. This is the
audit trail that lets every downstream signal be traced back to vetted inputs.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provenance (
    key            TEXT PRIMARY KEY,   -- e.g. 'yahoo:SPY' or 'fred:DGS10'
    symbol         TEXT NOT NULL,
    source         TEXT NOT NULL,
    kind           TEXT NOT NULL,      -- 'price' | 'series'
    pull_ts        TEXT NOT NULL,      -- UTC ISO8601
    start_date     TEXT,
    end_date       TEXT,
    n_rows         INTEGER,
    freq           TEXT,
    status         TEXT NOT NULL,      -- 'ok' | 'empty' | 'error'
    message        TEXT
);

CREATE TABLE IF NOT EXISTS qa_flags (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    key       TEXT NOT NULL,
    raised_ts TEXT NOT NULL,
    code      TEXT NOT NULL,           -- e.g. 'PRICE_DIVERGENCE', 'STALE', 'THIN_HISTORY'
    severity  TEXT NOT NULL,           -- 'info' | 'warn' | 'error'
    detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_qa_key ON qa_flags(key);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProvenanceDB:
    """Thin SQLite wrapper for provenance + QA flags."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- provenance ------------------------------------------------------
    def record_pull(
        self,
        *,
        key: str,
        symbol: str,
        source: str,
        kind: str,
        status: str,
        start_date: str | None = None,
        end_date: str | None = None,
        n_rows: int | None = None,
        freq: str | None = None,
        message: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO provenance
                (key, symbol, source, kind, pull_ts, start_date, end_date,
                 n_rows, freq, status, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                symbol=excluded.symbol, source=excluded.source, kind=excluded.kind,
                pull_ts=excluded.pull_ts, start_date=excluded.start_date,
                end_date=excluded.end_date, n_rows=excluded.n_rows, freq=excluded.freq,
                status=excluded.status, message=excluded.message
            """,
            (key, symbol, source, kind, _utcnow(), start_date, end_date,
             n_rows, freq, status, message),
        )
        self._conn.commit()

    def get_provenance(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM provenance WHERE key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def all_provenance(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM provenance ORDER BY key"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- QA flags --------------------------------------------------------
    def clear_flags(self, key: str) -> None:
        self._conn.execute("DELETE FROM qa_flags WHERE key = ?", (key,))
        self._conn.commit()

    def raise_flag(
        self, *, key: str, code: str, severity: str, detail: dict | str | None = None
    ) -> None:
        if isinstance(detail, dict):
            detail = json.dumps(detail)
        self._conn.execute(
            "INSERT INTO qa_flags (key, raised_ts, code, severity, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, _utcnow(), code, severity, detail),
        )
        self._conn.commit()

    def flags(self, key: str | None = None) -> list[dict[str, Any]]:
        if key:
            rows = self._conn.execute(
                "SELECT * FROM qa_flags WHERE key = ? ORDER BY raised_ts DESC", (key,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM qa_flags ORDER BY raised_ts DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProvenanceDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
