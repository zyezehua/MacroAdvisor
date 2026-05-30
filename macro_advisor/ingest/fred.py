"""FRED adapter using the keyless CSV endpoint.

``https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`` returns the full
history without an API key, which keeps the project dependency-free of credentials.
Yields are returned as a single ``value`` column (percent, as published).
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from macro_advisor.ingest.base import PullResult

log = logging.getLogger(__name__)

SOURCE = "fred"
_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT = 30


def fetch(series_id: str, start: str | None = None, end: str | None = None) -> PullResult:
    """Fetch one FRED series as a daily/observed ``value`` frame."""
    params = {"id": series_id}
    if start:
        params["cosd"] = start
    if end:
        params["coed"] = end
    try:
        resp = requests.get(_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # CSV has columns: DATE (or 'observation_date'), <SERIES_ID>
        date_col = df.columns[0]
        val_col = df.columns[1]
        df = df.rename(columns={date_col: "date", val_col: "value"})
        df["date"] = pd.to_datetime(df["date"])
        # FRED encodes missing observations as '.'
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).set_index("date").sort_index()
        if df.empty:
            return PullResult(series_id, SOURCE, "series", df=None, status="empty",
                              message="no observations", freq="D")
        return PullResult(series_id, SOURCE, "series", df=df, status="ok", freq="D")
    except Exception as exc:
        log.warning("fred fetch failed for %s: %s", series_id, exc)
        return PullResult(series_id, SOURCE, "series", df=None, status="error",
                          message=str(exc), freq="D")
