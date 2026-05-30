"""Stooq adapter (keyless CSV) — used as an independent mirror for cross-checking.

Stooq uses lowercase symbols with a market suffix for US equities/ETFs, e.g.
``spy.us``. Index symbols differ (``^spx``), so Stooq is primarily a mirror for the
ETF universe where it overlaps cleanly with Yahoo.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from macro_advisor.ingest.base import PullResult

log = logging.getLogger(__name__)

SOURCE = "stooq"
_BASE = "https://stooq.com/q/d/l/"
_TIMEOUT = 30


def _to_stooq_symbol(symbol: str) -> str | None:
    """Map a Yahoo-style ticker to a Stooq symbol; return None if unsupported."""
    if symbol.startswith("^") or "-" in symbol or "=" in symbol:
        return None  # indices / FX crosses: skip mirror
    return f"{symbol.lower()}.us"


def fetch(symbol: str, start: str | None = None, end: str | None = None) -> PullResult:
    stooq_sym = _to_stooq_symbol(symbol)
    if stooq_sym is None:
        return PullResult(symbol, SOURCE, "price", df=None, status="empty",
                          message="no stooq mapping", freq="1d")
    params = {"s": stooq_sym, "i": "d"}
    if start:
        params["d1"] = start.replace("-", "")
    if end:
        params["d2"] = end.replace("-", "")
    try:
        resp = requests.get(_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "Date" not in df.columns or df.empty:
            return PullResult(symbol, SOURCE, "price", df=None, status="empty",
                              message="no rows", freq="1d")
        df = df.rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["adj_close"] = df.get("close")  # Stooq close is already adjusted
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                if c in df.columns]
        return PullResult(symbol, SOURCE, "price", df=df[keep], status="ok", freq="1d")
    except Exception as exc:
        log.warning("stooq fetch failed for %s: %s", symbol, exc)
        return PullResult(symbol, SOURCE, "price", df=None, status="error",
                          message=str(exc), freq="1d")
