"""Yahoo Finance adapter (via yfinance).

Returns a normalized OHLCV frame with a tz-naive DatetimeIndex and columns:
``open, high, low, close, adj_close, volume``. Yahoo's ``Adj Close`` accounts for
splits/dividends and is the series used for return computation downstream.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from macro_advisor.ingest.base import PullResult

log = logging.getLogger(__name__)

SOURCE = "yahoo"

_COLMAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten yfinance output (which may be multi-indexed) to lowercase OHLCV."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # yfinance returns MultiIndex columns when auto_adjust=False; collapse to level 0.
    if isinstance(df.columns, pd.MultiIndex):
        # columns like ('Adj Close', 'SPY') -> take the price-field level
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=_COLMAP)
    keep = [c for c in _COLMAP.values() if c in df.columns]
    df = df[keep]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    # If adj_close missing (auto_adjust path), fall back to close.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    return df.dropna(how="all").sort_index()


def fetch(symbol: str, start: str | None = None, end: str | None = None,
          interval: str = "1d") -> PullResult:
    """Fetch one symbol's history from Yahoo Finance."""
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = _normalize(raw, symbol)
        if df.empty:
            return PullResult(symbol, SOURCE, "price", df=None, status="empty",
                              message="no rows returned", freq=interval)
        return PullResult(symbol, SOURCE, "price", df=df, status="ok", freq=interval)
    except Exception as exc:  # network / parsing / delisted
        log.warning("yahoo fetch failed for %s: %s", symbol, exc)
        return PullResult(symbol, SOURCE, "price", df=None, status="error",
                          message=str(exc), freq=interval)
