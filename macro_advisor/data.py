"""Read-side market-data access for the signal layer.

``MarketStore`` is the single gateway signals use to read vetted history. It wraps the
Phase-0 :class:`ParquetCache` (price/series frames) and :class:`ProvenanceDB` (QA flags)
and enforces the project's core integrity guarantee:

    *No signal is computed on unreconciled data.*

Any series carrying an ``error``-severity QA flag is refused (``DataIntegrityError``);
``info``/``warn`` flags pass through unchanged. Symbols map to provenance keys exactly as
the ingest pipeline records them: Yahoo prices as ``yahoo:{symbol}``, Treasury tenors as
``treasury:{tenor}``, FRED series as ``fred:{series}``.
"""
from __future__ import annotations

import logging

import pandas as pd

from macro_advisor.config import Config, load_config
from macro_advisor.storage import ParquetCache, ProvenanceDB

log = logging.getLogger(__name__)

_PRICE_COLS = ("adj_close", "close")


class DataIntegrityError(RuntimeError):
    """Raised when a requested series carries an error-severity QA flag."""


class MissingSeriesError(KeyError):
    """Raised when a required series is not present in the cache."""


class MarketStore:
    """Read-only accessor over the parquet cache + provenance DB."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.cache = ParquetCache(self.cfg.path("parquet_dir"), self.cfg.path("series_dir"))
        self.db = ProvenanceDB(self.cfg.path("db_path"))

    # -- integrity -------------------------------------------------------
    def _gate(self, key: str) -> None:
        """Refuse to serve a series that has an error-severity QA flag."""
        errors = [f for f in self.db.flags(key) if f["severity"] == "error"]
        if errors:
            codes = ", ".join(f["code"] for f in errors)
            raise DataIntegrityError(f"{key}: error-severity QA flag(s): {codes}")

    @staticmethod
    def _price_col(df: pd.DataFrame) -> str:
        for c in _PRICE_COLS:
            if c in df.columns:
                return c
        return df.columns[0]

    # -- price (Yahoo) ---------------------------------------------------
    def frame(self, symbol: str) -> pd.DataFrame:
        """Full OHLCV frame for a Yahoo price symbol."""
        df = self.cache.read(symbol, kind="price")
        if df is None or df.empty:
            raise MissingSeriesError(f"no cached price for {symbol!r}")
        self._gate(f"yahoo:{symbol}")
        return df

    def price(self, symbol: str) -> pd.Series:
        """Adjusted-close (fallback close) price series for a Yahoo symbol."""
        df = self.frame(symbol)
        return df[self._price_col(df)].rename(symbol).dropna()

    # -- yields (Treasury curve) -----------------------------------------
    def yield_(self, tenor: str) -> pd.Series:
        """Treasury par-yield series (percent) for a tenor id, e.g. ``UST10Y``."""
        df = self.cache.read(tenor, kind="series")
        if df is None or df.empty:
            raise MissingSeriesError(f"no cached yield for {tenor!r}")
        self._gate(f"treasury:{tenor}")
        col = "value" if "value" in df.columns else df.columns[0]
        return df[col].rename(tenor).dropna()

    # -- optional FRED extras --------------------------------------------
    def fred(self, series: str) -> pd.Series | None:
        """Best-effort FRED series (credit OAS, real yields). ``None`` if absent."""
        df = self.cache.read(series, kind="series")
        if df is None or df.empty:
            return None
        self._gate(f"fred:{series}")
        col = "value" if "value" in df.columns else df.columns[0]
        return df[col].rename(series).dropna()

    # -- convenience helpers ---------------------------------------------
    def has_price(self, symbol: str) -> bool:
        return self.cache.exists(symbol, "price")

    def has_yield(self, tenor: str) -> bool:
        return self.cache.exists(tenor, "series")

    def try_price(self, symbol: str) -> pd.Series | None:
        """``price`` that returns ``None`` instead of raising on missing/blocked data."""
        try:
            return self.price(symbol)
        except (MissingSeriesError, DataIntegrityError) as exc:
            log.info("skip price %s: %s", symbol, exc)
            return None

    def try_yield(self, tenor: str) -> pd.Series | None:
        try:
            return self.yield_(tenor)
        except (MissingSeriesError, DataIntegrityError) as exc:
            log.info("skip yield %s: %s", tenor, exc)
            return None

    def provenance(self) -> list[dict]:
        return self.db.all_provenance()

    def flags(self, key: str | None = None) -> list[dict]:
        return self.db.flags(key)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MarketStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
