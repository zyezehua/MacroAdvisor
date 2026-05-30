"""Parquet cache for time-series data.

One parquet file per series. Prices live under ``prices/`` (OHLCV frames), macro
series under ``series/`` (single-column value frames). Writes merge with any existing
cached history so incremental pulls extend coverage without re-downloading everything.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(symbol: str) -> str:
    """Make a filesystem-safe filename from a ticker/series id (e.g. '^GSPC', 'DX-Y.NYB')."""
    return _SAFE.sub("_", symbol)


class ParquetCache:
    def __init__(self, prices_dir: str | Path, series_dir: str | Path):
        self.prices_dir = Path(prices_dir)
        self.series_dir = Path(series_dir)
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self.series_dir.mkdir(parents=True, exist_ok=True)

    # -- path helpers ----------------------------------------------------
    def _file(self, symbol: str, kind: str) -> Path:
        base = self.prices_dir if kind == "price" else self.series_dir
        return base / f"{_safe_name(symbol)}.parquet"

    def exists(self, symbol: str, kind: str) -> bool:
        return self._file(symbol, kind).exists()

    # -- read / write ----------------------------------------------------
    def read(self, symbol: str, kind: str = "price") -> pd.DataFrame | None:
        path = self._file(symbol, kind)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return df.sort_index()

    def write(self, symbol: str, df: pd.DataFrame, kind: str = "price") -> pd.DataFrame:
        """Merge ``df`` with any cached history and persist. Returns the merged frame."""
        if df is None or df.empty:
            return df
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        existing = self.read(symbol, kind)
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = df
        merged.to_parquet(self._file(symbol, kind))
        return merged
