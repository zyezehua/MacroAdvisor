"""Shared fixtures for Phase-1 signal/stress tests.

Builds a ``MarketStore`` backed entirely by synthetic parquet + SQLite under ``tmp_path``,
with no network. Relies on ``Config.path`` joining ``REPO_ROOT / storage[key]``: because a
join with an absolute path discards the left operand, pointing the storage keys at absolute
tmp paths fully redirects the store off the repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.config import Config
from macro_advisor.data import MarketStore
from macro_advisor.storage import ParquetCache, ProvenanceDB

_SIGNALS = {"lookback_days": 60, "squash_k": 1.5, "neutral_band": 0.10}
_STRESS = {
    "weights": {"volatility": 0.25, "credit": 0.20, "rates": 0.20,
                "momentum": 0.15, "breadth": 0.10, "cross_asset": 0.10},
    "bands": {"calm": 30, "normal": 55, "elevated": 70, "stressed": 85},
    "logistic_k": 1.0,
}


def bdays(n: int, end: str = "2026-05-29") -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def price_frame(values: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """OHLCV frame from a close path (flat OHLC around close, valid bounds)."""
    close = pd.Series(values, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "adj_close": close, "volume": 1_000_000,
    }, index=idx)


def value_frame(values: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture
def store_factory(tmp_path):
    """Return a builder: prices={sym: OHLCV df}, series={id: value df}, flags=[...] -> MarketStore.

    Each call gets its own subdirectory so repeated builds in one test stay isolated.
    """
    counter = {"n": 0}

    def _build(prices: dict | None = None, series: dict | None = None,
               flags: list | None = None) -> MarketStore:
        counter["n"] += 1
        root = tmp_path / f"build{counter['n']}"
        pdir, sdir = root / "prices", root / "series"
        dbp = root / "ma.sqlite"
        cache = ParquetCache(pdir, sdir)
        for sym, df in (prices or {}).items():
            cache.write(sym, df, kind="price")
        for sid, df in (series or {}).items():
            cache.write(sid, df, kind="series")
        db = ProvenanceDB(dbp)
        for f in (flags or []):
            db.raise_flag(key=f["key"], code=f.get("code", "X"),
                          severity=f["severity"], detail=f.get("detail"))
        db.close()
        cfg = Config(
            settings={
                "storage": {"parquet_dir": str(pdir), "series_dir": str(sdir),
                            "db_path": str(dbp)},
                "signals": _SIGNALS, "stress": _STRESS,
            },
            universe={},
        )
        return MarketStore(cfg)
    return _build
