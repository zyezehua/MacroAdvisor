"""Shared helpers for signal modules: parameter access + common series lookups."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_advisor.data import MarketStore

# Broad-equity proxy preference: core-tier index first, then the tradable ETF.
_EQUITY_SYMBOLS = ("^GSPC", "SPY")
# SPDR sector ETFs used for breadth (present only in the backtest_equity tier).
SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")


@dataclass(frozen=True)
class Params:
    lookback: int
    squash_k: float
    neutral_band: float

    @classmethod
    def from_store(cls, store: MarketStore) -> "Params":
        s = store.cfg.signals
        return cls(
            lookback=int(s.get("lookback_days", 252)),
            squash_k=float(s.get("squash_k", 1.5)),
            neutral_band=float(s.get("neutral_band", 0.10)),
        )


def equity_price(store: MarketStore) -> tuple[pd.Series | None, str | None]:
    """Return (price_series, provenance_key) for the broad-equity proxy, or (None, None)."""
    for sym in _EQUITY_SYMBOLS:
        s = store.try_price(sym)
        if s is not None and not s.empty:
            return s, f"yahoo:{sym}"
    return None, None
