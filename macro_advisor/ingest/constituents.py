"""Index constituent fetcher for the LIVE-signal universe.

S&P 500 and Nasdaq-100 members are read from Wikipedia. These lists are *current*
membership only and are therefore used strictly for live signal generation — never
fed into historical OOS backtests (which would introduce survivorship bias). If the
fetch fails, we fall back to the seed list in ``universe.yaml``.
"""
from __future__ import annotations

import logging

import pandas as pd

from macro_advisor.config import load_config

log = logging.getLogger(__name__)

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _clean(tickers) -> list[str]:
    out: list[str] = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper().replace(".", "-")  # BRK.B -> BRK-B (Yahoo convention)
        if t and t.isascii():
            out.append(t)
    return list(dict.fromkeys(out))


def _from_wikipedia(url: str, symbol_col_candidates: tuple[str, ...]) -> list[str]:
    tables = pd.read_html(url)
    for tbl in tables:
        cols = {str(c).strip().lower(): c for c in tbl.columns}
        for cand in symbol_col_candidates:
            if cand in cols:
                return _clean(tbl[cols[cand]].tolist())
    raise ValueError(f"no symbol column found in tables at {url}")


def sp500() -> list[str]:
    try:
        return _from_wikipedia(_SP500_URL, ("symbol", "ticker"))
    except Exception as exc:
        log.warning("sp500 constituent fetch failed: %s — using seed", exc)
        return _seed()


def nasdaq100() -> list[str]:
    try:
        return _from_wikipedia(_NDX_URL, ("ticker", "symbol"))
    except Exception as exc:
        log.warning("nasdaq100 constituent fetch failed: %s — using seed", exc)
        return _seed()


def _seed() -> list[str]:
    cfg = load_config()
    return list(cfg.universe.get("live_signal", {}).get("fallback_seed", []))


def live_universe() -> list[str]:
    """Union of S&P 500 and Nasdaq-100 members (Yahoo ticker convention)."""
    return list(dict.fromkeys(sp500() + nasdaq100()))
