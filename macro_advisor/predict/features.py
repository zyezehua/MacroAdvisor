"""Causal per-(date, asset) feature panel for OOS prediction.

Two feature groups, both strictly backward-looking:
  * **market/macro** — every Phase-1 signal's score (volatility, credit, rates, breadth,
    dollar, …), identical across assets on a given date (the macro regime).
  * **asset-specific** — each ETF's own technicals (momentum, RSI, realized vol, distance
    from moving averages) computed from its own price via the Phase-1 ``transform`` helpers.

The result is a tidy long frame indexed by ``(date, symbol)`` whose columns are features. It is
the single feature source for both the per-asset return models and (its market columns) the
stress-path model. No forward information enters any column.
"""
from __future__ import annotations

import pandas as pd

from macro_advisor.data import MarketStore
from macro_advisor.signals import compute_all
from macro_advisor.signals import transform as tf

# default per-asset technical windows; overridden by cfg.predict["asset_feature_lookbacks"]
_DEFAULT_LOOKBACKS = (21, 63, 126)


def market_features(store: MarketStore) -> pd.DataFrame:
    """Wide frame of market/macro signal scores, indexed by date (one column per signal)."""
    signals = compute_all(store)
    cols = {name: sig.score for name, sig in signals.items()}
    return pd.DataFrame(cols).sort_index()


def asset_features(price: pd.Series, lookbacks=_DEFAULT_LOOKBACKS) -> pd.DataFrame:
    """Per-asset causal technicals from a single price series."""
    feats = {"rsi": (tf.wilder_rsi(price, 14) - 50.0) / 50.0}
    for w in lookbacks:
        feats[f"ret_{w}"] = tf.ret(price, w)
        feats[f"retz_{w}"] = tf.roll_z(tf.ret(price, 1), w)      # short-term return, z-scored
        feats[f"rvol_{w}"] = tf.realized_vol(price, w)
        feats[f"sma_gap_{w}"] = price / tf.sma(price, w) - 1.0   # distance above the MA
    return pd.DataFrame(feats).sort_index()


def build_panel(store: MarketStore, symbols: list[str],
                lookbacks=_DEFAULT_LOOKBACKS, min_coverage: float = 0.5) -> pd.DataFrame:
    """Assemble the long ``(date, symbol)`` feature panel for the given universe.

    Market features are broadcast to every asset and prefixed ``mkt_``; per-asset technicals are
    prefixed ``a_``. Feature *columns* whose non-null coverage is below ``min_coverage`` are
    dropped first — so a single short series (e.g. a FRED extra with only a few years in cache)
    can't gut the whole panel's history — then warm-up rows are dropped.
    """
    mkt = market_features(store).add_prefix("mkt_")
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        px = store.try_price(sym)
        if px is None or px.empty:
            continue
        a = asset_features(px, lookbacks).add_prefix("a_")
        joined = a.join(mkt, how="inner")
        joined.insert(0, "symbol", sym)
        joined.index.name = "date"
        frames.append(joined.reset_index())
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True).set_index(["date", "symbol"]).sort_index()
    feat = [c for c in panel.columns if c.startswith(("mkt_", "a_"))]
    keep = [c for c in feat if panel[c].notna().mean() >= min_coverage]
    return panel[keep].dropna()


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith(("mkt_", "a_"))]
