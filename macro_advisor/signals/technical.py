"""Technical signals on the broad-equity proxy: trend, momentum, RSI, breadth.

Orientation: +1 = risk-on (uptrend / strong momentum / broad participation),
-1 = risk-off. All transforms are causal (see ``transform`` module).
"""
from __future__ import annotations

import pandas as pd

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals._common import SECTOR_ETFS, Params, equity_price
from macro_advisor.signals.base import build_signal, SignalResult


def trend(store: MarketStore) -> SignalResult | None:
    px, key = equity_price(store)
    if px is None:
        return None
    p = Params.from_store(store)
    sma50, sma200 = tf.sma(px, 50), tf.sma(px, 200)
    dist = (px / sma200 - 1.0)                      # fractional distance above 200d SMA
    score = tf.squash(tf.roll_z(dist, p.lookback), p.squash_k)
    golden = bool(sma50.dropna().iloc[-1] > sma200.dropna().iloc[-1]) if not sma200.dropna().empty else False
    last_dist = float(dist.dropna().iloc[-1]) * 100 if not dist.dropna().empty else float("nan")
    attribution = (
        f"price {last_dist:+.1f}% vs 200d SMA, "
        f"{'50d>200d (golden cross)' if golden else '50d<200d (death cross)'}"
    )
    return build_signal(name="trend", category="technical", score=score, raw=dist,
                        attribution=attribution, inputs=[key], neutral_band=p.neutral_band)


def momentum(store: MarketStore) -> SignalResult | None:
    px, key = equity_price(store)
    if px is None:
        return None
    p = Params.from_store(store)
    blended = (tf.ret(px, 21) + tf.ret(px, 63)) / 2.0
    score = tf.squash(tf.roll_z(blended, p.lookback), p.squash_k)
    r1m = float(tf.ret(px, 21).dropna().iloc[-1]) * 100 if not tf.ret(px, 21).dropna().empty else float("nan")
    r3m = float(tf.ret(px, 63).dropna().iloc[-1]) * 100 if not tf.ret(px, 63).dropna().empty else float("nan")
    attribution = f"1m {r1m:+.1f}% / 3m {r3m:+.1f}% price momentum"
    return build_signal(name="momentum", category="technical", score=score, raw=blended,
                        attribution=attribution, inputs=[key], neutral_band=p.neutral_band)


def rsi(store: MarketStore) -> SignalResult | None:
    px, key = equity_price(store)
    if px is None:
        return None
    p = Params.from_store(store)
    r = tf.wilder_rsi(px, 14)
    score = tf.squash((r - 50.0) / 15.0, p.squash_k)   # ~15 ≈ 1 std of daily RSI
    last = float(r.dropna().iloc[-1]) if not r.dropna().empty else float("nan")
    zone = "overbought" if last >= 70 else "oversold" if last <= 30 else "neutral"
    attribution = f"RSI(14) = {last:.0f} ({zone})"
    return build_signal(name="rsi", category="technical", score=score, raw=r,
                        attribution=attribution, inputs=[key], neutral_band=p.neutral_band)


def breadth(store: MarketStore) -> SignalResult | None:
    """Fraction of SPDR sector ETFs trading above their own 50d SMA."""
    p = Params.from_store(store)
    above: list[pd.Series] = []
    used: list[str] = []
    for sym in SECTOR_ETFS:
        s = store.try_price(sym)
        if s is None or s.empty:
            continue
        above.append((s > tf.sma(s, 50)).astype(float))
        used.append(f"yahoo:{sym}")
    if len(above) < 4:                                  # need a meaningful cross-section
        return None
    frac = pd.concat(above, axis=1).mean(axis=1).dropna()
    score = tf.squash((frac - 0.5) * 4.0, p.squash_k)   # 0.5 = half participating = neutral
    last = float(frac.iloc[-1]) * 100 if not frac.empty else float("nan")
    attribution = f"{last:.0f}% of {len(used)} sectors above 50d SMA"
    return build_signal(name="breadth", category="technical", score=score, raw=frac,
                        attribution=attribution, inputs=used, neutral_band=p.neutral_band)


ALL = (trend, momentum, rsi, breadth)
