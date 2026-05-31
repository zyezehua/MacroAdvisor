"""Rates signals from the Treasury curve (+ FRED real yield/breakeven when present).

Orientation: +1 = risk-on, -1 = risk-off.
  * curve slope: deep inversion = recession risk = risk-off; positive/steepening = risk-on.
  * level move:  rising yields = tightening financial conditions = risk-off.
  * real yield:  rising real yields = tighter policy = risk-off.
  * breakeven:   falling breakevens = growth/deflation scare = risk-off.
"""
from __future__ import annotations

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals._common import Params
from macro_advisor.signals.base import build_signal, SignalResult


def _slope(store: MarketStore, long_tenor: str, short_tenor: str, name: str,
           label: str) -> SignalResult | None:
    lng, sht = store.try_yield(long_tenor), store.try_yield(short_tenor)
    if lng is None or sht is None:
        return None
    p = Params.from_store(store)
    slope = (lng - sht).dropna()                       # in percent (pp)
    score = tf.squash(tf.roll_z(slope, p.lookback), p.squash_k)
    last_bps = float(slope.iloc[-1]) * 100 if not slope.empty else float("nan")
    state = "inverted" if last_bps < 0 else "positive"
    attribution = f"{label} = {last_bps:+.0f} bps ({state})"
    return build_signal(name=name, category="rates", score=score, raw=slope,
                        attribution=attribution,
                        inputs=[f"treasury:{long_tenor}", f"treasury:{short_tenor}"],
                        neutral_band=p.neutral_band)


def curve_2s10s(store: MarketStore) -> SignalResult | None:
    return _slope(store, "UST10Y", "UST2Y", "curve_2s10s", "2s10s")


def curve_3m10y(store: MarketStore) -> SignalResult | None:
    return _slope(store, "UST10Y", "UST3M", "curve_3m10y", "3m10y")


def level_move(store: MarketStore) -> SignalResult | None:
    """1-month change in the 10Y yield. Rising yields = tightening = risk-off."""
    y = store.try_yield("UST10Y")
    if y is None:
        return None
    p = Params.from_store(store)
    chg = y.diff(21).dropna()                           # pp change over ~1m
    score = -tf.squash(tf.roll_z(chg, p.lookback), p.squash_k)
    last_bps = float(chg.iloc[-1]) * 100 if not chg.empty else float("nan")
    attribution = f"10Y yield 1m change {last_bps:+.0f} bps"
    return build_signal(name="level_move", category="rates", score=score, raw=chg,
                        attribution=attribution, inputs=["treasury:UST10Y"],
                        neutral_band=p.neutral_band)


def real_yield(store: MarketStore) -> SignalResult | None:
    """10Y real yield (FRED DFII10). Rising real yields = tighter = risk-off."""
    r = store.fred("DFII10")
    if r is None:
        return None
    p = Params.from_store(store)
    score = -tf.squash(tf.roll_z(r, p.lookback), p.squash_k)
    last = float(r.iloc[-1]) if not r.empty else float("nan")
    attribution = f"10Y real yield {last:+.2f}%"
    return build_signal(name="real_yield", category="rates", score=score, raw=r,
                        attribution=attribution, inputs=["fred:DFII10"],
                        neutral_band=p.neutral_band)


def breakeven(store: MarketStore) -> SignalResult | None:
    """10Y breakeven inflation (FRED T10YIE), 1m change. Falling = growth scare = risk-off."""
    be = store.fred("T10YIE")
    if be is None:
        return None
    p = Params.from_store(store)
    chg = be.diff(21).dropna()
    score = tf.squash(tf.roll_z(chg, p.lookback), p.squash_k)
    last = float(be.iloc[-1]) if not be.empty else float("nan")
    last_chg = float(chg.iloc[-1]) * 100 if not chg.empty else float("nan")
    attribution = f"10Y breakeven {last:.2f}% (1m {last_chg:+.0f} bps)"
    return build_signal(name="breakeven", category="rates", score=score, raw=be,
                        attribution=attribution, inputs=["fred:T10YIE"],
                        neutral_band=p.neutral_band)


ALL = (curve_2s10s, curve_3m10y, level_move, real_yield, breakeven)
