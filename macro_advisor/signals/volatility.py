"""Volatility signals: VIX level, implied-vs-realized term, Nasdaq vol premium, MOVE.

Orientation: +1 = risk-on (low/falling vol, implied above realized = calm),
-1 = risk-off (elevated vol / implied below realized / rates-vol spike).
"""
from __future__ import annotations

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals._common import Params, equity_price
from macro_advisor.signals.base import build_signal, SignalResult


def vix_level(store: MarketStore) -> SignalResult | None:
    vix = store.try_price("^VIX")
    if vix is None:
        return None
    p = Params.from_store(store)
    score = -tf.squash(tf.roll_z(vix, p.lookback), p.squash_k)   # high VIX -> risk_off
    pct = tf.roll_pct(vix, p.lookback)
    last, lp = float(vix.iloc[-1]), float(pct.dropna().iloc[-1]) * 100 if not pct.dropna().empty else float("nan")
    attribution = f"VIX {last:.1f} @ {lp:.0f}th pctile (1y)"
    return build_signal(name="vix_level", category="volatility", score=score, raw=vix,
                        attribution=attribution, inputs=["yahoo:^VIX"], neutral_band=p.neutral_band)


def vix_term(store: MarketStore) -> SignalResult | None:
    """Implied (VIX) vs realized vol of the equity proxy. Implied >> realized = calm."""
    vix = store.try_price("^VIX")
    px, key = equity_price(store)
    if vix is None or px is None:
        return None
    p = Params.from_store(store)
    rvol_pct = tf.realized_vol(px, 21) * 100.0                   # annualized %, comparable to VIX
    spread = (vix - rvol_pct).dropna()
    score = tf.squash(tf.roll_z(spread, p.lookback), p.squash_k)  # implied premium -> risk_on
    last = float(spread.iloc[-1]) if not spread.empty else float("nan")
    attribution = f"VIX minus 21d realized vol = {last:+.1f} pts ({'premium' if last >= 0 else 'discount'})"
    return build_signal(name="vix_term", category="volatility", score=score, raw=spread,
                        attribution=attribution, inputs=["yahoo:^VIX", key], neutral_band=p.neutral_band)


def vxn_premium(store: MarketStore) -> SignalResult | None:
    """Nasdaq vol (VXN) over S&P vol (VIX): tech-vol premium widening = risk-off."""
    vxn, vix = store.try_price("^VXN"), store.try_price("^VIX")
    if vxn is None or vix is None:
        return None
    p = Params.from_store(store)
    spread = (vxn - vix).dropna()
    score = -tf.squash(tf.roll_z(spread, p.lookback), p.squash_k)
    last = float(spread.iloc[-1]) if not spread.empty else float("nan")
    attribution = f"VXN-VIX spread = {last:+.1f} pts"
    return build_signal(name="vxn_premium", category="volatility", score=score, raw=spread,
                        attribution=attribution, inputs=["yahoo:^VXN", "yahoo:^VIX"], neutral_band=p.neutral_band)


def move(store: MarketStore) -> SignalResult | None:
    """ICE BofA MOVE — Treasury-implied vol. Elevated rates vol = risk-off."""
    mv = store.try_price("^MOVE")
    if mv is None:
        return None
    p = Params.from_store(store)
    score = -tf.squash(tf.roll_z(mv, p.lookback), p.squash_k)
    pct = tf.roll_pct(mv, p.lookback)
    last, lp = float(mv.iloc[-1]), float(pct.dropna().iloc[-1]) * 100 if not pct.dropna().empty else float("nan")
    attribution = f"MOVE {last:.0f} @ {lp:.0f}th pctile (1y)"
    return build_signal(name="move", category="volatility", score=score, raw=mv,
                        attribution=attribution, inputs=["yahoo:^MOVE"], neutral_band=p.neutral_band)


ALL = (vix_level, vix_term, vxn_premium, move)
