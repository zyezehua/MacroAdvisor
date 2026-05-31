"""Cross-asset signals: US dollar strength and the stock-bond correlation regime.

Orientation: +1 = risk-on, -1 = risk-off.
  * dollar:           a strengthening USD tightens global conditions = risk-off.
  * stock_bond_corr:  a positive equity-Treasury return correlation means bonds stop
                      hedging equities (both sell off together) = regime stress = risk-off.
"""
from __future__ import annotations

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals._common import Params, equity_price
from macro_advisor.signals.base import build_signal, SignalResult

_DOLLAR_SYMBOLS = ("DX-Y.NYB", "UUP")


def dollar(store: MarketStore) -> SignalResult | None:
    px = key = None
    for sym in _DOLLAR_SYMBOLS:
        s = store.try_price(sym)
        if s is not None and not s.empty:
            px, key = s, f"yahoo:{sym}"
            break
    if px is None:
        return None
    p = Params.from_store(store)
    mom = tf.ret(px, 21)
    score = -tf.squash(tf.roll_z(mom, p.lookback), p.squash_k)   # strong USD -> risk_off
    last = float(mom.dropna().iloc[-1]) * 100 if not mom.dropna().empty else float("nan")
    attribution = f"USD 1m {last:+.1f}% ({'strengthening' if last >= 0 else 'weakening'})"
    return build_signal(name="dollar", category="cross_asset", score=score, raw=px,
                        attribution=attribution, inputs=[key], neutral_band=p.neutral_band)


def stock_bond_corr(store: MarketStore) -> SignalResult | None:
    eq, eq_key = equity_price(store)
    tlt = store.try_price("TLT")
    if eq is None or tlt is None:
        return None
    p = Params.from_store(store)
    corr = tf.roll_corr(eq, tlt, 63).dropna()                    # ~3m rolling return corr
    if corr.empty:
        return None
    score = -tf.squash(tf.roll_z(corr, p.lookback), p.squash_k)  # positive corr -> risk_off
    last = float(corr.iloc[-1])
    attribution = f"stock-bond 3m corr = {last:+.2f} ({'bonds not hedging' if last > 0 else 'bonds hedging'})"
    return build_signal(name="stock_bond_corr", category="cross_asset", score=score, raw=corr,
                        attribution=attribution, inputs=[eq_key, "yahoo:TLT"],
                        neutral_band=p.neutral_band)


ALL = (dollar, stock_bond_corr)
