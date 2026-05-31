"""Credit signals: HY-vs-IG relative performance, plus FRED HY OAS when available.

Orientation: +1 = risk-on (HY outperforming IG, spreads tightening), -1 = risk-off
(HY underperforming, spreads widening — a classic early stress signal).
"""
from __future__ import annotations

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals._common import Params
from macro_advisor.signals.base import build_signal, SignalResult


def hy_ig(store: MarketStore) -> SignalResult | None:
    """HYG/LQD total-return ratio momentum: rising ratio = HY leading = risk-on."""
    hyg, lqd = store.try_price("HYG"), store.try_price("LQD")
    if hyg is None or lqd is None:
        return None
    p = Params.from_store(store)
    ratio = (hyg / lqd).dropna()
    mom = tf.ret(ratio, 21)                            # 1-month relative momentum
    score = tf.squash(tf.roll_z(mom, p.lookback), p.squash_k)
    last = float(mom.dropna().iloc[-1]) * 100 if not mom.dropna().empty else float("nan")
    attribution = f"HY/IG ratio 1m {last:+.1f}% ({'HY leading' if last >= 0 else 'HY lagging'})"
    return build_signal(name="hy_ig", category="credit", score=score, raw=ratio,
                        attribution=attribution, inputs=["yahoo:HYG", "yahoo:LQD"], neutral_band=p.neutral_band)


def hy_oas(store: MarketStore) -> SignalResult | None:
    """ICE BofA US HY OAS (FRED). Widening spread = risk-off. Skipped if not pulled."""
    oas = store.fred("BAMLH0A0HYM2")
    if oas is None:
        return None
    p = Params.from_store(store)
    score = -tf.squash(tf.roll_z(oas, p.lookback), p.squash_k)
    pct = tf.roll_pct(oas, p.lookback)
    last = float(oas.iloc[-1]) if not oas.empty else float("nan")
    lp = float(pct.dropna().iloc[-1]) * 100 if not pct.dropna().empty else float("nan")
    attribution = f"HY OAS {last:.2f}% @ {lp:.0f}th pctile (1y)"
    return build_signal(name="hy_oas", category="credit", score=score, raw=oas,
                        attribution=attribution, inputs=["fred:BAMLH0A0HYM2"], neutral_band=p.neutral_band)


ALL = (hy_ig, hy_oas)
