"""News / sentiment signals (Phase 3).

Two flavors, both **causal** and orientation-normalized to the project convention
(+1 = risk-on/favorable, -1 = risk-off/stress):

* **FRED hard-sentiment** — survey & financial-conditions series (U.Mich consumer sentiment,
  Chicago Fed NFCI, St. Louis Fed financial stress). Low frequency (weekly/monthly), so each
  is forward-filled to the trading calendar (**never back-filled**) before any trailing stat.
* **GDELT news tone** — average tone of global news for a query, with a news-*volume* spike
  treated as added risk-off. **Single-source** (no cross-check mirror): confirmatory only, and
  weighted modestly in the stress index.

For the OOS feature panel these series additionally get a publication-lag shift in
``predict/features.py``; the live dashboard read uses the latest available observation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_advisor.data import MarketStore
from macro_advisor.signals import transform as tf
from macro_advisor.signals.base import SignalResult, build_signal


@dataclass(frozen=True)
class SentParams:
    lookback: int
    change_window: int
    squash_k: float
    neutral_band: float
    smooth_days: int
    volume_weight: float

    @classmethod
    def from_store(cls, store: MarketStore) -> "SentParams":
        s = store.cfg.sentiment
        g = s.get("gdelt", {}) or {}
        return cls(
            lookback=int(s.get("lookback_days", 504)),
            change_window=int(s.get("change_window", 126)),
            squash_k=float(s.get("squash_k", 1.5)),
            neutral_band=float(s.get("neutral_band", 0.10)),
            smooth_days=int(g.get("smooth_days", 7)),
            volume_weight=float(g.get("volume_weight", 0.35)),
        )


# how far past the last observation a low-frequency reading is carried forward as "current":
# enough to bridge a monthly survey's release gap, but bounded so a discontinued/stale series
# lapses (and trips the STALE QA flag) instead of looking current forever.
_MAX_CARRY_BDAYS = 45


def _daily_ffill(s: pd.Series) -> pd.Series:
    """Reindex a low-frequency series onto business days, forward-filling only (causal).

    Extends a bounded window *past* the last observation so the latest known reading persists
    until the next release (a survey stays current between prints) — never inventing a future
    value, only carrying the last one forward."""
    s = s.dropna().sort_index()
    if s.empty:
        return s
    last = s.index.max()
    cap = last + pd.tseries.offsets.BDay(_MAX_CARRY_BDAYS)
    today = pd.Timestamp.now().normalize()
    end = last if today <= last else min(today, cap)
    bidx = pd.date_range(s.index.min(), end, freq="B", name=s.index.name or "date")
    return s.reindex(bidx, method="ffill").rename(s.name)


def _level_signal(level: pd.Series, p: SentParams, *, risk_on: bool, use_change: bool) -> pd.Series:
    """Causal score from a (daily-ffilled) level series. ``risk_on`` sets the orientation:
    True  -> high/rising value is risk-on (e.g. consumer sentiment);
    False -> high/rising value is risk-off (e.g. financial conditions/stress)."""
    z_level = tf.roll_z(level, p.lookback)
    z = z_level
    if use_change:
        z_chg = tf.roll_z(tf.ret(level, p.change_window), p.lookback)
        z = 0.6 * z_level + 0.4 * z_chg
    score = tf.squash(z, p.squash_k)
    return score if risk_on else -score


def consumer_sentiment(store: MarketStore) -> SignalResult | None:
    """U.Mich consumer sentiment (FRED UMCSENT). High/improving = risk-on."""
    raw = store.fred("UMCSENT")
    if raw is None:
        return None
    p = SentParams.from_store(store)
    level = _daily_ffill(raw)
    score = _level_signal(level, p, risk_on=True, use_change=True)
    z = float(tf.roll_z(level, p.lookback).dropna().iloc[-1]) if not level.dropna().empty else float("nan")
    last = float(level.dropna().iloc[-1]) if not level.dropna().empty else float("nan")
    attribution = f"U.Mich sentiment {last:.1f} ({z:+.1f}σ vs 2y; {'above' if z >= 0 else 'below'} trend)"
    return build_signal(name="consumer_sentiment", category="sentiment", score=score, raw=level,
                        attribution=attribution, inputs=["fred:UMCSENT"], neutral_band=p.neutral_band)


def financial_conditions(store: MarketStore) -> SignalResult | None:
    """Chicago Fed National Financial Conditions Index (FRED NFCI). Positive = tighter = risk-off."""
    raw = store.fred("NFCI")
    if raw is None:
        return None
    p = SentParams.from_store(store)
    level = _daily_ffill(raw)
    score = _level_signal(level, p, risk_on=False, use_change=False)
    last = float(level.dropna().iloc[-1]) if not level.dropna().empty else float("nan")
    attribution = f"NFCI {last:+.2f} ({'tight' if last > 0 else 'loose'} vs average financial conditions)"
    return build_signal(name="financial_conditions", category="sentiment", score=score, raw=level,
                        attribution=attribution, inputs=["fred:NFCI"], neutral_band=p.neutral_band)


def financial_stress(store: MarketStore) -> SignalResult | None:
    """St. Louis Fed Financial Stress Index (FRED STLFSI4). Elevated = risk-off."""
    raw = store.fred("STLFSI4")
    if raw is None:
        return None
    p = SentParams.from_store(store)
    level = _daily_ffill(raw)
    score = _level_signal(level, p, risk_on=False, use_change=False)
    last = float(level.dropna().iloc[-1]) if not level.dropna().empty else float("nan")
    attribution = f"STL financial stress {last:+.2f} ({'elevated' if last > 0 else 'subdued'})"
    return build_signal(name="financial_stress", category="sentiment", score=score, raw=level,
                        attribution=attribution, inputs=["fred:STLFSI4"], neutral_band=p.neutral_band)


def news_tone(store: MarketStore) -> SignalResult | None:
    """GDELT news tone for the configured query. Positive/rising tone = risk-on; a news-volume
    spike adds risk-off. Single-source (confirmatory)."""
    sources = store.cfg.news_sources()
    label = sources[0]["label"] if sources else "news_markets"
    df = store.news(label)
    if df is None or "value" not in df:
        return None
    p = SentParams.from_store(store)
    tone = df["value"].rolling(p.smooth_days, min_periods=1).mean().dropna()
    if tone.empty:
        return None
    score = tf.squash(tf.roll_z(tone, p.lookback), p.squash_k)
    if "volume" in df and df["volume"].notna().any() and p.volume_weight > 0:
        vol = df["volume"].rolling(p.smooth_days, min_periods=1).mean()
        vscore = tf.squash(tf.roll_z(vol, p.lookback), p.squash_k).reindex(score.index)
        score = (score - p.volume_weight * vscore.fillna(0.0)).clip(-1.0, 1.0)
    last = float(tone.iloc[-1])
    z = float(tf.roll_z(tone, p.lookback).dropna().iloc[-1]) if not tone.dropna().empty else float("nan")
    attribution = f"News tone {last:+.2f} ({z:+.1f}σ vs 2y; {'favorable' if z >= 0 else 'adverse'} coverage) [single-source]"
    return build_signal(name="news_tone", category="sentiment", score=score, raw=tone,
                        attribution=attribution, inputs=[f"gdelt:{label}"], neutral_band=p.neutral_band)


ALL = (consumer_sentiment, financial_conditions, financial_stress, news_tone)
