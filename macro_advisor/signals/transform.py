"""Causal time-series transforms — the leakage-safe core of the signal layer.

Every helper here is strictly **backward-looking**: the value at time ``t`` depends only
on observations at or before ``t``. There is no centering, no full-sample fit, and no
forward fill across the evaluation point. This is what lets the same signal code be reused
unchanged by the Phase-2 walk-forward OOS backtester without introducing look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# minimum fraction of a window that must be populated before a value is emitted
_MIN_FRAC = 0.5


def _min_periods(window: int) -> int:
    return max(2, int(window * _MIN_FRAC))


def ret(s: pd.Series, periods: int = 1) -> pd.Series:
    """Simple return over ``periods`` (backward difference)."""
    return s.pct_change(periods)


def log_ret(s: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(s).diff(periods)


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=_min_periods(window)).mean()


def roll_z(s: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: (x - rolling_mean) / rolling_std. Causal."""
    m = s.rolling(window, min_periods=_min_periods(window)).mean()
    sd = s.rolling(window, min_periods=_min_periods(window)).std()
    return (s - m) / sd.replace(0.0, np.nan)


def roll_pct(s: pd.Series, window: int) -> pd.Series:
    """Trailing percentile rank in [0, 1] of the latest value within the window.

    Uses ``rank(pct=True)`` over each trailing window and takes the last entry, so the
    rank at ``t`` only ever sees data up to ``t``.
    """
    mp = _min_periods(window)
    return s.rolling(window, min_periods=mp).apply(
        lambda w: w.rank(pct=True).iloc[-1], raw=False
    )


def squash(z: pd.Series, k: float = 1.5) -> pd.Series:
    """Map an unbounded z-score to [-1, 1] via tanh(z / k)."""
    return np.tanh(z / k)


def wilder_rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. Uses Wilder smoothing (EMA, alpha = 1/window)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # when avg_loss == 0 (only gains) RSI is 100
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def realized_vol(s: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Trailing realized volatility of daily log returns."""
    r = log_ret(s)
    vol = r.rolling(window, min_periods=_min_periods(window)).std()
    if annualize:
        vol = vol * np.sqrt(252.0)
    return vol


def roll_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Trailing Pearson correlation of two daily-return series (aligned, inner)."""
    ra, rb = ret(a), ret(b)
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    return joined.iloc[:, 0].rolling(window, min_periods=_min_periods(window)).corr(
        joined.iloc[:, 1]
    )
