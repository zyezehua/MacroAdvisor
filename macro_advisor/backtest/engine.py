"""Vectorized daily backtester driven by OOS directional predictions.

A position is taken each day from that day's OOS direction signal (vol-targeted within the risk
budget) and earns the asset's **next-day** return; trading costs apply to weight changes. Only
OOS-predicted dates trade, so the equity curve is genuinely out-of-sample. Daily rebalancing to
the horizon's directional signal is a deliberate 2a simplification (holding-period optimization
is a 2b/recommendation concern).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.backtest import metrics

ANN = 252.0


def daily_returns(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Wide date×symbol simple daily returns from a {symbol: price} mapping."""
    return pd.DataFrame({s: p.pct_change() for s, p in prices.items()}).sort_index()


def target_weights(preds: pd.DataFrame, returns_wide: pd.DataFrame, *,
                   prob_threshold: float, vol_target: float, max_position: float,
                   per_position_cap: float, max_leverage: float,
                   long_only: bool) -> pd.DataFrame:
    """Build a date×symbol weight matrix from OOS classifier predictions.

    Direction must clear ``prob_threshold``; size = vol_target / trailing asset vol, capped per
    position; gross exposure scaled to ``max_leverage``.
    """
    # raw directional conviction in {-1,0,+1}
    conf = preds.get("p_up").where(preds["pred"] == 1, preds.get("p_down")) if "p_up" in preds else None
    take = (conf >= prob_threshold) if conf is not None else pd.Series(True, index=preds.index)
    sign = preds["pred"].where(take, 0).astype(float)
    if long_only:
        sign = sign.clip(lower=0.0)
    sign_wide = sign.unstack("symbol").reindex(columns=returns_wide.columns).sort_index()

    # trailing annualized vol per asset (causal), for vol targeting
    vol = returns_wide.rolling(63, min_periods=20).std() * np.sqrt(ANN)
    size = (vol_target / vol).clip(upper=max_position)
    w = (sign_wide * size).reindex(returns_wide.index).fillna(0.0)
    w = w.clip(lower=-per_position_cap, upper=per_position_cap)

    gross = w.abs().sum(axis=1)
    scale = (max_leverage / gross).clip(upper=1.0).replace([np.inf, np.nan], 1.0)
    return w.mul(scale, axis=0)


def run(preds: pd.DataFrame, prices: dict[str, pd.Series], *, cfg, rf_daily=0.0) -> dict:
    """Backtest one model/horizon's OOS predictions. Returns equity, daily returns, and metrics."""
    rb, pcfg = cfg.risk_budget, cfg.predict
    bt = cfg.backtest
    pbt = pcfg.get("backtest", {})
    returns_wide = daily_returns(prices)

    w = target_weights(
        preds, returns_wide,
        prob_threshold=float(pcfg.get("prob_threshold", 0.55)),
        vol_target=float(pbt.get("vol_target_annual", 0.10)),
        max_position=float(pbt.get("max_position", 1.0)),
        per_position_cap=float(rb.get("per_position_cap", 0.15)),
        max_leverage=float(rb.get("max_leverage", 1.0)),
        long_only=bool(pbt.get("long_only", False)),
    )

    # weight at t earns the asset's return at t+1
    fwd = returns_wide.shift(-1)
    gross_ret = (w * fwd).sum(axis=1)

    # costs on turnover (sum of |Δweight|), per leg
    cost_rate = (float(bt.get("cost_bps_per_trade", 2.0)) + float(bt.get("slippage_bps", 1.0))) / 1e4
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    net_ret = (gross_ret - turnover * cost_rate).dropna()

    # restrict to the OOS window (skip pre-prediction zero-weight days, which would
    # otherwise dilute the metrics)
    first_oos = preds.index.get_level_values("date").min()
    net_ret = net_ret[net_ret.index >= first_oos]

    return {
        "returns": net_ret,
        "equity": metrics.equity_curve(net_ret),
        "metrics": metrics.summary(net_ret, rf_daily),
        "avg_gross": float(w.abs().sum(axis=1).mean()),
        "avg_turnover": float(turnover.mean()),
    }


def benchmark(prices: dict[str, pd.Series], symbol: str, index: pd.Index, rf_daily=0.0) -> dict:
    """Buy-&-hold benchmark (e.g. SPY) aligned to the strategy window."""
    r = prices[symbol].pct_change().reindex(index).dropna()
    return {"returns": r, "equity": metrics.equity_curve(r), "metrics": metrics.summary(r, rf_daily)}
