"""Backtester metrics + engine: hand-checked math and cost application."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.backtest import engine, metrics
from macro_advisor.config import load_config


def _ret(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)))


def test_metrics_hand_computed():
    r = _ret([0.01, -0.02, 0.03, -0.01, 0.02])
    # max drawdown from the cumulative equity
    eq = (1 + r).cumprod()
    assert metrics.max_drawdown(r) == pytest.approx(float((eq / eq.cummax() - 1).min()))
    # sortino uses downside deviation only
    ex = r
    dd = np.sqrt((ex[ex < 0] ** 2).mean())
    assert metrics.sortino(r) == pytest.approx(np.sqrt(252) * ex.mean() / dd)
    assert 0.0 <= metrics.hit_rate(r) <= 1.0


def test_sortino_zero_when_no_downside():
    assert metrics.sortino(_ret([0.01, 0.02, 0.0])) == 0.0


def test_engine_costs_reduce_return():
    """A flip-flopping position pays turnover costs, lowering net vs gross return."""
    cfg = load_config()
    dates = pd.bdate_range(end="2026-05-29", periods=300)
    # one asset, steadily rising so a long position is profitable gross
    price = pd.Series(np.linspace(100, 130, 300), index=dates, name="SPY")
    prices = {"SPY": price}
    # alternate long/flat predictions over the back half -> turnover every other day
    pred_dates = dates[150:]
    preds = pd.DataFrame({
        "pred": [1 if i % 2 == 0 else 0 for i in range(len(pred_dates))],
        "p_up": 0.9, "p_down": 0.0,
    }, index=pd.MultiIndex.from_product([pred_dates, ["SPY"]], names=["date", "symbol"]))

    out = engine.run(preds, prices, cfg=cfg, rf_daily=0.0)
    assert not out["returns"].empty
    # OOS window only (trimmed to first prediction date)
    assert out["returns"].index.min() >= pred_dates.min()
    assert out["avg_turnover"] > 0          # costs actually applied
    assert set(out["metrics"]) >= {"sortino", "sharpe", "max_drawdown", "hit_rate"}
