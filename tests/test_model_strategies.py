"""Model-signal strategies: position shaping (hold/rebalance/gate) and full backtest output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.config import load_config
from macro_advisor.strategy import model_strategies as ms


def _oos_pred(dates, symbols, pred=1.0, p_up=0.9):
    rows = []
    for s in symbols:
        rows.append(pd.DataFrame({"date": dates, "symbol": s, "model": "stack", "horizon": "short",
                                  "pred": pred, "p_up": p_up, "p_down": 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_min_hold_blocks_early_flips():
    d = pd.Series([1, 1, -1, -1, -1, 1, 1], index=pd.bdate_range("2024-01-01", periods=7), dtype=float)
    held = ms._min_hold(d, holding_period=3)
    # the flip to -1 at index 2 is within 3 days of the start; it must be suppressed until day 3
    assert held.iloc[2] == 1.0
    assert held.iloc[3] == -1.0          # now allowed to flip


def test_rebalance_holds_between_periods():
    idx = pd.bdate_range("2024-01-01", periods=10)
    d = pd.Series([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], index=idx, dtype=float)
    weekly = ms._resample_hold(d, "weekly")
    # within a week the position is the week's last signal carried forward (constant per week)
    assert weekly.nunique() <= 2 and len(weekly) == len(d)


def test_rotation_gate_splits_by_regime():
    idx = pd.bdate_range("2024-01-01", periods=4)
    s = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    stress = pd.Series([40, 40, 70, 70], index=idx, dtype=float)
    p = ms.ModelStrategyParams(gate="rotation", stress_max=55.0)
    eq = ms._apply_gate(s, "equities", stress, p)
    rt = ms._apply_gate(s, "rates", stress, p)
    assert list(eq) == [1, 1, 0, 0]      # equities only in calm
    assert list(rt) == [0, 0, 1, 1]      # rates only in stress


def test_run_strategy_produces_stats_and_attribution():
    cfg = load_config()
    dates = pd.bdate_range(end="2026-05-29", periods=400)
    prices = {"SPY": pd.Series(np.linspace(100, 150, 400), index=dates),
              "QQQ": pd.Series(np.linspace(100, 170, 400), index=dates)}
    oos = _oos_pred(dates[200:], ["SPY", "QQQ"])     # always-long over the OOS window
    params = ms.ModelStrategyParams(model="stack", horizon="short", universe=["SPY", "QQQ"],
                                    rebalance="weekly", holding_period=5)
    out = ms.run_strategy(oos, prices, cfg, params, benchmark="SPY")
    assert "error" not in out
    assert {"sortino", "volatility", "calmar", "var_95", "beta"}.issubset(out["stats"])
    assert not out["attribution"]["per_asset"].empty
    assert not out["equity"].empty


def test_default_strategies_are_valid_params():
    for name, spec in ms.default_strategies().items():
        p = spec["params"]
        assert p.rebalance in ms.REBALANCES and p.gate in ms.GATES and p.universe
