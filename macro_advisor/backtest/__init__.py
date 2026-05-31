"""Backtester (Phase 2): vectorized engine with costs + walk-forward harness."""
from macro_advisor.backtest import metrics
from macro_advisor.backtest.engine import benchmark, daily_returns, run

__all__ = ["run", "benchmark", "daily_returns", "metrics"]
