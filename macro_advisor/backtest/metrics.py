"""Performance metrics for a daily strategy-return series. All annualized at 252 trading days."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def _excess(returns: pd.Series, rf_daily: pd.Series | float) -> pd.Series:
    if isinstance(rf_daily, pd.Series):
        rf_daily = rf_daily.reindex(returns.index).fillna(0.0)
    return returns - rf_daily


def sharpe(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    ex = _excess(returns, rf_daily).dropna()
    sd = ex.std()
    return float(np.sqrt(ANN) * ex.mean() / sd) if sd else 0.0


def sortino(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    ex = _excess(returns, rf_daily).dropna()
    downside = ex[ex < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    return float(np.sqrt(ANN) * ex.mean() / dd) if dd else 0.0


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0


def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    if len(eq) < 2:
        return 0.0
    yrs = len(eq) / ANN
    return float(eq.iloc[-1] ** (1.0 / yrs) - 1.0)


def hit_rate(returns: pd.Series) -> float:
    r = returns[returns != 0].dropna()
    return float((r > 0).mean()) if len(r) else 0.0


def summary(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> dict:
    returns = returns.dropna()
    return {
        "sortino": round(sortino(returns, rf_daily), 3),
        "sharpe": round(sharpe(returns, rf_daily), 3),
        "max_drawdown": round(max_drawdown(returns), 4),
        "cagr": round(cagr(returns), 4),
        "hit_rate": round(hit_rate(returns), 4),
        "n_days": int(len(returns)),
    }
