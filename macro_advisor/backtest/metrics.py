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


# -- extended statistics (Phase 5) ------------------------------------------

def volatility(returns: pd.Series) -> float:
    """Annualized volatility of daily returns."""
    sd = returns.dropna().std()
    return float(sd * np.sqrt(ANN)) if sd == sd else 0.0


def calmar(returns: pd.Series) -> float:
    """CAGR / |max drawdown| — return per unit of worst peak-to-trough pain."""
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns) / mdd) if mdd else 0.0


def win_loss(returns: pd.Series) -> dict:
    """Average win, average loss, profit factor and payoff ratio over non-zero days."""
    r = returns[returns != 0].dropna()
    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else float("inf"),
        "payoff_ratio": round(avg_win / abs(avg_loss), 3) if avg_loss else float("inf"),
    }


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical daily VaR at ``level`` (reported as a positive loss fraction)."""
    r = returns.dropna()
    return float(-np.quantile(r, 1.0 - level)) if len(r) else 0.0


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Historical daily CVaR / expected shortfall (positive loss fraction)."""
    r = returns.dropna()
    if not len(r):
        return 0.0
    cutoff = np.quantile(r, 1.0 - level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else 0.0


def beta_alpha(returns: pd.Series, benchmark: pd.Series, rf_daily: pd.Series | float = 0.0) -> dict:
    """OLS beta and annualized alpha of the strategy vs a benchmark return series."""
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 2 or df["b"].var() == 0:
        return {"beta": 0.0, "alpha": 0.0}
    beta = float(df["r"].cov(df["b"]) / df["b"].var())
    alpha_daily = float(_excess(df["r"], rf_daily).mean() - beta * _excess(df["b"], rf_daily).mean())
    return {"beta": round(beta, 3), "alpha": round(alpha_daily * ANN, 4)}


def drawdown_duration(returns: pd.Series) -> int:
    """Longest underwater stretch, in trading days (peak to recovery)."""
    eq = equity_curve(returns)
    if eq.empty:
        return 0
    underwater = eq < eq.cummax()
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year × month table of compounded returns (fractions)."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    m = (1.0 + r).resample("ME").prod() - 1.0
    tbl = m.to_frame("ret")
    tbl["year"], tbl["month"] = tbl.index.year, tbl.index.month
    return tbl.pivot_table(index="year", columns="month", values="ret")


def extended(returns: pd.Series, rf_daily: pd.Series | float = 0.0, *,
             benchmark: pd.Series | None = None,
             exposure: pd.Series | None = None,
             turnover: pd.Series | None = None) -> dict:
    """Full statistics panel: the headline ``summary`` plus risk / win-loss / market stats."""
    returns = returns.dropna()
    out = {**summary(returns, rf_daily),
           "volatility": round(volatility(returns), 4),
           "calmar": round(calmar(returns), 3),
           "var_95": round(value_at_risk(returns), 4),
           "cvar_95": round(cvar(returns), 4),
           "skew": round(float(returns.skew()), 3) if len(returns) > 2 else 0.0,
           "kurtosis": round(float(returns.kurtosis()), 3) if len(returns) > 3 else 0.0,
           "dd_duration_days": drawdown_duration(returns),
           **win_loss(returns)}
    if benchmark is not None:
        out.update(beta_alpha(returns, benchmark, rf_daily))
    if exposure is not None and len(exposure):
        out["time_in_market"] = round(float((exposure.abs() > 1e-9).mean()), 4)
        out["avg_gross_exposure"] = round(float(exposure.abs().mean()), 3)
    if turnover is not None and len(turnover):
        out["annual_turnover"] = round(float(turnover.mean() * ANN), 2)
    return out
