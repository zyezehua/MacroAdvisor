"""Extended Phase-5 statistics: risk, win/loss, market stats, monthly table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.backtest import metrics


def _ret(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)))


def test_win_loss_and_profit_factor():
    r = _ret([0.02, -0.01, 0.04, -0.02, 0.0])     # wins 0.06, losses 0.03 -> PF 2.0
    wl = metrics.win_loss(r)
    assert wl["profit_factor"] == 2.0
    assert wl["avg_win"] > 0 > wl["avg_loss"]


def test_var_cvar_are_positive_losses():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500),
                  index=pd.bdate_range("2022-01-01", periods=500))
    var, cv = metrics.value_at_risk(r), metrics.cvar(r)
    assert var > 0 and cv >= var                    # expected shortfall is at least the VaR


def test_beta_alpha_self_is_one_and_zero():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2022-01-01", periods=300))
    ba = metrics.beta_alpha(b, b)
    assert ba["beta"] == pytest.approx(1.0, abs=1e-6)
    assert ba["alpha"] == pytest.approx(0.0, abs=1e-6)


def test_extended_has_all_blocks_and_monthly_table():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-01", periods=520)
    r = pd.Series(rng.normal(0.0004, 0.009, 520), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.008, 520), index=idx)
    expo = pd.Series(1.0, index=idx)
    ext = metrics.extended(r, benchmark=bench, exposure=expo, turnover=pd.Series(0.1, index=idx))
    for k in ("sortino", "volatility", "calmar", "var_95", "cvar_95", "beta", "alpha",
              "dd_duration_days", "profit_factor", "time_in_market", "annual_turnover"):
        assert k in ext
    assert ext["time_in_market"] == 1.0
    mt = metrics.monthly_returns(r)
    assert not mt.empty and mt.index.name == "year"
