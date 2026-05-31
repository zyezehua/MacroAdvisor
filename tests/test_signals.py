"""Signal families: directional correctness on constructed inputs + graceful skips."""
from __future__ import annotations

import numpy as np

from macro_advisor.signals import cross_asset, rates, technical, volatility
from macro_advisor.signals.base import RISK_OFF, RISK_ON
from tests.conftest import bdays, price_frame, value_frame

N = 180
IDX = bdays(N)


def _flat_then_spike(base, spike, n=N, tail=10):
    arr = np.full(n, float(base))
    arr[-tail:] = np.linspace(base, spike, tail)
    return arr


def test_all_scores_bounded(store_factory):
    store = store_factory(
        prices={"SPY": price_frame(np.linspace(400, 520, N), IDX),
                "^VIX": price_frame(_flat_then_spike(15, 35), IDX)},
    )
    for fn in (technical.trend, technical.momentum, technical.rsi, volatility.vix_level):
        res = fn(store)
        assert res is not None
        s = res.score.dropna()
        assert s.between(-1, 1).all()


def test_vix_spike_is_risk_off(store_factory):
    store = store_factory(prices={"^VIX": price_frame(_flat_then_spike(15, 35), IDX)})
    res = volatility.vix_level(store)
    assert res is not None
    assert res.latest_score < 0 and res.direction == RISK_OFF


def test_equity_uptrend_is_risk_on(store_factory):
    # steady ramp pulls price further above its lagging 200d SMA over time -> trend risk_on
    store = store_factory(prices={"SPY": price_frame(np.linspace(400, 520, N), IDX)})
    res = technical.trend(store)
    assert res is not None
    assert res.latest_score > 0 and res.direction == RISK_ON


def test_accelerating_momentum_is_risk_on(store_factory):
    # convex (accelerating) path -> recent returns exceed the trailing norm -> momentum risk_on
    convex = 400.0 + 0.004 * np.arange(N) ** 2
    store = store_factory(prices={"SPY": price_frame(convex, IDX)})
    mom = technical.momentum(store)
    assert mom is not None
    assert mom.latest_score > 0 and mom.direction == RISK_ON


def test_curve_inversion_is_risk_off(store_factory):
    # 10Y flat at 4.0; 2Y rises through it -> slope falls below its trailing norm (inverting)
    store = store_factory(series={
        "UST10Y": value_frame(np.full(N, 4.0), IDX),
        "UST2Y": value_frame(np.linspace(3.0, 4.3, N), IDX),
    })
    res = rates.curve_2s10s(store)
    assert res is not None
    assert res.latest_score < 0 and res.direction == RISK_OFF


def test_missing_inputs_skip(store_factory):
    store = store_factory()  # empty cache
    assert volatility.vix_level(store) is None
    assert technical.momentum(store) is None
    assert rates.curve_2s10s(store) is None
    assert cross_asset.stock_bond_corr(store) is None
