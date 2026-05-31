"""Recommendation engine: ensemble scoring, risk-budget caps, payoff illustration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.config import load_config
from macro_advisor.recommend import payoff, portfolio, score
from tests.conftest import bdays, price_frame


def _noisy_price(n=200, drift=0.03, seed=0):
    """Price with genuine up & down days so downside vol is well-defined."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift / n, 0.01, n)
    idx = bdays(n)
    return price_frame(100 * np.cumprod(1 + rets), idx)


@pytest.fixture
def store2(store_factory):
    return store_factory(prices={"AAA": _noisy_price(seed=1), "BBB": _noisy_price(seed=2)})


def _forecast():
    rows = [
        # AAA: both models bullish -> agree
        ("AAA", 1, 0.70, 0.10, 0.030, "linear"), ("AAA", 1, 0.65, 0.15, 0.020, "gbm"),
        # BBB: models disagree
        ("BBB", 1, 0.55, 0.30, 0.010, "linear"), ("BBB", -1, 0.20, 0.60, -0.015, "gbm"),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "pred", "p_up", "p_down", "exp_ret", "model"])
    df["horizon"] = "short"
    df["date"] = pd.Timestamp("2026-05-28")
    return df


def test_ensemble_agreement_flag():
    ens = score.ensemble_frame(_forecast(), "short").set_index("symbol")
    assert ens.loc["AAA", "agree"]            # both bullish
    assert not ens.loc["BBB", "agree"]        # disagree


def test_score_and_rank_qualifies_and_orders(store2):
    cfg = load_config()
    ens = score.ensemble_frame(_forecast(), "short")
    ranked = score.score_and_rank(ens, store2, cfg)
    assert "AAA" in set(ranked["symbol"])           # high-conviction bullish idea qualifies
    assert ranked["idea_score"].is_monotonic_decreasing
    assert (ranked["downside_vol"] > 0).all()


def test_pinning_forces_low_conviction_idea_in(store2):
    cfg = load_config().with_overrides(
        {"recommend": {"min_conviction": 0.99, "pinned_symbols": ["BBB"]}})
    ens = score.ensemble_frame(_forecast(), "short")
    ranked = score.score_and_rank(ens, store2, cfg)
    assert "BBB" in set(ranked["symbol"])           # pinned despite failing the 0.99 gate
    assert bool(ranked.set_index("symbol").loc["BBB", "pinned"])


def test_asset_class_filter_restricts(store2):
    # AAA/BBB map to "equities" (fillna); restricting to rates leaves nothing unless pinned
    cfg = load_config().with_overrides({"recommend": {"include_asset_classes": ["rates"]}})
    ens = score.ensemble_frame(_forecast(), "short")
    ranked = score.score_and_rank(ens, store2, cfg)
    assert ranked.empty
    cfg_eq = load_config().with_overrides({"recommend": {"include_asset_classes": ["equities"]}})
    assert not score.score_and_rank(ens, store2, cfg_eq).empty


def test_portfolio_respects_caps():
    cfg = load_config()
    rb = cfg.risk_budget
    # many high-score ideas across both classes to push against the caps
    ideas = pd.DataFrame({
        "symbol": [f"E{i}" for i in range(8)] + [f"R{i}" for i in range(4)],
        "direction": [1, -1, 1, 1, -1, 1, 1, -1] + [1, 1, -1, 1],
        "idea_score": np.linspace(1.0, 0.4, 12),
        "asset_class": ["equities"] * 8 + ["rates"] * 4,
    })
    alloc, summary = portfolio.build_portfolio(ideas, cfg)
    assert alloc["weight"].abs().max() <= rb["per_position_cap"] + 1e-9
    assert summary["gross"] <= rb["max_leverage"] + 1e-9
    for cls, g in summary["by_class"].items():
        assert g <= rb["per_asset_class_cap"] + 1e-9
    # dollar sizing consistent with weights and notional
    assert np.allclose(alloc["dollars"], (alloc["weight"] * rb["notional_usd"]).round(0))


def test_empty_ideas_portfolio():
    alloc, summary = portfolio.build_portfolio(pd.DataFrame(
        columns=["symbol", "direction", "idea_score", "asset_class"]), load_config())
    assert alloc.empty and summary["n_positions"] == 0


def test_payoff_long_call_shape():
    p = payoff.long_option(spot=100.0, bullish=True, otm_pct=0.03)
    x, y = np.array(p.x), np.array(p.y)
    k = p.markers["strike"]
    below, above = y[x < k], y[x > k]
    assert np.allclose(below, below[0])        # flat (just the premium) below the strike
    assert above[-1] > above[0]                # rising above the strike
    assert y.min() < 0                          # max loss = premium


def test_payoff_spread_is_capped():
    p = payoff.vertical_spread(spot=100.0, bullish=True, otm_pct=0.03, width_pct=0.05)
    y = np.array(p.y)
    assert y.max() < 100                        # defined, capped upside (not unbounded)
    assert y.min() < 0                          # defined max loss
