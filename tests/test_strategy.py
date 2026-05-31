"""Custom-strategy spec round-trip + evaluator (reuses the backtester) + leakage checks."""
from __future__ import annotations

import numpy as np
import pytest

from macro_advisor.config import load_config
from macro_advisor.strategy import evaluate, spec_from_json, spec_to_json
from macro_advisor.strategy.spec import Rule, StrategySpec
from tests.conftest import bdays, price_frame


def _trend_price(n=500, drift=0.0004, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    return price_frame(100 * np.cumprod(1 + rets), bdays(n))


def test_spec_json_roundtrip():
    spec = StrategySpec(
        name="t", universe=["SPY", "QQQ"],
        rules=[Rule(input="px_ret_63", op=">", threshold=0.0, weight=1.0)],
        direction="long_only", sizing="equal", rebalance="monthly",
        caps={"per_position_cap": 0.2, "max_leverage": 1.0},
    )
    back = spec_from_json(spec_to_json(spec))
    assert back.to_dict() == spec.to_dict()


def test_evaluate_runs_and_benchmarks(store_factory):
    store = store_factory(prices={"SPY": _trend_price(seed=1), "QQQ": _trend_price(seed=2)})
    spec = StrategySpec(
        name="trend", universe=["SPY", "QQQ"],
        rules=[Rule(input="px_sma_gap_50", op=">", threshold=0.0, weight=1.0)],
        direction="long_only", sizing="vol_target", rebalance="weekly",
    )
    out = evaluate(spec, store, load_config())
    assert "error" not in out
    assert not out["equity"].empty
    assert "SPY" in out["equity"].columns               # benchmark attached
    assert set(out["metrics"]["strategy"]) >= {"trend", "SPY (buy & hold)"}


def test_always_long_tracks_spy_next_day(store_factory):
    """An always-invested, full-weight long on SPY earns SPY's next-day return exactly
    (no cost except the single entry day) — a tight correctness + causality check."""
    store = store_factory(prices={"SPY": _trend_price(seed=3)})
    spec = StrategySpec(
        name="always_long", universe=["SPY"],
        rules=[Rule(input="px_rvol_21", op=">", threshold=-1.0, weight=1.0)],  # always true once warm
        direction="long_only", sizing="equal", rebalance="daily",
        caps={"per_position_cap": 1.0, "max_leverage": 1.0},
    )
    out = evaluate(spec, store, load_config())
    strat = out["returns"]
    spy_fwd = store.price("SPY").pct_change().shift(-1).reindex(strat.index)
    invested = strat[strat != 0]
    diff = (strat - spy_fwd).reindex(invested.index).abs()
    assert diff.median() < 1e-9                          # equals SPY next-day return
    assert diff.max() < 0.001                            # only the entry-day cost differs


def test_evaluator_is_leakage_free_to_truncation(store_factory):
    """Truncating *future* price history must not change past strategy returns (causality)."""
    full_px = _trend_price(n=500, seed=4)
    spec = StrategySpec(
        name="trunc", universe=["SPY"],
        rules=[Rule(input="px_ret_21", op=">", threshold=0.0, weight=1.0)],
        direction="long_only", sizing="vol_target", rebalance="daily",
    )
    cfg = load_config()
    full = evaluate(spec, store_factory(prices={"SPY": full_px}), cfg)["returns"]
    trunc_px = full_px.iloc[:380]
    trunc = evaluate(spec, store_factory(prices={"SPY": trunc_px}), cfg)["returns"]
    common = trunc.index[:-2]                            # drop the fwd-return boundary
    assert len(common) > 100
    assert np.allclose(full.reindex(common), trunc.reindex(common), atol=1e-12)


def test_empty_universe_rejected():
    with pytest.raises(ValueError):
        StrategySpec(name="x", universe=[], rules=[Rule("px_ret_21", ">", 0.0)]).validate()
