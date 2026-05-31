"""Evaluate a :class:`StrategySpec` by reusing the Phase-2 vectorized backtester.

Flow: resolve each rule's input (market-level signal / stress level, or a per-asset technical),
score every asset on every date, derive a (rebalanced) direction, synthesize a ``preds`` frame in
the shape the engine expects, and call :func:`backtest.engine.run` with a Config whose risk budget
reflects the spec. A SPY buy-&-hold benchmark is aligned to the strategy window.

Causality is inherited: signals are causal, per-asset technicals use the causal ``transform``
helpers, and the engine earns each weight the asset's *next-day* return. The user-chosen
thresholds are not walk-forward-fitted, so this is an in-sample exploration tool by design.
"""
from __future__ import annotations

import operator as _op

import numpy as np
import pandas as pd

from macro_advisor.backtest import engine, metrics
from macro_advisor.config import Config
from macro_advisor.data import MarketStore
from macro_advisor.signals import compute_all
from macro_advisor.signals import transform as tf
from macro_advisor.stress import compute_stress
from macro_advisor.strategy.spec import StrategySpec

_OPS = {">": _op.gt, ">=": _op.ge, "<": _op.lt, "<=": _op.le}

# per-asset technical inputs (prefix px_); computed from each asset's own price.
_ASSET_INPUTS = ("px_ret_21", "px_ret_63", "px_ret_126", "px_rsi",
                 "px_sma_gap_50", "px_sma_gap_200", "px_rvol_21")
_MARKET_EXTRA = {"stress_level": "Composite stress 0–100 (higher = more risk-off)"}


def available_inputs(store: MarketStore | None = None,
                     signal_names: list[str] | None = None) -> dict[str, dict[str, str]]:
    """Inputs the rule-builder can reference: market signal scores + stress level + per-asset
    technicals. ``signal_names`` lets the caller avoid recomputing signals."""
    if signal_names is None:
        signal_names = sorted(compute_all(store).keys()) if store is not None else []
    out: dict[str, dict[str, str]] = {}
    for name in signal_names:
        out[name] = {"kind": "market", "desc": f"signal score {name} (−1 risk-off … +1 risk-on)"}
    for name, desc in _MARKET_EXTRA.items():
        out[name] = {"kind": "market", "desc": desc}
    asset_desc = {
        "px_ret_21": "21d price return", "px_ret_63": "63d price return",
        "px_ret_126": "126d price return", "px_rsi": "Wilder RSI (0–100)",
        "px_sma_gap_50": "price vs 50d MA (fraction)", "px_sma_gap_200": "price vs 200d MA (fraction)",
        "px_rvol_21": "21d annualized realized vol",
    }
    for name in _ASSET_INPUTS:
        out[name] = {"kind": "asset", "desc": asset_desc[name]}
    return out


def _asset_input_frame(price: pd.Series) -> pd.DataFrame:
    """Per-asset technical inputs from one price series (all causal)."""
    return pd.DataFrame({
        "px_ret_21": tf.ret(price, 21),
        "px_ret_63": tf.ret(price, 63),
        "px_ret_126": tf.ret(price, 126),
        "px_rsi": tf.wilder_rsi(price, 14),
        "px_sma_gap_50": price / tf.sma(price, 50) - 1.0,
        "px_sma_gap_200": price / tf.sma(price, 200) - 1.0,
        "px_rvol_21": tf.realized_vol(price, 21),
    }).sort_index()


def _market_frame(store: MarketStore) -> pd.DataFrame:
    """Wide date-indexed frame of market-level inputs (signal scores + stress level)."""
    signals = compute_all(store)
    cols = {name: sig.score for name, sig in signals.items()}
    df = pd.DataFrame(cols).sort_index()
    try:
        stress = compute_stress(store, signals)
        df["stress_level"] = stress.history
    except Exception:  # stress needs overlapping components; degrade to signals only
        pass
    return df.sort_index()


def _resample_direction(direction: pd.Series, rebalance: str) -> pd.Series:
    """Hold the direction between rebalances (causal: use the period's last signal, carried fwd)."""
    if rebalance == "daily":
        return direction
    rule = {"weekly": "W-FRI", "monthly": "M"}[rebalance]
    periodic = direction.resample(rule).last()
    return periodic.reindex(direction.index, method="ffill")


def _strategy_cfg(cfg: Config, spec: StrategySpec) -> Config:
    """Config whose risk budget / sizing reflect the spec, for the engine to consume."""
    rb = {k: float(v) for k, v in spec.caps.items()
          if k in ("per_position_cap", "per_asset_class_cap", "max_leverage")}
    pbt: dict[str, float | bool] = {"long_only": spec.direction == "long_only"}
    if spec.sizing == "equal":
        # uniform sizing: blow past vol-targeting so every taken position clips to the per-position
        # cap, then the engine's gross-scaling renders them equal-weight.
        pbt["vol_target_annual"] = 10.0
        pbt["max_position"] = float(spec.caps.get("per_position_cap", cfg.risk_budget["per_position_cap"]))
    patch: dict = {"predict": {"prob_threshold": 0.5, "backtest": pbt}}
    if rb:
        patch["risk_budget"] = rb
    return cfg.with_overrides(patch)


def evaluate(spec: StrategySpec, store: MarketStore, cfg: Config) -> dict:
    """Backtest ``spec`` and return equity curves (strategy + SPY), metrics, and a summary."""
    spec.validate()
    prices: dict[str, pd.Series] = {}
    for sym in spec.universe:
        px = store.try_price(sym)
        if px is not None and not px.empty:
            prices[sym] = px
    if not prices:
        return {"error": "none of the strategy's universe symbols are in the data cache"}

    market = _market_frame(store)
    market_inputs = {r.input for r in spec.rules if r.input in market.columns}
    asset_input_names = {r.input for r in spec.rules if r.input in _ASSET_INPUTS}

    rows = []
    for sym, px in prices.items():
        idx = px.index
        ainputs = _asset_input_frame(px) if asset_input_names else pd.DataFrame(index=idx)
        score = pd.Series(0.0, index=idx)
        for r in spec.rules:
            if r.input in market_inputs:
                series = market[r.input].reindex(idx)
            elif r.input in asset_input_names:
                series = ainputs[r.input]
            else:
                continue  # unknown input -> contributes nothing
            cond = _OPS[r.op](series, r.threshold).reindex(idx).fillna(False)
            score = score.add(r.weight * cond.astype(float), fill_value=0.0)
        direction = np.sign(score)
        if spec.direction == "long_only":
            direction = direction.clip(lower=0.0)
        direction = _resample_direction(direction, spec.rebalance).fillna(0.0)
        sub = pd.DataFrame({"date": idx, "symbol": sym, "pred": direction.astype(float).values})
        rows.append(sub)

    preds = pd.concat(rows, ignore_index=True)
    preds["p_up"] = (preds["pred"] == 1).astype(float)
    preds["p_down"] = (preds["pred"] == -1).astype(float)
    preds = preds.set_index(["date", "symbol"]).sort_index()

    res = engine.run(preds, prices, cfg=_strategy_cfg(cfg, spec))
    strat_ret = res["returns"]
    if strat_ret.empty:
        return {"error": "strategy took no positions over the available history"}

    equity = {spec.name: res["equity"]}
    metric_rows = [{"strategy": spec.name, **res["metrics"],
                    "avg_gross": round(res["avg_gross"], 3), "avg_turnover": round(res["avg_turnover"], 4)}]
    spy = store.try_price("SPY")
    if spy is not None and not spy.empty:
        bench = engine.benchmark({"SPY": spy}, "SPY", index=strat_ret.index)
        equity["SPY"] = bench["equity"]
        metric_rows.append({"strategy": "SPY (buy & hold)", **bench["metrics"],
                            "avg_gross": 1.0, "avg_turnover": 0.0})

    equity_df = pd.DataFrame(equity).dropna(how="all").sort_index()
    return {
        "equity": equity_df,
        "metrics": pd.DataFrame(metric_rows),
        "returns": strat_ret,
        "summary": {
            "n_assets": len(prices),
            "first": str(strat_ret.index.min().date()),
            "last": str(strat_ret.index.max().date()),
            "objective": cfg.risk_budget.get("ranking_objective", "sortino"),
        },
    }


# re-export for callers that want the metrics helpers alongside the evaluator
__all__ = ["available_inputs", "evaluate", "metrics"]
