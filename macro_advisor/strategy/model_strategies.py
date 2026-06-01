"""Tunable strategies driven by the **OOS model predictions** (Phase 5).

Unlike the in-sample Strategy Lab (which scores raw signals with user-chosen thresholds), these
strategies trade the genuinely out-of-sample directional forecasts shipped in
``data/oos/oos_predictions.parquet`` (``date×symbol×model×horizon`` + ``pred``/``p_up``/``p_down``).
The app re-runs the vectorized backtester live as the user moves the knobs — pure pandas, no ML
dependency — so it stays OOS-credible while being fully interactive.

Knobs (``ModelStrategyParams``): model family, horizon, conviction (signal) threshold, rebalance /
roll frequency, **minimum holding period** (hysteresis, distinct from rebalance), long-short vs
long-only, sizing, leverage / per-position caps, trading costs, and an optional **stress gate**
(defensive-equity or risk-on/off rotation across the equity & rates sleeves).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macro_advisor.backtest import attribution, engine, metrics
from macro_advisor.config import Config

REBALANCES = ("daily", "weekly", "monthly")
GATES = (None, "defensive_equity", "rotation")


@dataclass
class ModelStrategyParams:
    model: str = "stack"
    horizon: str = "short"               # short | med_long
    conviction: float = 0.0              # min directional split max(p_up,p_down)/(p_up+p_down)
    rebalance: str = "daily"             # daily | weekly | monthly (roll frequency)
    holding_period: int = 0              # min trading days a position is held before it can flip
    direction: str = "long_short"        # long_short | long_only
    sizing: str = "vol_target"           # vol_target | equal
    vol_target: float = 0.10
    max_leverage: float = 1.0
    per_position_cap: float = 0.15
    cost_bps: float = 2.0
    slippage_bps: float = 1.0
    gate: str | None = None              # None | defensive_equity | rotation
    stress_max: float = 60.0             # stress gate threshold (0-100)
    universe: list[str] = field(default_factory=list)


# -- position shaping -------------------------------------------------------

def _resample_hold(direction: pd.Series, rebalance: str) -> pd.Series:
    """Hold the direction between rebalances (causal: carry the period's last signal forward)."""
    if rebalance == "daily":
        return direction
    rule = {"weekly": "W-FRI", "monthly": "ME"}[rebalance]
    return direction.resample(rule).last().reindex(direction.index, method="ffill")


def _min_hold(direction: pd.Series, holding_period: int) -> pd.Series:
    """Enforce a minimum holding period: a position can't change within ``holding_period`` days
    of its last change (simple hysteresis), so the holding period is decoupled from rebalance."""
    if holding_period <= 1:
        return direction
    vals = direction.to_numpy(dtype=float)
    out = vals.copy()
    held, last_change = vals[0], 0
    for i in range(1, len(vals)):
        if vals[i] != held and (i - last_change) >= holding_period:
            held, last_change = vals[i], i
        out[i] = held
    return pd.Series(out, index=direction.index)


def _apply_gate(sym_dir: pd.Series, asset_class: str, stress: pd.Series | None,
                params: ModelStrategyParams) -> pd.Series:
    """Stress-regime gate. ``defensive_equity`` zeroes equity longs in high stress; ``rotation``
    runs equities only in low stress and rates only in high stress (each on the model direction)."""
    if params.gate is None or stress is None:
        return sym_dir
    st = stress.reindex(sym_dir.index).ffill()
    if params.gate == "defensive_equity" and asset_class == "equities":
        return sym_dir.where(~((sym_dir > 0) & (st >= params.stress_max)), 0.0)
    if params.gate == "rotation":
        active = st < params.stress_max if asset_class == "equities" else st >= params.stress_max
        return sym_dir.where(active, 0.0)
    return sym_dir


def build_preds(oos_pred: pd.DataFrame, params: ModelStrategyParams, *,
                asset_class: dict[str, str] | None = None,
                stress: pd.Series | None = None) -> pd.DataFrame:
    """Shape the OOS model directions into an engine-ready ``preds`` frame for one strategy."""
    df = oos_pred[(oos_pred["model"] == params.model) & (oos_pred["horizon"] == params.horizon)].copy()
    if params.universe:
        df = df[df["symbol"].isin(params.universe)]
    if df.empty:
        return pd.DataFrame()
    denom = (df["p_up"] + df["p_down"]).replace(0, np.nan)
    df["conv"] = df[["p_up", "p_down"]].max(axis=1) / denom
    df.loc[df["conv"] < params.conviction, "pred"] = 0.0

    amap = asset_class or {}
    rows = []
    for sym, g in df.groupby("symbol"):
        s = g.set_index("date")["pred"].astype(float).sort_index()
        s = _apply_gate(s, amap.get(sym, "equities"), stress, params)
        s = _resample_hold(s, params.rebalance)
        s = _min_hold(s, params.holding_period)
        if params.direction == "long_only":
            s = s.clip(lower=0.0)
        rows.append(pd.DataFrame({"date": s.index, "symbol": sym, "pred": s.to_numpy()}))
    preds = pd.concat(rows, ignore_index=True)
    preds["p_up"] = (preds["pred"] == 1).astype(float)
    preds["p_down"] = (preds["pred"] == -1).astype(float)
    return preds.set_index(["date", "symbol"]).sort_index()


def _strategy_cfg(cfg: Config, params: ModelStrategyParams) -> Config:
    """Config whose risk budget / sizing / costs reflect the strategy params for the engine."""
    pbt: dict = {"long_only": params.direction == "long_only"}
    if params.sizing == "equal":
        pbt["vol_target_annual"], pbt["max_position"] = 10.0, params.per_position_cap
    else:
        pbt["vol_target_annual"], pbt["max_position"] = params.vol_target, 1.0
    return cfg.with_overrides({
        "risk_budget": {"per_position_cap": params.per_position_cap, "max_leverage": params.max_leverage},
        "predict": {"prob_threshold": 0.5, "backtest": pbt},   # conviction already applied here
        "backtest": {"cost_bps_per_trade": params.cost_bps, "slippage_bps": params.slippage_bps},
    })


def run_strategy(oos_pred: pd.DataFrame, prices: dict[str, pd.Series], cfg: Config,
                 params: ModelStrategyParams, *, asset_class: dict[str, str] | None = None,
                 stress: pd.Series | None = None, benchmark: str = "SPY",
                 rf_daily=0.0) -> dict:
    """Backtest one model-signal strategy; return equity, full stats, PnL attribution, monthly table."""
    preds = build_preds(oos_pred, params, asset_class=asset_class, stress=stress)
    if preds.empty or (preds["pred"] != 0).sum() == 0:
        return {"error": "strategy took no positions over the available OOS history"}
    sub = {s: p for s, p in prices.items() if s in preds.index.get_level_values("symbol")}
    if not sub:
        return {"error": "none of the strategy's symbols are in the price cache"}

    res = engine.run(preds, sub, cfg=_strategy_cfg(cfg, params), rf_daily=rf_daily)
    ret = res["returns"]
    if ret.empty:
        return {"error": "strategy took no positions over the available OOS history"}

    bench_ret = None
    equity = {params.model + " strategy": res["equity"]}
    if benchmark in prices:
        b = engine.benchmark(prices, benchmark, ret.index, rf_daily=rf_daily)
        bench_ret, equity[benchmark] = b["returns"], b["equity"]

    stats = metrics.extended(ret, rf_daily, benchmark=bench_ret,
                             exposure=res.get("gross_exposure"), turnover=res.get("turnover"))
    return {
        "equity": pd.DataFrame(equity).dropna(how="all").sort_index(),
        "stats": stats,
        "attribution": attribution.attribute(res),
        "monthly": metrics.monthly_returns(ret),
        "returns": ret,
        "window": (str(ret.index.min().date()), str(ret.index.max().date())),
    }


def default_strategies() -> dict[str, dict]:
    """Named starter strategies (params + a one-line description) for the dashboard picker."""
    eq = ["SPY", "QQQ", "IWM"]
    rates = ["IEF", "TLT"]
    return {
        "Ensemble Directional — Equities": {
            "params": ModelStrategyParams(model="stack", horizon="short", rebalance="weekly",
                                          universe=eq),
            "desc": "Trade the stacked-ensemble short-horizon direction on equity ETFs, vol-targeted."},
        "Cross-Asset Ensemble": {
            "params": ModelStrategyParams(model="stack", horizon="med_long", rebalance="weekly",
                                          universe=eq + rates),
            "desc": "Medium-horizon ensemble direction across equities + rates."},
        "Stress-Gated Equity Trend": {
            "params": ModelStrategyParams(model="stack", horizon="short", direction="long_only",
                                          conviction=0.55, rebalance="weekly", gate="defensive_equity",
                                          stress_max=60.0, universe=eq),
            "desc": "Long equities on an up signal, but stand aside when composite stress ≥ 60."},
        "Risk-On/Off Rotation": {
            "params": ModelStrategyParams(model="stack", horizon="med_long", direction="long_only",
                                          rebalance="monthly", gate="rotation", stress_max=55.0,
                                          universe=eq + rates),
            "desc": "Equities in calm regimes, rotate to duration (rates) when stress ≥ 55."},
        "High-Conviction Ensemble": {
            "params": ModelStrategyParams(model="stack", horizon="short", conviction=0.62,
                                          rebalance="weekly", universe=eq + rates),
            "desc": "Only act on high-conviction calls (leans on Phase-4 calibration)."},
    }


def class_map(cfg: Config) -> dict[str, str]:
    """Symbol -> asset class (equities / rates), for the stress gates."""
    m = {s: "equities" for s in cfg.yahoo_symbols("backtest_equity")}
    m.update({s: "rates" for s in cfg.yahoo_symbols("backtest_rates")})
    return m
