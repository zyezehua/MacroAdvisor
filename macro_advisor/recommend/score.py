"""Score + rank trade ideas from the OOS forecasts.

Consumes ``data/oos/forecast.parquet`` (per-asset direction / p_up / p_down / expected return,
per model × horizon) and ranks ideas by a **Sortino-spirit** risk-adjusted score:

    idea_score = direction · expected_return / downside_vol

where ``downside_vol`` is the asset's trailing downside deviation (annualized). An *ensemble*
combines the two model families; per-model lists are also available for the detail view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.config import Config
from macro_advisor.data import MarketStore

EQUITIES, RATES = "equities", "rates"


def class_map(cfg: Config) -> dict[str, str]:
    """Map each universe symbol to its asset class (for the per-asset-class cap)."""
    m = {s: EQUITIES for s in cfg.yahoo_symbols("backtest_equity")}
    m.update({s: RATES for s in cfg.yahoo_symbols("backtest_rates")})
    return m


def downside_vol_latest(price: pd.Series, window: int) -> float:
    """Trailing annualized downside deviation of daily returns (latest value)."""
    r = price.pct_change().dropna().iloc[-window:]
    downside = r[r < 0]
    if downside.empty:
        return float("nan")
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(252.0))


def _sign(x: float) -> int:
    return int(np.sign(x))


def ensemble_frame(forecast: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Per-symbol ensemble of the model forecasts for one horizon."""
    h = forecast[forecast["horizon"] == horizon]
    rows = []
    for sym, g in h.groupby("symbol"):
        ens_up, ens_down, ens_ret = g["p_up"].mean(), g["p_down"].mean(), g["exp_ret"].mean()
        per = {f"{r['model']}_dir": _sign(r["pred"]) for _, r in g.iterrows()}
        per.update({f"{r['model']}_conf": max(r["p_up"], r["p_down"]) for _, r in g.iterrows()})
        signs = {_sign(r["pred"]) for _, r in g.iterrows() if r["pred"] != 0}
        rows.append({"symbol": sym, "p_up": ens_up, "p_down": ens_down, "exp_ret": ens_ret,
                     "agree": len(signs) == 1 and len(g) > 1, **per})
    return pd.DataFrame(rows)


def single_frame(forecast: pd.DataFrame, horizon: str, model: str) -> pd.DataFrame:
    h = forecast[(forecast["horizon"] == horizon) & (forecast["model"] == model)]
    return h[["symbol", "p_up", "p_down", "exp_ret"]].assign(agree=False).reset_index(drop=True)


def score_and_rank(frame: pd.DataFrame, store: MarketStore, cfg: Config) -> pd.DataFrame:
    """Add direction/conviction/downside-vol/idea_score/asset_class; filter + rank."""
    rc = cfg.recommend
    window = int(rc.get("downside_vol_window", 63))
    min_conv = float(rc.get("min_conviction", cfg.predict.get("prob_threshold", 0.5)))
    min_dvol = float(rc.get("min_downside_vol", 0.03))
    require_agree = bool(rc.get("require_agreement", False))
    exclude = set(rc.get("exclude_symbols", []))
    cmap = class_map(cfg)

    out = frame[~frame["symbol"].isin(exclude)].copy()
    out["direction"] = np.where(out["p_up"] >= out["p_down"], 1, -1)
    # conviction = directional split (how lopsided up vs down is), robust to the 3-class
    # models putting most probability mass on the "flat" class.
    denom = (out["p_up"] + out["p_down"]).replace(0, np.nan)
    out["conviction"] = out[["p_up", "p_down"]].max(axis=1) / denom
    out["asset_class"] = out["symbol"].map(cmap).fillna("equities")
    dvol = {}
    for sym in out["symbol"]:
        px = store.try_price(sym)
        dvol[sym] = downside_vol_latest(px, window) if px is not None else float("nan")
    # floor the downside vol so near-cash ETFs don't get inflated risk-adjusted scores
    out["downside_vol"] = out["symbol"].map(dvol).clip(lower=min_dvol)
    out["idea_score"] = out["direction"] * out["exp_ret"] / out["downside_vol"]

    qualified = out["conviction"] >= min_conv
    if require_agree and "agree" in out:
        qualified &= out["agree"]
    out = out[qualified & out["downside_vol"].notna() & out["idea_score"].notna()]
    out = out.sort_values("idea_score", ascending=False).reset_index(drop=True)
    return out.head(int(rc.get("max_ideas", 10)))
