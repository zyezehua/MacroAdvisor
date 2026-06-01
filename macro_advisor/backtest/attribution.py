"""PnL attribution for a backtest (Phase 5).

Decomposes the net return stream produced by :func:`backtest.engine.run` into the pieces a user
actually wants to interrogate:

  * **per-asset** — gross PnL, trading cost and net PnL contributed by each symbol;
  * **long vs short** — how much gross PnL came from long legs vs short legs;
  * **waterfall** — gross PnL → costs → net, as additive cumulative-return contributions.

Everything is expressed as a **cumulative return contribution** (sum of daily ``weightₜ · returnₜ``),
which is additive across assets and reconciles to the strategy's cumulative simple return — so the
bars literally add up to the headline number. Pure pandas over the engine's per-asset frames.
"""
from __future__ import annotations

import pandas as pd


def attribute(result: dict) -> dict:
    """Build per-asset / long-short / waterfall attribution from an ``engine.run`` result."""
    contrib: pd.DataFrame = result.get("contrib", pd.DataFrame())
    cost: pd.DataFrame = result.get("cost_by_asset", pd.DataFrame())
    weights: pd.DataFrame = result.get("weights", pd.DataFrame())
    if contrib.empty:
        return {"per_asset": pd.DataFrame(), "long_short": {}, "waterfall": {}}

    gross_by_asset = contrib.sum(axis=0)
    cost_by_asset = cost.sum(axis=0) if not cost.empty else gross_by_asset * 0.0
    per_asset = pd.DataFrame({
        "gross_pnl": gross_by_asset,
        "cost": cost_by_asset,
        "net_pnl": gross_by_asset - cost_by_asset,
    }).sort_values("net_pnl", ascending=False)
    per_asset.index.name = "symbol"

    # long vs short gross PnL: split each asset's daily contribution by its position sign.
    # pandas .sum() skips NaN (the final shifted-return row is all-NaN but survives dropna).
    long_gross = float(contrib.where(weights > 0).sum().sum()) if not weights.empty else float("nan")
    short_gross = float(contrib.where(weights < 0).sum().sum()) if not weights.empty else float("nan")

    gross_total = float(gross_by_asset.sum())
    cost_total = float(cost_by_asset.sum())
    waterfall = {"gross_pnl": round(gross_total, 4),
                 "cost": round(-cost_total, 4),
                 "net_pnl": round(gross_total - cost_total, 4)}

    return {
        "per_asset": per_asset.round(4).reset_index(),
        "long_short": {"long_gross": round(float(long_gross), 4),
                       "short_gross": round(float(short_gross), 4)},
        "waterfall": waterfall,
    }
