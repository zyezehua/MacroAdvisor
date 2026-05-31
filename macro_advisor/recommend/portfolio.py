"""Construct a risk-budgeted portfolio from ranked ideas.

Sizes positions in proportion to their risk-adjusted ``idea_score``, then enforces the locked
risk budget — per-position cap, per-asset-class cap, gross leverage — and converts to dollar
amounts against the notional. The cap enforcement is order-deterministic and guarantees every
output respects all three limits (asserted in the tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.config import Config


def build_portfolio(ideas: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Return (allocation rows, summary). ``ideas`` must have idea_score/direction/asset_class."""
    rb = cfg.risk_budget
    notional = float(rb.get("notional_usd", 250_000))
    pos_cap = float(rb.get("per_position_cap", 0.15))
    class_cap = float(rb.get("per_asset_class_cap", 0.60))
    max_lev = float(rb.get("max_leverage", 1.0))

    if ideas.empty:
        return pd.DataFrame(columns=["symbol", "direction", "weight", "dollars", "asset_class"]), \
            {"gross": 0.0, "notional": notional, "n_positions": 0, "by_class": {}}

    df = ideas.copy()
    score = df["idea_score"].clip(lower=0)
    mag = (score / score.sum()) * max_lev if score.sum() > 0 else pd.Series(0.0, index=df.index)
    w = df["direction"].to_numpy() * mag.to_numpy()

    # 1) per-position cap
    w = np.clip(w, -pos_cap, pos_cap)
    df["w"] = w

    # 2) per-asset-class gross cap
    for cls, idx in df.groupby("asset_class").groups.items():
        gross_cls = df.loc[idx, "w"].abs().sum()
        if gross_cls > class_cap:
            df.loc[idx, "w"] *= class_cap / gross_cls

    # 3) gross leverage cap
    gross = df["w"].abs().sum()
    if gross > max_lev:
        df["w"] *= max_lev / gross

    df = df[df["w"].abs() > 1e-9].copy()
    df["direction"] = np.where(df["w"] >= 0, "long", "short")
    df["weight"] = df["w"].round(4)
    df["dollars"] = (df["weight"] * notional).round(0)   # consistent with the displayed weight
    alloc = df[["symbol", "direction", "weight", "dollars", "asset_class"]].reset_index(drop=True)

    by_class = df.groupby("asset_class")["w"].apply(lambda s: round(float(s.abs().sum()), 4)).to_dict()
    summary = {"gross": round(float(df["w"].abs().sum()), 4),
               "net": round(float(df["w"].sum()), 4),
               "notional": notional, "n_positions": int(len(alloc)), "by_class": by_class}
    return alloc, summary
