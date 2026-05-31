"""Composite market-stress index with per-component decomposition.

Pipeline:
  1. Re-orient each signal into *stress space*: ``stress = -score`` so +1 = maximum
     stress (risk-off) and -1 = calm (risk-on).
  2. Group signals into six components (volatility, credit, rates, momentum, breadth,
     cross_asset) and average the members of each.
  3. Blend components with **fixed expert weights** (``config/settings.yaml`` ``stress:``),
     renormalizing per-date over whichever components are present, into a latent in [-1, 1].
  4. Map the latent to a 0-100 level via a logistic curve and label it with configurable bands.

The result carries the full history, the latest level + label, the per-component
contributions (which sum to the latent), and the top signal drivers — so every reading is
traceable back to the signals and, through them, to vetted data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macro_advisor.data import MarketStore
from macro_advisor.signals import compute_all
from macro_advisor.signals.base import SignalResult

# Map each signal name to its stress component. The six components match the weight keys
# in settings.yaml. Anything unmapped falls back to the signal's own category.
COMPONENT_OF: dict[str, str] = {
    # volatility
    "vix_level": "volatility", "vix_term": "volatility",
    "vxn_premium": "volatility", "move": "volatility",
    # credit
    "hy_ig": "credit", "hy_oas": "credit",
    # rates
    "curve_2s10s": "rates", "curve_3m10y": "rates",
    "level_move": "rates", "real_yield": "rates", "breakeven": "rates",
    # momentum (directional technicals)
    "trend": "momentum", "momentum": "momentum", "rsi": "momentum",
    # breadth
    "breadth": "breadth",
    # cross-asset
    "dollar": "cross_asset", "stock_bond_corr": "cross_asset",
}

_DEFAULT_WEIGHTS = {
    "volatility": 0.25, "credit": 0.20, "rates": 0.20,
    "momentum": 0.15, "breadth": 0.10, "cross_asset": 0.10,
}
_DEFAULT_BANDS = {"calm": 30, "normal": 55, "elevated": 70, "stressed": 85}
_LATENT_SCALE = 3.0     # spreads a [-1, 1] latent across the logistic's responsive range


@dataclass
class ComponentContribution:
    component: str
    weight: float           # renormalized weight actually applied (latest date)
    stress: float           # component mean stress at the latest date, [-1, 1]
    contribution: float     # weight * stress (sums across components to the latent)


@dataclass
class StressResult:
    level: float                                  # 0-100, latest
    label: str
    latent: float                                 # [-1, 1], latest
    asof: pd.Timestamp
    history: pd.Series                            # 0-100 over time
    components: list[ComponentContribution]
    top_drivers: list[str]                        # human-readable, most stressful first
    n_signals: int
    component_history: pd.DataFrame = field(default_factory=pd.DataFrame)


def _band_label(level: float, bands: dict[str, float]) -> str:
    if level <= bands["calm"]:
        return "calm"
    if level <= bands["normal"]:
        return "normal"
    if level <= bands["elevated"]:
        return "elevated"
    if level <= bands["stressed"]:
        return "stressed"
    return "crisis"


def _component_frame(signals: dict[str, SignalResult]) -> pd.DataFrame:
    """Average each component's member signals (in stress space) into one column per component."""
    by_comp: dict[str, list[pd.Series]] = {}
    for name, sig in signals.items():
        comp = COMPONENT_OF.get(name, sig.category)
        by_comp.setdefault(comp, []).append((-sig.score).rename(name))   # stress = -score
    cols = {}
    for comp, members in by_comp.items():
        cols[comp] = pd.concat(members, axis=1).mean(axis=1, skipna=True)
    return pd.DataFrame(cols).sort_index()


def _latent(comp_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Per-date weighted mean of component stress, renormalizing over present components."""
    w = pd.Series({c: weights.get(c, 0.0) for c in comp_df.columns}, dtype=float)
    mask = comp_df.notna()
    num = comp_df.mul(w, axis=1).sum(axis=1, skipna=True)
    den = mask.mul(w, axis=1).sum(axis=1)
    return (num / den.replace(0.0, np.nan)).rename("latent")


def compute_stress(store: MarketStore,
                   signals: dict[str, SignalResult] | None = None) -> StressResult:
    """Compute the composite stress index from the live signal set."""
    signals = signals if signals is not None else compute_all(store)
    cfg = store.cfg.stress
    weights = {**_DEFAULT_WEIGHTS, **(cfg.get("weights") or {})}
    bands = {**_DEFAULT_BANDS, **(cfg.get("bands") or {})}
    logistic_k = float(cfg.get("logistic_k", 1.0))

    if not signals:
        raise ValueError("no signals available to compute stress")

    comp_df = _component_frame(signals)
    latent = _latent(comp_df, weights).dropna()
    if latent.empty:
        raise ValueError("stress latent is empty (no overlapping component history)")

    level_hist = 100.0 / (1.0 + np.exp(-logistic_k * _LATENT_SCALE * latent))
    level_hist = level_hist.rename("stress")

    asof = pd.Timestamp(latent.index.max())
    latent_now = float(latent.iloc[-1])
    level_now = float(level_hist.iloc[-1])

    # latest-date contributions, with weights renormalized over present components
    last_row = comp_df.loc[asof] if asof in comp_df.index else comp_df.iloc[-1]
    present = last_row.dropna().index.tolist()
    wsum = sum(weights.get(c, 0.0) for c in present) or 1.0
    contribs = [
        ComponentContribution(
            component=c,
            weight=weights.get(c, 0.0) / wsum,
            stress=float(last_row[c]),
            contribution=(weights.get(c, 0.0) / wsum) * float(last_row[c]),
        )
        for c in present
    ]
    contribs.sort(key=lambda x: x.contribution, reverse=True)

    # top signal drivers: largest latest stress (most risk-off) first
    ranked = sorted(
        signals.values(),
        key=lambda s: (-(s.latest_score) if not np.isnan(s.latest_score) else -np.inf),
        reverse=True,
    )
    top_drivers = [
        f"[{COMPONENT_OF.get(s.name, s.category)}] {s.name}: {s.attribution} "
        f"({'+stress' if -s.latest_score > 0 else '-stress'} {-s.latest_score:+.2f})"
        for s in ranked
        if not np.isnan(s.latest_score)
    ]

    return StressResult(
        level=level_now,
        label=_band_label(level_now, bands),
        latent=latent_now,
        asof=asof,
        history=level_hist,
        components=contribs,
        top_drivers=top_drivers,
        n_signals=len(signals),
        component_history=comp_df,
    )
