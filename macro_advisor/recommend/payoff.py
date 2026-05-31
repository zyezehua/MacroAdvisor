"""Structured-payoff *illustrations* — payoff-at-expiry shapes only.

This is deliberately **not** an option pricer: there are no greeks, no implied vol, no fair value.
It draws the expiry P&L shape of a simple structure (a long call/put for an outright directional
view, or a vertical spread for a defined-risk version) so the dashboard can show *how* a view could
be expressed. Premiums are nominal placeholders for the diagram, clearly labelled as illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Payoff:
    label: str
    x: list[float]           # underlying price grid
    y: list[float]           # P&L at expiry (per 1 unit of underlying notional)
    markers: dict = field(default_factory=dict)   # e.g. {"spot": .., "strike": ..}


def _grid(spot: float, span: float = 0.20, n: int = 81) -> np.ndarray:
    return np.linspace(spot * (1 - span), spot * (1 + span), n)


def vertical_spread(spot: float, bullish: bool, otm_pct: float = 0.03,
                    width_pct: float = 0.05) -> Payoff:
    """Defined-risk directional structure (debit call spread / put spread). Illustration only."""
    x = _grid(spot)
    if bullish:
        k_long, k_short = spot * (1 + otm_pct), spot * (1 + otm_pct + width_pct)
        intrinsic = np.clip(x - k_long, 0, None) - np.clip(x - k_short, 0, None)
    else:
        k_long, k_short = spot * (1 - otm_pct), spot * (1 - otm_pct - width_pct)
        intrinsic = np.clip(k_long - x, 0, None) - np.clip(k_short - x, 0, None)
    premium = spot * width_pct * 0.35                      # nominal debit, illustrative only
    y = intrinsic - premium
    kind = "call" if bullish else "put"
    return Payoff(label=f"{kind} debit spread (illustrative)", x=list(x), y=list(np.round(y, 2)),
                  markers={"spot": spot, "k_long": round(k_long, 2), "k_short": round(k_short, 2)})


def long_option(spot: float, bullish: bool, otm_pct: float = 0.03) -> Payoff:
    """Outright long call/put. Illustration only."""
    x = _grid(spot)
    k = spot * (1 + otm_pct) if bullish else spot * (1 - otm_pct)
    intrinsic = np.clip(x - k, 0, None) if bullish else np.clip(k - x, 0, None)
    premium = spot * 0.02                                  # nominal, illustrative only
    y = intrinsic - premium
    kind = "call" if bullish else "put"
    return Payoff(label=f"long {kind} (illustrative)", x=list(x), y=list(np.round(y, 2)),
                  markers={"spot": spot, "strike": round(k, 2)})


def illustrate(spot: float, direction: int, *, otm_pct: float = 0.03,
               defined_risk: bool = True) -> Payoff:
    """Pick a structure for the trade direction (+1 long / -1 short view)."""
    bullish = direction >= 0
    return (vertical_spread(spot, bullish, otm_pct) if defined_risk
            else long_option(spot, bullish, otm_pct))
