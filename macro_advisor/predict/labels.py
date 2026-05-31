"""Forward-looking labels for OOS prediction.

Labels are intentionally *future* (the thing we forecast); the no-leakage discipline applies to
**features** (see ``features``/``transform``) and to the walk-forward split (see ``walkforward``),
never to labels. Each label at date ``t`` summarizes the window ``(t, t+h]`` and is therefore only
realized at ``t+h`` — the walk-forward engine embargoes around that horizon so a training row's
label can't peek into its test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# label kinds used across the predict layer
DIRECTION = "direction"   # classification: -1 / 0 / +1
MAGNITUDE = "magnitude"   # regression: forward return
STRESS = "stress"         # regression: forward change in the stress level


def forward_return(price: pd.Series, h: int) -> pd.Series:
    """Return over the next ``h`` trading days: price[t+h]/price[t] - 1, indexed at t."""
    return price.shift(-h) / price - 1.0


def direction(fwd_ret: pd.Series, neutral_band: float, h: int) -> pd.Series:
    """Map a forward return to -1/0/+1. The band scales with sqrt(h) (vol ~ sqrt-time)."""
    band = neutral_band * np.sqrt(h)
    out = pd.Series(0, index=fwd_ret.index, dtype="int64")
    out[fwd_ret > band] = 1
    out[fwd_ret < -band] = -1
    return out.where(fwd_ret.notna())


def asset_labels(price: pd.Series, h: int, neutral_band: float) -> pd.DataFrame:
    """Per-asset forward return + direction for one horizon, indexed by date."""
    fwd = forward_return(price, h)
    return pd.DataFrame({"fwd_ret": fwd, "direction": direction(fwd, neutral_band, h)})


def stress_label(level: pd.Series, h: int) -> pd.Series:
    """Forward change in the composite stress level over ``h`` days, indexed at t."""
    return (level.shift(-h) - level).rename("fwd_stress_chg")
