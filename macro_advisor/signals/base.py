"""Signal contract + shared construction helper.

Every signal returns a :class:`SignalResult`: a *full historical* causal score series in
``[-1, 1]`` (orientation: **+1 = risk-on/favorable, -1 = risk-off/stress**), the underlying
raw measure for charts/audit, the latest direction, a one-line human attribution, and the
provenance keys it consumed. Returning the whole series (not just the latest value) is what
makes signals reusable by the Phase-2 backtester.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

RISK_ON = "risk_on"
RISK_OFF = "risk_off"
NEUTRAL = "neutral"

CATEGORIES = ("technical", "volatility", "credit", "rates", "cross_asset")


@dataclass
class SignalResult:
    name: str
    category: str
    score: pd.Series                 # causal, [-1, 1]; +risk_on / -risk_off
    raw: pd.Series                   # underlying measure
    direction: str                   # latest: risk_on | risk_off | neutral
    attribution: str                 # one-line explanation
    asof: pd.Timestamp
    inputs: list[str] = field(default_factory=list)

    @property
    def latest_score(self) -> float:
        return float(self.score.dropna().iloc[-1]) if not self.score.dropna().empty else float("nan")


def direction_of(score: float, neutral_band: float) -> str:
    if pd.isna(score) or abs(score) <= neutral_band:
        return NEUTRAL
    return RISK_ON if score > 0 else RISK_OFF


def build_signal(
    *,
    name: str,
    category: str,
    score: pd.Series,
    raw: pd.Series,
    attribution: str,
    inputs: list[str],
    neutral_band: float,
) -> SignalResult:
    """Assemble a :class:`SignalResult`, deriving latest direction + asof from ``score``."""
    score = score.dropna()
    asof = pd.Timestamp(score.index.max()) if not score.empty else pd.NaT
    latest = float(score.iloc[-1]) if not score.empty else float("nan")
    return SignalResult(
        name=name,
        category=category,
        score=score,
        raw=raw,
        direction=direction_of(latest, neutral_band),
        attribution=attribution,
        asof=asof,
        inputs=inputs,
    )
