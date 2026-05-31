"""Serialize / deserialize strategy specs (JSON), for save-load in the dashboard.

Persistence is intentionally file-format only (a JSON blob): the dashboard keeps specs in
``st.session_state`` and offers download/upload, so nothing is written to the HF-synced data
cache. A couple of starter presets are provided to seed the rule-builder UI.
"""
from __future__ import annotations

import json

from macro_advisor.strategy.spec import Rule, StrategySpec


def spec_to_json(spec: StrategySpec, *, indent: int = 2) -> str:
    return json.dumps(spec.to_dict(), indent=indent)


def spec_from_json(text: str) -> StrategySpec:
    return StrategySpec.from_dict(json.loads(text)).validate()


def presets() -> dict[str, StrategySpec]:
    """A few starter strategies to illustrate the rule grammar."""
    return {
        "Risk-on trend (equities)": StrategySpec(
            name="Risk-on trend",
            universe=["SPY", "QQQ", "IWM"],
            rules=[
                Rule(input="stress_level", op="<", threshold=55.0, weight=1.0),
                Rule(input="px_sma_gap_200", op=">", threshold=0.0, weight=1.0),
            ],
            direction="long_only", sizing="vol_target", rebalance="weekly",
        ),
        "Defensive duration (rates)": StrategySpec(
            name="Defensive duration",
            universe=["IEF", "TLT"],
            rules=[
                Rule(input="stress_level", op=">", threshold=60.0, weight=1.0),
            ],
            direction="long_only", sizing="vol_target", rebalance="weekly",
        ),
    }
