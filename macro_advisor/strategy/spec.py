"""Strategy specification — a serializable description of a custom rules-based strategy.

A strategy scores each asset on each date as a weighted sum of **rule indicators**:

    score(date, asset) = Σ_i  weight_i · [ input_i(date, asset)  op_i  threshold_i ]

where the bracket is +1 when the condition holds and 0 otherwise. The per-asset direction is
``sign(score)`` (clipped to long-only if requested); a position is taken whenever the score is
non-zero. Sizing, rebalance cadence and risk caps are part of the spec so the evaluator can map
it onto the existing backtester.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPERATORS = (">", ">=", "<", "<=")
DIRECTIONS = ("long_short", "long_only")
SIZINGS = ("vol_target", "equal")
REBALANCES = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class Rule:
    input: str                 # an available input name (see strategy.evaluator.available_inputs)
    op: str                    # one of OPERATORS
    threshold: float
    weight: float = 1.0        # signed weight; negative flips the indicator's contribution

    def to_dict(self) -> dict[str, Any]:
        return {"input": self.input, "op": self.op, "threshold": self.threshold, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rule":
        return cls(input=str(d["input"]), op=str(d["op"]),
                   threshold=float(d["threshold"]), weight=float(d.get("weight", 1.0)))


@dataclass(frozen=True)
class StrategySpec:
    name: str
    universe: list[str]
    rules: list[Rule]
    direction: str = "long_short"       # long_short | long_only
    sizing: str = "vol_target"          # vol_target | equal
    rebalance: str = "weekly"           # daily | weekly | monthly
    caps: dict[str, float] = field(default_factory=dict)   # optional risk-budget overrides

    def validate(self) -> "StrategySpec":
        if not self.universe:
            raise ValueError("strategy universe is empty")
        if not self.rules:
            raise ValueError("strategy has no rules")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.sizing not in SIZINGS:
            raise ValueError(f"sizing must be one of {SIZINGS}")
        if self.rebalance not in REBALANCES:
            raise ValueError(f"rebalance must be one of {REBALANCES}")
        for r in self.rules:
            if r.op not in OPERATORS:
                raise ValueError(f"rule op must be one of {OPERATORS}, got {r.op!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "universe": list(self.universe),
            "rules": [r.to_dict() for r in self.rules],
            "direction": self.direction,
            "sizing": self.sizing,
            "rebalance": self.rebalance,
            "caps": dict(self.caps),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategySpec":
        return cls(
            name=str(d.get("name", "custom")),
            universe=[str(s) for s in d.get("universe", [])],
            rules=[Rule.from_dict(r) for r in d.get("rules", [])],
            direction=str(d.get("direction", "long_short")),
            sizing=str(d.get("sizing", "vol_target")),
            rebalance=str(d.get("rebalance", "weekly")),
            caps={str(k): float(v) for k, v in (d.get("caps") or {}).items()},
        )
