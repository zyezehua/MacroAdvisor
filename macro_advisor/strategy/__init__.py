"""Custom-strategy rules-builder (Phase 3).

Compose a directional strategy from the signal library + per-asset technicals, then backtest
it in-app by reusing the Phase-2 vectorized engine. The rules are causal (signals are causal,
the engine earns next-day returns), but the *thresholds are user-chosen* — this is an explicit
in-sample exploration tool, **not** a walk-forward-validated model.
"""
from macro_advisor.strategy.evaluator import available_inputs, evaluate
from macro_advisor.strategy.library import spec_from_json, spec_to_json
from macro_advisor.strategy.spec import Rule, StrategySpec

__all__ = [
    "StrategySpec", "Rule", "evaluate", "available_inputs", "spec_to_json", "spec_from_json",
]
