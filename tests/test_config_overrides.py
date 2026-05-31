"""Config.with_overrides: deep-merge, immutability of the source."""
from __future__ import annotations

from macro_advisor.config import load_config


def test_with_overrides_deep_merges():
    cfg = load_config()
    base_cap = cfg.risk_budget["per_position_cap"]
    base_leverage = cfg.risk_budget["max_leverage"]

    cfg2 = cfg.with_overrides({"risk_budget": {"per_position_cap": 0.25}})
    assert cfg2.risk_budget["per_position_cap"] == 0.25
    # sibling keys in the same nested block are preserved
    assert cfg2.risk_budget["max_leverage"] == base_leverage
    # original is untouched (frozen + deep-copied)
    assert cfg.risk_budget["per_position_cap"] == base_cap


def test_empty_patch_returns_self():
    cfg = load_config()
    assert cfg.with_overrides({}) is cfg


def test_recommend_overrides_flow_through():
    cfg = load_config()
    cfg2 = cfg.with_overrides({"recommend": {"min_conviction": 0.9, "pinned_symbols": ["SPY"]}})
    assert cfg2.recommend["min_conviction"] == 0.9
    assert cfg2.recommend["pinned_symbols"] == ["SPY"]
    # an untouched recommend key survives the merge
    assert "max_ideas" in cfg2.recommend
