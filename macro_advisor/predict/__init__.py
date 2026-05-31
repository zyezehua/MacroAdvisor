"""Prediction layer (Phase 2): label construction + walk-forward OOS engine.

All labels are computed from data available at decision time only; validation uses
expanding/rolling walk-forward with purged + embargoed splits to prevent leakage.
"""
from macro_advisor.predict.features import build_panel, feature_columns
from macro_advisor.predict.walkforward import final_forecast, walk_forward

__all__ = ["build_panel", "feature_columns", "walk_forward", "final_forecast"]
