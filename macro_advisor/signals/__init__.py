"""Signal library (Phase 1).

Quant/factor, technical, fundamental, policy and hard-data sentiment signals. Every
signal returns a normalized score plus a one-line attribution string so downstream
conclusions remain traceable.
"""
from macro_advisor.signals.base import (
    CATEGORIES,
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    SignalResult,
)
from macro_advisor.signals.registry import compute_all

__all__ = [
    "SignalResult",
    "compute_all",
    "CATEGORIES",
    "RISK_ON",
    "RISK_OFF",
    "NEUTRAL",
]
