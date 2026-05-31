"""Composite market-stress index (Phase 1) with per-component decomposition."""
from macro_advisor.stress.index import (
    ComponentContribution,
    StressResult,
    compute_stress,
)

__all__ = ["compute_stress", "StressResult", "ComponentContribution"]
