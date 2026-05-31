"""Signal registry — the single entry point that runs every signal against a store.

Signals that lack their inputs (e.g. FRED extras not pulled, or a core-only refresh
without sector ETFs) return ``None`` and are skipped with a log note, so the live signal
set degrades gracefully with the data actually on hand.
"""
from __future__ import annotations

import logging

from macro_advisor.data import MarketStore
from macro_advisor.signals import credit, cross_asset, rates, technical, volatility
from macro_advisor.signals.base import SignalResult

log = logging.getLogger(__name__)

# Ordered so the output reads volatility -> credit -> rates -> momentum/technical -> cross-asset.
_FAMILIES = (volatility, credit, rates, technical, cross_asset)


def compute_all(store: MarketStore) -> dict[str, SignalResult]:
    """Compute every available signal. Returns name -> SignalResult for those that fired."""
    out: dict[str, SignalResult] = {}
    for fam in _FAMILIES:
        for fn in fam.ALL:
            try:
                res = fn(store)
            except Exception as exc:                       # never let one signal abort the rest
                log.warning("signal %s.%s failed: %s", fam.__name__, fn.__name__, exc)
                continue
            if res is None:
                log.info("signal %s skipped (inputs unavailable)", fn.__name__)
                continue
            out[res.name] = res
    return out
