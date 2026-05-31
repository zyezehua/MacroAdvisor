#!/usr/bin/env python
"""CLI to compute the signal library + composite stress index from cached data.

Reads only the local parquet/SQLite cache (run ``scripts/pull_data.py`` first to populate
it). Prints the headline stress level, per-component decomposition, the full signal table,
and the top drivers — the same numbers the dashboard renders.

Examples:
    python scripts/compute_signals.py
    python scripts/compute_signals.py --verbose
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_advisor.data import MarketStore                      # noqa: E402
from macro_advisor.signals import compute_all                   # noqa: E402
from macro_advisor.stress import compute_stress                 # noqa: E402

_DIR_GLYPH = {"risk_on": "▲ on ", "risk_off": "▼ off", "neutral": "· neu"}


def main() -> int:
    ap = argparse.ArgumentParser(description="MacroAdvisor signal + stress engine")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-n", "--top", type=int, default=8, help="number of top drivers to show")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    store = MarketStore()
    try:
        signals = compute_all(store)
        if not signals:
            print("No signals computed — is the data cache populated? "
                  "Run: python scripts/pull_data.py --full")
            return 1
        stress = compute_stress(store, signals)
    finally:
        store.close()

    print(f"\n=== MARKET STRESS: {stress.level:5.1f} / 100  [{stress.label.upper()}] "
          f"  (as of {stress.asof.date()}, {stress.n_signals} signals) ===")

    print("\n-- Component decomposition (weight x stress, sums to latent "
          f"{stress.latent:+.3f}) --")
    for c in stress.components:
        bar = _bar(c.contribution)
        print(f"  {c.component:12s} w={c.weight:4.2f}  stress={c.stress:+.2f}  "
              f"contrib={c.contribution:+.3f}  {bar}")

    print("\n-- Signals --")
    for name, s in sorted(signals.items(), key=lambda kv: (kv[1].category, kv[0])):
        glyph = _DIR_GLYPH.get(s.direction, s.direction)
        print(f"  {s.category:11s} {name:16s} {s.latest_score:+.2f} {glyph}  {s.attribution}")

    print(f"\n-- Top {args.top} drivers (most stressful first) --")
    for line in stress.top_drivers[: args.top]:
        print(f"  {line}")
    return 0


def _bar(x: float, width: int = 20) -> str:
    """A centered text bar for a value in roughly [-1, 1]."""
    n = int(round(abs(x) * width))
    n = min(n, width)
    return ("+" if x >= 0 else "-") * n


if __name__ == "__main__":
    raise SystemExit(main())
