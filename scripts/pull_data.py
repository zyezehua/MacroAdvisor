#!/usr/bin/env python
"""CLI to run the data pipeline and print a coverage + QA summary.

Examples:
    python scripts/pull_data.py --core            # light/intraday scope
    python scripts/pull_data.py --full            # full backtest universe + Treasury curve
    python scripts/pull_data.py --smoke           # tiny fixed set, fast end-to-end check
    python scripts/pull_data.py --full --fred-extras   # also try optional FRED series
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow running as a plain script (no install needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_advisor.ingest.pipeline import DataPipeline  # noqa: E402


def _summarize(pipe: DataPipeline, results: dict) -> int:
    ok = sum(1 for r in results.values() if r.ok)
    bad = [k for k, r in results.items() if not r.ok]
    print(f"\n=== Pull summary: {ok}/{len(results)} series OK ===")
    for key, res in sorted(results.items()):
        s, e, n = res.coverage()
        flags = pipe.db.flags(key)
        flag_str = ", ".join(f"{f['code']}({f['severity']})" for f in flags) or "-"
        marker = "OK " if res.ok else "!! "
        print(f"{marker}{key:22s} {str(s):>10}->{str(e):<10} n={n:<6} {flag_str}")
    if bad:
        print(f"\nFailed/empty: {', '.join(bad)}")
    # surface error-severity QA flags across the DB
    errors = [f for f in pipe.db.flags() if f["severity"] == "error"]
    if errors:
        print(f"\n*** {len(errors)} ERROR-severity QA flag(s) — inspect before using data ***")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="MacroAdvisor data pipeline")
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--core", action="store_true", help="core stress universe")
    scope.add_argument("--full", action="store_true", help="full backtest universe + FRED")
    scope.add_argument("--smoke", action="store_true", help="tiny fixed set for a fast check")
    ap.add_argument("--start", default=None, help="override history start (YYYY-MM-DD)")
    ap.add_argument("--fred-extras", action="store_true",
                    help="also try optional FRED series (best-effort; skipped if blocked)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pipe = DataPipeline()
    try:
        if args.smoke:
            results = {}
            for sym in ("SPY", "TLT", "^VIX"):
                r = pipe.pull_equity(sym, start=args.start or "2020-01-01")
                results[r.key] = r
            # exercise the Treasury-curve path on a short window
            results.update(pipe.pull_treasury(start=args.start or "2023-01-01"))
        elif args.core:
            results = pipe.run_core(start=args.start)
        else:
            results = pipe.run_full(start=args.start, fred_extras=args.fred_extras)
        return _summarize(pipe, results)
    finally:
        pipe.close()


if __name__ == "__main__":
    raise SystemExit(main())
