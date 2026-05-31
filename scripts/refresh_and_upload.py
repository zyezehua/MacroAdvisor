#!/usr/bin/env python
"""Refresh the data cache and upload it to the Hugging Face Hub dataset repo.

This is the entry point the GitHub Actions cron calls. It reuses the existing pipeline
(``scripts/pull_data.py`` logic) to pull data, then syncs the local cache to HF Hub so the
deployed dashboard can download it.

Examples:
    python scripts/refresh_and_upload.py --scope full     # post-close: full universe + FRED
    python scripts/refresh_and_upload.py --scope core      # intraday: core stress universe
    python scripts/refresh_and_upload.py --scope core --no-upload   # local dry-run (pull only)

Upload requires a write token in the HF_TOKEN env var. Without it (and without --no-upload)
the script pulls, prints the summary, and skips the upload with a clear note.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_advisor.ingest.pipeline import DataPipeline   # noqa: E402
from macro_advisor.storage import remote                 # noqa: E402

# reuse the coverage/QA summary printer from the sibling pull script
from pull_data import _summarize                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh MacroAdvisor cache + upload to HF Hub")
    ap.add_argument("--scope", choices=("full", "core"), default="full",
                    help="full = universe + Treasury curve + FRED extras; core = stress core")
    ap.add_argument("--no-upload", action="store_true", help="pull only; skip the HF upload")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pipe = DataPipeline()
    try:
        if args.scope == "full":
            results = pipe.run_full(fred_extras=True)
        else:
            results = pipe.run_core()
        rc = _summarize(pipe, results)
    finally:
        pipe.close()

    if args.no_upload:
        print("\n[--no-upload] skipping HF upload.")
        return rc
    if not os.getenv("HF_TOKEN"):
        print("\nHF_TOKEN not set — skipping upload. Set HF_TOKEN to push the cache to HF Hub.")
        return rc

    repo = remote.upload_cache(message=f"Refresh cache ({args.scope})")
    print(f"\nUploaded cache to https://huggingface.co/datasets/{repo}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
