#!/usr/bin/env python
"""Launch the MacroAdvisor Streamlit dashboard.

Thin wrapper so the app can be started without remembering the streamlit invocation:

    python scripts/run_dashboard.py            # equivalent to:
    streamlit run macro_advisor/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macro_advisor" / "dashboard" / "app.py"


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("streamlit is not installed. Run: pip install -r requirements.txt")
        return 1
    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    return stcli.main()  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
