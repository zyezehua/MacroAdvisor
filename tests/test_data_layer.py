"""Offline tests for the Phase 0 data layer (no network).

Run: pytest -q   (or: python -m pytest tests/)
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from macro_advisor.config import load_config
from macro_advisor.crosscheck import check_series, reconcile_levels
from macro_advisor.storage import ParquetCache, ProvenanceDB


# --- config -----------------------------------------------------------------
def test_config_loads_and_excludes_mirror_tickers():
    cfg = load_config()
    assert cfg.risk_budget["notional_usd"] == 250000
    assert cfg.horizons["short"]["max_days"] == 5
    syms = cfg.yahoo_symbols("backtest_rates")
    # ETFs included; CBOE yield-mirror tickers must NOT be in the generic price pull
    assert "TLT" in syms
    assert "^TNX" not in syms and "^FVX" not in syms


# --- storage: parquet cache -------------------------------------------------
def _frame(start, n, val=100.0):
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({"adj_close": np.linspace(val, val + n, n)}, index=idx)


def test_parquet_roundtrip_and_merge(tmp_path):
    cache = ParquetCache(tmp_path / "prices", tmp_path / "series")
    a = _frame("2024-01-01", 10)
    cache.write("^TEST", a, kind="price")          # weird symbol -> safe filename
    b = _frame("2024-01-08", 10)                    # overlaps a
    merged = cache.write("^TEST", b, kind="price")
    out = cache.read("^TEST", kind="price")
    assert out is not None
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates             # overlap de-duplicated
    assert len(out) == len(merged)


# --- storage: provenance db -------------------------------------------------
def test_provenance_and_flags(tmp_path):
    db = ProvenanceDB(tmp_path / "p.sqlite")
    db.record_pull(key="yahoo:SPY", symbol="SPY", source="yahoo", kind="price",
                   status="ok", start_date="2020-01-01", end_date="2020-12-31",
                   n_rows=252, freq="1d")
    assert db.get_provenance("yahoo:SPY")["n_rows"] == 252
    db.raise_flag(key="yahoo:SPY", code="STALE", severity="warn", detail={"age_days": 9})
    assert db.flags("yahoo:SPY")[0]["code"] == "STALE"
    db.clear_flags("yahoo:SPY")
    assert db.flags("yahoo:SPY") == []
    db.close()


# --- crosscheck: single series ---------------------------------------------
def test_check_series_flags():
    # thin + OHLC-inconsistent + nonpositive
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "open": [10, 10, 10, 10, 10],
        "high": [9, 11, 11, 11, 11],     # row0 high < low => inconsistent
        "low": [10, 9, 9, 9, 9],
        "close": [10, 10, 10, -1, 10],   # one nonpositive
        "adj_close": [10, 10, 10, -1, 10],
    }, index=idx)
    codes = {f.code for f in check_series(df, staleness_days=4, min_history_days=250)}
    assert "THIN_HISTORY" in codes
    assert "OHLC_INCONSISTENT" in codes
    assert "NONPOSITIVE_PRICE" in codes
    assert "STALE" in codes              # 2024 data vs now


# --- crosscheck: level reconciliation --------------------------------------
def test_reconcile_levels():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    base = pd.Series(np.linspace(4.0, 4.5, 100), index=idx)
    near = base + 0.001                              # within tol
    far = base + 0.50                                # far outside tol
    assert reconcile_levels(base, near, abs_tol=0.12)[0].code == "RECONCILED"
    flags = reconcile_levels(base, far, abs_tol=0.12)
    assert flags[0].code == "LEVEL_DIVERGENCE"
    assert flags[0].severity == "error"             # 100% diverging
