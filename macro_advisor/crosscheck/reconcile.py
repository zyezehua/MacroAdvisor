"""Data-quality checks.

Two families:
  * single-series checks (staleness, thin history, non-monotonic/dup index)
  * cross-source reconciliation (Yahoo vs Stooq close divergence)

Each check returns ``Flag`` records. Nothing here mutates data — the pipeline decides
how to act (warn vs block) based on severity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class Flag:
    code: str
    severity: str        # 'info' | 'warn' | 'error'
    detail: dict


def _price_col(df: pd.DataFrame) -> str:
    for c in ("adj_close", "close", "value"):
        if c in df.columns:
            return c
    return df.columns[0]


def check_series(
    df: pd.DataFrame,
    *,
    staleness_days: int,
    min_history_days: int,
) -> list[Flag]:
    """Single-series sanity checks."""
    flags: list[Flag] = []
    if df is None or df.empty:
        return [Flag("EMPTY", "error", {"reason": "no data"})]

    # thin history
    if len(df) < min_history_days:
        flags.append(Flag("THIN_HISTORY", "warn",
                          {"n_rows": int(len(df)), "min": min_history_days}))

    # staleness — last observation vs now
    last = pd.Timestamp(df.index.max())
    age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - last.to_pydatetime()).days
    if age_days > staleness_days:
        flags.append(Flag("STALE", "warn",
                          {"last": str(last.date()), "age_days": int(age_days)}))

    # duplicate / non-monotonic index
    if not df.index.is_monotonic_increasing:
        flags.append(Flag("NON_MONOTONIC", "warn", {}))
    if df.index.has_duplicates:
        flags.append(Flag("DUP_INDEX", "warn",
                          {"n_dups": int(df.index.duplicated().sum())}))

    # absurd jumps (possible bad print): >40% single-day move on a price series
    col = _price_col(df)
    if col in ("adj_close", "close"):
        rets = df[col].pct_change().abs()
        n_jumps = int((rets > 0.40).sum())
        if n_jumps:
            worst = float(rets.max())
            flags.append(Flag("PRICE_JUMP", "info",
                              {"n_jumps": n_jumps, "worst_pct": round(worst, 3)}))

    # OHLC internal consistency (keyless self-check replacing an external mirror):
    # high must bound low/open/close, and prices must be positive.
    if {"high", "low"}.issubset(df.columns):
        bad = (df["high"] < df["low"])
        for side in ("open", "close"):
            if side in df.columns:
                bad = bad | (df["high"] < df[side]) | (df["low"] > df[side])
        n_bad = int(bad.sum())
        if n_bad:
            flags.append(Flag("OHLC_INCONSISTENT", "warn", {"n_rows": n_bad}))
    if col in df.columns:
        n_nonpos = int((df[col] <= 0).sum())
        if n_nonpos:
            flags.append(Flag("NONPOSITIVE_PRICE", "error", {"n_rows": n_nonpos}))
    return flags


def reconcile_levels(
    primary: pd.Series,
    mirror: pd.Series,
    *,
    abs_tol: float,
    code: str = "LEVEL_DIVERGENCE",
) -> list[Flag]:
    """Compare two level series (e.g. Treasury 10Y vs Yahoo ^TNX) by absolute diff.

    Used for yields where the meaningful tolerance is in absolute units (bp), not
    relative. ``abs_tol`` is in the series' own units (percent for yields).
    """
    joined = pd.concat(
        [primary.rename("p"), mirror.rename("m")], axis=1, join="inner"
    ).dropna()
    if joined.empty:
        return [Flag("NO_OVERLAP", "info", {"reason": "no common dates"})]
    diff = (joined["p"] - joined["m"]).abs()
    n_div = int((diff > abs_tol).sum())
    frac = n_div / len(joined)
    if n_div:
        severity = "error" if frac > 0.05 else "warn"
        return [Flag(code, severity, {
            "n_overlap": int(len(joined)),
            "n_diverging": n_div,
            "frac_diverging": round(frac, 4),
            "max_abs_diff": round(float(diff.max()), 4),
            "abs_tol": abs_tol,
        })]
    return [Flag("RECONCILED", "info", {"n_overlap": int(len(joined))})]


def reconcile_prices(
    primary: pd.DataFrame,
    mirror: pd.DataFrame,
    *,
    rel_tol: float,
) -> list[Flag]:
    """Compare primary (Yahoo) vs mirror (Stooq) adjusted close on overlapping dates."""
    flags: list[Flag] = []
    if primary is None or primary.empty or mirror is None or mirror.empty:
        return [Flag("NO_MIRROR", "info", {"reason": "mirror unavailable"})]

    pc, mc = _price_col(primary), _price_col(mirror)
    joined = pd.concat(
        [primary[pc].rename("p"), mirror[mc].rename("m")], axis=1, join="inner"
    ).dropna()
    if joined.empty:
        return [Flag("NO_OVERLAP", "info", {"reason": "no common dates"})]

    rel = (joined["p"] - joined["m"]).abs() / joined["m"].replace(0, np.nan)
    n_div = int((rel > rel_tol).sum())
    frac = n_div / len(joined)
    if n_div:
        severity = "error" if frac > 0.05 else "warn"
        flags.append(Flag("PRICE_DIVERGENCE", severity, {
            "n_overlap": int(len(joined)),
            "n_diverging": n_div,
            "frac_diverging": round(frac, 4),
            "max_rel_diff": round(float(rel.max()), 4),
            "rel_tol": rel_tol,
        }))
    else:
        flags.append(Flag("RECONCILED", "info",
                          {"n_overlap": int(len(joined))}))
    return flags
