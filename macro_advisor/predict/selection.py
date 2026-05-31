"""Leakage-safe model selection helpers (Phase 4).

The outer walk-forward (``walkforward``) is what makes the *backtest* out-of-sample. But several
ML-uplift steps — probability calibration, hyperparameter tuning, stacking — need their own
*inner* validation **within a training fold**. If that inner split were a naive random/K-fold,
overlapping forward-looking labels (window ``(t, t+h]``) would leak across the inner boundary and
quietly inflate the chosen hyperparameters / calibration map.

This module provides **purged** inner splits: validation folds are separated from training rows by
a ``purge`` gap of ``horizon + embargo`` trading days on *both* sides, so no training row's label
window can overlap a validation row's. Everything here operates purely on the rows handed to it
(the outer training block) and never sees the outer test block, so the outer OOS guarantee holds.

Splits are returned as positional ``(train_idx, val_idx)`` integer arrays so they drop straight
into scikit-learn's ``cv=`` argument and into out-of-fold prediction loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row_ranks(dates) -> tuple[np.ndarray, int]:
    """Map each row's date to the ordinal rank of that date among the unique sorted dates.

    Ranks are in *trading-day* units (one step per distinct date), so a ``purge`` expressed in
    trading days is just a gap in rank space — robust to multiple rows sharing a date (the panel
    is ``(date, symbol)``) and to non-contiguous calendars.
    """
    idx = pd.DatetimeIndex(np.asarray(dates))
    uniq = idx.unique().sort_values()
    rank_of = pd.Series(np.arange(len(uniq)), index=uniq)
    return idx.map(rank_of).to_numpy(), len(uniq)


def purged_kfold(dates, n_splits: int, purge: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold over *dates*: contiguous date blocks as validation, neighbours purged.

    For each of ``n_splits`` contiguous date folds, the validation set is that fold's rows and the
    training set is every row at least ``purge`` trading days *before or after* the fold — the
    purge gap on both sides removes rows whose label window overlaps the validation block. Folds
    with no usable train or val rows are skipped. Each row appears in exactly one validation fold,
    so this also yields a full out-of-fold cover (used by stacking).
    """
    if n_splits < 2:
        raise ValueError("purged_kfold needs n_splits >= 2")
    row_rank, n_dates = _row_ranks(dates)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in np.array_split(np.arange(n_dates), n_splits):
        if len(fold) == 0:
            continue
        lo, hi = int(fold[0]), int(fold[-1])
        val_idx = np.where((row_rank >= lo) & (row_rank <= hi))[0]
        train_idx = np.where((row_rank < lo - purge) | (row_rank > hi + purge))[0]
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def purged_holdout(dates, purge: int, val_frac: float = 0.25
                   ) -> tuple[np.ndarray, np.ndarray]:
    """A single time-ordered purged holdout: earliest dates train, latest ``val_frac`` validate.

    Training rows end a ``purge`` gap before the validation block begins, so the most recent — and
    therefore most calibration-relevant — rows are held out without any label overlap. Returns
    empty arrays if either side would be empty.
    """
    row_rank, n_dates = _row_ranks(dates)
    if n_dates < 2:
        return np.array([], int), np.array([], int)
    val_start = int(np.ceil(n_dates * (1.0 - val_frac)))
    val_start = min(max(val_start, 1), n_dates - 1)
    val_idx = np.where(row_rank >= val_start)[0]
    train_idx = np.where(row_rank < val_start - purge)[0]
    return train_idx, val_idx
