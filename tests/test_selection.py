"""Purged inner splits keep overlapping labels out of the validation fold."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict import selection


def _dates(n_dates=120, n_sym=2):
    d = pd.bdate_range(end="2026-01-01", periods=n_dates)
    return np.repeat(d.values, n_sym)          # (date, symbol) panel: each date repeats per symbol


def test_purged_kfold_partitions_validation_and_purges_neighbours():
    dates = _dates()
    purge = 7
    splits = selection.purged_kfold(dates, n_splits=5, purge=purge)
    assert len(splits) >= 2

    rank, n = selection._row_ranks(dates)
    covered = set()
    for train_idx, val_idx in splits:
        # train/val disjoint
        assert not (set(train_idx) & set(val_idx))
        covered |= set(val_idx)
        # the gap between the validation block and the nearest training row exceeds the purge
        vlo, vhi = rank[val_idx].min(), rank[val_idx].max()
        gaps = np.r_[vlo - rank[train_idx][rank[train_idx] < vlo],
                     rank[train_idx][rank[train_idx] > vhi] - vhi]
        if len(gaps):
            assert gaps.min() > purge
    # every row lands in exactly one validation fold (full out-of-fold cover)
    assert covered == set(range(len(dates)))


def test_purged_holdout_is_time_ordered_with_gap():
    dates = _dates()
    purge = 6
    train_idx, val_idx = selection.purged_holdout(dates, purge=purge, val_frac=0.25)
    rank, _ = selection._row_ranks(dates)
    assert len(train_idx) and len(val_idx)
    # validation is the most recent block; training ends a purge gap before it
    assert rank[val_idx].min() > rank[train_idx].max()
    assert rank[val_idx].min() - rank[train_idx].max() > purge
