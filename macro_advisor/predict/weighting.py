"""Leakage-safe training sample weights (Phase 4, workstream C).

Two effects, combined multiplicatively and normalised to mean 1 over each training fold:

  * **Recency decay** — an exponential half-life in trading days, so the model leans on the most
    recent (most regime-relevant) history without discarding the long tail. ``halflife=0`` is off.
  * **Label uniqueness** (López-de-Prado style) — forward-return labels span overlapping windows
    ``(t, t+h]``; heavily-overlapping rows carry near-duplicate information. Each row is weighted by
    its *average uniqueness* = mean inverse concurrency over its own label window, so a cluster of
    overlapping rows collectively counts about as much as one independent observation.

Both terms are computed only from the rows handed in (a single training fold) and from past/sample
dates — nothing here looks forward of the fold, so the OOS guarantee is untouched. The factory
returns a ``weight_fn(dates, y)`` matching the hook the walk-forward engine calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _date_ranks(dates) -> tuple[np.ndarray, int]:
    idx = pd.DatetimeIndex(np.asarray(dates))
    uniq = idx.unique().sort_values()
    rank_of = pd.Series(np.arange(len(uniq)), index=uniq)
    return idx.map(rank_of).to_numpy(), len(uniq)


def _avg_uniqueness(row_rank: np.ndarray, n_dates: int, horizon: int) -> np.ndarray:
    """Average label uniqueness per *date rank*, mapped back to rows.

    A label at rank ``r`` occupies forward ranks ``r+1 .. r+h``. Concurrency at rank ``k`` is the
    number of labels whose window covers ``k``; uniqueness of ``r`` is the mean of ``1/concurrency``
    over its window. Computed on the unique-date axis (symbols share the calendar) for O(D) cost.
    """
    h = max(int(horizon), 1)
    D = n_dates
    # concurrency[k] = #{ r : r+1 <= k <= r+h } = #{ r in [k-h, k-1] } among present labels.
    present = np.bincount(row_rank, minlength=D).astype(float)
    present = (present > 0).astype(float)               # one label per date on the date axis
    csum = np.concatenate([[0.0], np.cumsum(present)])
    conc = np.zeros(D)
    for k in range(D):
        lo, hi = max(0, k - h), k - 1                   # r range whose window covers k
        if hi >= lo:
            conc[k] = csum[hi + 1] - csum[lo]
    conc = np.where(conc > 0, conc, 1.0)
    inv = 1.0 / conc
    inv_csum = np.concatenate([[0.0], np.cumsum(inv)])
    uniq_by_rank = np.ones(D)
    for r in range(D):
        lo, hi = r + 1, min(r + h, D - 1)               # this label's forward window
        if hi >= lo:
            uniq_by_rank[r] = (inv_csum[hi + 1] - inv_csum[lo]) / (hi - lo + 1)
    return uniq_by_rank[row_rank]


def make_weight_fn(cfg_block: dict, horizon: int):
    """Build ``weight_fn(dates, y) -> np.ndarray`` from the ``predict.sample_weight`` config.

    Returns ``None`` when weighting is disabled — the engine then trains unweighted.
    """
    if not cfg_block or not cfg_block.get("enabled"):
        return None
    halflife = float(cfg_block.get("recency_halflife_days", 0) or 0)
    use_uniq = bool(cfg_block.get("uniqueness", True))

    def weight_fn(dates, y=None) -> np.ndarray:
        row_rank, n = _date_ranks(dates)
        w = np.ones(len(row_rank), dtype=float)
        if halflife > 0:
            age = (n - 1) - row_rank                     # 0 = most recent date in the fold
            w *= np.power(0.5, age / halflife)
        if use_uniq:
            w *= _avg_uniqueness(row_rank, n, horizon)
        m = w.mean()
        return w / m if m > 0 else w                     # normalise to mean 1 (keeps reg strength)

    return weight_fn
