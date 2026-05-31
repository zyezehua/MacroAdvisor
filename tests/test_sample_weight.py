"""Training sample weights: recency decay + label uniqueness, normalised and leakage-safe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_advisor.predict import weighting


def _dates(n_dates=200, n_sym=2):
    return np.repeat(pd.bdate_range(end="2026-01-01", periods=n_dates).values, n_sym)


def test_disabled_returns_none():
    assert weighting.make_weight_fn({"enabled": False}, horizon=5) is None
    assert weighting.make_weight_fn({}, horizon=5) is None


def test_weights_normalised_and_recency_monotone():
    dates = _dates()
    fn = weighting.make_weight_fn(
        {"enabled": True, "recency_halflife_days": 40, "uniqueness": False}, horizon=5)
    w = fn(dates)
    assert np.isclose(w.mean(), 1.0)              # normalised to mean 1
    assert (w > 0).all()
    # most recent date weighted more than the oldest under recency decay
    rank, _ = weighting._date_ranks(dates)
    assert w[rank == rank.max()].mean() > w[rank == 0].mean()


def test_uniqueness_downweights_overlapping_interior():
    dates = _dates(n_dates=120)
    fn = weighting.make_weight_fn(
        {"enabled": True, "recency_halflife_days": 0, "uniqueness": True}, horizon=10)
    w = fn(dates)
    assert np.isclose(w.mean(), 1.0)
    rank, n = weighting._date_ranks(dates)
    # the very last labels have shorter (more unique) forward windows than the dense interior
    assert w[rank >= n - 2].mean() >= w[(rank > 20) & (rank < n - 20)].mean()
