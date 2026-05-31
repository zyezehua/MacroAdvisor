"""Causal transforms: leakage (causality) guarantees + numeric correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.signals import transform as tf


def _series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-05-29", periods=n)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)


@pytest.mark.parametrize("fn", [
    lambda s: tf.roll_z(s, 60),
    lambda s: tf.sma(s, 50),
    lambda s: tf.wilder_rsi(s, 14),
    lambda s: tf.realized_vol(s, 21),
])
def test_causality_no_lookahead(fn):
    """Value at date t must be identical whether computed on full or truncated history."""
    s = _series()
    cut = s.index[150]
    full = fn(s)
    trunc = fn(s.loc[:cut])
    assert full.loc[cut] == pytest.approx(trunc.loc[cut], nan_ok=True)


def test_roll_z_matches_manual():
    s = _series(seed=3)
    w = 60
    z = tf.roll_z(s, w)
    t = s.index[120]
    window = s.loc[:t].iloc[-w:]
    expected = (s.loc[t] - window.mean()) / window.std()
    assert z.loc[t] == pytest.approx(expected)


def test_rsi_bounds_and_extremes():
    idx = pd.bdate_range(end="2026-05-29", periods=60)
    up = pd.Series(np.arange(1, 61, dtype=float), index=idx)     # strictly increasing
    rsi = tf.wilder_rsi(up, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    assert rsi.iloc[-1] == pytest.approx(100.0)                  # only gains -> RSI 100


def test_realized_vol_value():
    idx = pd.bdate_range(end="2026-05-29", periods=40)
    s = _series(n=40, seed=1)
    rv = tf.realized_vol(s, 21, annualize=True)
    manual = tf.log_ret(s).iloc[-21:].std() * np.sqrt(252)
    assert rv.iloc[-1] == pytest.approx(manual)
