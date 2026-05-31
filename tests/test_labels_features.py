"""Labels + per-asset feature construction: correctness and causality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.predict import features, labels


def _price(n=200, seed=0):
    idx = pd.bdate_range(end="2026-05-29", periods=n)
    idx.name = "date"
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="X")


def test_forward_return_is_future_window():
    p = pd.Series([100.0, 110.0, 121.0, 133.1], index=pd.bdate_range("2026-01-01", periods=4))
    fwd = labels.forward_return(p, 1)
    assert fwd.iloc[0] == pytest.approx(0.10)      # 100 -> 110
    assert pd.isna(fwd.iloc[-1])                     # no future point


def test_direction_band_and_sqrt_scaling():
    fwd = pd.Series([0.05, -0.05, 0.0001, np.nan], index=pd.bdate_range("2026-01-01", periods=4))
    d = labels.direction(fwd, neutral_band=0.01, h=4)   # band = 0.01*2 = 0.02
    assert list(d.iloc[:3]) == [1, -1, 0]
    assert pd.isna(d.iloc[-1])


def test_asset_features_are_causal():
    """A feature at date t must not change when future prices are appended."""
    p = _price(220, seed=1)
    cut = p.index[150]
    full = features.asset_features(p)
    trunc = features.asset_features(p.loc[:cut])
    common = [c for c in full.columns]
    for c in common:
        assert full.loc[cut, c] == pytest.approx(trunc.loc[cut, c], nan_ok=True)


def test_stress_label_forward_change():
    lvl = pd.Series([50.0, 55.0, 60.0, 40.0], index=pd.bdate_range("2026-01-01", periods=4))
    chg = labels.stress_label(lvl, 1)
    assert chg.iloc[0] == pytest.approx(5.0)
    assert pd.isna(chg.iloc[-1])
