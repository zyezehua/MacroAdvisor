"""Classifier probability calibration is leakage-safe and produces valid probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_advisor.predict.models import make_model


def _xy_dates(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range(end="2026-01-01", periods=n // 2).values, 2)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = np.sign(X["f1"] - 0.5 * X["f2"] + rng.normal(0, 0.5, n)).astype(int)
    return X, pd.Series(y), dates


@pytest.mark.parametrize("name", ["linear", "gbm"])
def test_calibration_engages_and_returns_valid_probs(name):
    X, y, dates = _xy_dates()
    params = {"calibrate": {"enabled": True, "method": "isotonic", "cv_splits": 4}}
    m = make_model(name, "clf", params=params).fit(X, y, dates=dates, purge=5)
    assert type(m._fitted).__name__ == "CalibratedClassifierCV"   # calibration path used
    preds = m.predict(X)
    assert ((preds["p_up"] >= 0) & (preds["p_up"] <= 1)).all()
    assert ((preds["p_down"] >= 0) & (preds["p_down"] <= 1)).all()
    # attribution still comes from the base explainer (one column per feature)
    assert list(m.attribution(X.head(3)).columns) == ["f1", "f2"]


def test_calibration_skipped_without_dates_or_when_disabled():
    X, y, _ = _xy_dates()
    off = make_model("gbm", "clf", params={"calibrate": {"enabled": False}}).fit(X, y, dates=None)
    assert type(off._fitted).__name__ != "CalibratedClassifierCV"
    # enabled but no dates -> cannot purge -> falls back to uncalibrated
    nod = make_model("gbm", "clf", params={"calibrate": {"enabled": True}}).fit(X, y, dates=None)
    assert type(nod._fitted).__name__ != "CalibratedClassifierCV"
